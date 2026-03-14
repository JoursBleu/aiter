# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import torch
from aiter import dtypes
from aiter.ops.triton._triton_kernels.quant.quant import (
    _static_per_tensor_quant_fp8_i8_kernel,
    _dynamic_per_tensor_quant_fp8_i8_kernel,
    _dynamic_per_token_quant_fp8_i8_kernel,
    _dynamic_mxfp4_quant_kernel,
    _mxfp4_quant_op,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

__all__ = [
    "static_per_tensor_quant_fp8_i8",
    "dynamic_per_tensor_quant_fp8_i8",
    "dynamic_per_token_quant_fp8_i8",
    "dynamic_mxfp4_quant",
    "_mxfp4_quant_op",
]


_LOGGER = AiterTritonLogger()

# Single-entry output buffer cache to avoid repeated allocation in benchmark loops.
# Key: (M, N, device_index, shuffle); Value: (x_fp4, blockscale_or_shuffled)
_buf_cache = {}

# Launch config cache: pure-Python heuristic decisions cached by (M, N).
_launch_config_cache = {}


def _get_launch_config(M, N):
    """Return (BLOCK_SIZE_M, BLOCK_SIZE_N, NUM_ITER, NUM_STAGES, NUM_WARPS, grid)."""
    key = (M, N)
    cached = _launch_config_cache.get(key)
    if cached is not None:
        return cached

    # Inline cdiv and next_power_of_2 to avoid function call overhead.
    def _cdiv(a, b):
        return (a + b - 1) // b

    def _np2(x):
        p = 1
        while p < x:
            p <<= 1
        return p

    NUM_ITER = 1
    NUM_STAGES = 1

    if N <= 1024:
        if M <= 4:
            bm = 4
            bn = min(512, _np2(N))
            bn = max(32, bn)
            nw = 4 if bn >= 256 else 2
        elif M <= 16:
            bm = 8
            bn = min(256, _np2(N))
            bn = max(32, bn)
            nw = 2 if bn <= 64 else 4
        elif M <= 32:
            bm = 32
            bn = 64 if N >= 64 else 32
            nw = 2
        else:
            bm = min(32, _np2(M))
            bn = min(256, _np2(N))
            bn = max(32, bn)
            nw = 4
    elif N <= 4096:
        if M <= 8:
            bm = _np2(M)
            bn = 128
            nw = 4
        elif M <= 32:
            bm = _np2(M)
            bn = 128
            nw = 4
        elif M <= 128:
            bm = 32
            bn = 128
            nw = 4
        else:
            bm = 16
            bn = 128
            nw = 4
    else:
        if M <= 16:
            bm = _np2(M)
            bn = 128
            nw = 4
            NUM_ITER = 1
        elif M <= 64:
            bm = 32
            bn = 128
            nw = 4
            NUM_ITER = 2
        else:
            bm = 32
            bn = 128
            nw = 4
            NUM_ITER = 4
            NUM_STAGES = 2

    grid = (_cdiv(M, bm), _cdiv(N, bn * NUM_ITER))
    result = (bm, bn, NUM_ITER, NUM_STAGES, nw, grid)
    _launch_config_cache[key] = result
    return result


def _get_or_alloc_buffers(M, N, device, shuffle, MXFP4_QUANT_BLOCK_SIZE=32):
    """Return cached output buffers and pre-computed views if shapes match.

    For shuffle=True, returns (x_fp4, bs_flat, x_fp4_view, bs_view) where
    views are pre-computed to avoid per-call view() overhead.
    For shuffle=False, returns (x_fp4, bs).
    """
    dev_idx = device.index if device.index is not None else 0
    key = (M, N, dev_idx, shuffle)
    cached = _buf_cache.get(key)
    if cached is not None:
        return cached

    x_fp4 = torch.empty((M, N // 2), dtype=torch.uint8, device=device)
    if shuffle:
        _cdiv = lambda a, b: (a + b - 1) // b
        scaleN = _cdiv(N, MXFP4_QUANT_BLOCK_SIZE)
        scaleN_pad = _cdiv(scaleN, 8) * 8
        M_padded = _cdiv(M, 256) * 256
        bs = torch.empty((M_padded * scaleN_pad,), dtype=torch.uint8, device=device)
        bs.fill_(127)
        # Pre-compute views once (these share storage with the underlying tensors)
        x_fp4_view = x_fp4.view(dtypes.fp4x2)
        bs_view = bs.view(M_padded, scaleN_pad).view(dtypes.fp8_e8m0)
        result = (x_fp4, bs, x_fp4_view, bs_view)
    else:
        bs = torch.empty(
            ((N + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE, M),
            dtype=torch.uint8,
            device=device,
        ).T
        result = (x_fp4, bs)

    # Evict previous entries to bound memory; keep only this key
    _buf_cache.clear()
    _buf_cache[key] = result
    return result


def static_per_tensor_quant_fp8_i8(
    qx: torch.Tensor, x_in: torch.Tensor, scale_in: torch.Tensor
):
    """
    Quantizes tensor using the provided scale to int8 or fp8

    Parameters:
    - qx: Output tensor of same shape as x_in. Must be fp8 or int8 dtype and allocated by the caller
    - x_in: Input tensor of shape (M, N).
    - scale_in: Input Scale tensor of shape (1,) and dtype fp32

    Returns:
    - qx: Quantized output values.
    """
    _LOGGER.info(f"STAIC_PER_TENSOR_QUANT_FP8_I8: x={tuple(x_in.shape)}")
    assert scale_in.numel() == 1  # only single scale value
    rows = x_in.shape[0]
    cols = x_in.shape[1]
    NUM_COL_POW2 = triton.next_power_of_2(cols)
    grid = lambda meta: (rows,)  # noqa: E731
    _static_per_tensor_quant_fp8_i8_kernel[grid](
        qx, x_in, scale_in, cols, x_in.stride(0), NUM_COL_POW2=NUM_COL_POW2
    )

    return qx


def dynamic_per_tensor_quant_fp8_i8(
    qx: torch.Tensor, x_in: torch.Tensor, scale_out: torch.Tensor
):
    """
    Calculate per tensor scale and then uses the scale to quantize input tensor to fp8 or int8

    Parameters:
    - x_in: Input tensor of shape (M, N).
    - qx: Output tensor of same shape as x_in. Must be fp8 or int8 dtype and allocated by the caller
    - scale_out: Output scale tensor of shape (1,), dtype fp32 and allocated by the caller

    Returns:
    - qx: Quantized output values of shape (M, N) with dtype fp8 or int8
    - scale_out: Single scale value of shape (1,)
    """
    _LOGGER.info(f"DYNAMIC_PER_TENSOR_QUANT_FP8_I8: x={tuple(x_in.shape)}")
    rows = x_in.shape[0]
    cols = x_in.shape[1]
    NUM_COL_POW2 = triton.next_power_of_2(cols)
    grid = lambda meta: (rows,)  # noqa: E731
    _dynamic_per_tensor_quant_fp8_i8_kernel[grid](
        x_in,
        scale_out,
        cols,
        x_in.stride(0),
        NUM_COL_POW2=NUM_COL_POW2,
        DTYPE_MAX=(
            torch.finfo(qx.dtype).max
            if torch.is_floating_point(qx)
            else torch.iinfo(qx.dtype).max
        ),
    )

    _static_per_tensor_quant_fp8_i8_kernel[grid](
        qx, x_in, scale_out, cols, x_in.stride(0), NUM_COL_POW2=NUM_COL_POW2
    )

    return qx, scale_out


def dynamic_per_token_quant_fp8_i8(
    qx: torch.Tensor,
    x_in: torch.Tensor,
    scale_out: torch.Tensor,
):
    """
    Quantizes tensor using the provided scale

    Parameters:
    - x_in: Input tensor of shape (M, N).
    - dtype_max: Optional parameter which specifies the max value of the dtype of x_in.
    - qx: Output tensor of same shape as x_in. Must be fp8 dtype and allocated by the caller
    - scale_out: Output scale tensor of shape (M,) dtype fp32 and allocated by the caller

    Returns:
    - qx: Quantized output values.
    - scale_out: Scale tensor of shape (M, )
    """
    _LOGGER.info(f"DYNAMIC_PER_TOKEN_QUANT_FP8_I8: x={tuple(x_in.shape)}")
    rows = x_in.shape[0]
    cols = x_in.shape[1]
    NUM_COL_POW2 = triton.next_power_of_2(cols)
    grid = lambda meta: (rows,)  # noqa: E731
    _dynamic_per_token_quant_fp8_i8_kernel[grid](
        qx,
        scale_out,
        x_in,
        cols,
        x_in.stride(0),
        NUM_COL_POW2=NUM_COL_POW2,
        DTYPE_MAX=(
            torch.finfo(qx.dtype).max
            if torch.is_floating_point(qx)
            else torch.iinfo(qx.dtype).max
        ),
    )

    return qx, scale_out


def dynamic_mxfp4_quant(
    x: torch.Tensor, scaling_mode: str = "even", shuffle: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to MX FP4 format.

    Args:
        x: The input tensor, typically fp16 or bf16.
        scaling_mode: The method to calculate MX block scaling.
            - "even" (default): `even_round` in `quark.torch.quantization.utils`.
            - etc.
        shuffle: If True, produce blockscale in the shuffled layout expected
            by CK gemm_a4w4 (bpreshuffle=True), avoiding a separate
            preshuffle_scales call.
    Returns:
        A tuple of (x_fp4, blockscale_e8m0).
        When shuffle=True, blockscale is returned in shuffled layout with
        shape (M_padded, scaleN_pad) and dtype fp8_e8m0.
    """
    # _LOGGER.info removed: f-string formatting overhead in hot path
    # Assume x is 2D-Tensor for now
    M, N = x.shape

    assert (N // 2) % 2 == 0

    # This is fixed by spec for MXFP4. Do not tune this.
    MXFP4_QUANT_BLOCK_SIZE = 32

    _cdiv = lambda a, b: (a + b - 1) // b
    if shuffle:
        scaleN = _cdiv(N, MXFP4_QUANT_BLOCK_SIZE)
        scaleN_pad = _cdiv(scaleN, 8) * 8
        M_padded = _cdiv(M, 256) * 256
        x_fp4, blockscale_shuffled, x_fp4_view, bs_view = _get_or_alloc_buffers(
            M, N, x.device, True, MXFP4_QUANT_BLOCK_SIZE
        )
    else:
        x_fp4, blockscale_e8m0 = _get_or_alloc_buffers(
            M, N, x.device, False, MXFP4_QUANT_BLOCK_SIZE
        )

    # Cached launch config — avoids repeated Python branching and triton.cdiv calls.
    BLOCK_SIZE_M, BLOCK_SIZE_N, NUM_ITER, NUM_STAGES, NUM_WARPS, grid = \
        _get_launch_config(M, N)

    if shuffle:
        _dynamic_mxfp4_quant_kernel[grid](
            x,
            x_fp4,
            blockscale_shuffled,
            *x.stride(),
            *x_fp4.stride(),
            0, 0,  # bs strides unused in shuffle mode
            M=M,
            N=N,
            MXFP4_QUANT_BLOCK_SIZE=MXFP4_QUANT_BLOCK_SIZE,
            SCALING_MODE=0,
            SHUFFLE=True,
            scaleN=scaleN,
            scaleN_pad=scaleN_pad,
            M_padded=M_padded,
            NUM_ITER=NUM_ITER,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            NUM_STAGES=NUM_STAGES,
            num_warps=NUM_WARPS,
            waves_per_eu=0,
            num_stages=1,
        )
        return (x_fp4_view, bs_view)
    else:
        _dynamic_mxfp4_quant_kernel[grid](
            x,
            x_fp4,
            blockscale_e8m0,
            *x.stride(),
            *x_fp4.stride(),
            *blockscale_e8m0.stride(),
            M=M,
            N=N,
            MXFP4_QUANT_BLOCK_SIZE=MXFP4_QUANT_BLOCK_SIZE,
            SCALING_MODE=0,
            NUM_ITER=NUM_ITER,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            NUM_STAGES=NUM_STAGES,
            num_warps=NUM_WARPS,
            waves_per_eu=0,
            num_stages=1,
        )
        return (x_fp4, blockscale_e8m0)
