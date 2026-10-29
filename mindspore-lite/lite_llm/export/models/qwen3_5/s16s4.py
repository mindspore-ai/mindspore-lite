# Adapted from HMOS-Device-AI converter/qwen3.5-35b-a3b; see source_provenance.json.
#!/usr/bin/env python3
"""S16S4 packing and reference math for Qwen3.5 GPTQ experts."""

from __future__ import annotations

import numpy as np

from .gptq import unpack_int4


N_TILE = 128
K_TILE = 64
GROUP_SIZE = 128


def signed_weight(tensors: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return signed-S4 ``[N,K]`` values and FP16 ``[K/G,N]`` scales."""
    qweight = tensors["qweight"]
    qzeros = tensors["qzeros"]
    scales = np.ascontiguousarray(tensors["scales"], dtype=np.float16)
    quantized_nk = unpack_int4(qweight, axis=0).T
    zeros = unpack_int4(qzeros, axis=1)
    n, k = quantized_nk.shape
    if zeros.shape != scales.shape or scales.shape != (k // GROUP_SIZE, n):
        raise ValueError(
            f"inconsistent GPTQ projection: weight={quantized_nk.shape} "
            f"zeros={zeros.shape} scales={scales.shape}"
        )
    if not np.all(zeros == 8):
        raise ValueError("MatmulS16S4 adaptation requires symmetric zero point 8")
    weight_nk = quantized_nk.astype(np.int16) - 8
    return np.ascontiguousarray(weight_nk.astype(np.int8)), scales


def pack_s4_nz(weight_kn: np.ndarray) -> np.ndarray:
    """Pack logical signed-S4 ``[K,N]`` into L311 NZ ``[N,K/2]`` bytes."""
    if weight_kn.ndim != 2 or weight_kn.dtype != np.int8:
        raise ValueError("weight must be INT8 [K,N]")
    k, n = weight_kn.shape
    if n % N_TILE or k % K_TILE:
        raise ValueError(f"S4 NZ requires N%{N_TILE}=0 and K%{K_TILE}=0: {(k, n)}")
    if np.any(weight_kn < -8) or np.any(weight_kn > 7):
        raise ValueError("weight contains values outside signed int4")
    reordered = (
        weight_kn.T.reshape(n // N_TILE, N_TILE // 16, 16, k // K_TILE, K_TILE)
        .transpose(0, 3, 1, 2, 4)
        .copy()
        .reshape(-1)
    )
    low = reordered[0::2].astype(np.uint8) & np.uint8(0x0F)
    high = (reordered[1::2].astype(np.uint8) & np.uint8(0x0F)) << 4
    return np.ascontiguousarray((low | high).view(np.int8).reshape(n, k // 2))


def fixpipe_parameters(scales: np.ndarray, activation_scale: float) -> np.ndarray:
    """Encode ``[groups,N]`` scales as L311 64-bit FixPipe records."""
    if scales.ndim != 2 or scales.shape[1] % N_TILE:
        raise ValueError(f"invalid grouped scales shape: {scales.shape}")
    groups, n = scales.shape
    effective = scales.astype(np.float32) / np.float32(activation_scale)
    ordered = effective.T.reshape(n // N_TILE, N_TILE, groups).transpose(0, 2, 1)
    records = np.zeros((*ordered.shape, 2), dtype=np.float32)
    records[..., 0] = ordered
    return np.ascontiguousarray(
        records.view(np.int8).reshape((n // N_TILE, groups, N_TILE, 8))
    )


def corrected_bias(
    weight_nk: np.ndarray, scales: np.ndarray, activation_scale: float
) -> np.ndarray:
    """Compensate the activation low-byte XOR used by the L311 kernel."""
    n, k = weight_nk.shape
    groups, scale_n = scales.shape
    if scale_n != n or k != groups * GROUP_SIZE:
        raise ValueError(f"weight/scales mismatch: {weight_nk.shape}, {scales.shape}")
    grouped = weight_nk.reshape(n, groups, GROUP_SIZE)
    correction = np.zeros(n, dtype=np.float64)
    for group in range(groups):
        weight_sum = grouped[:, group].sum(axis=1, dtype=np.int64)
        correction += (
            weight_sum.astype(np.float64)
            * scales[group].astype(np.float64)
            / float(activation_scale)
        )
    return np.ascontiguousarray((128.0 * correction).astype(np.float16))


def materialize_projection(
    tensors: dict[str, np.ndarray], activation_scale: float
) -> dict[str, np.ndarray]:
    weight_nk, scales = signed_weight(tensors)
    return {
        "w": pack_s4_nz(weight_nk.T),
        "bias": corrected_bias(weight_nk, scales, activation_scale),
        "w_scale": fixpipe_parameters(scales, activation_scale),
    }


def emulate_linear(
    x: np.ndarray,
    weight_nk: np.ndarray,
    scales: np.ndarray,
    activation_scale: float,
) -> np.ndarray:
    """Mirror S16 low/high S8 MAD and per-group FP16 FixPipe accumulation."""
    quantized = np.clip(
        np.rint(x.astype(np.float32) * np.float32(activation_scale)),
        -32768,
        32767,
    ).astype(np.int16)
    unsigned = quantized.view(np.uint16)
    low = (((unsigned & 0xFF) ^ 0x80).astype(np.uint8)).view(np.int8)
    high = (quantized.astype(np.int32) >> 8).astype(np.int8)
    m, k = quantized.shape
    n, weight_k = weight_nk.shape
    groups = k // GROUP_SIZE
    if weight_k != k or scales.shape != (groups, n):
        raise ValueError("S16S4 reference shape mismatch")
    low_acc = np.zeros((m, n), dtype=np.float16)
    high_acc = np.zeros((m, n), dtype=np.float16)
    for group in range(groups):
        begin = group * GROUP_SIZE
        end = begin + GROUP_SIZE
        weights = weight_nk[:, begin:end].astype(np.int32)
        dequant = scales[group].astype(np.float32) / np.float32(activation_scale)
        low_group = (
            (low[:, begin:end].astype(np.int32) @ weights.T).astype(np.float32)
            * dequant
        ).astype(np.float16)
        high_group = (
            (high[:, begin:end].astype(np.int32) @ weights.T).astype(np.float32)
            * dequant
        ).astype(np.float16)
        low_acc = (low_acc.astype(np.float32) + low_group.astype(np.float32)).astype(
            np.float16
        )
        high_acc = (
            high_acc.astype(np.float32) + high_group.astype(np.float32)
        ).astype(np.float16)
    bias = corrected_bias(weight_nk, scales, activation_scale)
    high_scaled = (high_acc.astype(np.float32) * 256.0).astype(np.float16)
    return (
        high_scaled.astype(np.float32)
        + low_acc.astype(np.float32)
        + bias.reshape(1, n).astype(np.float32)
    ).astype(np.float16)
