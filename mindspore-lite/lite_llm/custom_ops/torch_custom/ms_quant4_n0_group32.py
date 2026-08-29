# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Torch eager and ONNX definitions for ``MsQuant4N0Group32``."""

from __future__ import annotations

import numpy as np
import torch


GROUP_SIZE = 32
# S1 blob v2 layout constants (DESIGN.md Contract, Q10)
F = 16               # fractal 行列（NZ 分形 16×16）
BASE_K = 256         # kt：mad 的 K tile
K_SUB = 1024         # K-sub tile（packed/scales 块步长）
BATCH_ROWS = 64      # dequant 批处理行数（N 方向）


def parse_input1_shape(input1_shape):
    """Parse the ``input1_shape`` attribute into ``(K, N)``.

    Accepts a ``"K,N"`` string (optionally wrapped in ``[...]``) or a
    ``(K, N)`` sequence, as produced by the ONNX plugin attribute.
    """
    if isinstance(input1_shape, str):
        cleaned = input1_shape.replace("[", " ").replace("]", " ")
        parts = [part for part in cleaned.replace(",", " ").split() if part]
        if len(parts) != 2:
            raise ValueError("input1_shape must contain K and N")
        return tuple(map(int, parts))
    k_dim, n_dim = map(int, input1_shape)
    return k_dim, n_dim


def quantize_pack(weight: np.ndarray) -> np.ndarray:
    """S1 blob v2 repack entry (GGUF weight export). Alias of the v2 quantize."""
    return quantize_weight_g32_4bit(weight)


def quantize_weight_g32_4bit(weight: np.ndarray) -> np.ndarray:
    """Quantize a [K, N] float weight into the S1 blob v2 layout.

    Numerically identical to v1 (same g32 scale and quantized values); only
    the storage layout changes (DESIGN.md Contract, Q10):

    ``blob v2 = [ packed | scales ]``

    - packed: signed int8 in value domain -8..7, stored as raw bytes in the
      UINT8 blob, **full byte per K-ordered value** (v1 的 nibble 打包与 2
      值粒度交织全部离线解出)，块序 ``[N-tile][K-sub tile][row(64)]
      [k(K_SUB, K 序)]``，不足处按 K_SUB / 64 补齐（kernel 用 ktSize/nCnt
      截断，不读填充区）。Zero-centering is done offline so the kernel
      needs Cast(s8->half)+Mul only.
    - scales: fp16, ``[N-tile][K-sub][row(64)][group(32)]``，g32 粒度不变。
    """
    matrix = np.asarray(weight, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("weight must have shape [K, N]")
    k_dim, n_dim = matrix.shape
    if k_dim <= 0 or k_dim % GROUP_SIZE != 0:
        raise ValueError("K must be a positive multiple of 32")
    if n_dim <= 0 or n_dim % 16 != 0:
        raise ValueError("N must be a positive multiple of 16")

    rows = np.ascontiguousarray(matrix.T)  # [N, K]
    groups = rows.reshape(n_dim, k_dim // GROUP_SIZE, GROUP_SIZE)
    scales_f32 = np.max(np.abs(groups), axis=2) / 7.0
    scales_f32 = np.where(scales_f32 == 0.0, 1.0, scales_f32)
    scales = scales_f32.astype(np.float16)
    # Store centered signed values as raw int8 bytes.  The external tensor
    # remains UINT8 for the ONNX/OM contract; the kernel reinterprets the
    # payload as int8.  This removes the runtime Adds(-8) pass.
    quant = np.clip(
        np.rint(groups / scales_f32[..., None]).astype(np.int32),
        -8,
        7,
    ).astype(np.int8)  # [N, K/32, 32]

    ksubs = (k_dim + K_SUB - 1) // K_SUB
    ntiles = (n_dim + BATCH_ROWS - 1) // BATCH_ROWS
    q = quant.reshape(n_dim, k_dim)  # [N, K] K 序全字节

    # packed [ntiles][ksubs][64][K_SUB]，K 序，按 K_SUB 补齐
    packed = np.zeros((ntiles, ksubs, BATCH_ROWS, K_SUB), dtype=np.int8)
    for t in range(ntiles):
        row0 = t * BATCH_ROWS
        n_here = min(BATCH_ROWS, n_dim - row0)
        for s in range(ksubs):
            k0 = s * K_SUB
            k_here = min(K_SUB, k_dim - k0)
            packed[t, s, :n_here, :k_here] = q[row0 : row0 + n_here, k0 : k0 + k_here]

    # scales [ntiles][ksubs][64][32] fp16，g32 粒度，按 K_SUB 补齐
    groups_per_sub = K_SUB // GROUP_SIZE
    scale_flat = scales.reshape(n_dim, k_dim // GROUP_SIZE)  # [N, K/32]
    sc = np.zeros((ntiles, ksubs, BATCH_ROWS, groups_per_sub), dtype=np.float16)
    for t in range(ntiles):
        row0 = t * BATCH_ROWS
        n_here = min(BATCH_ROWS, n_dim - row0)
        for s in range(ksubs):
            g0 = s * groups_per_sub
            g_here = min(groups_per_sub, k_dim // GROUP_SIZE - g0)
            sc[t, s, :n_here, :g_here] = scale_flat[row0 : row0 + n_here, g0 : g0 + g_here]

    # Keep the public blob dtype UINT8 while preserving the int8 bit pattern.
    return np.concatenate(
        [packed.view(np.uint8).reshape(-1), sc.view(np.uint8).reshape(-1)]
    ).astype(np.uint8, copy=False)


def dequantize_weight_g32_4bit(
    blob: np.ndarray, input1_shape: tuple[int, int] | list[int]
) -> np.ndarray:
    """Dequantize the S1 v2 blob back to the FP16 [K, N] weight (golden)."""
    k_dim, n_dim = map(int, input1_shape)
    if k_dim <= 0 or k_dim % GROUP_SIZE != 0 or n_dim <= 0 or n_dim % 16 != 0:
        raise ValueError("input1_shape requires K%32==0 and N%16==0")
    raw = np.ascontiguousarray(blob, dtype=np.uint8).reshape(-1)
    ksubs = (k_dim + K_SUB - 1) // K_SUB
    ntiles = (n_dim + BATCH_ROWS - 1) // BATCH_ROWS
    packed_bytes = ntiles * ksubs * BATCH_ROWS * K_SUB
    scale_bytes = ntiles * ksubs * BATCH_ROWS * (K_SUB // GROUP_SIZE) * 2
    if raw.size != packed_bytes + scale_bytes:
        raise ValueError("v2 packed weight byte size does not match input1_shape")
    # The packed section is centered int8 represented as raw UINT8 bytes.
    packed = raw[:packed_bytes].view(np.int8).reshape(
        ntiles, ksubs, BATCH_ROWS, K_SUB
    )
    scales = raw[packed_bytes:].view(np.float16).reshape(
        ntiles, ksubs, BATCH_ROWS, K_SUB // GROUP_SIZE
    )
    rows = np.empty((n_dim, k_dim), dtype=np.float16)
    for t in range(ntiles):
        row0 = t * BATCH_ROWS
        n_here = min(BATCH_ROWS, n_dim - row0)
        for s in range(ksubs):
            k0 = s * K_SUB
            k_here = min(K_SUB, k_dim - k0)
            g_here = (k_here + GROUP_SIZE - 1) // GROUP_SIZE
            vals = packed[t, s, :n_here, :k_here].astype(np.float32)
            sc = scales[t, s, :n_here, :g_here].astype(np.float32)
            grp = np.repeat(sc, GROUP_SIZE, axis=1)[:, :k_here]
            rows[row0 : row0 + n_here, k0 : k0 + k_here] = (
                vals * grp
            ).astype(np.float16)
    return rows.T.copy()

class MsQuant4N0Group32(torch.autograd.Function):
    """W4A16 group-32 weight-only MatMul."""

    @staticmethod
    def forward(ctx, x, weight, input1_shape):
        """forward: helper."""
        del ctx
        if x.dtype != torch.float16 or weight.dtype != torch.uint8:
            raise TypeError("x must be FP16 and weight must be UINT8")
        if x.ndim < 2 or weight.ndim != 1:
            raise ValueError("x rank must be >= 2 and weight rank must be 1")
        if isinstance(input1_shape, str):
            cleaned = input1_shape.replace("[", " ").replace("]", " ")
            parts = [part for part in cleaned.replace(",", " ").split() if part]
            if len(parts) != 2:
                raise ValueError("input1_shape must contain K and N")
            k_dim, n_dim = map(int, parts)
        else:
            k_dim, n_dim = map(int, input1_shape)
        if x.shape[-1] != k_dim:
            raise ValueError("x last dimension must equal input1_shape[0]")
        dequant = dequantize_weight_g32_4bit(weight.cpu().numpy(), (k_dim, n_dim))
        result = torch.matmul(x.to(torch.float32), torch.from_numpy(dequant).to(
            device=x.device, dtype=torch.float32
        ))
        return result.to(torch.float16)

    @staticmethod
    def symbolic(g, x, weight, input1_shape):
        """symbolic: helper."""
        if isinstance(input1_shape, str):
            cleaned = input1_shape.replace("[", " ").replace("]", " ")
            shape = [int(part) for part in cleaned.replace(",", " ").split()]
        else:
            shape = [int(value) for value in input1_shape]
        shape_string = f"{shape[0]},{shape[1]}"
        output = g.op(
            "custom::MsQuant4N0Group32",
            x,
            weight,
            input1_shape_s=shape_string,
        )
        x_sizes = x.type().sizes()
        if x_sizes is not None:
            output.setType(
                output.type().with_dtype(torch.float16).with_sizes(
                    list(x_sizes[:-1]) + [shape[1]]
                )
            )
        return output
