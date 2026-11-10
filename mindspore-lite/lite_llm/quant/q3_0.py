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
"""lite_llm-owned Q3_0 (GGUF type 41) wire format and block quantization.

The operator package (MsQuant3N0Group32) owns memory layout / offline_reorder only.
Export and GGUF producers call quantize_q3_0_blocks here, then
MsQuant3N0Group32.offline_reorder for NPU planes. GGUF readers and writers call
register_private_q3_0_type here before handling type-41 tensors.
"""

from __future__ import annotations

from enum import IntEnum
from importlib import import_module

import numpy as np

GGUF_TYPE = 41  # lite_llm-private GGUF tensor type
BLOCK_SIZE = 32  # values per block
SCALE_BYTES = 2
QH_BYTES = 4
QS_BYTES = 8
BLOCK_BYTES = SCALE_BYTES + QH_BYTES + QS_BYTES  # bytes per block (==14)
QH_OFFSET = SCALE_BYTES
QS_OFFSET = SCALE_BYTES + QH_BYTES


def register_private_q3_0_type():
    """Extend the GGUF enum without overwriting an upstream wire type.

    GGUF has no public registration API. Replace its module-level enum references
    together, preserving existing integer values and quantization tables. Never
    infer the need for registration from a parser exception's wording.
    """
    import gguf

    original = gguf.GGMLQuantizationType
    members = original.__members__
    occupied = [member for member in members.values() if member.value == GGUF_TYPE]
    if occupied:
        if occupied[0].name != "Q3_0" or gguf.GGML_QUANT_SIZES.get(occupied[0]) != (BLOCK_SIZE, BLOCK_BYTES):
            raise ValueError("GGUF type 41 conflicts with private Q3_0")
        return occupied[0]
    if "Q3_0" in members or GGUF_TYPE in gguf.GGML_QUANT_SIZES:
        raise ValueError("GGUF Q3_0 name or type 41 is already occupied")
    modules = [import_module(name) for name in (
        "gguf", "gguf.constants", "gguf.gguf_reader", "gguf.gguf_writer", "gguf.quants"
    )]
    extended = IntEnum("GGMLQuantizationType", {
        **{name: member.value for name, member in members.items()}, "Q3_0": GGUF_TYPE
    })
    gguf.GGML_QUANT_SIZES[extended.Q3_0] = (BLOCK_SIZE, BLOCK_BYTES)
    for module in modules:
        if getattr(module, "GGMLQuantizationType", None) is original:
            module.GGMLQuantizationType = extended
    return extended.Q3_0


def _shape(input1_shape):
    """Validate [K, N] against the MsQuant3N0Group32 layout contract."""
    if not isinstance(input1_shape, (tuple, list)) or len(input1_shape) != 2:
        raise ValueError("input1_shape must be a list/tuple [K, N]")
    if any(not isinstance(v, (int, np.integer)) or isinstance(v, (bool, np.bool_))
           for v in input1_shape):
        raise ValueError("input1_shape must contain integers")
    k_dim, n_dim = map(int, input1_shape)
    if k_dim <= 0 or k_dim > 16384 or k_dim % BLOCK_SIZE or n_dim <= 0 or n_dim % 16:
        raise ValueError("input1_shape requires 0<K<=16384, K%32==0 and positive N%16==0")
    if n_dim * k_dim // BLOCK_SIZE * BLOCK_BYTES > np.iinfo(np.uint32).max:
        raise ValueError("weight exceeds kernel address limits")
    return k_dim, n_dim


def quantize_q3_0_blocks(weight: np.ndarray) -> np.ndarray:
    """Quantize finite [K, N] FP32 weights to canonical GGUF type-41 blocks.

    Numeric contract: first signed absolute maximum, d = vmax / -4, FP32
    reciprocal and trunc(x/d + 4.5), clamp to [0, 7]. Only the stored scale is
    rounded to FP16; codes use the original FP32 scale. Returns a 1-D UINT8
    ndarray of N*K/32*14 bytes (row-major N of K/32 blocks).
    """
    matrix = np.asarray(weight, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("weight must have shape [K, N]")
    k_dim, n_dim = _shape(matrix.shape)
    if not np.isfinite(matrix).all():
        raise ValueError("weight must contain only finite FP32 values")
    groups = np.ascontiguousarray(matrix.T).reshape(-1, BLOCK_SIZE)
    first = np.argmax(np.abs(groups), axis=1)
    maxima = groups[np.arange(groups.shape[0]), first]
    scales_f32 = maxima / np.float32(-4)
    scales = scales_f32.astype("<f2")
    if not np.isfinite(scales).all():
        raise ValueError("Q3 scale exceeds FP16 range")
    inverse = np.zeros(scales.shape, dtype=np.float32)
    np.divide(np.float32(1), scales_f32, out=inverse, where=scales_f32 != 0)
    codes = np.clip(np.trunc(groups * inverse[:, None] + np.float32(4.5)), 0, 7).astype(np.uint8)
    blocks = np.empty((n_dim * k_dim // BLOCK_SIZE, BLOCK_BYTES), dtype=np.uint8)
    blocks[:, :SCALE_BYTES] = scales.view(np.uint8).reshape(-1, SCALE_BYTES)
    blocks[:, QH_OFFSET:QS_OFFSET] = np.bitwise_or.reduce(
        ((codes.reshape(-1, 4, 8) >> 2) & 1) << np.arange(8, dtype=np.uint8), axis=2
    )
    blocks[:, QS_OFFSET:] = np.bitwise_or.reduce(
        (codes.reshape(-1, 8, 4) & 3) << (2 * np.arange(4, dtype=np.uint8)), axis=2
    )
    return blocks.reshape(-1)
