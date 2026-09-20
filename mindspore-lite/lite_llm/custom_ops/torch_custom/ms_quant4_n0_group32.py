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
"""Torch/ONNX W4A16 interface and canonical Q4_0 compact phase4 NZF weights.

Canonical blocks are [N, K/32, 18]: little-endian FP16 scale, then 16 bytes
pairing unsigned codes j and j+16. NZF contains all packed cells followed by
all scale cells, in N64/K1024 cell order, without padding. Each live nc/kc
cell contains [kc/16, nc/16] fractals and [nc, kc/32] FP16 scales.

For a flattened signed-int4 16x16 fractal f, the phase4 wire permutation is
byte[4*i+p] = (f[64*p+2*i] & 15) | ((f[64*p+2*i+1] & 15) << 4).
Repacking preserves every code and scale bit with bounded cell scratch.
"""

from __future__ import annotations

import numpy as np
import torch


def parse_input1_shape(input1_shape):
    """Parse the ONNX attribute string or logical [K, N] sequence."""
    if isinstance(input1_shape, str):
        cleaned = input1_shape.replace("[", " ").replace("]", " ")
        input1_shape = tuple(int(part) for part in cleaned.replace(",", " ").split())
    return MsQuant4N0Group32._shape(input1_shape)  # pylint: disable=protected-access


# Inference-only autograd.Function: backward/JVP hooks are intentionally absent.
class MsQuant4N0Group32(torch.autograd.Function):  # pylint: disable=abstract-method
    """W4A16 group-32 MatMul and its single weight-preparation API."""


    @staticmethod
    def _shape(input1_shape):
        """Validate the logical matrix shape without truncating dimensions."""
        if not isinstance(input1_shape, (tuple, list)) or len(input1_shape) != 2:
            raise ValueError("input1_shape must contain K and N")
        if any(not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
               for value in input1_shape):
            raise ValueError("input1_shape dimensions must be integers")
        k_dim, n_dim = map(int, input1_shape)
        if k_dim <= 0 or k_dim % 32 or n_dim <= 0 or n_dim % 16:
            raise ValueError("input1_shape requires positive K%32==0 and N%16==0")
        return k_dim, n_dim

    @staticmethod
    def weight_blob_size(k_dim: int, n_dim: int) -> int:
        """Return the exact compact UINT8 length for logical [K, N]."""
        k_dim, n_dim = MsQuant4N0Group32._shape((k_dim, n_dim))
        return n_dim * k_dim // MsQuant4N0Group32.GROUP_SIZE * 18

    @staticmethod
    def _byte_stream(data, size):
        """Accept byte-preserving input only, including read-only/strided arrays."""
        if isinstance(data, bytes):
            raw = np.frombuffer(data, dtype=np.uint8)
        elif isinstance(data, np.ndarray) and data.dtype == np.uint8:
            raw = np.ascontiguousarray(data).reshape(-1)
        else:
            raise ValueError("weight data must be bytes or a UINT8 ndarray")
        if raw.size != size:
            raise ValueError("weight byte size does not match input1_shape")
        return raw

    @staticmethod
    def _cells(k_dim, n_dim):
        """Yield live cell bounds and their packed/scale byte ranges."""
        for n0 in range(0, n_dim, MsQuant4N0Group32.N_TILE):
            nc = min(MsQuant4N0Group32.N_TILE, n_dim - n0)
            for k0 in range(0, k_dim, MsQuant4N0Group32.K_SUB):
                kc = min(MsQuant4N0Group32.K_SUB, k_dim - k0)
                packed = n0 * k_dim // 2 + nc * k0 // 2
                scale = n_dim * k_dim // 2 + n0 * k_dim // 16 + nc * k0 // 16
                yield n0, k0, nc, kc, slice(packed, packed + nc * kc // 2), slice(scale, scale + nc * kc // 16)

    @staticmethod
    def repack_q4_0_to_nzf(blocks: bytes | np.ndarray, input1_shape: tuple[int, int] | list[int]) -> np.ndarray:
        """Losslessly reorder canonical Q4_0 blocks to compact phase4 NZF.

        Codes become signed wire nibbles via XOR 8; raw scale bytes are copied,
        including signed zero and NaN payloads. Scratch is at most one N64/K1024
        cell, not a matrix-sized expansion of the packed UINT8 input.
        """
        k_dim, n_dim = MsQuant4N0Group32._shape(input1_shape)
        size = MsQuant4N0Group32.weight_blob_size(k_dim, n_dim)
        canonical = MsQuant4N0Group32._byte_stream(blocks, size).reshape(n_dim, k_dim // 32, 18)
        result = np.empty(size, dtype=np.uint8)
        scratch = np.empty(min(n_dim, 64) * min(k_dim, 1024), dtype=np.uint8)
        for n0, k0, nc, kc, packed_range, scale_range in MsQuant4N0Group32._cells(k_dim, n_dim):
            source = canonical[n0:n0 + nc, k0 // 32:(k0 + kc) // 32]
            tile = scratch[:nc * kc].reshape(nc, kc // 32, 32)
            np.bitwise_and(source[..., 2:], 15, out=tile[..., :16])
            np.right_shift(source[..., 2:], 4, out=tile[..., 16:])
            tile ^= np.uint8(8)
            phases = tile.reshape(nc // 16, 16, kc // 16, 16).transpose(2, 0, 1, 3)
            phases = phases.reshape(kc // 16, nc // 16, 4, 64)
            packed = result[packed_range].reshape(kc // 16, nc // 16, 32, 4)
            np.left_shift(phases[..., 1::2].swapaxes(-2, -1), 4, out=packed)
            np.bitwise_or(packed, phases[..., ::2].swapaxes(-2, -1), out=packed)
            result[scale_range].reshape(nc, kc // 32, 2)[:] = source[..., :2]
        return result

    @staticmethod
    def quantize_q4_0_blocks(weight: np.ndarray) -> np.ndarray:
        """Quantize finite floating [K, N] with llama.cpp Q4_0 reference math.

        First signed absolute maximum determines d=vmax/-8. Codes use FP32
        reciprocal/multiply/add and truncation, before d is rounded to FP16.
        All-zero groups store -0 scales and offset-binary code 8.
        """
        matrix = np.asarray(weight)
        if matrix.ndim != 2 or not np.issubdtype(matrix.dtype, np.floating):
            raise ValueError("weight must be a floating matrix [K, N]")
        k_dim, n_dim = MsQuant4N0Group32._shape(matrix.shape)
        matrix = matrix.astype(np.float32, copy=False)
        if not np.all(np.isfinite(matrix)):
            raise ValueError("weight must contain only finite FP32 values")
        groups = np.ascontiguousarray(matrix.T).reshape(n_dim, k_dim // 32, 32)
        extrema = np.argmax(np.abs(groups), axis=2)
        vmax = np.take_along_axis(groups, extrema[..., None], axis=2)[..., 0]
        vmax = np.where(vmax == np.float32(0), np.float32(0), vmax)
        scales = vmax / np.float32(-8)
        reciprocal = np.zeros_like(scales)
        np.divide(np.float32(1), scales, out=reciprocal, where=scales != 0)
        shifted = groups * reciprocal[..., None]
        shifted += np.float32(8.5)
        codes = shifted.astype(np.uint8)
        np.minimum(codes, np.uint8(15), out=codes)
        blocks = np.empty((n_dim, k_dim // 32, 18), dtype=np.uint8)
        blocks[..., :2] = scales.astype("<f2").view(np.uint8).reshape(n_dim, k_dim // 32, 2)
        blocks[..., 2:] = codes[..., :16] | (codes[..., 16:] << 4)
        return blocks.reshape(-1)

    @staticmethod
    def quantize_weight_g32_4bit(weight: np.ndarray) -> np.ndarray:
        """Quantize floating [K, N] to canonical Q4_0 and pack compact NZF."""
        blocks = MsQuant4N0Group32.quantize_q4_0_blocks(weight)
        return MsQuant4N0Group32.repack_q4_0_to_nzf(blocks, np.shape(weight))

    @staticmethod
    def dequantize_weight_g32_4bit(blob: np.ndarray, input1_shape: tuple[int, int] | list[int]) -> np.ndarray:
        """Decode compact phase4 NZF to FP16 [K, N] for eager evaluation."""
        k_dim, n_dim = MsQuant4N0Group32._shape(input1_shape)
        raw = MsQuant4N0Group32._byte_stream(blob, MsQuant4N0Group32.weight_blob_size(k_dim, n_dim))
        result = np.empty((k_dim, n_dim), dtype=np.float16)
        scratch = np.empty(min(n_dim, 64) * min(k_dim, 1024), dtype=np.uint8)
        for n0, k0, nc, kc, packed_range, scale_range in MsQuant4N0Group32._cells(k_dim, n_dim):
            phase_bytes = raw[packed_range].reshape(kc // 16, nc // 16, 32, 4).swapaxes(-2, -1)
            phases = scratch[:nc * kc].reshape(kc // 16, nc // 16, 4, 64)
            np.bitwise_and(phase_bytes, 15, out=phases[..., ::2])
            np.right_shift(phase_bytes, 4, out=phases[..., 1::2])
            rows = phases.reshape(kc // 16, nc // 16, 16, 16).transpose(1, 2, 0, 3).reshape(nc, kc)
            signed = (rows.astype(np.int8) ^ 8) - 8
            values = signed.reshape(nc, kc // 32, 32).astype(np.float32)
            scales = raw[scale_range].view("<f2").reshape(nc, kc // 32)
            values *= scales[..., None].astype(np.float32)
            result[k0:k0 + kc, n0:n0 + nc] = values.reshape(nc, kc).T
        return result

    @staticmethod
    def forward(ctx, x, weight, input1_shape):  # pylint: disable=arguments-differ
        """Evaluate MatMul with the same compact blob consumed by the device."""
        del ctx
        if x.dtype != torch.float16 or weight.dtype != torch.uint8:
            raise TypeError("x must be FP16 and weight must be UINT8")
        if x.ndim < 2 or weight.ndim != 1:
            raise ValueError("x rank must be >= 2 and weight rank must be 1")
        k_dim, n_dim = parse_input1_shape(input1_shape)
        if x.shape[-1] != k_dim:
            raise ValueError("x last dimension must equal input1_shape[0]")
        dequant = MsQuant4N0Group32.dequantize_weight_g32_4bit(weight.cpu().numpy(), (k_dim, n_dim))
        result = torch.matmul(x.to(torch.float32), torch.from_numpy(dequant).to(
            device=x.device, dtype=torch.float32
        ))
        return result.to(torch.float16)

    @staticmethod
    def symbolic(g, x, weight, input1_shape):
        """Preserve the custom ONNX op, two inputs and logical shape attribute."""
        shape = parse_input1_shape(input1_shape)
        output = g.op(
            "custom::MsQuant4N0Group32", x, weight, input1_shape_s=f"{shape[0]},{shape[1]}"
        )
        x_sizes = x.type().sizes()
        if x_sizes is not None:
            output.setType(output.type().with_dtype(torch.float16).with_sizes(list(x_sizes[:-1]) + [shape[1]]))
        return output
