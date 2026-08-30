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
"""Torch eager and ONNX definitions for ``MsRotaryPosEmb``."""

from __future__ import annotations

import torch


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = torch.chunk(value, 2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class MsRotaryPosEmb(torch.autograd.Function):
    """Apply NeoX half-split RoPE to BNSD query and key tensors."""

    @staticmethod
    def forward(ctx, query, key, cos, sin):
        """forward: helper."""
        del ctx
        tensors = (query, key, cos, sin)
        if any(tensor.dtype != torch.float16 for tensor in tensors):
            raise TypeError("query, key, cos and sin must be FP16")
        if query.ndim != 4 or key.ndim != 4 or cos.ndim != 3 or sin.ndim != 3:
            raise ValueError("query/key must be rank 4 and cos/sin rank 3")
        if cos.shape != sin.shape:
            raise ValueError("cos and sin shapes must match")
        if (query.shape[0] != key.shape[0] or query.shape[2] != key.shape[2]
                or query.shape[3] != key.shape[3]):
            raise ValueError("query and key B, S and D dimensions must match")
        expected = (query.shape[0], query.shape[2], query.shape[3])
        if cos.shape != expected or query.shape[3] % 2 != 0:
            raise ValueError("cos/sin must be [B,S,D] and D must be even")
        cos_fp32 = cos.to(torch.float32).unsqueeze(1)
        sin_fp32 = sin.to(torch.float32).unsqueeze(1)
        query_fp32 = query.to(torch.float32)
        key_fp32 = key.to(torch.float32)
        query_out = query_fp32 * cos_fp32 + _rotate_half(query_fp32) * sin_fp32
        key_out = key_fp32 * cos_fp32 + _rotate_half(key_fp32) * sin_fp32
        return query_out.to(torch.float16), key_out.to(torch.float16)

    @staticmethod
    def symbolic(g, query, key, cos, sin):
        """symbolic: helper."""
        query_out, key_out = g.op(
            "custom::MsRotaryPosEmb", query, key, cos, sin, outputs=2
        )
        query_out.setType(
            query_out.type().with_dtype(torch.float16).with_sizes(
                query.type().sizes()
            )
        )
        key_out.setType(
            key_out.type().with_dtype(torch.float16).with_sizes(
                key.type().sizes()
            )
        )
        return query_out, key_out
