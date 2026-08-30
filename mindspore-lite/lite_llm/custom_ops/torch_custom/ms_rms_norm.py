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
"""Torch eager and ONNX adapters for ``custom::MsRmsNorm``."""

from __future__ import annotations

import math

import torch


class MsRmsNorm(torch.autograd.Function):
    """FP16 RMSNorm over the last dimension, with optional gamma."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w: torch.Tensor | None = None,
        epsilon: float = 1.0e-6,
    ) -> torch.Tensor:
        """forward: helper."""
        del ctx
        if x.dtype != torch.float16:
            raise TypeError(f"x must be float16, got {x.dtype}")
        if x.ndim == 0 or x.shape[-1] <= 0:
            raise ValueError("x must have rank >= 1 and a non-empty last dimension")
        if x.shape[-1] % 16 != 0:
            raise ValueError("the last x dimension must be a multiple of 16")
        if x.shape[-1] > 8192:
            raise ValueError("the last x dimension must be <= 8192")
        if w is not None:
            if w.dtype != torch.float16:
                raise TypeError(f"w must be float16, got {w.dtype}")
            if w.ndim != 1 or w.shape[0] != x.shape[-1]:
                raise ValueError(f"w must be [{x.shape[-1]}], got {list(w.shape)}")
        epsilon = float(epsilon)
        if not math.isfinite(epsilon) or epsilon < 0.0:
            raise ValueError("epsilon must be finite and non-negative")

        x_fp32 = x.to(torch.float32)
        mean_square = torch.mean(x_fp32 * x_fp32, dim=-1, keepdim=True)
        y = x_fp32 * torch.rsqrt(mean_square + epsilon)
        if w is not None:
            y = y * w.to(torch.float32)
        return y.to(torch.float16)

    @staticmethod
    def symbolic(g, x, w=None, epsilon: float = 1.0e-6):
        if w is None:
            output = g.op("custom::MsRmsNorm", x, epsilon_f=epsilon)
        else:
            output = g.op("custom::MsRmsNorm", x, w, epsilon_f=epsilon)
        output.setType(
            output.type().with_dtype(torch.float16).with_sizes(x.type().sizes())
        )
        return output
