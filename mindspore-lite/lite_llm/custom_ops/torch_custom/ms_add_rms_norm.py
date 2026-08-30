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
"""Torch eager and ONNX definitions for ``MsAddRmsNorm``."""

from __future__ import annotations

import torch


class MsAddRmsNorm(torch.autograd.Function):
    """Residual addition followed by RMSNorm, returning norm and sum."""

    @staticmethod
    def forward(ctx, x, residual, gamma, epsilon: float = 1.0e-6):
        """forward: helper."""
        del ctx
        if x.shape != residual.shape or x.ndim < 2:
            raise ValueError("x and residual must have the same rank >= 2 and shape")
        if gamma.ndim != 1 or gamma.shape[0] != x.shape[-1]:
            raise ValueError("gamma must be [K], matching the last input dimension")
        summed = (x + residual).to(torch.float16)
        summed_fp32 = summed.to(torch.float32)
        mean_square = torch.mean(
            summed_fp32 * summed_fp32, dim=-1, keepdim=True
        )
        normalized = (
            summed_fp32
            * torch.rsqrt(mean_square + epsilon)
            * gamma.to(torch.float32)
        ).to(torch.float16)
        return normalized, summed

    @staticmethod
    def symbolic(g, x, residual, gamma, epsilon: float = 1.0e-6):
        """symbolic: helper."""
        normalized, summed = g.op(
            "custom::MsAddRmsNorm",
            x,
            residual,
            gamma,
            epsilon_f=epsilon,
            outputs=2,
        )
        sizes = x.type().sizes()
        normalized.setType(
            normalized.type().with_dtype(torch.float16).with_sizes(sizes)
        )
        summed.setType(summed.type().with_dtype(torch.float16).with_sizes(sizes))
        return normalized, summed
