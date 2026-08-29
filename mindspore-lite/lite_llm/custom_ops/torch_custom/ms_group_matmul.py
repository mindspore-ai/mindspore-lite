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
"""Torch eager and ONNX definitions for ``MsGroupMatmul``."""

from __future__ import annotations

import torch


class MsGroupMatmul(torch.autograd.Function):
    """Grouped-query-attention FP16 matrix multiplication."""

    @staticmethod
    def forward(ctx, x1, x2, trans_b):
        """forward: helper."""
        del ctx
        if x1.dtype != torch.float16 or x2.dtype != torch.float16:
            raise TypeError("x1 and x2 must be FP16")
        if x1.ndim != 4 or x2.ndim != 4:
            raise ValueError("x1 and x2 must be rank-4")
        if x1.shape[0] != x2.shape[0] or x1.shape[1] % x2.shape[1] != 0:
            raise ValueError("batch/head dimensions are incompatible")
        repeats = x1.shape[1] // x2.shape[1]
        repeated = x2.repeat_interleave(repeats, dim=1)
        rhs = repeated.transpose(-2, -1) if trans_b else repeated
        if x1.shape[-1] != rhs.shape[-2]:
            raise ValueError("matrix K dimensions are incompatible")
        return torch.matmul(x1.to(torch.float32), rhs.to(torch.float32)).to(
            torch.float16
        )

    @staticmethod
    def symbolic(g, x1, x2, trans_b):
        """symbolic: helper."""
        value = "True" if trans_b else "False"
        output = g.op("custom::MsGroupMatmul", x1, x2, trans_b_s=value)
        x1_sizes = x1.type().sizes()
        x2_sizes = x2.type().sizes()
        if x1_sizes is not None and x2_sizes is not None:
            n_dim = x2_sizes[-2] if trans_b else x2_sizes[-1]
            sizes = list(x1_sizes[:-1]) + [n_dim]
            output.setType(output.type().with_dtype(torch.float16).with_sizes(sizes))
        return output
