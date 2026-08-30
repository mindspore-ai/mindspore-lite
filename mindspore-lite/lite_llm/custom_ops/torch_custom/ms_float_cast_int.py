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
"""Torch eager and ONNX definitions for ``MsFloatCastInt``."""

from __future__ import annotations

import torch


class MsFloatCastInt(torch.autograd.Function):
    """Truncate FP16 values toward zero and saturate them to INT8."""

    @staticmethod
    def forward(ctx, x):
        del ctx
        if x.dtype != torch.float16:
            raise TypeError("x must be FP16")
        if x.ndim < 2:
            raise ValueError("x rank must be at least two")
        return torch.clamp(torch.trunc(x), -128, 127).to(torch.int8)

    @staticmethod
    def symbolic(g, x):
        output = g.op("custom::MsFloatCastInt", x)
        output.setType(
            output.type().with_dtype(torch.int8).with_sizes(x.type().sizes())
        )
        return output
