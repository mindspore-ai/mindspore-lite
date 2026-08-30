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
"""Torch eager and ONNX definitions for ``MsAddSoftmax``."""

from __future__ import annotations

import torch


class MsAddSoftmax(torch.autograd.Function):
    """FP16 broadcast add followed by FP32 last-axis softmax."""

    @staticmethod
    def forward(ctx, scores, mask):
        """forward: helper."""
        del ctx
        if scores.ndim < 3 or mask.ndim != scores.ndim:
            raise ValueError("scores and mask must have the same rank >= 3")
        if tuple(mask.shape[-2:]) != tuple(scores.shape[-2:]):
            raise ValueError("mask must match the final [M, K] dimensions")
        if any(dim != 1 for dim in mask.shape[:-2]):
            raise ValueError("mask leading dimensions must all be 1")
        summed = (scores + mask).to(torch.float16)
        return torch.softmax(summed.to(torch.float32), dim=-1).to(torch.float16)

    @staticmethod
    def symbolic(g, scores, mask):
        output = g.op("custom::MsAddSoftmax", scores, mask)
        output.setType(
            output.type().with_dtype(torch.float16).with_sizes(scores.type().sizes())
        )
        return output
