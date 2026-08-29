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
"""Torch eager and ONNX definitions for the ``MsScatterND`` custom operator."""

from __future__ import annotations

import torch


class MsScatterND(torch.autograd.Function):
    """Overwrite sequence slices and export a ``custom::MsScatterND`` node."""

    @staticmethod
    def forward(
        ctx,
        data: torch.Tensor,
        indices: torch.Tensor,
        updates: torch.Tensor,
        layout: str = "SND",
    ) -> torch.Tensor:
        """forward: helper."""
        del ctx
        if layout == "BNSD":
            if (data.ndim != 4 or data.shape[0] != 1 or indices.ndim != 1
                    or indices.shape[0] != 1 or updates.ndim != 4
                    or updates.shape[0] != 1
                    or updates.shape[1] != data.shape[1]
                    or updates.shape[3] != data.shape[3]):
                raise ValueError(
                    "BNSD expects data [1,N,S,H], indices [1], "
                    "and updates [1,N,L,H]"
                )
            result = data.clone()
            position = int(indices[0])
            if 0 <= position < data.shape[2]:
                rows = min(updates.shape[2], data.shape[2] - position)
                result[:, :, position:position + rows, :] = updates[:, :, :rows, :]
            return result
        if layout != "SND":
            raise ValueError("layout must be SND or BNSD")
        if data.ndim != 3:
            raise ValueError(f"data must be [S, N, H], got {tuple(data.shape)}")
        if indices.ndim != 2 or indices.shape[1] != 1:
            raise ValueError(f"indices must be [U, 1], got {tuple(indices.shape)}")
        expected_updates = (indices.shape[0], data.shape[1], data.shape[2])
        if tuple(updates.shape) != expected_updates:
            raise ValueError(
                f"updates must be {expected_updates}, got {tuple(updates.shape)}"
            )

        result = data.clone()
        seq_len = data.shape[0]
        for update_index, raw_index in enumerate(indices[:, 0].tolist()):
            sequence_index = int(raw_index)
            if sequence_index < 0:
                sequence_index += seq_len
            if 0 <= sequence_index < seq_len:
                result[sequence_index] = updates[update_index]
        return result

    @staticmethod
    def symbolic(g, data, indices, updates, layout: str = "SND"):
        output = g.op(
            "custom::MsScatterND", data, indices, updates, layout_s=layout
        )
        output.setType(
            output.type().with_dtype(torch.float16).with_sizes(data.type().sizes())
        )
        return output
