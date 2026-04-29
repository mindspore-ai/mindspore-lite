#!/usr/bin/env python3
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Ulysses Sequence Parallel communication primitive: all_to_all_4d.

Split/gather along head and sequence dimensions for multi-card attention.
Forward  (scatter=2, gather=1): [B, S/P, H, D] → [B, S, H/P, D]
Reverse  (scatter=1, gather=2): [B, S, H/P, D] → [B, S/P, H, D]
"""
import torch
import torch.distributed as dist


def all_to_all_4d(
    x: torch.Tensor,
    scatter_idx: int = 2,
    gather_idx: int = 1,
    group=None,
    use_sync: bool = False
) -> torch.Tensor:
    """
    All-to-all communication for 4D tensors.
    :param x:
    :param scatter_idx:
    :param gather_idx:
    :param group:
    :param use_sync:
    :return: all-to-all result
    """
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return x

    assert scatter_idx in (1, 2) and gather_idx in (1, 2)
    assert scatter_idx != gather_idx

    # Step 1: split scatter dim into [P, scatter_dim/P]
    x = x.unflatten(scatter_idx, (world_size, -1))

    # Step 2: permute P to dim 0 for all_to_all
    dims = list(range(x.dim()))
    dims.remove(scatter_idx)
    x = x.permute([scatter_idx] + dims).contiguous()

    # Step 3: AllToAll exchange
    out = torch.empty_like(x)
    dist.all_to_all_single(out, x, group=group)
    if use_sync:
        torch.npu.synchronize()

    # Step 4: merge P into gather dim
    if gather_idx == 1:
        # gather along seq: [P, B, S', H', D] → [B, P*S', H', D]
        out = out.permute(1, 0, 2, 3, 4).contiguous()
        out = out.reshape(out.shape[0], world_size * out.shape[2], out.shape[3], out.shape[4])
    else:
        # gather along heads: [P, B, S', H', D] → [B, S', P*H', D]
        out = out.permute(1, 2, 0, 3, 4).contiguous()
        out = out.reshape(out.shape[0], out.shape[1], world_size * out.shape[3], out.shape[4])

    return out
