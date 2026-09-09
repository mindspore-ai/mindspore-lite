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
Ulysses Sequence Parallel communication and utility primitives.

``all_to_all_4d`` exchanges head/sequence dimensions for multi-card
attention:
Forward  (scatter=2, gather=1): [B, S/P, H, D] -> [B, S, H/P, D]
Reverse  (scatter=1, gather=2): [B, S, H/P, D] -> [B, S/P, H, D]

The remaining helpers are the sequence-parallel utilities shared by
SP-style model adapters: sequence all-gather, per-rank sequence slicing
with padding bookkeeping, and GQA KV-head index computation for
Ulysses-style attention.
"""
import torch
import torch.distributed as dist

from ._initializer import get_rank, get_world_size


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

    if scatter_idx not in (1, 2) or gather_idx not in (1, 2):
        raise ValueError(
            f"scatter_idx and gather_idx must be in (1, 2), "
            f"but got scatter_idx={scatter_idx}, gather_idx={gather_idx}.")
    if scatter_idx == gather_idx:
        raise ValueError(
            f"scatter_idx must not equal gather_idx, "
            f"but got scatter_idx={scatter_idx}, gather_idx={gather_idx}.")

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


def get_sp_size() -> int:
    """Return the SP world size (the distributed world size)."""
    return get_world_size()


def get_sp_rank() -> int:
    """Return the SP rank (the distributed rank)."""
    return get_rank()


def all_gather_seq(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """All-gather a local sequence chunk back to the full sequence.

    ``x`` is a shard along ``dim``; the gathered tensor interleaves rank
    chunks in rank order (matching ``torch.chunk(x, world, dim)[rank]``).
    """
    world_size = get_sp_size()
    if world_size == 1:
        return x
    # all_gather_into_tensor concatenates along dim 0 of the flat output:
    # move the sequence dim first, gather, then move it back.
    x_t = x.transpose(dim, 0).contiguous()
    out_t = torch.empty(x_t.shape[0] * world_size, *x_t.shape[1:],
                        dtype=x.dtype, device=x.device)
    dist.all_gather_into_tensor(out_t, x_t)
    return out_t.transpose(0, dim).contiguous()


def pad_split_seq(x: torch.Tensor, seq_pad: int, dim: int = 1):
    """Pad then slice the full sequence into this rank's chunk.

    Returns ``(local_chunk, padded)`` where ``padded`` is True when padding
    rows were appended. ``x`` is the FULL (unsharded) sequence; with
    ``seq_pad=0`` this is a plain per-rank slice.
    """
    world_size = get_sp_size()
    rank = get_sp_rank()
    if world_size == 1:
        return x, False
    if seq_pad > 0:
        pad_shape = list(x.shape)
        pad_shape[dim] = seq_pad
        zeros = x.new_zeros(pad_shape)
        x = torch.cat([x, zeros], dim=dim)
    s_local = x.shape[dim] // world_size
    slices = [slice(None)] * x.dim()
    slices[dim] = slice(rank * s_local, (rank + 1) * s_local)
    return x[tuple(slices)].contiguous(), seq_pad > 0


def gqa_kv_head_index(num_q_heads_local: int, q_heads_per_kv: int,
                      device) -> torch.Tensor:
    """Global KV-head index for each local query head (Ulysses GQA).

    After ``all_to_all_4d`` the query heads on this rank are the contiguous
    block ``[rank * num_q_heads_local, (rank+1) * num_q_heads_local)`` of the
    full head order; each maps to its KV group ``global_q_head // q_heads_per_kv``.
    """
    rank = get_sp_rank()
    local_q = torch.arange(num_q_heads_local, device=device)
    global_q = rank * num_q_heads_local + local_q
    return (global_q // q_heads_per_kv).long()
