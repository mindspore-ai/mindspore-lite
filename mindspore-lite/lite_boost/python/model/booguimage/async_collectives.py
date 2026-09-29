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
Async SP communication primitives for the BooguImage comm-compute overlap kernels.

Two styles, used by ``comm_compute_overlap.py``:

- ``*_async``: issue the collective with ``async_op=True`` on the calling
  stream, wait and post-process INSIDE the function. When called inside a
  comm-stream context (``with torch.npu.stream(comm)``) the wait and the
  permute/reshape run on the comm stream, so the main stream only syncs via
  events and stays free for compute.

- ``*_launch`` / ``*_finalize``: split style for real comm|compute overlap.
  ``launch`` posts the collective and returns the RAW output plus a work
  handle; ``finalize`` waits and restores the 4-D layout. Compute can be
  interleaved between the two.
"""

import torch
import torch.distributed as dist

from lite_boost.parallel.context_parallel import merge_a2a_4d


def all_to_all_4d_async(x, scatter_idx=2, gather_idx=1, group=None):
    """All-to-all a 4-D tensor, scatter ``scatter_idx`` and gather ``gather_idx``.

    Returns ``(out, work)``; ``work`` is ``None`` when world <= 1.
    """
    group = group or dist.group.WORLD
    world = dist.get_world_size(group)
    if world <= 1:
        return x, None

    x = x.unflatten(scatter_idx, (world, -1))
    dims = list(range(x.dim()))
    dims.remove(scatter_idx)
    x = x.permute([scatter_idx] + dims).contiguous()
    out = torch.empty_like(x)
    work = dist.all_to_all_single(out, x, group=group, async_op=True)
    work.wait()

    return merge_a2a_4d(out, world, gather_idx), work


def all_gather_seq_4d_async(x, group=None):
    """All-gather a [B, S, N, D] tensor along the sequence dim.

    Returns ``(out, work)``; ``work`` is ``None`` when world <= 1.
    """
    group = group or dist.group.WORLD
    world = dist.get_world_size(group)
    if world <= 1:
        return x, None

    b, s, n, d = x.shape
    x_t = x.permute(1, 0, 2, 3).contiguous()
    out_t = torch.empty(s * world, b, n, d, dtype=x.dtype, device=x.device)
    work = dist.all_gather_into_tensor(out_t, x_t, group=group, async_op=True)
    work.wait()
    out = out_t.permute(1, 0, 2, 3).contiguous()
    return out, work


def all_to_all_4d_launch(x, scatter_idx=2, gather_idx=1, group=None):
    """Post the a2a collective; pair with ``all_to_all_4d_finalize``.

    Returns ``(raw_out, work, gather_idx)``; ``work`` is ``None`` when
    world <= 1.
    """
    group = group or dist.group.WORLD
    world = dist.get_world_size(group)
    if world <= 1:
        return x, None, gather_idx

    x = x.unflatten(scatter_idx, (world, -1))
    dims = list(range(x.dim()))
    dims.remove(scatter_idx)
    x = x.permute([scatter_idx] + dims).contiguous()
    out = torch.empty_like(x)
    work = dist.all_to_all_single(out, x, group=group, async_op=True)
    return out, work, gather_idx


def all_to_all_4d_finalize(raw_out, work, gather_idx):
    """Wait for the a2a posted by ``all_to_all_4d_launch`` and restore 4-D."""
    if work is None:
        return raw_out
    world = dist.get_world_size()
    work.wait()
    return merge_a2a_4d(raw_out, world, gather_idx)


def all_gather_kv_launch(key, value, group=None):
    """Gather K and V along the sequence dim as ONE concatenated tensor.

    Returns ``(raw_out_t, work, (b, s, kv, d))``; on world == 1 returns
    ``((key, value), None, None)``. Pairs with ``all_gather_kv_finalize``.
    """
    group = group or dist.group.WORLD
    world = dist.get_world_size(group)
    if world <= 1:
        return (key, value), None, None

    b, s, kv, d = key.shape
    kv_t = torch.cat([key, value], dim=2).contiguous()      # [B, S, 2*kv, D]
    kv_t = kv_t.permute(1, 0, 2, 3).contiguous()            # [S, B, 2*kv, D]
    out_t = torch.empty(s * world, b, 2 * kv, d, dtype=key.dtype, device=key.device)
    work = dist.all_gather_into_tensor(out_t, kv_t, group=group, async_op=True)
    return out_t, work, (b, s, kv, d)


def all_gather_kv_finalize(raw_out_t, work, args):
    """Wait for the gather posted by ``all_gather_kv_launch``; split K/V."""
    if work is None:
        return raw_out_t    # (key, value) tuple from launch's world<=1 path
    _, _, kv, _ = args
    work.wait()
    kv_full = raw_out_t.permute(1, 0, 2, 3).contiguous()    # [B, S_full, 2*kv, D]
    key_full = kv_full[:, :, :kv, :].contiguous()
    value_full = kv_full[:, :, kv:, :].contiguous()
    return key_full, value_full
