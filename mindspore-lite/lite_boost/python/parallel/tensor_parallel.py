#!/usr/bin/env python3
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
"""
Tensor-parallel (TP) utility primitives.

HCCL communicator-name lookup with caching, fused NPU matmul-all-reduce
helpers, and GQA (grouped-query attention) weight sharding: column-wise /
row-wise ``nn.Linear`` sharding, overlapping KV-head partitioning, and
per-rank K/V expansion for GQA models.
"""
import os
import torch
import torch.distributed as dist

from ._initializer import all_reduce, get_rank, get_world_size, is_distributed_active

_hcom_cache = {}


def get_hcom_name(rank=None):
    """Return the HCCL communicator name for ``rank`` (cached), or None when TP is inactive."""
    if not is_distributed_active():
        return None
    if rank is None:
        rank = get_rank()
    if rank in _hcom_cache:
        return _hcom_cache[rank]
    default_pg = dist.group.WORLD
    hcom = None
    try:
        hcom = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    except AttributeError:
        try:
            hcom = default_pg.get_hccl_comm_name(rank)
        except AttributeError:
            hcom = None
    _hcom_cache[rank] = hcom
    return hcom


def mm_all_reduce(x, weight, bias=None, *, hcom=None, reduce_op="sum"):
    """Fused matmul + all-reduce via ``npu_mm_all_reduce_base``; falls back to plain linear."""
    import torch_npu

    if os.environ.get("LB_TP_FP32_REDUCE")=="1":
        out_dtype = x.dtype
        partial = torch.nn.functional.linear(x.float(), weight.float(), bias.float() if bias is not None else None)
        all_reduce(partial)
        return partial.to(out_dtype)

    if os.environ.get("LB_TP_BF16_SEPARATE")=="1":
        partial = torch.nn.functional.linear(x, weight, bias)
        all_reduce(partial)
        return partial

    if hcom is None:
        hcom = get_hcom_name()
    if hcom is None:
        return torch.nn.functional.linear(x, weight, bias)
    x2 = weight.transpose(0, 1).contiguous()
    if bias is not None:
        return torch_npu.npu_mm_all_reduce_base(x, x2, hcom, reduce_op=reduce_op, bias=bias)
    return torch_npu.npu_mm_all_reduce_base(x, x2, hcom, reduce_op=reduce_op)


def concat_all_reduce_split(
    x1, weight1, bias1,
    x2, weight2, bias2,
    *,
    cat_dim=1, out_dim=-1,
):
    """Run two linears, all-reduce the concatenated partials, then split per rank along ``out_dim``."""
    if not is_distributed_active():
        out1 = torch.nn.functional.linear(x1, weight1, bias1)
        out2 = torch.nn.functional.linear(x2, weight2, bias2)
        return out1, out2

    rank = get_rank()
    ws = get_world_size()

    if os.environ.get("LB_TP_FP32_REDUCE") == "1":
        out_dtype = x1.dtype
        partial1 = torch.nn.functional.linear(x1.float(), weight1.float(), bias1.float() if bias1 is not None else None)
        partial2 = torch.nn.functional.linear(x2.float(), weight2.float(), bias2.float() if bias2 is not None else None)
        combined = torch.cat([partial1, partial2], dim=cat_dim).to(out_dtype)
    else:
        partial1 = torch.nn.functional.linear(x1, weight1, bias1)
        partial2 = torch.nn.functional.linear(x2, weight2, bias2)
        combined = torch.cat([partial1, partial2], dim=cat_dim)
    all_reduce(combined)

    len1 = partial1.shape[cat_dim]
    len2 = partial2.shape[cat_dim]
    out1_full = combined.narrow(cat_dim, 0, len1)
    out2_full = combined.narrow(cat_dim, len1, len2)
    n = out1_full.shape[out_dim]
    chunk_size = n // ws
    out1 = out1_full.narrow(out_dim, rank * chunk_size, chunk_size)
    out2 = out2_full.narrow(out_dim, rank * chunk_size, chunk_size)
    return out1, out2


def shard_colwise(linear, rank=None, world_size=None):
    """Shard a Linear along the output dimension in place for the given TP ``rank``."""
    if linear is None:
        return
    chunk = linear.out_features // world_size
    start = rank * chunk
    end = start + chunk
    linear.weight.data = linear.weight.data[start:end, :]
    linear.out_features = chunk
    if linear.bias is not None:
        linear.bias.data = linear.bias.data[start:end]


def shard_rowwise(linear, rank=None, world_size=None):
    """Shard a Linear along the input dimension in place for the given TP ``rank``."""
    if linear is None:
        return
    chunk = linear.in_features // world_size
    start = rank * chunk
    end = start + chunk
    linear.weight.data = linear.weight.data[:, start:end]
    linear.in_features = chunk


def get_overlap_kv_heads_per_rank(rank=None, world_size=None, num_kv_heads=None):
    """Return ``(start_head, actual_heads)`` of the overlapping KV-head chunk assigned to ``rank``."""
    heads_per_rank = (num_kv_heads + world_size - 1) // world_size
    start_head = rank * (heads_per_rank - 1) if rank > 0 else 0
    end_head = min(start_head + heads_per_rank, num_kv_heads)
    actual_heads = end_head - start_head
    return start_head, actual_heads


def shard_kv_overlap(linear, rank=None, world_size=None, num_kv_heads=None, head_dim=None):
    """Shard a KV-projection Linear in place using overlapping head partitioning."""
    if linear is None:
        return
    start_head, actual_heads = get_overlap_kv_heads_per_rank(rank, world_size, num_kv_heads)
    start_row = start_head * head_dim
    end_row = (start_head + actual_heads) * head_dim
    linear.weight.data = linear.weight.data[start_row:end_row, :]
    linear.out_features = actual_heads * head_dim
    if linear.bias is not None:
        linear.bias.data = linear.bias.data[start_row:end_row]


def gqa_expand_kv_overlap(key, value, num_q_heads, num_kv_heads, tp_kv_start_head, tp_full_kv_heads):
    """Expand per-rank K/V to local query heads; falls back to plain GQA expansion when TP is off."""
    if not is_distributed_active():
        factor = num_q_heads // num_kv_heads
        return (
            key.repeat_interleave(factor, dim=-3),
            value.repeat_interleave(factor, dim=-3),
        )
    tp_rank = get_rank()
    tp_world_size = get_world_size()

    full_q_heads = num_q_heads * tp_world_size
    heads_per_kv_group = full_q_heads // tp_full_kv_heads

    local_q = torch.arange(num_q_heads, device=key.device)
    global_q = tp_rank * num_q_heads + local_q
    global_kv_idx = global_q // heads_per_kv_group
    local_kv_idx = global_kv_idx - tp_kv_start_head
    local_kv_idx = local_kv_idx.clamp(min=0, max=num_kv_heads - 1)
    return key[:, local_kv_idx], value[:, local_kv_idx]
