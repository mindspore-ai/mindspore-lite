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
lite_boost parallel module
"""

__all__ = [
    "initialize_usp",
    "is_distributed_active",
    "get_world_size",
    "get_rank",
    "all_reduce",
    "broadcast",
    "mm_all_reduce",
    "concat_all_reduce_split",
    "get_hcom_name",
    "shard_colwise",
    "shard_rowwise",
    "shard_kv_overlap",
    "get_overlap_kv_heads_per_rank",
    "gqa_expand_kv_overlap",
    "all_to_all_4d",
    "get_sp_size",
    "get_sp_rank",
    "all_gather_seq",
    "pad_split_seq",
    "gqa_kv_head_index",
]

from ._initializer import (
    initialize_usp,
    is_distributed_active,
    get_world_size,
    get_rank,
    all_reduce,
    broadcast,
)
from .tensor_parallel import (
    mm_all_reduce,
    concat_all_reduce_split,
    get_hcom_name,
    shard_colwise,
    shard_rowwise,
    get_overlap_kv_heads_per_rank,
    shard_kv_overlap,
    gqa_expand_kv_overlap,
)
from .context_parallel import (
    all_to_all_4d,
    get_sp_size,
    get_sp_rank,
    all_gather_seq,
    pad_split_seq,
    gqa_kv_head_index,
)
