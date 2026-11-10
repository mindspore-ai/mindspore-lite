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
"""LiteBoost adapter for Qwen3.5-9B (hybrid GatedDeltaNet + full-attention LLM)."""
from .boost import boost_qwen3_5_9b
from .model import (
    _chunk_wrapper,
    _recurrent_wrapper,
    qwen3_5_full_attn_forward,
    qwen3_5_full_attn_forward_cp,
    shard_qwen3_5_cp,
    shard_qwen3_5_tp,
)

__all__ = [
    "boost_qwen3_5_9b",
    "shard_qwen3_5_tp",
    "shard_qwen3_5_cp",
    "qwen3_5_full_attn_forward",
    "qwen3_5_full_attn_forward_cp",
    "_chunk_wrapper",
    "_recurrent_wrapper",
]
