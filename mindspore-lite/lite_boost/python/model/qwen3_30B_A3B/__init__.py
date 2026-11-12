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
"""LiteBoost adapter for Qwen3-30B-A3B (Qwen3MoeForCausalLM, 48-layer MoE)."""
from .boost import boost_qwen3_30B_A3B
from .hf_loader import load_qwen3_moe_bf16
from .model import (
    qwen3_30B_full_attn_forward,
    shard_qwen3_30B_tp,
)

__all__ = [
    "boost_qwen3_30B_A3B",
    "load_qwen3_moe_bf16",
    "shard_qwen3_30B_tp",
    "qwen3_30B_full_attn_forward",
]
