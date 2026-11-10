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
"""Hugging Face bf16 checkpoint loader for Qwen3-MoE (Qwen3-30B-A3B)."""
from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def load_qwen3_moe_bf16(
    model_dir: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "eager",
):
    """Load a bf16 Qwen3-MoE checkpoint on CPU for the LiteBoost adapter.

    The checkpoint must be a standard Hugging Face directory (``config.json`` +
    ``*.safetensors``); nothing is converted on disk. The four pinning choices
    are what the adapter relies on:

    * ``dtype=bfloat16`` — the 61 GB checkpoint would double in fp32;
    * ``low_cpu_mem_usage`` — avoids a second full copy while loading shards;
    * ``attn_implementation="eager"`` — attention is replaced by the LiteBoost
      forwards, so no fused backend is selected at load time;
    * ``experts_implementation="eager"`` — leaves the plain per-expert loop in
      place instead of relying on a fused kernel that the reference path and
      the in-place TP sharder do not need.

    ``Qwen3MoeExperts`` stores experts as packed 3-D ``gate_up_proj`` /
    ``down_proj`` tensors; the adapter slices those tensors in place.
    """
    from transformers import Qwen3MoeForCausalLM

    model = Qwen3MoeForCausalLM.from_pretrained(
        model_dir,
        dtype=dtype,
        attn_implementation=attn_implementation,
        experts_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.eval()
    logger.info(
        "%s", f"  HF bf16 loader: {model.config.num_hidden_layers} layers, "
        f"E={model.config.num_experts}, I={model.config.moe_intermediate_size}, "
        f"H={model.config.hidden_size} ({dtype})")
    return model
