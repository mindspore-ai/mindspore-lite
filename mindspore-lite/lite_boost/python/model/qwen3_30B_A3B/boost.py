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
"""Qwen3-30B-A3B (Qwen3MoeForCausalLM) LiteBoost boost entry point."""
from __future__ import annotations

import logging
import os
import types

import torch.distributed as dist

from .model import (
    _patch_ffn,
    _patch_rms_norm,
    qwen3_30B_full_attn_forward,
    qwen3_30B_eager_attn_forward,
    shard_qwen3_30B_tp,
    _get_text_model,
    patch_fused_moe,
)

logger = logging.getLogger(__name__)


def boost_qwen3_30B_A3B(model, *, world_size: int | None = None,
                         mode: str = "tp",
                         fuse_rms_norm: bool = False,
                         fuse_ffn: bool = False,
                         fuse_moe: bool = False,
                         fuse_moe_router: bool = False,
                         use_eager: bool = False):
    """Patch a Qwen3-30B-A3B model in place for LiteBoost NPU inference."""
    if world_size is None:
        if dist is not None and dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1

    if mode not in ("tp", "cp"):
        raise ValueError(f"mode must be 'tp' or 'cp', got {mode!r}")
    if mode == "cp":
        if world_size > 1:
            import warnings
            warnings.warn(
                "Qwen3-30B-A3B does not yet support context-parallel; "
                "falling back to tensor-parallel (mode='tp')."
            )
            mode = "tp"

    text_model = _get_text_model(model)

    attn_fwd = qwen3_30B_eager_attn_forward if use_eager else qwen3_30B_full_attn_forward
    for _, layer in enumerate(text_model.layers):
        layer.self_attn.forward = types.MethodType(
            attn_fwd, layer.self_attn
        )

    if world_size > 1:
        rank = dist.get_rank() if (dist is not None and dist.is_initialized()) else 0
        if mode == "tp":
            shard_qwen3_30B_tp(model, rank, world_size)

    if fuse_rms_norm:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import (
            Qwen3MoeRMSNorm,
        )
        patched = 0
        for layer in text_model.layers:
            if hasattr(layer, "input_layernorm") and isinstance(
                    layer.input_layernorm, Qwen3MoeRMSNorm):
                _patch_rms_norm(layer.input_layernorm)
                patched += 1
            if hasattr(layer, "post_attention_layernorm") and isinstance(
                    layer.post_attention_layernorm, Qwen3MoeRMSNorm):
                _patch_rms_norm(layer.post_attention_layernorm)
                patched += 1
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                if hasattr(attn, "q_norm") and isinstance(
                        attn.q_norm, Qwen3MoeRMSNorm):
                    _patch_rms_norm(attn.q_norm)
                    patched += 1
                if hasattr(attn, "k_norm") and isinstance(
                        attn.k_norm, Qwen3MoeRMSNorm):
                    _patch_rms_norm(attn.k_norm)
                    patched += 1
        if hasattr(text_model, "norm") and isinstance(
                text_model.norm, Qwen3MoeRMSNorm):
            _patch_rms_norm(text_model.norm)
            patched += 1

    if fuse_rms_norm and os.environ.get("LB_ADD_RMSNORM", "1") == "1":
        from .model import _patch_residual_layer_forward
        patched_layer = 0
        for layer in text_model.layers:
            if _patch_residual_layer_forward(layer):
                patched_layer += 1
        if patched_layer:
            logger.info("%s", f"  fuse_add_rmsnorm=True ({patched_layer} decoder layers → "
                  f"npu_add_rms_norm residual fusion)")

    if fuse_ffn:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import (
            Qwen3MoeMLP,
        )
        patched_ffn = 0
        for layer in text_model.layers:
            mlp = getattr(layer, "mlp", None)
            if isinstance(mlp, Qwen3MoeMLP):
                _patch_ffn(mlp)
                patched_ffn += 1

    if fuse_moe:
        moe_mode_env = os.environ.get("LB_MOE_FUSED", "2")
        if moe_mode_env != "0":
            moe_mode = {"2": "cann", "1": "sort"}.get(moe_mode_env, "cann")
            decode_grouped = (
                os.environ.get("LB_MOE_DECODE_GROUPED", "1") == "1"
            )
            patched_moe, patched_router = patch_fused_moe(
                text_model, mode=moe_mode, decode_grouped=decode_grouped,
                fuse_router=fuse_moe_router, owner="boost",
            )
            router_msg = f", {patched_router} routers fused" if fuse_moe_router else ""
            logger.info(
                "%s", f"  fuse_moe=True ({patched_moe} experts modules → "
                f"{moe_mode} flow, decode_grouped={decode_grouped}{router_msg})"
            )

    return model
