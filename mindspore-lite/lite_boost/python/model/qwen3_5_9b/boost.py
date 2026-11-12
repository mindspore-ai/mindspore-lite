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
"""LiteBoost adaptation entry point for Qwen3.5-9B on Ascend A2 NPU."""

from __future__ import annotations

# The adapter patches HuggingFace internals in place (registered buffers, swapped
# forwards, per-layer overrides) and has to keep the upstream call signatures,
# so touching protected members and ignoring some arguments is deliberate here.
# pylint: disable=protected-access,unused-argument

import logging
import types

import torch
from torch import nn

from lite_boost.parallel._initializer import get_rank, get_world_size

from .model import (
    _chunk_wrapper,
    _recurrent_wrapper,
    _patch_ffn,
    _patch_ffn_down_layout,
    _patch_ffn_down_w8a16,
    _patch_ffn_gate_up_w8a16,
    _patch_linear_matmul,
    _patch_inproj_ba,
    _patch_inproj_4way,
    _patch_a_log_precompute,
    _patch_fused_conv_silu,
    _patch_rms_norm,
    _patch_rms_norm_gated,
    _patch_vision_attention,
    _patch_full_attn_proj_fuse,
    install_layer_scoped_kernels,
    qwen3_5_full_attn_forward,
    qwen3_5_full_attn_forward_static,
    qwen3_5_text_model_forward_graph,
    torch_causal_conv1d_update_graph,
    shard_qwen3_5_cp,
    shard_qwen3_5_tp,
    _get_text_model,
)

logger = logging.getLogger(__name__)


def _require_custom_ops(op_names):
    """Fail fast when an AscendC op this configuration needs is not registered.

    The adapter has no stock fallback: a missing op means the wheel was built
    without it or ``liblite_boost_ops.so`` never loaded, and running on the
    slower reference kernels would silently change both latency and numerics.
    """
    # Importing .model above loads lite_boost.ops, which registers the ops.
    missing = [name for name in op_names
               if not hasattr(torch.ops.lite_boost, name)]
    if not missing:
        return
    raise RuntimeError(
        "Qwen3.5-9B LiteBoost requires the AscendC ops "
        f"{', '.join(missing)}, but they are not registered. Rebuild/install "
        "the wheel from mindspore-lite/lite_boost with `bash build.sh` (it "
        "bundles liblite_boost_ops.so and the custom-op vendor package), and "
        "source the matching CANN set_env.sh before running. See "
        "LITE_BOOST_OPS_LIB if the library lives outside the wheel."
    )


def boost_qwen3_5_9b(model, config: dict | None = None, *,
                     world_size: int | None = None, mode: str = "tp",
                     graph_mode: bool = False, fuse_rms_norm: bool = False,
                     fuse_ffn: bool = False, fuse_rms_norm_gated: bool = False,
                     fuse_inproj_ba: bool = False,
                     bucketed_prefill: bool = False,
                     fuse_vision_attn: bool = False,
                     opt_down_proj_layout: bool = False,
                     opt_fias_decode: bool = False,
                     opt_w8a16_down_proj: bool = False,
                     opt_w8a16_gate_up: bool = False,
                     opt_recurrent_op_f2: bool = False,
                     opt_inproj_4way: bool = False,
                     opt_a_log_precompute: bool = False,
                     opt_fused_conv_silu: bool = False,
                     opt_full_attn_proj_fuse: bool = False,
                     opt_chunk_qk_l2norm_sink: bool = False,
                     opt_chunk_out_zero_pad: bool = False):
    """Patch a Qwen3.5-9B model in place for LiteBoost NPU inference.

    ``config`` exists so BoostManager can dispatch here like it does for the
    other adapters; this model selects its optimizations through the explicit
    keyword flags instead of YAML sections, so the dict is not read.
    """
    if world_size is None:
        world_size = get_world_size()

    if mode not in ("tp", "cp"):
        raise ValueError(f"mode must be 'tp' or 'cp', got {mode!r}")

    required = ["chunk_gated_delta_rule", "recurrent_gated_delta_rule"]
    if opt_fused_conv_silu and world_size == 1:
        required.append("fused_causal_conv_silu")
    if opt_chunk_qk_l2norm_sink:
        required.append("fused_qk_l2norm_chunk_gated_delta_rule")
    _require_custom_ops(required)

    text_model = _get_text_model(model)
    layer_types = text_model.config.layer_types

    # HF calls the GDN kernels through module globals; redirect them to the
    # per-layer replacements installed below (see install_layer_scoped_kernels).
    install_layer_scoped_kernels()

    for i, layer in enumerate(text_model.layers):
        if layer_types[i] == "linear_attention":
            la = layer.linear_attn
            la.causal_conv1d_fn = None
            if graph_mode:
                la.causal_conv1d_update = torch_causal_conv1d_update_graph
            la.chunk_gated_delta_rule = _chunk_wrapper(la)
            la._lb_chunk_qk_l2norm_sink = opt_chunk_qk_l2norm_sink
            la._lb_chunk_out_zero_pad = opt_chunk_out_zero_pad
            if la.num_v_heads % la.num_k_heads == 0:
                la._lb_gqa_native = True
            la.recurrent_gated_delta_rule = _recurrent_wrapper(la, opt_f2=opt_recurrent_op_f2)
        elif layer_types[i] == "full_attention":
            fwd = qwen3_5_full_attn_forward_static if graph_mode else qwen3_5_full_attn_forward
            layer.self_attn.forward = types.MethodType(fwd, layer.self_attn)
            layer.self_attn._lite_boost_graph_mode = graph_mode
        else:
            raise ValueError(
                f"Unknown layer_type {layer_types[i]!r} at layer {i}; "
                f"expected 'linear_attention' or 'full_attention'."
            )

    if world_size > 1:
        rank = get_rank()
        if mode == "tp":
            shard_qwen3_5_tp(model, rank, world_size)
        else:
            shard_qwen3_5_cp(model, rank, world_size)

    if graph_mode:
        lm = text_model.language_model if hasattr(text_model, "language_model") else text_model
        lm._forward_eager = lm.forward
        lm.forward = types.MethodType(qwen3_5_text_model_forward_graph, lm)

    if fuse_rms_norm:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
        patched = 0
        for layer in text_model.layers:
            if hasattr(layer, "input_layernorm") and isinstance(layer.input_layernorm, Qwen3_5RMSNorm):
                _patch_rms_norm(layer.input_layernorm)
                patched += 1
            if hasattr(layer, "post_attention_layernorm") and isinstance(
                    layer.post_attention_layernorm, Qwen3_5RMSNorm):
                _patch_rms_norm(layer.post_attention_layernorm)
                patched += 1
            if hasattr(layer, "self_attn"):
                attn = layer.self_attn
                if hasattr(attn, "q_norm") and isinstance(attn.q_norm, Qwen3_5RMSNorm):
                    _patch_rms_norm(attn.q_norm)
                    patched += 1
                if hasattr(attn, "k_norm") and isinstance(attn.k_norm, Qwen3_5RMSNorm):
                    _patch_rms_norm(attn.k_norm)
                    patched += 1
        if hasattr(text_model, "norm") and isinstance(text_model.norm, Qwen3_5RMSNorm):
            _patch_rms_norm(text_model.norm)
            patched += 1
        lm = text_model.language_model if hasattr(text_model, "language_model") else None
        if lm is not None and hasattr(lm, "norm") and isinstance(lm.norm, Qwen3_5RMSNorm):
            _patch_rms_norm(lm.norm)
            patched += 1

    if fuse_ffn:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP
        patched_ffn = 0
        for layer in text_model.layers:
            if hasattr(layer, "mlp") and isinstance(layer.mlp, Qwen3_5MLP):
                _patch_ffn(layer.mlp)
                patched_ffn += 1

    if fuse_rms_norm_gated:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNormGated
        patched_gated = 0
        for layer in text_model.layers:
            la = getattr(layer, "linear_attn", None)
            if la is not None and hasattr(la, "norm") and isinstance(la.norm, Qwen3_5RMSNormGated):
                _patch_rms_norm_gated(la.norm)
                patched_gated += 1

    if fuse_inproj_ba:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
        patched_ba = 0
        for layer in text_model.layers:
            la = getattr(layer, "linear_attn", None)
            if la is not None and isinstance(la, Qwen3_5GatedDeltaNet):
                _patch_inproj_ba(la)
                patched_ba += 1

    if opt_inproj_4way:
        if world_size > 1:
            logger.info("%s", f"[boost] WARNING: opt_inproj_4way is 1p-only (world_size={world_size}), skipping")
        elif not fuse_inproj_ba:
            logger.info("[boost] WARNING: opt_inproj_4way requires fuse_inproj_ba, skipping")
        else:
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
            patched_4way = 0
            for layer in text_model.layers:
                la = getattr(layer, "linear_attn", None)
                if la is not None and isinstance(la, Qwen3_5GatedDeltaNet):
                    _patch_inproj_4way(la)
                    patched_4way += 1

    if opt_a_log_precompute:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
        patched_a_log = 0
        for layer in text_model.layers:
            la = getattr(layer, "linear_attn", None)
            if la is not None and isinstance(la, Qwen3_5GatedDeltaNet):
                _patch_a_log_precompute(la)
                patched_a_log += 1
        if patched_a_log > 0:
            logger.info("%s", f"[boost] opt_a_log_precompute applied to {patched_a_log} linear-attn layers")

    if opt_fused_conv_silu:
        if world_size > 1:
            logger.info("%s", f"[boost] WARNING: opt_fused_conv_silu is 1p-only (world_size={world_size}), skipping")
        else:
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
            patched_f1 = 0
            for layer in text_model.layers:
                la = getattr(layer, "linear_attn", None)
                if la is not None and isinstance(la, Qwen3_5GatedDeltaNet):
                    if _patch_fused_conv_silu(la):
                        patched_f1 += 1
            if patched_f1 > 0:
                logger.info("%s", f"[boost] opt_fused_conv_silu applied to {patched_f1} linear-attn layers")
            else:
                logger.info("[boost] WARNING: opt_fused_conv_silu found no layer (op not registered? "
                      "rebuild liblite_boost_ops.so + F1 vendor on ASCEND_CUSTOM_OPP_PATH)")

    if opt_full_attn_proj_fuse:
        if world_size > 1:
            logger.info("%s", f"[boost] WARNING: opt_full_attn_proj_fuse is 1p-only "
                          f"(world_size={world_size}), skipping")
        else:
            patched_f6 = 0
            for layer in text_model.layers:
                attn = getattr(layer, "self_attn", None)
                if attn is not None and getattr(attn, "q_proj", None) is not None \
                        and getattr(attn, "k_proj", None) is not None \
                        and getattr(attn, "v_proj", None) is not None:
                    _patch_full_attn_proj_fuse(attn)
                    patched_f6 += 1
            if patched_f6 > 0:
                logger.info("%s", f"[boost] opt_full_attn_proj_fuse applied to {patched_f6} full-attn layers")

    if fuse_vision_attn:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionAttention
        conditional = getattr(model, "model", None)
        visual = getattr(conditional, "visual", None) if conditional is not None else None
        if visual is None:
            pass
        else:
            patched_vis = 0
            for blk in visual.blocks:
                attn = getattr(blk, "attn", None)
                if attn is not None and isinstance(attn, Qwen3_5VisionAttention):
                    _patch_vision_attention(attn)
                    patched_vis += 1

    if bucketed_prefill:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
        lm = text_model.language_model if hasattr(text_model, "language_model") else text_model
        lm._bucketed_prefill = True
        tagged_la = 0
        tagged_fa = 0
        for layer in text_model.layers:
            la = getattr(layer, "linear_attn", None)
            if la is not None and isinstance(la, Qwen3_5GatedDeltaNet):
                la._static_actual_seq_len = None
                la._static_conv_state_idx = None
                tagged_la += 1
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                attn._bucketed_prefill = True
                attn._prefill_mask = None
                tagged_fa += 1

    if opt_down_proj_layout:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP
        patched_down = 0
        patched_o = 0
        for layer in text_model.layers:
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and isinstance(mlp, Qwen3_5MLP) and hasattr(mlp, "_fused_gate_up_weight"):
                _patch_ffn_down_layout(mlp)
                patched_down += 1
            attn = getattr(layer, "self_attn", None)
            if attn is not None and hasattr(attn, "o_proj") and isinstance(attn.o_proj, nn.Linear):
                if type(attn.o_proj).__name__ != "_RowParallelLinear":
                    _patch_linear_matmul(attn.o_proj)
                    patched_o += 1

    if opt_fias_decode:
        tagged_fias = 0
        for layer in text_model.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None and hasattr(attn, "_lite_boost_graph_mode") and attn._lite_boost_graph_mode:
                attn._opt_fias_decode = True
                tagged_fias += 1

    if opt_w8a16_down_proj:
        if world_size > 1 and mode == "tp":
            logger.info("%s", f"[boost] opt_w8a16_down_proj is 1p/CP-only (tp world_size={world_size}), skipping")
        else:
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP
            patched_w8a16 = 0
            for layer in text_model.layers:
                mlp = getattr(layer, "mlp", None)
                if mlp is not None and isinstance(mlp, Qwen3_5MLP) and hasattr(mlp, "_fused_gate_up_weight"):
                    _patch_ffn_down_w8a16(mlp)
                    patched_w8a16 += 1

    if opt_w8a16_gate_up:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP
        patched_gate = 0
        for layer in text_model.layers:
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and isinstance(mlp, Qwen3_5MLP) and hasattr(mlp, "_fused_gate_up_weight"):
                _patch_ffn_gate_up_w8a16(mlp)
                patched_gate += 1

    return model
