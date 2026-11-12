#!/usr/bin/env python3
# modified from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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
"""Qwen3.5-9B LiteBoost adapter: kernel wrappers, full-attention forward, TP/CP sharding.

The forwards here are drop-in replacements for the ones in
``transformers.models.qwen3_5.modeling_qwen3_5`` (transformers 5.19.0: the
reference Qwen3.5 implementation, whose GDN kernels are vendored from the FLA
library). Module structure, signatures, cache layouts and the numerics of the
unfused paths follow that file; this adapter only swaps the compute path to
Ascend kernels, adds static-cache / graph-capture entry points, and introduces
tensor- and context-parallel sharding.
"""

from __future__ import annotations

# The adapter patches HuggingFace internals in place (registered buffers, swapped
# forwards, per-layer overrides) and has to keep the upstream call signatures,
# so touching protected members and ignoring some arguments is deliberate here.
# pylint: disable=protected-access,unused-argument

import math
import types
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from lite_boost.layers.attention import flash_attention
from lite_boost.ops import chunk_gated_delta_rule, recurrent_gated_delta_rule
from lite_boost.parallel._initializer import is_distributed_active
from lite_boost.parallel.context_parallel import all_gather_seq
from lite_boost.parallel.tensor_parallel import (
    get_hcom_name,
    mm_all_reduce,
    shard_colwise,
    shard_rowwise,
)


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """FLA-library L2 normalization (matches HF's `l2norm` in modeling_qwen3_5)."""
    return torch.nn.functional.normalize(x, dim=dim, eps=eps)


# Module-level kernel functions that HF's Qwen3_5GatedDeltaNet.forward calls
# directly (see modeling_qwen3_5.py). LiteBoost installs its replacements as
# attributes of the same name on each layer, so these globals need to be
# redirected to the layer that is currently running.
_GDN_GLOBAL_KERNELS = (
    "torch_chunk_gated_delta_rule",
    "torch_recurrent_gated_delta_rule",
    "causal_conv1d_fn",
    "causal_conv1d_update",
)


def install_layer_scoped_kernels():
    """Redirect HF's module-global GDN kernel calls to the per-layer overrides.

    transformers' ``Qwen3_5GatedDeltaNet.forward`` calls the *module-level*
    ``torch_chunk_gated_delta_rule`` / ``torch_recurrent_gated_delta_rule`` /
    ``causal_conv1d_fn`` / ``causal_conv1d_update``. The ``@use_kernelized_func``
    decorator only rebinds those calls to ``self.<name>`` when the optional
    ``kernels`` package is active, so on a stock install LiteBoost's per-layer
    replacements would never be reached. This installs:

    * a thin ``Qwen3_5GatedDeltaNet.forward`` wrapper that records which layer's
      forward is running, and
    * a dispatcher for each of the four globals that forwards to the same-named
      attribute of that layer, falling back to the original HF function when the
      attribute is missing or ``None`` (``None`` is how boost disables a kernel).

    Idempotent; called once from ``boost_qwen3_5_9b`` before the per-layer patch
    loop. Layers whose ``forward`` is replaced wholesale by a boost patch bypass
    the scope wrapper and call ``self.<kernel>`` directly, which is equivalent.
    """
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen3_5_modeling

    if getattr(qwen3_5_modeling, "_lb_layer_scope_installed", False):
        return
    gdn_cls = qwen3_5_modeling.Qwen3_5GatedDeltaNet
    orig_forward = gdn_cls.forward

    def _scoped_forward(self, *args, **kwargs):
        prev = getattr(qwen3_5_modeling, "_lb_active_layer", None)
        qwen3_5_modeling._lb_active_layer = self
        try:
            return orig_forward(self, *args, **kwargs)
        finally:
            qwen3_5_modeling._lb_active_layer = prev

    gdn_cls.forward = _scoped_forward
    qwen3_5_modeling._lb_active_layer = None

    for name in _GDN_GLOBAL_KERNELS:
        original = getattr(qwen3_5_modeling, name)
        # HF's kernelised form drops the historical `torch_` prefix
        # (`torch_chunk_gated_delta_rule` -> `self.chunk_gated_delta_rule`), which is
        # also the name LiteBoost installs; accept either spelling.
        attr_names = (name, name[len("torch_"):]) if name.startswith("torch_") else (name,)

        def _dispatch(*args, _names=attr_names, _original=original, **kwargs):
            layer = getattr(qwen3_5_modeling, "_lb_active_layer", None)
            kernel = None
            if layer is not None:
                for attr in _names:
                    kernel = getattr(layer, attr, None)
                    if kernel is not None:
                        break
            return (kernel if kernel is not None else _original)(*args, **kwargs)

        setattr(qwen3_5_modeling, name, _dispatch)

    qwen3_5_modeling._lb_layer_scope_installed = True


def _fused_rms_norm_forward(self, x):
    """Fused RMSNorm forward using `torch_npu.npu_rms_norm` (single aclnn op)."""
    import torch_npu
    out, _ = torch_npu.npu_rms_norm(x, self._fused_gamma, epsilon=self.eps)
    return out


def _patch_rms_norm(rms_norm_module):
    """Patch a single `Qwen3_5RMSNorm` instance to use the fused forward."""
    with torch.no_grad():
        gamma = (1.0 + rms_norm_module.weight.float()).to(rms_norm_module.weight.dtype).contiguous()
    rms_norm_module.register_buffer("_fused_gamma", gamma, persistent=False)
    rms_norm_module.forward = types.MethodType(_fused_rms_norm_forward, rms_norm_module)


def _fused_ffn_forward(self, x):
    """Fused FFN forward: gate_proj + up_proj → single MatMul, silu+mul → npu_swiglu."""
    import torch_npu
    num_tokens = x.numel() // x.shape[-1] if x.dim() >= 1 else 1
    if hasattr(self, "_gate_up_w_int8_T") and num_tokens <= 8:
        x_2d = x.view(-1, x.shape[-1])
        fused_2d = torch_npu.npu_weight_quant_batchmatmul(
            x_2d, self._gate_up_w_int8_T, self._gate_up_antiquant_scale,
        )
        fused = fused_2d.view(*x.shape[:-1], -1)
    else:
        fused = torch.nn.functional.linear(x, self._fused_gate_up_weight)
    act = torch_npu.npu_swiglu(fused, dim=-1)
    tp_row_parallel = getattr(self.down_proj, "world", 1) > 1
    if hasattr(self, "_down_w_int8_T") and not tp_row_parallel:
        act_2d = act.view(-1, act.shape[-1])
        out_2d = torch_npu.npu_weight_quant_batchmatmul(
            act_2d, self._down_w_int8_T, self._down_antiquant_scale,
        )
        return out_2d.view(*act.shape[:-1], -1)
    if hasattr(self, "_down_weight_T") and not tp_row_parallel:
        return torch.matmul(act, self._down_weight_T)
    return self.down_proj(act)


def _patch_ffn(mlp_module):
    """Patch a single `Qwen3_5MLP` instance to use the fused FFN forward."""
    with torch.no_grad():
        fused_w = torch.cat(
            [mlp_module.gate_proj.weight, mlp_module.up_proj.weight], dim=0
        ).contiguous()
    mlp_module.register_buffer("_fused_gate_up_weight", fused_w, persistent=False)
    mlp_module.forward = types.MethodType(_fused_ffn_forward, mlp_module)


def _linear_matmul_forward(self, x):
    """Opt D5: nn.Linear forward using pre-transposed weight + torch.matmul."""
    out = torch.matmul(x, self._weight_T)
    if self.bias is not None:
        out = out + self.bias
    return out


def _patch_linear_matmul(linear_module):
    """Opt D5: Patch a single `nn.Linear` to use pre-transposed weight + matmul."""
    with torch.no_grad():
        w_trans = linear_module.weight.t().contiguous()
    linear_module.register_buffer("_weight_T", w_trans, persistent=False)
    linear_module.forward = types.MethodType(_linear_matmul_forward, linear_module)


def _patch_ffn_down_layout(mlp_module):
    """Opt D5: Pre-transpose down_proj weight for matmul layout in fused FFN."""
    with torch.no_grad():
        down_w_trans = mlp_module.down_proj.weight.t().contiguous()
    mlp_module.register_buffer("_down_weight_T", down_w_trans, persistent=False)


def _patch_ffn_down_w8a16(mlp_module):
    """Opt A2: Quantize down_proj weight to int8 + per-channel scale."""
    with torch.no_grad():
        w = mlp_module.down_proj.weight
        scale = w.abs().amax(dim=1) / 127.0
        w_int8 = (w / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
        w_int8_trans = w_int8.t().contiguous()
        antiquant_scale = scale.to(torch.bfloat16).unsqueeze(0)
    mlp_module.register_buffer("_down_w_int8_T", w_int8_trans, persistent=False)
    mlp_module.register_buffer("_down_antiquant_scale", antiquant_scale, persistent=False)


def _patch_ffn_gate_up_w8a16(mlp_module):
    """Opt A3: Quantize gate_up (fused) weight to int8 + per-channel scale."""
    with torch.no_grad():
        w = mlp_module._fused_gate_up_weight
        scale = w.abs().amax(dim=1) / 127.0
        w_int8 = (w / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
        w_int8_trans = w_int8.t().contiguous()
        antiquant_scale = scale.to(torch.bfloat16).unsqueeze(0)
    mlp_module.register_buffer("_gate_up_w_int8_T", w_int8_trans, persistent=False)
    mlp_module.register_buffer("_gate_up_antiquant_scale", antiquant_scale, persistent=False)
    if hasattr(mlp_module, "gate_proj") and hasattr(mlp_module.gate_proj, "weight"):
        del mlp_module.gate_proj.weight
    if hasattr(mlp_module, "up_proj") and hasattr(mlp_module.up_proj, "weight"):
        del mlp_module.up_proj.weight


def _fused_inproj_ba_forward(
    self,
    hidden_states: torch.Tensor,
    cache_params=None,
    attention_mask=None,
    **kwargs,
):
    """Fused linear-attn forward: `in_proj_b` + `in_proj_a` → single MatMul + chunk."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        apply_mask_to_padding_states,
    )

    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, seq_len, _ = hidden_states.shape

    use_precomputed_states = (
        cache_params is not None
        and cache_params.has_previous_state(self.layer_idx)
        and seq_len == 1
    )
    multi_round_prefill = (
        cache_params is not None
        and cache_params.has_previous_state(self.layer_idx)
        and seq_len > 1
        and getattr(self, "_multi_round_prefill", False)
    )

    conv_state = None
    recurrent_state = None
    if use_precomputed_states:
        # transformers 5.x keeps the linear-attention states in per-state dicts
        # (`conv_states[state_idx]`), not in a bare tensor.
        conv_state = cache_params.layers[self.layer_idx].conv_states[0]
        recurrent_state = cache_params.layers[self.layer_idx].recurrent_states[0]
    if multi_round_prefill:
        recurrent_state = cache_params.layers[self.layer_idx].recurrent_states[0]
        conv_state = cache_params.layers[self.layer_idx].conv_states[0]

    if hasattr(self, "_fused_qkv_z_ba_weight") and self._fused_qkv_z_ba_weight is not None:
        combined = torch.nn.functional.linear(hidden_states, self._fused_qkv_z_ba_weight)
        mixed_qkv, z, ba = torch.split(
            combined, self._inproj_4way_sizes, dim=-1
        )
        x_conv = combined
    else:
        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states)
        ba = torch.nn.functional.linear(hidden_states, self._fused_ba_weight)
        x_conv = mixed_qkv
    mixed_qkv = mixed_qkv.transpose(1, 2)
    z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)
    b, a = torch.chunk(ba, 2, dim=-1)

    if use_precomputed_states:
        mixed_qkv = self.causal_conv1d_update(
            mixed_qkv,
            conv_state,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            self.activation,
        )
    elif multi_round_prefill:
        left_ctx = conv_state[:, :, 1:]
        padded = torch.cat([left_ctx, mixed_qkv], dim=-1)
        conv_out = F.silu(self.conv1d(padded))
        mixed_qkv = conv_out[:, :, 3:3 + seq_len]
        static_idx = getattr(self, "_static_conv_state_idx", None)
        if torch.is_tensor(static_idx) and static_idx.numel() == 4:
            conv_state.copy_(
                torch.index_select(padded, -1, static_idx)
            )
    else:
        if (getattr(self, "_lb_fused_conv_silu", False)
                and batch_size == 1
                and seq_len >= self.conv_kernel_size):
            seq_len8 = None
            if getattr(self, "_lb_bucketed_exec", False):
                s8 = getattr(self, "_static_seq_len8", None)
                if torch.is_tensor(s8) and s8.numel() == 8:
                    seq_len8 = s8
            if seq_len8 is None:
                s8 = getattr(self, "_eager_seq_len8", None)
                if not (torch.is_tensor(s8) and s8.numel() == 8
                        and s8.device == x_conv.device):
                    s8 = torch.zeros(8, dtype=torch.int32, device=x_conv.device)
                    self._eager_seq_len8 = s8
                s8.fill_(seq_len)
                seq_len8 = s8
            conv_bias = getattr(self, "_conv_bias_z", None)
            if conv_bias is None:
                conv_bias = self.conv1d.bias
            y, conv_state_tm = torch.ops.lite_boost.fused_causal_conv_silu(
                x_conv, self._conv_w_t, conv_bias, seq_len8)
            if cache_params is not None:
                cache_params.update_conv_state(conv_state_tm.transpose(1, 2), self.layer_idx)
            mixed_qkv = y.transpose(1, 2)
        elif self.causal_conv1d_fn is not None:
            if cache_params is not None:
                static_idx = getattr(self, "_static_prefill_conv_idx", None)
                if (getattr(self, "_lb_bucketed_exec", False)
                        and torch.is_tensor(static_idx)
                        and static_idx.numel() == self.conv_kernel_size):
                    conv_state = torch.index_select(mixed_qkv, -1, static_idx)
                else:
                    conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                conv_state = cache_params.update_conv_state(conv_state, self.layer_idx)
            mixed_qkv = self.causal_conv1d_fn(
                x=mixed_qkv,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
                seq_idx=None,
            )
        else:
            if cache_params is not None:
                static_idx = getattr(self, "_static_prefill_conv_idx", None)
                if (getattr(self, "_lb_bucketed_exec", False)
                        and torch.is_tensor(static_idx)
                        and static_idx.numel() == self.conv_kernel_size):
                    conv_state = torch.index_select(mixed_qkv, -1, static_idx)
                else:
                    conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                conv_state = cache_params.update_conv_state(conv_state, self.layer_idx)
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv,
        [self.key_dim, self.key_dim, self.value_dim],
        dim=-1,
    )

    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

    beta = b.sigmoid()
    if hasattr(self, "_A_neg_exp") and self._A_neg_exp is not None:
        g = self._A_neg_exp * F.softplus(a.float() + self.dt_bias)
    else:
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    if self.num_v_heads // self.num_k_heads > 1:
        if (not use_precomputed_states
                and getattr(self, "_lb_gqa_native", False)):
            pass
        else:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    if not use_precomputed_states:
        core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
            query, key, value,
            g=g, beta=beta,
            initial_state=recurrent_state if multi_round_prefill else None,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
        )
    else:
        core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
            query, key, value,
            g=g, beta=beta,
            initial_state=recurrent_state,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
        )

    if cache_params is not None:
        cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

    return self.out_proj(core_attn_out)


def _patch_inproj_ba(linear_attn_module):
    """Patch a single `Qwen3_5GatedDeltaNet` instance to merge in_proj_b + in_proj_a."""
    with torch.no_grad():
        fused_ba = torch.cat(
            [linear_attn_module.in_proj_b.weight, linear_attn_module.in_proj_a.weight],
            dim=0,
        ).contiguous()
    linear_attn_module.register_buffer("_fused_ba_weight", fused_ba, persistent=False)
    # The fused forward reaches the single-token conv step through `self.`, while
    # transformers' own forward calls the module-level kernel; bind the graph-safe
    # implementation here so the fused forward also works outside graph mode.
    linear_attn_module.causal_conv1d_update = torch_causal_conv1d_update_graph
    linear_attn_module.forward = types.MethodType(
        _fused_inproj_ba_forward, linear_attn_module
    )


def _patch_inproj_4way(linear_attn_module):
    """Merge in_proj_qkv + in_proj_z + fused_ba into a single MatMul (Opt 5d)."""
    with torch.no_grad():
        qkv_w = linear_attn_module.in_proj_qkv.weight
        z_w = linear_attn_module.in_proj_z.weight
        ba_w = linear_attn_module._fused_ba_weight
        merged = torch.cat([qkv_w, z_w, ba_w], dim=0).contiguous()
    linear_attn_module.register_buffer(
        "_fused_qkv_z_ba_weight", merged, persistent=False
    )
    linear_attn_module._inproj_4way_sizes = [qkv_w.shape[0], z_w.shape[0], ba_w.shape[0]]


def _patch_a_log_precompute(linear_attn_module):
    """Precompute `-A_log.float().exp()` at patch time (Opt 19)."""
    with torch.no_grad():
        a_neg_exp = (-linear_attn_module.A_log.float().exp()).contiguous()
    linear_attn_module.register_buffer(
        "_A_neg_exp", a_neg_exp, persistent=False
    )


def _patch_fused_conv_silu(linear_attn_module):
    """Enable the FusedCausalConvSilu AscendC op for this layer's prefill (F1)."""
    try:
        torch.ops.lite_boost.fused_causal_conv_silu
    except (AttributeError, RuntimeError):
        return False
    with torch.no_grad():
        w_t = linear_attn_module.conv1d.weight.squeeze(1).transpose(0, 1).contiguous()
    linear_attn_module.register_buffer("_conv_w_t", w_t, persistent=False)
    if linear_attn_module.conv1d.bias is None:
        zb = torch.zeros(
            linear_attn_module.conv1d.weight.shape[0],
            dtype=linear_attn_module.conv1d.weight.dtype,
            device=linear_attn_module.conv1d.weight.device,
        )
        linear_attn_module.register_buffer("_conv_bias_z", zb, persistent=False)
    linear_attn_module._lb_fused_conv_silu = True
    return True


def _fused_rms_norm_gated_forward(self, hidden_states, gate=None):
    """Fused RMSNormGated forward: `npu_rms_norm` + `F.silu(gate, fp32)` + mul."""
    import torch_npu
    norm_out, _ = torch_npu.npu_rms_norm(
        hidden_states, self._fused_gamma, epsilon=self.variance_epsilon
    )
    gate_act = torch.nn.functional.silu(gate.float())
    return (norm_out * gate_act).to(hidden_states.dtype)


def _patch_rms_norm_gated(norm_module):
    """Patch a single `Qwen3_5RMSNormGated` instance to use the fused forward."""
    with torch.no_grad():
        gamma = norm_module.weight.float().to(norm_module.weight.dtype).contiguous()
    norm_module.register_buffer("_fused_gamma", gamma, persistent=False)
    norm_module.forward = types.MethodType(_fused_rms_norm_gated_forward, norm_module)


def _vision_attn_forward(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: torch.Tensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    **kwargs,
) -> torch.Tensor:
    """NPU flash-attention forward for `Qwen3_5VisionAttention`."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb_vision

    seq_length = hidden_states.shape[0]
    query_states, key_states, value_states = (
        self.qkv(hidden_states)
        .reshape(seq_length, 3, self.num_heads, -1)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_vision(
        query_states, key_states, cos, sin,
    )

    head_dim = query_states.shape[-1]
    pad_to = (head_dim + 15) // 16 * 16
    pad_n = pad_to - head_dim

    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    q_chunks = torch.split(query_states, lengths, dim=0)
    k_chunks = torch.split(key_states, lengths, dim=0)
    v_chunks = torch.split(value_states, lengths, dim=0)

    max_len = max(lengths)
    q_padded = torch.stack(
        [F.pad(c, (0, 0, 0, max_len - c.shape[0])) for c in q_chunks], dim=0,
    )
    k_padded = torch.stack(
        [F.pad(c, (0, 0, 0, max_len - c.shape[0])) for c in k_chunks], dim=0,
    )
    v_padded = torch.stack(
        [F.pad(c, (0, 0, 0, max_len - c.shape[0])) for c in v_chunks], dim=0,
    )

    if pad_n > 0:
        q_padded = F.pad(q_padded, (0, pad_n))
        k_padded = F.pad(k_padded, (0, pad_n))
        v_padded = F.pad(v_padded, (0, pad_n))

    q_lens = torch.tensor(lengths, dtype=torch.int32, device=query_states.device)
    k_lens = q_lens

    attn_output = flash_attention(
        q_padded, k_padded, v_padded,
        q_lens=q_lens,
        k_lens=k_lens,
        softmax_scale=self.scaling,
        causal=False,
        dtype=q_padded.dtype,
    )

    attn_output = attn_output[..., :head_dim]
    attn_chunks = [attn_output[i, :lengths[i]] for i in range(len(lengths))]
    attn_output = torch.cat(attn_chunks, dim=0)
    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    attn_output = self.proj(attn_output)
    return attn_output


def _patch_vision_attention(attn_module):
    """Patch a single `Qwen3_5VisionAttention` instance to use NPU FA."""
    attn_module.forward = types.MethodType(_vision_attn_forward, attn_module)


class _RowParallelLinear(nn.Module):
    """Wrap a row-sharded `nn.Linear` so its forward fuses matmul + HCCL all-reduce."""

    def __init__(self, linear: nn.Linear, rank: int, world: int):
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.rank = rank
        self.world = world
        self._hcom = None

    def forward(self, x):
        """Row-parallel linear: local matmul then all-reduce across TP ranks."""
        if self.world <= 1:
            return F.linear(x, self.weight, self.bias)
        bias = self.bias if self.rank == 0 else None
        if self._hcom is None and is_distributed_active():
            self._hcom = get_hcom_name()
        return mm_all_reduce(x, self.weight, bias, hcom=self._hcom)


def _zeros_state(layer: "nn.Module", batch_size: int, device, dtype) -> torch.Tensor:
    """Zero recurrent state `[B, num_v_heads, head_k_dim, head_v_dim]`."""
    return torch.zeros(
        batch_size, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim,
        device=device, dtype=dtype,
    )


def _chunk_gated_delta_rule_tnd(layer, query, key, value, g, beta,
                                initial_state, output_final_state,
                                use_qk_l2norm_in_kernel, scale_value):
    """Opt F2 (prefill): B==1 direct TND call into the CANN chunk op."""
    import torch_npu

    seq_len = query.shape[1]
    q_heads, k_dim = query.shape[-2], query.shape[-1]
    v_heads, v_dim = value.shape[-2], value.shape[-1]
    device, dtype = query.device, query.dtype
    low_dtype = dtype if dtype in (torch.bfloat16, torch.float16) else torch.bfloat16

    q_tnd = query.reshape(seq_len, q_heads, k_dim)
    k_tnd = key.reshape(seq_len, q_heads, k_dim)
    kernel_l2norm = (use_qk_l2norm_in_kernel
                     and getattr(layer, "_lb_chunk_qk_l2norm_sink", False))
    if use_qk_l2norm_in_kernel and not kernel_l2norm:
        q_tnd = l2norm(q_tnd, dim=-1)
        k_tnd = l2norm(k_tnd, dim=-1)
    else:
        q_tnd = q_tnd.contiguous()
        k_tnd = k_tnd.contiguous()
    v_tnd = value.reshape(seq_len, v_heads, v_dim).contiguous().to(low_dtype)
    beta_tnd = None
    if beta is not None:
        beta_tnd = beta.reshape(seq_len, v_heads).contiguous().to(low_dtype)
    g_tnd = None
    if g is not None:
        g_tnd = g.reshape(seq_len, v_heads).float().contiguous()
    q_tnd = q_tnd.to(low_dtype)
    k_tnd = k_tnd.to(low_dtype)

    if initial_state is None:
        state_cann = torch.zeros(1, v_heads, v_dim, k_dim, device=device, dtype=low_dtype)
    else:
        state_cann = initial_state.to(device=device, dtype=low_dtype) \
                              .transpose(-1, -2).contiguous()
    if torch_npu.get_npu_format(state_cann) != 2:
        state_cann = torch_npu.npu_format_cast(state_cann, 2)

    static_buf = getattr(layer, "_static_actual_seq_len", None)
    use_static_seq = static_buf is not None and static_buf.numel() > 0
    if use_static_seq:
        actual_seq_lengths = static_buf
    else:
        actual_seq_lengths = torch.full((1,), seq_len, dtype=torch.int32, device=device)

    out_zero_pad = getattr(layer, "_lb_chunk_out_zero_pad", False)
    if kernel_l2norm:
        # Norm+CGDR fusion: q/k are passed RAW (normalized in-kernel), and the
        # pack-row zeroing is optionally done in-kernel too.
        out_tnd, state_out_cann = torch.ops.lite_boost.fused_qk_l2norm_chunk_gated_delta_rule(
            q_tnd, k_tnd, v_tnd, beta_tnd, state_cann,
            actual_seq_lengths, g_tnd, float(scale_value), out_zero_pad,
        )
        pad_in_python = use_static_seq and not out_zero_pad
    else:
        # Plain chunk op: no in-kernel norm / pad handling, so the pad rows are
        # masked here.
        out_tnd, state_out_cann = torch.ops.lite_boost.chunk_gated_delta_rule(
            q_tnd, k_tnd, v_tnd, beta_tnd, state_cann,
            actual_seq_lengths, g_tnd, float(scale_value),
        )
        pad_in_python = use_static_seq
    if pad_in_python:
        pad_rows = getattr(layer, "_static_pad_rows", None)
        if pad_rows is None or pad_rows.numel() != seq_len:
            pos = torch.arange(seq_len, dtype=torch.int32, device=device)
            pad_rows = pos >= actual_seq_lengths
        out_tnd = out_tnd.masked_fill(pad_rows.view(seq_len, 1, 1), 0)
    out_bthd = out_tnd.reshape(1, seq_len, v_heads, v_dim)
    state_out = None
    if output_final_state:
        state_out = state_out_cann.transpose(-1, -2).contiguous()
    return out_bthd, state_out


def _chunk_wrapper(layer: "nn.Module"):
    """Return a HF-signature chunk kernel that calls the LiteBoost aclnn op."""
    head_k_dim = layer.head_k_dim
    scale_value = 1.0 / math.sqrt(head_k_dim)

    def _fn(query, key, value, g=None, beta=None, initial_state=None,
            output_final_state=False, use_qk_l2norm_in_kernel=True, **kwargs):
        # HF forwards extra kwargs (cu_seqlens, cache_position, ...) that the
        # single-sequence TND path does not model; they are None/empty here.
        del kwargs
        if query.shape[0] == 1:
            return _chunk_gated_delta_rule_tnd(
                layer, query, key, value, g, beta, initial_state,
                output_final_state, use_qk_l2norm_in_kernel, scale_value,
            )
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()
        if use_qk_l2norm_in_kernel:
            q = l2norm(q, dim=-1)
            k = l2norm(k, dim=-1)
        beta_bnsd = beta.transpose(1, 2).contiguous() if beta is not None else None
        g_bnsd = g.transpose(1, 2).contiguous() if g is not None else None

        batch, _, seq_len, _ = q.shape
        device, dtype = q.device, q.dtype
        if initial_state is None:
            state_in = _zeros_state(layer, batch, device, dtype)
        else:
            state_in = initial_state.to(device=device, dtype=dtype)

        static_buf = getattr(layer, "_static_actual_seq_len", None)
        if static_buf is not None and static_buf.numel() > 0:
            actual_seq_lengths = static_buf
        else:
            actual_seq_lengths = torch.full((batch,), seq_len, dtype=torch.int32, device=device)

        out_bnsd, state_out = chunk_gated_delta_rule(
            q, k, v, beta_bnsd, state_in, actual_seq_lengths,
            g=g_bnsd, scale_value=scale_value,
        )
        out_bthd = out_bnsd.transpose(1, 2).contiguous()
        if not output_final_state:
            state_out = None
        return out_bthd, state_out

    return _fn


def _recurrent_wrapper(layer: "nn.Module", opt_f2: bool = False):
    """Return a HF-signature recurrent kernel that calls the LiteBoost aclnn op."""
    head_k_dim = layer.head_k_dim
    scale_value = 1.0 / math.sqrt(head_k_dim)

    def _fn(query, key, value, g=None, beta=None, initial_state=None,
            output_final_state=False, use_qk_l2norm_in_kernel=True, **kwargs):
        # HF forwards extra kwargs (cu_seqlens, ...); the recurrent op takes none.
        del kwargs
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        if use_qk_l2norm_in_kernel:
            q = l2norm(q, dim=-1)
            k = l2norm(k, dim=-1)
        beta_bnsd = beta.transpose(1, 2) if beta is not None else None
        g_bnsd = g.transpose(1, 2) if g is not None else None

        batch, _, seq_len, _ = q.shape
        device, dtype = q.device, q.dtype
        if initial_state is None:
            state_in = _zeros_state(layer, batch, device, dtype)
        else:
            state_in = initial_state.to(device=device, dtype=dtype)

        actual_seq_lengths = torch.full((batch,), seq_len, dtype=torch.int32, device=device)
        ssm_state_indices = torch.arange(batch, dtype=torch.int32, device=device)
        num_accepted_tokens = actual_seq_lengths
        gk = torch.zeros(batch, layer.num_v_heads, seq_len, head_k_dim,
                         dtype=torch.float32, device=device)

        if opt_f2 and seq_len == 1:
            import torch_npu
            q_heads = q.shape[1]
            v_heads = v.shape[1]
            k_dim = q.shape[3]
            v_dim = v.shape[3]
            q_tnd = q.transpose(1, 2).reshape(-1, q_heads, k_dim)
            k_tnd = k.transpose(1, 2).reshape(-1, q_heads, k_dim)
            v_tnd = v.transpose(1, 2).reshape(-1, v_heads, v_dim)
            beta_tnd = beta_bnsd.transpose(1, 2).reshape(-1, v_heads) if beta_bnsd is not None else None
            g_tnd = g_bnsd.transpose(1, 2).reshape(-1, v_heads) if g_bnsd is not None else None
            gk_tnd = gk.transpose(1, 2).reshape(-1, v_heads, k_dim)
            state_cann = state_in.transpose(-1, -2).contiguous()
            if torch_npu.get_npu_format(state_cann) != 2:
                state_cann = torch_npu.npu_format_cast(state_cann, 2)
            seq_lengths_int = actual_seq_lengths.int().contiguous()
            out_tnd, state_out_cann = torch.ops.lite_boost.recurrent_gated_delta_rule(
                q_tnd, k_tnd, v_tnd, beta_tnd, state_cann,
                seq_lengths_int, ssm_state_indices, g_tnd, gk_tnd,
                num_accepted_tokens, scale_value,
            )
            out_bnsd = out_tnd.reshape(batch, seq_len, v_heads, v_dim).transpose(1, 2).contiguous()
            state_out = state_out_cann.transpose(-1, -2).contiguous()
        else:
            out_bnsd, state_out = recurrent_gated_delta_rule(
                q, k, v, beta_bnsd, state_in, actual_seq_lengths,
                ssm_state_indices, g_bnsd, gk, num_accepted_tokens,
                scale_value=scale_value,
            )
        out_bthd = out_bnsd.transpose(1, 2)
        if not output_final_state:
            state_out = None
        return out_bthd, state_out

    return _fn


def torch_causal_conv1d_update_graph(
    hidden_states,
    conv_state,
    weight,
    bias=None,
    activation=None,
):
    """Graph-safe depthwise causal_conv1d_update for decode (T=1)."""
    _, _, _ = hidden_states.shape
    _ = conv_state.shape[-1]
    _ = weight.shape[-1]
    w = weight.squeeze(1)

    hidden_d = hidden_states.to(weight.dtype)
    window = torch.cat([conv_state[:, :, 1:], hidden_d], dim=-1)
    conv_state.copy_(window)

    out = (window * w).sum(dim=-1, keepdim=True)
    if bias is not None:
        out = out + bias.view(1, -1, 1)
    if activation is not None:
        out = F.silu(out)
    out = out.to(hidden_states.dtype)
    return out


def _patch_full_attn_proj_fuse(attn_module):
    """Build fused q|k|v + gate weights for a full-attn layer (Opt F6, 1p only)."""
    with torch.no_grad():
        w_q = attn_module.q_proj.weight
        d = attn_module.head_dim
        hq = w_q.shape[0] // 2 // d
        w_q_v = w_q.view(hq, 2, d, -1)
        w_qkv = torch.cat(
            [w_q_v[:, 0].reshape(hq * d, -1),
             attn_module.k_proj.weight,
             attn_module.v_proj.weight], dim=0
        ).contiguous()
        w_gate = w_q_v[:, 1].reshape(hq * d, -1).contiguous()
        attn_module.register_buffer("_lb_qkv_weight", w_qkv, persistent=False)
        attn_module.register_buffer("_lb_gate_weight", w_gate, persistent=False)
        attn_module._lb_proj_fused = True
    return True


def _full_attn_project(self, hidden_states):
    """q/k/v (+gate) projections for full-attn layers, pre-norm, [B, T, H, D]."""
    input_shape = hidden_states.shape[:-1]
    d = self.head_dim
    if getattr(self, "_lb_proj_fused", False):
        hq = self._lb_gate_weight.shape[0] // d
        nkv_d = self.k_proj.out_features
        qkv = torch.nn.functional.linear(hidden_states, self._lb_qkv_weight)
        gate = torch.nn.functional.linear(hidden_states, self._lb_gate_weight)
        q, k, v = torch.split(qkv, [hq * d, nkv_d, nkv_d], dim=-1)
        hidden_shape = (*input_shape, -1, d)
        return q.view(hidden_shape), gate, k.view(hidden_shape), v.view(hidden_shape)
    hidden_shape = (*input_shape, -1, d)
    query_states, gate = torch.chunk(
        self.q_proj(hidden_states).view(*input_shape, -1, d * 2), 2, dim=-1
    )
    gate = gate.reshape(*input_shape, -1)
    return (
        query_states.view(hidden_shape), gate,
        self.k_proj(hidden_states).view(hidden_shape),
        self.v_proj(hidden_states).view(hidden_shape),
    )


_PFA_CAUSAL_MASKS = {}


def _pfa_causal_mask(seq_len, kv_len, device):
    """Cached bool causal mask of shape [1, 1, seq_len, kv_len] for PFA.

    ``True`` means masked, which is the convention `npu_prompt_flash_attention`
    expects for ``sparse_mode=0``.  Query row ``i`` sees key rows ``[0,
    kv_len - seq_len + i]``: the leading ``kv_len - seq_len`` columns belong to
    an already-complete cached prefix (chunked prefill over a KV cache) and are
    left fully visible.  Masks are cached per (length, device) so the steady
    state costs a dict lookup.
    """
    key = (seq_len, kv_len, str(device))
    mask = _PFA_CAUSAL_MASKS.get(key)
    if mask is None:
        prefix = kv_len - seq_len
        causal = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1,
        )
        if prefix > 0:
            causal = torch.cat(
                (torch.zeros(seq_len, prefix, dtype=torch.bool, device=device), causal),
                dim=1,
            )
        mask = causal.unsqueeze(0).unsqueeze(0)
        _PFA_CAUSAL_MASKS[key] = mask
    return mask


def _pfa_full_attn(query_states, key_states, value_states, scaling):
    """Causal prompt-flash-attention over BNSD query/key/value tensors.

    `lite_boost.flash_attention` documents -- and the NPU backend implements --
    global attention only: its `causal` flag is a GPU-backend feature, so
    routing causal prefill through it silently attends to future tokens.  Call
    `npu_prompt_flash_attention` with an explicit causal mask instead.  GQA is
    handled by the kernel itself (`num_key_value_heads`), so k/v stay at their
    native head count.  Returns a BNSD tensor like the inputs.
    """
    import torch_npu

    return torch_npu.npu_prompt_flash_attention(
        query_states, key_states, value_states,
        num_heads=query_states.shape[1],
        num_key_value_heads=key_states.shape[1],
        input_layout="BNSD",
        scale_value=scaling,
        atten_mask=_pfa_causal_mask(
            query_states.shape[2], key_states.shape[2], query_states.device),
        sparse_mode=0,
        pre_tokens=2147483647,
        next_tokens=2147483647,
    )


def qwen3_5_full_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values=None,
    **kwargs,
):
    """Drop-in `Qwen3_5Attention.forward` on fused `npu_prompt_flash_attention`."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

    input_shape = hidden_states.shape[:-1]

    q_raw, gate, k_raw, v_raw = _full_attn_project(self, hidden_states)
    query_states = self.q_norm(q_raw).transpose(1, 2)
    key_states = self.k_norm(k_raw).transpose(1, 2)
    value_states = v_raw.transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

    attn_output = _pfa_full_attn(query_states, key_states, value_states, self.scaling)
    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = attn_output * torch.sigmoid(gate)
    attn_output = self.o_proj(attn_output)
    return attn_output, None

_STATIC_BUFFERS = {}


def _get_static_buffers(layer_idx, batch, max_cache_len, device):
    """Return (or lazily create) the per-layer static mask buffer."""
    key = (layer_idx, str(device))
    buf = _STATIC_BUFFERS.get(key)
    if buf is None or buf["max_cache_len"] != max_cache_len or buf["mask"].shape[0] != batch:
        mask = torch.full((batch, 1, max_cache_len), -1e4, dtype=torch.float16, device=device)
        buf = {"mask": mask, "max_cache_len": max_cache_len}
        _STATIC_BUFFERS[key] = buf
    return buf


def qwen3_5_full_attn_forward_static(
    self,
    hidden_states: torch.Tensor,
    position_embeddings,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values=None,
    **kwargs,
):
    """Static-cache full-attn forward for graph-mode decode (T=1 only)."""
    seq_len = hidden_states.shape[1]
    if seq_len > 1:
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        input_shape = hidden_states.shape[:-1]
        q_raw, gate, k_raw, v_raw = _full_attn_project(self, hidden_states)
        query_states = self.q_norm(q_raw).transpose(1, 2)
        key_states = self.k_norm(k_raw).transpose(1, 2)
        value_states = v_raw.transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_cache_full = None
        value_cache_full = None
        if past_key_values is not None:
            static_cp = getattr(self, "_static_cache_position", None)
            if static_cp is not None and static_cp.shape[0] == seq_len:
                cache_position = static_cp
            else:
                cache_position = torch.arange(seq_len, device=hidden_states.device)
            update_ret = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_position=cache_position,
            )
            if isinstance(update_ret, tuple) and len(update_ret) == 2:
                key_cache_full, value_cache_full = update_ret

        if getattr(self, "_bucketed_prefill", False):
            import torch_npu
            q_bnsd = query_states
            num_heads_local = q_bnsd.shape[1]
            mask_shape = self._prefill_mask.shape if self._prefill_mask is not None else None
            if mask_shape is not None and mask_shape[-1] != seq_len and \
               key_cache_full is not None:
                k_bnsd = key_cache_full
                v_bnsd = value_cache_full
            else:
                k_bnsd = key_states
                v_bnsd = value_states
            num_kv_heads_local = k_bnsd.shape[1]
            attn_out = torch_npu.npu_prompt_flash_attention(
                q_bnsd, k_bnsd, v_bnsd,
                num_heads=num_heads_local,
                num_key_value_heads=num_kv_heads_local,
                input_layout="BNSD_BSND",
                scale_value=self.scaling,
                atten_mask=self._prefill_mask,
                pre_tokens=2147483647,
                next_tokens=2147483647,
                sparse_mode=0,
            )
            attn_output = attn_out.reshape(*input_shape, -1)
            attn_output = attn_output * torch.sigmoid(gate)
            attn_output = self.o_proj(attn_output)
            return attn_output, None

        attn_output = _pfa_full_attn(query_states, key_states, value_states, self.scaling)
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    import torch_npu
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states, gate = torch.chunk(
        self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
    )
    gate = gate.reshape(*input_shape, -1)
    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    key_cache_full, value_cache_full = past_key_values.update(
        key_states, value_states, self.layer_idx
    )

    cur_len_tensor = past_key_values.get_seq_length()
    if not torch.is_tensor(cur_len_tensor):
        cur_len_tensor = torch.tensor([cur_len_tensor], dtype=torch.int64, device=hidden_states.device)

    batch = hidden_states.shape[0]
    max_cache_len = key_cache_full.shape[2]
    buf = _get_static_buffers(self.layer_idx, batch, max_cache_len, hidden_states.device)
    static_mask = buf["mask"]

    new_pos = (cur_len_tensor - 1).clamp(min=0).to(torch.int64).view(1)
    static_mask.index_copy_(
        2,
        new_pos,
        torch.zeros(batch, 1, 1, dtype=static_mask.dtype, device=static_mask.device),
    )

    q_bnsd = query_states
    k_bnsd = key_cache_full
    v_bnsd = value_cache_full

    num_heads_local = q_bnsd.shape[1]
    num_kv_heads_local = k_bnsd.shape[1]

    attn_out = torch_npu.npu_prompt_flash_attention(
        q_bnsd, k_bnsd, v_bnsd,
        num_heads=num_heads_local,
        num_key_value_heads=num_kv_heads_local,
        input_layout="BNSD_BSND",
        scale_value=self.scaling,
        atten_mask=static_mask,
        pre_tokens=2147483647,
        next_tokens=2147483647,
        sparse_mode=0,
    )

    attn_output = attn_out.reshape(*input_shape, -1)
    attn_output = attn_output * torch.sigmoid(gate)
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def qwen3_5_text_model_forward_graph(
    self,
    input_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    past_key_values=None,
    inputs_embeds: torch.Tensor | None = None,
    use_cache: bool | None = None,
    **kwargs,
):
    """Graph-safe `Qwen3_5TextModel.forward` for decode (T=1)."""
    seq_len = (inputs_embeds.shape[1] if inputs_embeds is not None
               else input_ids.shape[1] if input_ids is not None else 0)
    if seq_len > 1:
        if not getattr(self, "_bucketed_prefill", False):
            return self._forward_eager(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                **kwargs,
            )

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if position_ids is None:
        past_seen = (past_key_values.get_seq_length()
                     if past_key_values is not None else 0)
        position_ids = torch.arange(
            inputs_embeds.shape[1], device=inputs_embeds.device,
        ) + past_seen
        position_ids = position_ids.view(1, 1, -1).expand(
            4, inputs_embeds.shape[0], -1)
    if position_ids is not None and position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        mrope_position_ids = position_ids[1:]
    else:
        text_position_ids = None
        mrope_position_ids = position_ids

    causal_mask = None
    if seq_len > 1:
        linear_attn_mask = attention_mask
    else:
        # transformers 5.19 builds this mask inside `create_recurrent_attention_mask`,
        # which deliberately returns None for a single decode token (a generated
        # token is never padding) -- the pre-5.19 `_update_linear_attn_mask` helper
        # this graph path was written against no longer exists.
        linear_attn_mask = None

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, mrope_position_ids)

    for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
        layer_mask = linear_attn_mask if self.config.layer_types[i] == "linear_attention" else causal_mask
        hidden_states = decoder_layer(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=layer_mask,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

    hidden_states = self.norm(hidden_states)
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ModelOutputWithPast
    return Qwen3_5ModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        hidden_states=None,
        attentions=None,
    )


def _shard_in_proj_qkv_and_conv(layer: "nn.Module", rank: int, world: int):
    """Head-structured column-shard of `in_proj_qkv` + matching `conv1d` channels."""
    lin = layer.in_proj_qkv
    w = lin.weight.data
    kd, vd = layer.key_dim, layer.value_dim
    assert kd % world == 0 and vd % world == 0, "key_dim/value_dim must divide world"
    kd_r, vd_r = kd // world, vd // world
    q_part = w[rank * kd_r:(rank + 1) * kd_r]
    k_part = w[kd + rank * kd_r:kd + (rank + 1) * kd_r]
    v_part = w[2 * kd + rank * vd_r:2 * kd + (rank + 1) * vd_r]
    new_w = torch.cat([q_part, k_part, v_part], dim=0).contiguous()
    lin.weight = nn.Parameter(new_w.to(lin.weight.device), requires_grad=False)

    conv_w = layer.conv1d.weight.data
    cq = conv_w[rank * kd_r:(rank + 1) * kd_r]
    ck = conv_w[kd + rank * kd_r:kd + (rank + 1) * kd_r]
    cv = conv_w[2 * kd + rank * vd_r:2 * kd + (rank + 1) * vd_r]
    new_conv_w = torch.cat([cq, ck, cv], dim=0).contiguous()
    layer.conv1d.weight = nn.Parameter(new_conv_w.to(layer.conv1d.weight.device),
                                       requires_grad=False)
    layer.conv1d.in_channels = kd_r * 2 + vd_r
    layer.conv1d.out_channels = kd_r * 2 + vd_r
    layer.conv1d.groups = kd_r * 2 + vd_r


def shard_qwen3_5_tp(model, rank: int, world: int):
    """In-place tensor-parallel shard of a `Qwen3_5ForConditionalGeneration` / `ForCausalLM`."""
    text_model = _get_text_model(model)
    layers = text_model.layers
    layer_types = text_model.config.layer_types

    for i, layer in enumerate(layers):
        if layer_types[i] == "full_attention":
            attn = layer.self_attn
            n_heads = attn.config.num_attention_heads
            n_kv = attn.config.num_key_value_heads
            assert n_heads % world == 0 and n_kv % world == 0
            shard_colwise(attn.q_proj, rank, world)
            shard_colwise(attn.k_proj, rank, world)
            shard_colwise(attn.v_proj, rank, world)
            shard_rowwise(attn.o_proj, rank, world)
            attn.o_proj = _RowParallelLinear(attn.o_proj, rank, world)
        elif layer_types[i] == "linear_attention":
            la = layer.linear_attn
            n_v = la.num_v_heads
            n_k = la.num_k_heads
            assert n_v % world == 0 and n_k % world == 0
            _shard_in_proj_qkv_and_conv(la, rank, world)
            shard_colwise(la.in_proj_z, rank, world)
            shard_colwise(la.in_proj_b, rank, world)
            shard_colwise(la.in_proj_a, rank, world)
            shard_rowwise(la.out_proj, rank, world)
            la.out_proj = _RowParallelLinear(la.out_proj, rank, world)
            n_v_r = n_v // world
            la.A_log = nn.Parameter(la.A_log.data[rank * n_v_r:(rank + 1) * n_v_r].contiguous(),
                                    requires_grad=False)
            la.dt_bias = nn.Parameter(la.dt_bias.data[rank * n_v_r:(rank + 1) * n_v_r].contiguous(),
                                      requires_grad=False)
            la.num_v_heads = n_v_r
            la.num_k_heads = n_k // world
            la.key_dim = la.head_k_dim * la.num_k_heads
            la.value_dim = la.head_v_dim * la.num_v_heads
            la.conv_dim = la.key_dim * 2 + la.value_dim
        mlp = layer.mlp
        shard_colwise(mlp.gate_proj, rank, world)
        shard_colwise(mlp.up_proj, rank, world)
        shard_rowwise(mlp.down_proj, rank, world)
        mlp.down_proj = _RowParallelLinear(mlp.down_proj, rank, world)
    return model


def qwen3_5_full_attn_forward_cp(
    self,
    hidden_states: torch.Tensor,
    position_embeddings,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values=None,
    **kwargs,
):
    """Head-split CP version of the full-attention forward (multi-p cp mode)."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        apply_rotary_pos_emb, repeat_kv,
    )

    world = dist.get_world_size() if (dist is not None and dist.is_initialized()) else 1
    if world == 1:
        return qwen3_5_full_attn_forward(
            self, hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs
        )

    seq_len = hidden_states.shape[1]
    if seq_len < world:
        if getattr(self, "_lite_boost_graph_mode", False):
            return qwen3_5_full_attn_forward_static(
                self, hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs
            )
        return qwen3_5_full_attn_forward(
            self, hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs
        )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states, gate = torch.chunk(
        self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
    )
    gate = gate.reshape(*input_shape, -1)
    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    key_cache_full = None
    value_cache_full = None
    if past_key_values is not None:
        update_ret = past_key_values.update(key_states, value_states, self.layer_idx)
        if isinstance(update_ret, tuple) and len(update_ret) == 2:
            key_cache_full, value_cache_full = update_ret

    rank = dist.get_rank()
    num_heads = query_states.size(1)
    num_kv_heads = key_states.size(1)

    if getattr(self, "_bucketed_prefill", False) and \
       getattr(self, "_prefill_mask", None) is not None:
        import torch_npu
        qh = num_heads // world
        kvh = num_kv_heads // world
        q_sl = slice(rank * qh, (rank + 1) * qh)
        kv_sl = slice(rank * kvh, (rank + 1) * kvh)
        mask_shape = self._prefill_mask.shape
        if mask_shape[-1] != seq_len and key_cache_full is not None:
            k_bnsd = key_cache_full[:, kv_sl]
            v_bnsd = value_cache_full[:, kv_sl]
        else:
            k_bnsd = key_states[:, kv_sl]
            v_bnsd = value_states[:, kv_sl]
        attn_r = torch_npu.npu_prompt_flash_attention(
            query_states[:, q_sl], k_bnsd, v_bnsd,
            num_heads=qh,
            num_key_value_heads=kvh,
            input_layout="BNSD_BSND",
            scale_value=self.scaling,
            atten_mask=self._prefill_mask,
            pre_tokens=2147483647,
            next_tokens=2147483647,
            sparse_mode=0,
        )
        attn_output = all_gather_seq(attn_r, dim=2)
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    if key_cache_full is not None:
        key_states, value_states = key_cache_full, value_cache_full
    kv_group = self.num_key_value_groups
    if kv_group > 1:
        key_states = repeat_kv(key_states, kv_group)
        value_states = repeat_kv(value_states, kv_group)
    hp = query_states.size(1) // world
    sl = slice(rank * hp, (rank + 1) * hp)
    attn_r = _pfa_full_attn(
        query_states[:, sl], key_states[:, sl], value_states[:, sl], self.scaling,
    )
    attn_output = all_gather_seq(attn_r, dim=1).transpose(1, 2)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attn_output * torch.sigmoid(gate)
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def shard_qwen3_5_cp(model, rank: int, world: int):
    """Context-parallel (Ulysses) mode: swap the 8 full-attention forwards to the USP version."""
    text_model = _get_text_model(model)
    layer_types = text_model.config.layer_types
    for i, layer in enumerate(text_model.layers):
        if layer_types[i] == "full_attention":
            layer.self_attn.forward = types.MethodType(
                qwen3_5_full_attn_forward_cp, layer.self_attn
            )
    return model


def _get_text_model(model):
    """Locate the `Qwen3_5TextModel` inside a ForConditionalGeneration / ForCausalLM."""
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "language_model"):
            return inner.language_model
        if hasattr(inner, "layers"):
            return inner
    raise AttributeError(
        f"Could not locate the text decoder on {type(model).__name__}; "
        f"expected .model.layers (ForCausalLM) or .model.language_model.layers "
        f"(ForConditionalGeneration)."
    )
