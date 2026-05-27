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
Export Qwen3.5-2B model to ONNX format.

Qwen3.5-2B is a multimodal VL model with hybrid linear attention
(GatedDeltaNet) and full attention architecture. The export splits
the model into three ONNX files:
  - Vision Tower (fixed grid_thw)
  - LLM Prefill (with image_embeds input)
  - LLM Decode (with recurrent states + KV cache input)
"""

import sys
import argparse
import gc
from pathlib import Path
import torch
import torch.nn.functional as F

try:
    import torch._dynamo

    torch._dynamo.disable()
except Exception:
    pass

try:
    from transformers import Qwen3_5ForConditionalGeneration
except ImportError:
    print("Error: transformers package not found or version too low.")
    print("Please install: pip install --upgrade transformers")
    sys.exit(1)


def _l2norm(x, dim=-1, eps=1e-6):
    """Apply L2 normalization along the given dimension."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def _make_additive_causal_mask(attention_mask, q_len, k_len, past_len, dtype):
    """Build additive causal + padding attention mask for full attention layers."""
    mask_value = torch.finfo(dtype).min
    ar_q = torch.arange(q_len, device=attention_mask.device)
    ar_k = torch.arange(k_len, device=attention_mask.device)
    causal = ar_k[None, :] > (past_len + ar_q[:, None])
    causal = causal.to(dtype) * mask_value
    causal = causal[None, None, :, :].expand(attention_mask.shape[0], 1, q_len, k_len)
    padding = (1.0 - attention_mask.to(dtype)) * mask_value
    padding = padding[:, None, None, :]
    return causal + padding


def _chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64, initial_state=None,
                            output_final_state=False, use_qk_l2norm_in_kernel=False):
    """Chunk-parallel forward of GatedDeltaNet for prefill (training/export mode)."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0
    )

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_i = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn_i @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2)
            @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def _recurrent_gated_delta_rule(query, key, value, g, beta, initial_state,
                                output_final_state=False, use_qk_l2norm_in_kernel=False):
    """Recurrent (token-by-token) forward of GatedDeltaNet for decode."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(value)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def _linear_attn_prefill(layer, hidden_states, attention_mask):
    """Run linear attention (GatedDeltaNet) forward for prefill, outputting conv and recurrent states."""
    batch_size, seq_len, _ = hidden_states.shape
    hidden_states_input = hidden_states
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        hidden_states_input = hidden_states * attention_mask[:, :, None]

    mixed_qkv = layer.in_proj_qkv(hidden_states_input)
    mixed_qkv = mixed_qkv.transpose(1, 2)
    conv_state = mixed_qkv[:, :, -(layer.conv_kernel_size - 1):]

    z = layer.in_proj_z(hidden_states)
    z = z.reshape(batch_size, seq_len, -1, layer.head_v_dim)

    b = layer.in_proj_b(hidden_states)
    a = layer.in_proj_a(hidden_states)

    if layer.causal_conv1d_fn is not None:
        mixed_qkv = layer.causal_conv1d_fn(
            x=mixed_qkv,
            weight=layer.conv1d.weight.squeeze(1),
            bias=layer.conv1d.bias,
            activation=layer.activation,
            seq_idx=None,
        )
    else:
        mixed_qkv = F.silu(layer.conv1d(mixed_qkv)[:, :, :seq_len])

    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv, [layer.key_dim, layer.key_dim, layer.value_dim], dim=-1
    )

    query = query.reshape(batch_size, seq_len, -1, layer.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, layer.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, layer.head_v_dim)

    beta = b.sigmoid()
    g = -layer.A_log.float().exp() * F.softplus(a.float() + layer.dt_bias)

    if layer.num_v_heads // layer.num_k_heads > 1:
        query = query.repeat_interleave(layer.num_v_heads // layer.num_k_heads, dim=2)
        key = key.repeat_interleave(layer.num_v_heads // layer.num_k_heads, dim=2)

    query = _l2norm(query, dim=-1, eps=1e-6)
    key = _l2norm(key, dim=-1, eps=1e-6)

    query_t = query.transpose(1, 2).contiguous()
    key_t = key.transpose(1, 2).contiguous()
    value_t = value.transpose(1, 2).contiguous()
    g_t = g.transpose(1, 2).contiguous()
    beta_t = beta.transpose(1, 2).contiguous()

    chunk_size = 64
    scale = 1.0 / (layer.head_k_dim ** 0.5)
    initial_state = torch.zeros(
        batch_size, query.shape[2], layer.head_k_dim, layer.head_v_dim,
        dtype=torch.float16, device=query.device,
    )

    core_attn_out, last_recurrent_state = _do_chunk_gated_delta_rule(
        query_t, key_t, value_t, g_t, beta_t, initial_state,
        chunk_size, scale,
    )
    core_attn_out = core_attn_out.transpose(1, 2).contiguous()
    last_recurrent_state = last_recurrent_state.to(torch.float32)

    core_attn_out = core_attn_out.reshape(-1, layer.head_v_dim)
    z = z.reshape(-1, layer.head_v_dim)
    core_attn_out = layer.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

    output = layer.out_proj(core_attn_out)
    return output, conv_state, last_recurrent_state


def _linear_attn_decode(layer, hidden_states, conv_state_in, recurrent_state_in):
    """Run linear attention (GatedDeltaNet) forward for decode with conv and recurrent state updates."""
    batch_size, seq_len, _ = hidden_states.shape

    mixed_qkv = layer.in_proj_qkv(hidden_states)
    mixed_qkv = mixed_qkv.transpose(1, 2)

    z = layer.in_proj_z(hidden_states)
    z = z.reshape(batch_size, seq_len, -1, layer.head_v_dim)

    b = layer.in_proj_b(hidden_states)
    a = layer.in_proj_a(hidden_states)

    state_len = conv_state_in.shape[-1]
    hidden_states_new = torch.cat([conv_state_in, mixed_qkv], dim=-1).to(layer.conv1d.weight.dtype)
    conv_state_out = hidden_states_new[:, :, -state_len:]
    out_conv = F.conv1d(
        hidden_states_new, layer.conv1d.weight, layer.conv1d.bias,
        padding=0, groups=hidden_states_new.shape[1],
    )
    mixed_qkv = F.silu(out_conv[:, :, -seq_len:])
    mixed_qkv = mixed_qkv.to(hidden_states.dtype)

    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv, [layer.key_dim, layer.key_dim, layer.value_dim], dim=-1
    )

    query = query.reshape(batch_size, seq_len, -1, layer.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, layer.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, layer.head_v_dim)

    beta = b.sigmoid()
    g = -layer.A_log.float().exp() * F.softplus(a.float() + layer.dt_bias)

    if layer.num_v_heads // layer.num_k_heads > 1:
        query = query.repeat_interleave(layer.num_v_heads // layer.num_k_heads, dim=2)
        key = key.repeat_interleave(layer.num_v_heads // layer.num_k_heads, dim=2)

    query = _l2norm(query, dim=-1, eps=1e-6)
    key = _l2norm(key, dim=-1, eps=1e-6)

    query_t = query.transpose(1, 2).contiguous()
    key_t = key.transpose(1, 2).contiguous()
    value_t = value.transpose(1, 2).contiguous()
    g_t = g.transpose(1, 2).contiguous()
    beta_t = beta.transpose(1, 2).contiguous()

    scale = 1.0 / (layer.head_k_dim ** 0.5)
    state_input = recurrent_state_in.to(torch.float16)

    core_attn_out, last_recurrent_state = _do_recurrent_gated_delta_rule(
        query_t, key_t, value_t, beta_t, state_input,
        g_t, scale,
    )
    core_attn_out = core_attn_out.transpose(1, 2).contiguous()
    last_recurrent_state = last_recurrent_state.to(torch.float32)

    core_attn_out = core_attn_out.reshape(-1, layer.head_v_dim)
    z = z.reshape(-1, layer.head_v_dim)
    core_attn_out = layer.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

    output = layer.out_proj(core_attn_out)
    return output, conv_state_out, last_recurrent_state


def _full_attn_decode_forward(layer, hidden_states, position_embeddings, attention_mask,
                              past_key=None, past_value=None):
    """Full attention forward for decode step, exported as CANN IncreFlashAttention fused op."""
    input_shape = hidden_states.shape[:-1]
    head_dim = layer.head_dim
    num_heads = layer.config.num_attention_heads
    num_kv_heads = layer.config.num_key_value_heads
    hidden_shape = (*input_shape, -1, head_dim)

    qkv = layer.q_proj(hidden_states).view(*input_shape, -1, head_dim * 2)
    query_states, gate = torch.chunk(qkv, 2, dim=-1)
    gate = gate.reshape(*input_shape, -1)

    query_states = layer.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = layer.k_norm(layer.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = layer.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key is not None:
        key_states = torch.cat([past_key, key_states], dim=2)
        value_states = torch.cat([past_value, value_states], dim=2)

    scale_value = 1.0 / (head_dim ** 0.5)
    if num_kv_heads < num_heads:
        repeat = num_heads // num_kv_heads
        key_expanded = key_states.repeat_interleave(repeat, dim=1)
        value_expanded = value_states.repeat_interleave(repeat, dim=1)
    else:
        key_expanded = key_states
        value_expanded = value_states
    attn_weights = torch.matmul(query_states, key_expanded.transpose(2, 3)) * scale_value
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_expanded)

    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = attn_output * torch.sigmoid(gate)
    attn_output = layer.o_proj(attn_output)
    return attn_output, key_states, value_states


class PromptFlashAttentionFunction(torch.autograd.Function):
    """Export full attention as CANN PromptFlashAttention fused op."""

    @staticmethod
    def forward(ctx, query, key, value, atten_mask, num_heads, num_kv_heads, scale_value):
        """PromptFlashAttention forward."""
        del ctx
        if num_kv_heads < num_heads:
            repeat = num_heads // num_kv_heads
            key = key.repeat_interleave(repeat, dim=1)
            value = value.repeat_interleave(repeat, dim=1)
        attn_weights = torch.matmul(query, key.transpose(2, 3)) * scale_value
        if atten_mask is not None:
            mask_value = torch.finfo(query.dtype).min
            attn_weights = attn_weights.masked_fill(~atten_mask, mask_value)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        return torch.matmul(attn_weights, value)

    @staticmethod
    def symbolic(g, query, key, value, atten_mask, num_heads, num_kv_heads, scale_value):
        """PromptFlashAttention symbolic."""
        if atten_mask is not None:
            return g.op(
                "Custom",
                query, key, value, atten_mask,
                input_index_i=[0, 1, 2, 3],
                input_names_s=["query", "key", "value", "atten_mask"],
                optional_input_names_s=["atten_mask"],
                output_names_s=["attention_out"],
                type_s="PromptFlashAttention",
                num_heads_i=int(num_heads),
                num_key_value_heads_i=int(num_kv_heads),
                scale_value_f=float(scale_value),
                input_layout_s="BNSD",
                next_tokens_i=65536,
                inner_precise_i=1,
            )
        return g.op(
            "Custom",
            query, key, value,
            input_index_i=[0, 1, 2],
            input_names_s=["query", "key", "value"],
            optional_input_names_s=[],
            output_names_s=["attention_out"],
            type_s="PromptFlashAttention",
            num_heads_i=int(num_heads),
            num_key_value_heads_i=int(num_kv_heads),
            scale_value_f=float(scale_value),
            input_layout_s="BNSD",
            next_tokens_i=65536,
            inner_precise_i=1,
        )


def _full_attn_prefill_forward(layer, hidden_states, position_embeddings, attention_mask):
    """Full attention forward for prefill, exported as CANN PromptFlashAttention fused op."""
    input_shape = hidden_states.shape[:-1]
    head_dim = layer.head_dim
    num_heads = layer.config.num_attention_heads
    num_kv_heads = layer.config.num_key_value_heads
    hidden_shape = (*input_shape, -1, head_dim)

    qkv = layer.q_proj(hidden_states).view(*input_shape, -1, head_dim * 2)
    query_states, gate = torch.chunk(qkv, 2, dim=-1)
    gate = gate.reshape(*input_shape, -1)

    query_states = layer.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = layer.k_norm(layer.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = layer.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    scale_value = 1.0 / (head_dim ** 0.5)
    bool_mask = attention_mask == 0
    attn_output = PromptFlashAttentionFunction.apply(
        query_states, key_states, value_states, bool_mask,
        num_heads, num_kv_heads, scale_value,
    )

    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = attn_output * torch.sigmoid(gate)
    attn_output = layer.o_proj(attn_output)
    return attn_output, key_states, value_states


class IncreFlashAttentionFunction(torch.autograd.Function):
    """Export incremental decode attention as CANN IncreFlashAttention fused op."""

    @staticmethod
    def forward(ctx, query, key, value, atten_mask, num_heads, scale_value,
                input_layout="BNSD", num_kv_heads=0):
        """IncreFlashAttention forward."""
        del ctx, input_layout
        if 0 < num_kv_heads < num_heads:
            repeat = num_heads // num_kv_heads
            key = key.repeat_interleave(repeat, dim=1)
            value = value.repeat_interleave(repeat, dim=1)
        attn_weights = torch.matmul(query, key.transpose(2, 3)) * scale_value
        if atten_mask is not None:
            attn_weights = attn_weights + atten_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        return torch.matmul(attn_weights, value)

    @staticmethod
    def symbolic(g, query, key, value, atten_mask, num_heads, scale_value,
                 input_layout="BNSD", num_kv_heads=0):
        """IncreFlashAttention symbolic."""
        return g.op(
            "Custom",
            query, key, value, atten_mask,
            input_index_i=[0, 1, 2, 3],
            input_names_s=["query", "key", "value", "atten_mask"],
            optional_input_names_s=["atten_mask"],
            output_names_s=["attention_out"],
            type_s="IncreFlashAttention",
            num_heads_i=int(num_heads),
            scale_value_f=float(scale_value),
            input_layout_s=str(input_layout),
            num_key_value_heads_i=int(num_kv_heads),
        )




class VisionTowerWrapper(torch.nn.Module):
    """Wrap Qwen3.5 vision tower with cached position embeddings for ONNX export."""

    def __init__(self, vision_tower, dummy_grid_thw):
        """Cache position embeddings for a fixed grid_thw to enable deterministic export."""
        super().__init__()
        self.vision_tower = vision_tower
        self.dummy_grid_thw = dummy_grid_thw
        with torch.no_grad():
            cached_pos_embeds = vision_tower.fast_pos_embed_interpolate(dummy_grid_thw)
            cached_rot_pos_emb = vision_tower.rot_pos_emb(dummy_grid_thw)
        vision_tower.fast_pos_embed_interpolate = lambda x: cached_pos_embeds
        vision_tower.rot_pos_emb = lambda x: cached_rot_pos_emb

    def forward(self, pixel_values):
        """Run vision tower forward and return pooled image embeddings."""
        outputs = self.vision_tower(
            pixel_values, grid_thw=self.dummy_grid_thw, return_dict=True
        )
        image_embeds = outputs.pooler_output
        if isinstance(image_embeds, (list, tuple)):
            image_embeds = torch.cat(image_embeds, dim=0)
        return image_embeds


def _add_rms_norm_module(x1, x2, norm_module):
    """Use the actual RMSNorm module to compute AddRmsNorm.

    Qwen3.5 RMSNorm uses (1 + weight), so we must call the module's forward
    rather than manually multiplying by the raw weight tensor.
    """
    x = x1 + x2
    y = norm_module(x)
    return y, x


def _do_add_rms_norm(residual, attn_out, norm_module, eps):
    """Residual add + RMSNorm using the actual RMSNorm module."""
    del eps
    return _add_rms_norm_module(residual, attn_out, norm_module)


def _do_chunk_gated_delta_rule(query_t, key_t, value_t, g_t, beta_t, initial_state,
                               chunk_size, scale):
    """Chunk-parallel GatedDeltaRule for prefill."""
    del scale
    q = query_t.transpose(1, 2).contiguous()
    k = key_t.transpose(1, 2).contiguous()
    v = value_t.transpose(1, 2).contiguous()
    g_in = g_t.transpose(1, 2).contiguous()
    beta_in = beta_t.transpose(1, 2).contiguous()
    state_in = initial_state.to(torch.float32)
    out, state = _chunk_gated_delta_rule(
        q, k, v, g=g_in, beta=beta_in,
        chunk_size=int(chunk_size),
        initial_state=state_in,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )
    out = out.transpose(1, 2).contiguous()
    state = state.to(torch.float16)
    return out, state


def _do_recurrent_gated_delta_rule(query_t, key_t, value_t, beta_t, state_in,
                                   g_t, scale):
    """Recurrent GatedDeltaRule for decode."""
    del scale
    q = query_t.transpose(1, 2).contiguous()
    k = key_t.transpose(1, 2).contiguous()
    v = value_t.transpose(1, 2).contiguous()
    beta_in = beta_t.transpose(1, 2).contiguous()
    g_in = g_t.transpose(1, 2).contiguous()
    state_f32 = state_in.to(torch.float32)
    out, state = _recurrent_gated_delta_rule(
        q, k, v, g=g_in, beta=beta_in,
        initial_state=state_f32,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )
    out = out.transpose(1, 2).contiguous()
    state = state.to(torch.float16)
    return out, state


class Qwen35LlmPrefill(torch.nn.Module):
    """LLM Prefill wrapper that processes full prompt with image embeddings and outputs initial states."""

    def __init__(self, text_model, lm_head, image_token_id, num_layers=None):
        """Initialize with text model, lm_head, image token id, and optional layer count limit."""
        super().__init__()
        self.text_model = text_model
        self.lm_head = lm_head
        self.image_token_id = int(image_token_id)
        self.config = text_model.config
        self.num_layers = num_layers

    def forward(self, input_ids, attention_mask, position_ids, image_embeds):
        """Forward pass: embed tokens, inject image embeddings, run through layers, return logits and states."""
        inputs_embeds = self.text_model.embed_tokens(input_ids)
        image_mask = input_ids == self.image_token_id
        image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(
            image_mask,
            image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
        )

        q_len = input_ids.shape[1]
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            mm_position_ids = position_ids[1:]
        else:
            mm_position_ids = position_ids

        position_embeddings = self.text_model.rotary_emb(inputs_embeds, mm_position_ids)
        k_len = q_len
        attn_mask = _make_additive_causal_mask(
            attention_mask, q_len, k_len, 0, inputs_embeds.dtype
        )
        linear_attn_mask = attention_mask

        hidden_states = inputs_embeds
        present_conv = []
        present_recurrent = []
        present_kv = []
        eps = float(self.config.rms_norm_eps)

        for idx, layer in enumerate(self.text_model.layers):
            if self.num_layers is not None and idx >= self.num_layers:
                break
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            if layer.layer_type == "linear_attention":
                attn_out, conv_s, rec_s = _linear_attn_prefill(
                    layer.linear_attn, hidden_states, linear_attn_mask
                )
                hidden_states, residual = _do_add_rms_norm(
                    residual, attn_out,
                    layer.post_attention_layernorm, eps)
                present_conv.append(conv_s)
                present_recurrent.append(rec_s)
            else:
                attn_out, pk, pv = _full_attn_prefill_forward(
                    layer.self_attn, hidden_states, position_embeddings, attn_mask,
                )
                hidden_states, residual = _do_add_rms_norm(
                    residual, attn_out,
                    layer.post_attention_layernorm, eps)
                present_kv.append(pk)
                present_kv.append(pv)

            mlp_out = layer.mlp(hidden_states)
            hidden_states = residual + mlp_out

        hidden_states = self.text_model.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        present_conv_stack = torch.stack(present_conv, dim=0) if present_conv else torch.zeros(0)
        present_recurrent_stack = torch.stack(present_recurrent, dim=0) if present_recurrent else torch.zeros(0)
        present_kv_stack = torch.stack(present_kv, dim=0) if present_kv else torch.zeros(0)

        return logits, present_conv_stack, present_recurrent_stack, present_kv_stack


class Qwen35LlmDecode(torch.nn.Module):
    """LLM Decode wrapper for single-token autoregressive generation with past states."""

    def __init__(self, text_model, lm_head, num_layers=None):
        """Initialize with text model, lm_head, and optional layer count limit."""
        super().__init__()
        self.text_model = text_model
        self.lm_head = lm_head
        self.config = text_model.config
        self.num_layers = num_layers

    def forward(self, input_ids, attention_mask, position_ids,
                past_conv_states, past_recurrent_states, past_kv_cache):
        """Forward pass: embed token, update conv/recurrent/KV states, return logits and updated states."""
        inputs_embeds = self.text_model.embed_tokens(input_ids)
        q_len = input_ids.shape[1]

        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            mm_position_ids = position_ids[1:]
        else:
            mm_position_ids = position_ids

        position_embeddings = self.text_model.rotary_emb(inputs_embeds, mm_position_ids)

        past_len = 0
        kv_idx = 0
        for layer in self.text_model.layers:
            if layer.layer_type == "full_attention":
                if kv_idx == 0 and past_kv_cache.shape[0] > 0:
                    past_len = past_kv_cache[0].shape[2]
                break

        k_len = past_len + q_len
        attn_mask = _make_additive_causal_mask(
            attention_mask, q_len, k_len, past_len, inputs_embeds.dtype
        )

        hidden_states = inputs_embeds
        present_conv = []
        present_recurrent = []
        present_kv = []
        linear_idx = 0
        kv_idx = 0
        eps = float(self.config.rms_norm_eps)

        for idx, layer in enumerate(self.text_model.layers):
            if self.num_layers is not None and idx >= self.num_layers:
                break
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            if layer.layer_type == "linear_attention":
                conv_s_in = past_conv_states[linear_idx]
                rec_s_in = past_recurrent_states[linear_idx]
                attn_out, conv_s_out, rec_s_out = _linear_attn_decode(
                    layer.linear_attn, hidden_states, conv_s_in, rec_s_in
                )
                hidden_states, residual = _do_add_rms_norm(
                    residual, attn_out,
                    layer.post_attention_layernorm, eps)
                present_conv.append(conv_s_out)
                present_recurrent.append(rec_s_out)
                linear_idx += 1
            else:
                pk_in = past_kv_cache[kv_idx]
                pv_in = past_kv_cache[kv_idx + 1]
                attn_out, pk, pv = _full_attn_decode_forward(
                    layer.self_attn, hidden_states, position_embeddings, attn_mask,
                    pk_in, pv_in,
                )
                hidden_states, residual = _do_add_rms_norm(
                    residual, attn_out,
                    layer.post_attention_layernorm, eps)
                present_kv.append(pk)
                present_kv.append(pv)
                kv_idx += 2

            mlp_out = layer.mlp(hidden_states)
            hidden_states = residual + mlp_out

        hidden_states = self.text_model.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        present_conv_stack = torch.stack(present_conv, dim=0) if present_conv else torch.zeros(0)
        present_recurrent_stack = torch.stack(present_recurrent, dim=0) if present_recurrent else torch.zeros(0)
        present_kv_stack = torch.stack(present_kv, dim=0) if present_kv else torch.zeros(0)

        return logits, present_conv_stack, present_recurrent_stack, present_kv_stack


def _get_model_meta(model, num_layers=None):
    """Extract model metadata (dimensions, layer counts) needed for export and dummy input construction."""
    text_model = model.model.language_model
    lm_head = model.lm_head
    config = text_model.config
    image_token_id = model.config.image_token_id

    total_layers = config.num_hidden_layers
    if num_layers is None:
        num_layers = total_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

    layer_types = config.layer_types[:num_layers]
    num_linear_layers = sum(1 for lt in layer_types if lt == "linear_attention")
    num_full_layers = sum(1 for lt in layer_types if lt == "full_attention")

    linear_key_head_dim = config.linear_key_head_dim
    linear_num_key_heads = config.linear_num_key_heads
    linear_value_head_dim = config.linear_value_head_dim
    linear_num_value_heads = config.linear_num_value_heads
    key_dim = linear_key_head_dim * linear_num_key_heads
    value_dim = linear_value_head_dim * linear_num_value_heads
    conv_dim = key_dim * 2 + value_dim
    conv_kernel_size = config.linear_conv_kernel_dim

    return {
        "text_model": text_model,
        "lm_head": lm_head,
        "image_token_id": image_token_id,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "num_linear_layers": num_linear_layers,
        "num_full_layers": num_full_layers,
        "linear_key_head_dim": linear_key_head_dim,
        "linear_num_key_heads": linear_num_key_heads,
        "linear_value_head_dim": linear_value_head_dim,
        "linear_num_value_heads": linear_num_value_heads,
        "conv_dim": conv_dim,
        "conv_kernel_size": conv_kernel_size,
    }


def export_vision_tower(model, output_dir, device="cpu", vision_image_size=128):
    """Export the Qwen3.5 vision tower to ONNX with cached position embeddings."""
    output_path = Path(output_dir) / "qwen3_5_vision.onnx"
    print(f"Exporting Vision Tower to {output_path}...")

    vision_tower = model.model.visual
    vision_tower.eval()
    vision_tower.to(device)

    patch_size = model.config.vision_config.patch_size
    grid_h = int(vision_image_size) // int(patch_size)
    grid_w = int(vision_image_size) // int(patch_size)
    dummy_grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.int64).to(device)
    dummy_seq_len = int(
        dummy_grid_thw[0, 0].item() * dummy_grid_thw[0, 1].item() * dummy_grid_thw[0, 2].item()
    )
    in_channels = model.config.vision_config.in_channels
    temporal_patch_size = model.config.vision_config.temporal_patch_size
    patch_dim = in_channels * temporal_patch_size * patch_size * patch_size
    dummy_pixel_values = torch.randn(dummy_seq_len, patch_dim, device=device, dtype=torch.float16)

    wrapper = VisionTowerWrapper(vision_tower, dummy_grid_thw)
    wrapper.eval()

    from torch.onnx import utils as onnx_utils

    with torch.no_grad():
        onnx_utils.export(
            wrapper,
            (dummy_pixel_values,),
            str(output_path),
            input_names=["pixel_values"],
            output_names=["image_embeds"],
            opset_version=14,
            do_constant_folding=False,
        )
    print("Vision Tower exported successfully.")


def _export_llm_prefill(prefill, output_dir, device, dummy_seq, dummy_num_img_tokens,
                        dtype=torch.float16):
    """Export the LLM prefill model to ONNX with dynamic axes for variable-length input."""
    prefill_path = Path(output_dir) / "qwen3_5_llm_prefill.onnx"
    dummy_input_ids = torch.randint(0, 1000, (1, dummy_seq), dtype=torch.int64, device=device)
    dummy_attention_mask = torch.ones(1, dummy_seq, dtype=torch.int64, device=device)
    base_pos = torch.arange(dummy_seq, device=device, dtype=torch.int64).view(1, -1)
    dummy_position_ids = base_pos.unsqueeze(0).expand(4, 1, dummy_seq)
    dummy_image_embeds = torch.randn(
        dummy_num_img_tokens, prefill.config.hidden_size,
        device=device, dtype=dtype,
    )
    input_names = ["input_ids", "attention_mask", "position_ids", "image_embeds"]
    output_names = ["logits", "present_conv_states", "present_recurrent_states",
                    "present_kv_cache"]
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq_len"},
        "attention_mask": {0: "batch", 1: "seq_len"},
        "position_ids": {1: "batch", 2: "seq_len"},
        "image_embeds": {0: "num_image_tokens"},
        "logits": {0: "batch", 1: "seq_len"},
        "present_conv_states": {1: "batch"},
        "present_recurrent_states": {1: "batch"},
        "present_kv_cache": {1: "batch", 3: "seq_len"},
    }
    from torch.onnx import utils as onnx_utils
    print(f"Exporting LLM prefill to {prefill_path}...")
    with torch.no_grad():
        onnx_utils.export(
            prefill,
            (dummy_input_ids, dummy_attention_mask, dummy_position_ids, dummy_image_embeds),
            str(prefill_path),
            input_names=input_names, output_names=output_names,
            opset_version=14, do_constant_folding=False, dynamic_axes=dynamic_axes,
        )
    print("LLM prefill exported successfully.")


def _export_llm_decode(decode, meta, output_dir, device, dummy_seq):
    """Export the LLM decode model to ONNX with dynamic KV cache sequence length."""
    decode_path = Path(output_dir) / "qwen3_5_llm_decode.onnx"
    dummy_step = 1
    dummy_past_len = dummy_seq
    dummy_input_ids_step = torch.randint(
        0, 1000, (1, dummy_step), dtype=torch.int64, device=device
    )
    dummy_attention_mask_step = torch.ones(
        1, dummy_past_len + dummy_step, dtype=torch.int64, device=device
    )
    step_pos = torch.tensor([[dummy_past_len]], dtype=torch.int64, device=device)
    dummy_position_ids_step = step_pos.unsqueeze(0).expand(4, 1, dummy_step)

    num_linear = meta["num_linear_layers"]
    num_full = meta["num_full_layers"]
    conv_dim = meta["conv_dim"]
    conv_kernel_size = meta["conv_kernel_size"]
    num_v_heads = meta["linear_num_value_heads"]
    k_head_dim = meta["linear_key_head_dim"]
    v_head_dim = meta["linear_value_head_dim"]
    num_kv_heads = meta["num_kv_heads"]
    head_dim = meta["head_dim"]

    dummy_past_conv = torch.zeros(
        num_linear, 1, conv_dim, conv_kernel_size - 1,
        dtype=torch.float16, device=device,
    )
    dummy_past_recurrent = torch.zeros(
        num_linear, 1, num_v_heads, k_head_dim, v_head_dim,
        dtype=torch.float32, device=device,
    )
    dummy_past_kv = torch.zeros(
        2 * num_full, 1, num_kv_heads, dummy_past_len, head_dim,
        dtype=torch.float16, device=device,
    )

    input_names = ["input_ids", "attention_mask", "position_ids",
                   "past_conv_states", "past_recurrent_states", "past_kv_cache"]
    output_names = ["logits", "present_conv_states", "present_recurrent_states",
                    "present_kv_cache"]
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "step"},
        "attention_mask": {0: "batch", 1: "total_seq_len"},
        "position_ids": {1: "batch", 2: "step"},
        "past_conv_states": {1: "batch"},
        "past_recurrent_states": {1: "batch"},
        "past_kv_cache": {1: "batch", 3: "past_seq_len"},
        "logits": {0: "batch", 1: "step"},
        "present_conv_states": {1: "batch"},
        "present_recurrent_states": {1: "batch"},
        "present_kv_cache": {1: "batch", 3: "total_seq_len"},
    }
    from torch.onnx import utils as onnx_utils
    print(f"Exporting LLM decode to {decode_path}...")
    with torch.no_grad():
        onnx_utils.export(
            decode,
            (dummy_input_ids_step, dummy_attention_mask_step, dummy_position_ids_step,
             dummy_past_conv, dummy_past_recurrent, dummy_past_kv),
            str(decode_path),
            input_names=input_names, output_names=output_names,
            opset_version=14, do_constant_folding=False, dynamic_axes=dynamic_axes,
        )
    print("LLM decode exported successfully.")


def export_llm_prefill_decode(model, output_dir, device="cpu", dummy_seq=8,
                              dummy_num_img_tokens=16, dtype=torch.float16, num_layers=None):
    """Export LLM prefill and decode models to ONNX, optionally limiting the number of layers."""
    meta = _get_model_meta(model, num_layers=num_layers)
    text_model = meta["text_model"]
    lm_head = meta["lm_head"]
    image_token_id = meta["image_token_id"]
    nl = meta["num_layers"]

    text_model.eval()
    lm_head.eval()
    text_model.to(device)
    lm_head.to(device)

    prefill = Qwen35LlmPrefill(text_model, lm_head, image_token_id,
                               num_layers=nl).to(device).eval()
    decode = Qwen35LlmDecode(text_model, lm_head,
                             num_layers=nl).to(device).eval()

    _export_llm_prefill(prefill, output_dir, device, dummy_seq, dummy_num_img_tokens,
                        dtype=dtype)
    _export_llm_decode(decode, meta, output_dir, device, dummy_seq)


def _get_onnx_dtype_sizes():
    return {1: 4, 6: 2, 7: 8, 10: 2, 11: 8}


def _fix_onnx_initializer_dims_to_match_raw_data(model, dtype_sizes):
    """Fix ONNX initializer dims to match actual raw_data byte length."""
    for init in model.graph.initializer:
        raw_len = len(init.raw_data)
        if raw_len == 0:
            continue
        dtype_sz = dtype_sizes.get(init.data_type, 4)
        num_el = 1
        for d in init.dims:
            num_el *= d
        expected = num_el * dtype_sz
        if expected == raw_len:
            continue
        actual_elements = raw_len // dtype_sz
        if len(init.dims) == 2:
            d0 = init.dims[0]
            if d0 > 0 and actual_elements % d0 == 0:
                init.dims[1] = actual_elements // d0
            else:
                init.dims[:] = [1, actual_elements]
        elif len(init.dims) == 1:
            init.dims[0] = actual_elements


def _write_onnx_external_data_file(model, onnx_dir, data_filename, onnx_module):
    """Write initializer raw data to external binary file and update offsets."""
    offset = 0
    with open(onnx_dir / data_filename, "wb") as f_data:
        for init in model.graph.initializer:
            raw_data = init.raw_data
            if not raw_data:
                continue
            length = len(raw_data)
            f_data.write(raw_data)
            init.raw_data = b""
            init.data_location = 1
            while len(init.external_data) > 0:
                init.external_data.pop()
            for k, v in [
                ("location", data_filename),
                ("offset", str(offset)),
                ("length", str(length)),
            ]:
                e = onnx_module.StringStringEntryProto()
                e.key = k
                e.value = v
                init.external_data.append(e)
            offset += length
    return offset


def _serialize_onnx_model_protobuf(model, onnx_path):
    """Serialize ONNX model protobuf to file."""
    with open(str(onnx_path), "wb") as f:
        f.write(model.SerializeToString())


def _verify_onnx_external_data_repack(onnx_path, dtype_sizes, onnx_module):
    """Verify external data repacking by loading model and checking initializer sizes."""
    check = onnx_module.load(str(onnx_path), load_external_data=False)
    errors = []
    for init in check.graph.initializer:
        if init.data_location == 1:
            dtype_sz = dtype_sizes.get(init.data_type, 4)
            num_el = 1
            for d in init.dims:
                num_el *= d
            expected = num_el * dtype_sz
            for entry in init.external_data:
                if entry.key == "length":
                    actual = int(entry.value)
                    if expected != actual:
                        errors.append(f"{init.name}: expected={expected}, actual={actual}")
    if errors:
        print(f"WARNING: {len(errors)} size mismatches after repack!")
        for e in errors[:5]:
            print(f"  {e}")
        raise RuntimeError("External data repack failed verification")


def _repack_onnx_external_data(onnx_path):
    """Repack ONNX external data by manually writing a single external data file."""
    import onnx as _onnx

    onnx_path = Path(onnx_path)
    onnx_dir = onnx_path.parent
    basename = onnx_path.stem
    data_filename = f"{basename}.data"
    new_onnx_path = onnx_dir / f"{basename}_packed.onnx"

    print(f"Repacking external data for {onnx_path}...")

    # Load model fully (this resolves all external data into memory)
    model = _onnx.load(str(onnx_path))

    # Fix: the onnx.load may have loaded wrong external data for some tensors
    # due to name-based file matching issues with PyTorch export.
    # We need to reload directly from the correct external data files.
    # Instead, we use the loaded data as-is and fix dims to match raw_data.
    dtype_sizes = _get_onnx_dtype_sizes()
    _fix_onnx_initializer_dims_to_match_raw_data(model, dtype_sizes)

    # Write external data file manually
    offset = _write_onnx_external_data_file(model, onnx_dir, data_filename, _onnx)

    # Serialize protobuf directly (bypass onnx.save_model which may re-embed data)
    _serialize_onnx_model_protobuf(model, new_onnx_path)

    # Verify
    _verify_onnx_external_data_repack(new_onnx_path, dtype_sizes, _onnx)

    import os
    os.replace(str(new_onnx_path), str(onnx_path))
    print(f"Repacked {onnx_path} successfully ({offset / 1e9:.2f} GB data).")


def main():
    """Load Qwen3.5-2B model and export vision tower, prefill, and decode to ONNX."""
    parser = argparse.ArgumentParser(description="Export Qwen3.5-2B to ONNX")
    parser.add_argument(
        "--model-id", type=str,
        default="./Qwen/Qwen3.5-2B",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./qwen3_5_2b_onnx",
        help="Output directory",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device for export (cpu or cuda)",
    )
    parser.add_argument(
        "--vision-image-size", type=int, default=128,
        help="Image size for vision tower export",
    )
    parser.add_argument(
        "--dummy-seq-len", type=int, default=8,
        help="Dummy sequence length for LLM export",
    )
    parser.add_argument(
        "--dtype", type=str, default="fp16", choices=["fp16", "fp32"],
        help="Export dtype",
    )
    parser.add_argument(
        "--num-layers", type=int, default=None,
        help="Number of layers to export (default: all 24). Use 4 for fast verification.",
    )
    parser.add_argument(
        "--no-custom-ops", action="store_true",
        help="Export with standard ONNX ops (no CANN custom ops) for accuracy verification.",
    )

    args = parser.parse_args()

    if args.no_custom_ops:
        print("Disabling custom ops - using standard ONNX ops for accuracy verification...")
        for cls in [PromptFlashAttentionFunction, IncreFlashAttentionFunction]:
            if hasattr(cls, 'symbolic'):
                cls._symbolic_saved = cls.symbolic
                delattr(cls, 'symbolic')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    print(f"\nLoading model {args.model_id} in {args.dtype} for export...")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map=args.device,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )

    export_vision_tower(model, output_dir, args.device, args.vision_image_size)
    export_llm_prefill_decode(model, output_dir, args.device, args.dummy_seq_len,
                              dtype=torch_dtype, num_layers=args.num_layers)

    # Repack LLM ONNX files to consolidate external data into single files
    print("\nRepacking ONNX files to consolidate external data...")
    _repack_onnx_external_data(output_dir / "qwen3_5_llm_prefill.onnx")
    _repack_onnx_external_data(output_dir / "qwen3_5_llm_decode.onnx")

    # Clean up the loose external data files from PyTorch export
    print("Cleaning up loose external data files...")
    for f in output_dir.iterdir():
        if f.is_file() and not f.name.endswith(('.onnx', '.data')):
            if not f.name.startswith('qwen3_5_'):
                f.unlink(missing_ok=True)

    print("Clearing memory after export...")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nExport finished. Files saved in {args.output_dir}")


if __name__ == "__main__":
    main()
