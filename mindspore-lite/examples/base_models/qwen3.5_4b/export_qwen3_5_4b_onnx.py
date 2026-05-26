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
Export Qwen3.5-4B model to ONNX format.

Qwen3.5-4B is a multimodal VL model with hybrid linear attention
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
    """
    L2 normalize the input tensor.
    """
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def _make_additive_causal_mask(attention_mask, q_len, k_len, past_len, dtype):
    """
    Make additive causal mask for the attention layer.
    """
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
    """
    Chunk gated delta rule for the attention layer.
    """
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
    """
    Recurrent gated delta rule for the attention layer.
    """
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
    """
    Linear attention prefill for the attention layer.
    """
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

    core_attn_out, last_recurrent_state = _chunk_gated_delta_rule(
        query, key, value, g=g, beta=beta,
        initial_state=None, output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    core_attn_out = core_attn_out.reshape(-1, layer.head_v_dim)
    z = z.reshape(-1, layer.head_v_dim)
    core_attn_out = layer.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

    output = layer.out_proj(core_attn_out)
    return output, conv_state, last_recurrent_state


def _linear_attn_decode(layer, hidden_states, conv_state_in, recurrent_state_in):
    """
    Linear attention decode for the attention layer.
    """
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

    core_attn_out, last_recurrent_state = _recurrent_gated_delta_rule(
        query, key, value, g=g, beta=beta,
        initial_state=recurrent_state_in, output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    core_attn_out = core_attn_out.reshape(-1, layer.head_v_dim)
    z = z.reshape(-1, layer.head_v_dim)
    core_attn_out = layer.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

    output = layer.out_proj(core_attn_out)
    return output, conv_state_out, last_recurrent_state


def _full_attn_forward(layer, hidden_states, position_embeddings, attention_mask,
                       past_key=None, past_value=None):
    """
    Full attention forward for the attention layer.
    """
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

    key_states_for_attn = key_states
    value_states_for_attn = value_states
    if num_kv_heads < num_heads:
        key_states_for_attn = key_states.repeat_interleave(num_heads // num_kv_heads, dim=1)
        value_states_for_attn = value_states.repeat_interleave(num_heads // num_kv_heads, dim=1)

    scaling = getattr(layer, "scaling", 1.0 / (head_dim ** 0.5))
    attn_weights = torch.matmul(query_states, key_states_for_attn.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states_for_attn)
    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = attn_output * torch.sigmoid(gate)
    attn_output = layer.o_proj(attn_output)
    return attn_output, key_states, value_states


class PromptFlashAttentionFunction(torch.autograd.Function):
    """Export full attention as CANN PromptFlashAttention fused op."""

    @staticmethod
    def forward(ctx, query, key, value, atten_mask, num_heads, num_kv_heads, scale_value):
        """
        PromptFlashAttention forward function.
        """
        del ctx
        if num_kv_heads < num_heads:
            repeat = num_heads // num_kv_heads
            key = key.repeat_interleave(repeat, dim=1)
            value = value.repeat_interleave(repeat, dim=1)
        attn_weights = torch.matmul(query, key.transpose(2, 3)) * scale_value
        if atten_mask is not None:
            attn_weights = attn_weights + atten_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        return torch.matmul(attn_weights, value)

    @staticmethod
    def symbolic(g, query, key, value, atten_mask, num_heads, num_kv_heads, scale_value):
        """
        PromptFlashAttention symbolic function.
        """
        if atten_mask is not None:
            return g.op(
                "PromptFlashAttention",
                query, key, value, atten_mask,
                num_heads_i=int(num_heads),
                num_key_value_heads_i=int(num_kv_heads),
                scale_value_f=float(scale_value),
                input_layout_s="BNSD",
                next_tokens_i=65536,
                inner_precise_i=1,
            )
        return g.op(
            "PromptFlashAttention",
            query, key, value,
            num_heads_i=int(num_heads),
            num_key_value_heads_i=int(num_kv_heads),
            scale_value_f=float(scale_value),
            input_layout_s="BNSD",
            next_tokens_i=65536,
            inner_precise_i=1,
        )


def _full_attn_prefill_forward(layer, hidden_states, position_embeddings, attention_mask):
    """
    Full attention prefill forward for the attention layer.
    """
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
    attn_output = PromptFlashAttentionFunction.apply(
        query_states, key_states, value_states, attention_mask,
        num_heads, num_kv_heads, scale_value,
    )

    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = attn_output * torch.sigmoid(gate)
    attn_output = layer.o_proj(attn_output)
    return attn_output, key_states, value_states


class VisionTowerWrapper(torch.nn.Module):
    """
    Vision tower wrapper for the attention layer.
    """
    def __init__(self, vision_tower, dummy_grid_thw):
        """
        Initialize the vision tower wrapper.
        """
        super().__init__()
        self.vision_tower = vision_tower
        self.dummy_grid_thw = dummy_grid_thw
        with torch.no_grad():
            cached_pos_embeds = vision_tower.fast_pos_embed_interpolate(dummy_grid_thw)
            cached_rot_pos_emb = vision_tower.rot_pos_emb(dummy_grid_thw)
        vision_tower.fast_pos_embed_interpolate = lambda x: cached_pos_embeds
        vision_tower.rot_pos_emb = lambda x: cached_rot_pos_emb

    def forward(self, pixel_values):
        """
        Forward function for the vision tower wrapper.
        """
        outputs = self.vision_tower(
            pixel_values, grid_thw=self.dummy_grid_thw, return_dict=True
        )
        image_embeds = outputs.pooler_output
        if isinstance(image_embeds, (list, tuple)):
            image_embeds = torch.cat(image_embeds, dim=0)
        return image_embeds


class Qwen35LlmPrefill(torch.nn.Module):
    """
    Qwen3.5 LLM prefill model for the attention layer.
    """
    def __init__(self, text_model, lm_head, image_token_id):
        """
        Initialize the Qwen3.5 LLM prefill model.
        """
        super().__init__()
        self.text_model = text_model
        self.lm_head = lm_head
        self.image_token_id = int(image_token_id)
        self.config = text_model.config

    def forward(self, input_ids, attention_mask, position_ids, image_embeds):
        """
        Forward function for the Qwen3.5 LLM prefill model.
        """
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

        for layer in self.text_model.layers:
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            if layer.layer_type == "linear_attention":
                attn_out, conv_s, rec_s = _linear_attn_prefill(
                    layer.linear_attn, hidden_states, linear_attn_mask
                )
                hidden_states = residual + attn_out
                present_conv.append(conv_s)
                present_recurrent.append(rec_s)
            else:
                attn_out, pk, pv = _full_attn_forward(
                    layer.self_attn, hidden_states, position_embeddings, attn_mask,
                )
                hidden_states = residual + attn_out
                present_kv.append(pk)
                present_kv.append(pv)

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = residual + layer.mlp(hidden_states)

        hidden_states = self.text_model.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        present_conv_stack = torch.stack(present_conv, dim=0) if present_conv else torch.zeros(0)
        present_recurrent_stack = torch.stack(present_recurrent, dim=0) if present_recurrent else torch.zeros(0)
        present_kv_stack = torch.stack(present_kv, dim=0) if present_kv else torch.zeros(0)

        return logits, present_conv_stack, present_recurrent_stack, present_kv_stack


class Qwen35LlmDecode(torch.nn.Module):
    """
    Qwen3.5 LLM decode model for the attention layer.
    """
    def __init__(self, text_model, lm_head):
        """
        Initialize the Qwen3.5 LLM decode model.
        """
        super().__init__()
        self.text_model = text_model
        self.lm_head = lm_head
        self.config = text_model.config

    def forward(self, input_ids, attention_mask, position_ids,
                past_conv_states, past_recurrent_states, past_kv_cache):
        """
        Forward function for the Qwen3.5 LLM decode model.
        """
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

        for layer in self.text_model.layers:
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            if layer.layer_type == "linear_attention":
                conv_s_in = past_conv_states[linear_idx]
                rec_s_in = past_recurrent_states[linear_idx]
                attn_out, conv_s_out, rec_s_out = _linear_attn_decode(
                    layer.linear_attn, hidden_states, conv_s_in, rec_s_in
                )
                hidden_states = residual + attn_out
                present_conv.append(conv_s_out)
                present_recurrent.append(rec_s_out)
                linear_idx += 1
            else:
                pk_in = past_kv_cache[kv_idx]
                pv_in = past_kv_cache[kv_idx + 1]
                attn_out, pk, pv = _full_attn_forward(
                    layer.self_attn, hidden_states, position_embeddings, attn_mask,
                    pk_in, pv_in,
                )
                hidden_states = residual + attn_out
                present_kv.append(pk)
                present_kv.append(pv)
                kv_idx += 2

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = residual + layer.mlp(hidden_states)

        hidden_states = self.text_model.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        present_conv_stack = torch.stack(present_conv, dim=0) if present_conv else torch.zeros(0)
        present_recurrent_stack = torch.stack(present_recurrent, dim=0) if present_recurrent else torch.zeros(0)
        present_kv_stack = torch.stack(present_kv, dim=0) if present_kv else torch.zeros(0)

        return logits, present_conv_stack, present_recurrent_stack, present_kv_stack


def _get_model_meta(model):
    """
    Get model metadata for the attention layer.
    """
    text_model = model.model.language_model
    lm_head = model.lm_head
    config = text_model.config
    image_token_id = model.config.image_token_id

    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

    num_linear_layers = sum(1 for lt in config.layer_types if lt == "linear_attention")
    num_full_layers = sum(1 for lt in config.layer_types if lt == "full_attention")

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
    """
    Export the vision tower of the Qwen3.5 model to ONNX format.
    """
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
            do_constant_folding=True,
        )
    print("Vision Tower exported successfully.")


def _export_llm_prefill(prefill, output_dir, device, dummy_seq, dummy_num_img_tokens,
                        dtype=torch.float16):
    """
    Export the LLM prefill model of the Qwen3.5 model to ONNX format.
    """
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
            opset_version=14, do_constant_folding=True, dynamic_axes=dynamic_axes,
        )
    print("LLM prefill exported successfully.")


def _export_llm_decode(decode, meta, output_dir, device, dummy_seq):
    """
    Export the LLM decode model of the Qwen3.5 model to ONNX format.
    """
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
            opset_version=14, do_constant_folding=True, dynamic_axes=dynamic_axes,
        )
    print("LLM decode exported successfully.")


def export_llm_prefill_decode(model, output_dir, device="cpu", dummy_seq=8,
                              dummy_num_img_tokens=16, dtype=torch.float16):
    """
    Export the LLM prefill and decode models of the Qwen3.5 model to ONNX format.
    """
    meta = _get_model_meta(model)
    text_model = meta["text_model"]
    lm_head = meta["lm_head"]
    image_token_id = meta["image_token_id"]

    text_model.eval()
    lm_head.eval()
    text_model.to(device)
    lm_head.to(device)

    prefill = Qwen35LlmPrefill(text_model, lm_head, image_token_id).to(device).eval()
    decode = Qwen35LlmDecode(text_model, lm_head).to(device).eval()

    _export_llm_prefill(prefill, output_dir, device, dummy_seq, dummy_num_img_tokens,
                        dtype=dtype)
    _export_llm_decode(decode, meta, output_dir, device, dummy_seq)


def main():
    """
    Main function for exporting Qwen3.5-4B model to ONNX format.
    """
    parser = argparse.ArgumentParser(description="Export Qwen3.5-4B to ONNX")
    parser.add_argument(
        "--model-id", type=str,
        default="./Qwen3.5-4B",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./qwen3_5_4b_onnx",
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

    args = parser.parse_args()

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
                              dtype=torch_dtype)

    print("Clearing memory after export...")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nExport finished. Files saved in {args.output_dir}")


if __name__ == "__main__":
    main()
