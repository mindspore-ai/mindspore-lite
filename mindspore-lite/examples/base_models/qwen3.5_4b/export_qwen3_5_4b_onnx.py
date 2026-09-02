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


# Keep the original graph as the default and enable the fused linear-attention
# operators explicitly when exporting for Ascend 310P.
USE_CUSTOM_RGDR = False
USE_CUSTOM_CGDR = False


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


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    """Export the 310P seven-input ChunkGatedDeltaRule Custom operator."""

    @staticmethod
    def forward(ctx, query, key, value, beta, state, actual_seq_lengths, g_decay):
        """Run the PyTorch reference implementation used during ONNX export."""
        del ctx, actual_seq_lengths
        batch_size = state.shape[0]
        sequence_length = query.shape[0] // batch_size
        num_key_heads = query.shape[1]
        num_value_heads = value.shape[1]
        key_dim = query.shape[2]
        value_dim = value.shape[2]
        query_4d = query.reshape(
            batch_size, sequence_length, num_key_heads, key_dim
        )
        key_4d = key.reshape(
            batch_size, sequence_length, num_key_heads, key_dim
        )
        value_4d = value.reshape(
            batch_size, sequence_length, num_value_heads, value_dim
        )
        beta_3d = beta.reshape(batch_size, sequence_length, num_value_heads)
        g_3d = g_decay.reshape(batch_size, sequence_length, num_value_heads)
        if num_value_heads > num_key_heads:
            repeat = num_value_heads // num_key_heads
            query_4d = query_4d.repeat_interleave(repeat, dim=2)
            key_4d = key_4d.repeat_interleave(repeat, dim=2)
        state_model = state.transpose(-2, -1).contiguous().float()
        out, final_state = _chunk_gated_delta_rule(
            query_4d,
            key_4d,
            value_4d,
            g=g_3d,
            beta=beta_3d,
            chunk_size=64,
            initial_state=state_model,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        )
        out = out.reshape(-1, num_value_heads, value_dim).to(torch.float16)
        final_state = final_state.transpose(-2, -1).contiguous().to(torch.float16)
        return out, final_state

    @staticmethod
    def symbolic(graph, query, key, value, beta, state, actual_seq_lengths, g_decay):
        """Export ChunkGatedDeltaRule as an ONNX Custom node."""
        scale_value = 1.0 / (128.0 ** 0.5)
        out, final_state = graph.op(
            "Custom",
            query,
            key,
            value,
            beta,
            state,
            actual_seq_lengths,
            g_decay,
            type_s="ChunkGatedDeltaRule",
            input_names_s=[
                "query", "key", "value", "beta", "initial_state",
                "actual_seq_lengths", "g",
            ],
            optional_input_names_s=["g"],
            output_names_s=["out", "final_state"],
            output_num_i=2,
            input_index_i=list(range(7)),
            scale_value_f=scale_value,
            dtype_i=10,
            outputs=2,
        )
        out.setType(value.type().with_dtype(torch.float16))
        final_state.setType(state.type().with_dtype(torch.float16))
        return out, final_state


def _chunk_gated_delta_rule_custom(query, key, value, g, beta, initial_state=None,
                                   output_final_state=False,
                                   use_qk_l2norm_in_kernel=False):
    """Qwen3.5 adapter for the 310P seven-input CGDR Custom operator."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)

    batch_size, sequence_length, num_key_heads, key_dim = query.shape
    num_value_heads = value.shape[2]
    value_dim = value.shape[3]
    token_count = batch_size * sequence_length
    query_flat = query.reshape(token_count, num_key_heads, key_dim).contiguous().half()
    key_flat = key.reshape(token_count, num_key_heads, key_dim).contiguous().half()
    value_flat = value.reshape(
        token_count, num_value_heads, value_dim
    ).contiguous().half()
    beta_flat = beta.reshape(token_count, num_value_heads).contiguous().half()
    g_flat = g.reshape(token_count, num_value_heads).contiguous().float()

    if initial_state is None:
        state_custom = torch.zeros(
            batch_size, num_value_heads, value_dim, key_dim,
            dtype=torch.float16, device=query.device,
        )
    else:
        state_custom = initial_state.transpose(-2, -1).contiguous().half()
    query_shape = torch.onnx.operators.shape_as_tensor(query)
    actual_seq_lengths = query_shape[1:2].to(torch.int32).expand(batch_size)
    core_attn_out, state_out = ChunkGatedDeltaRuleFunction.apply(
        query_flat,
        key_flat,
        value_flat,
        beta_flat,
        state_custom,
        actual_seq_lengths,
        g_flat,
    )
    core_attn_out = core_attn_out.reshape(
        batch_size, sequence_length, num_value_heads, value_dim
    ).to(initial_dtype)
    state_out = state_out.transpose(-2, -1).contiguous().float()
    if not output_final_state:
        state_out = None
    return core_attn_out, state_out


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


class RecurrentGatedDeltaRuleFunction(torch.autograd.Function):
    """Export the 310P RecurrentGatedDeltaRule Ascend C Custom operator.

    The custom operator stores its recurrent matrix as [B, NV, DV, DK], while
    the HuggingFace Qwen3.5 implementation stores it as [B, NV, DK, DV].  The
    caller performs the two layout conversions and keeps the model-facing state
    FP32.  The currently available kernel accepts an FP16 state input.
    """

    @staticmethod
    def forward(ctx, query, key, value, beta, state, actual_seq_lengths,
                ssm_state_indices, g_decay, num_accepted_tokens):
        """Run the PyTorch reference implementation used during ONNX export."""
        del ctx, ssm_state_indices, num_accepted_tokens
        scale_value = 1.0 / (128.0 ** 0.5)
        query_fp32 = query.float()
        key_fp32 = key.float()
        value_fp32 = value.float()
        beta_fp32 = beta.float()
        state_fp32 = state.float().clone()
        decay_fp32 = g_decay.float().exp()

        token_count, num_key_heads, _ = query_fp32.shape
        num_value_heads = value_fp32.shape[1]
        result = torch.zeros_like(value_fp32)
        seq_start = 0
        for batch_idx in range(actual_seq_lengths.numel()):
            seq_len = int(actual_seq_lengths[batch_idx].item())
            seq_end = min(seq_start + seq_len, token_count)
            for value_head in range(num_value_heads):
                key_head = value_head // (num_value_heads // num_key_heads)
                recurrent = state_fp32[batch_idx, value_head]
                for token_idx in range(seq_start, seq_end):
                    recurrent = recurrent * decay_fp32[token_idx, value_head]
                    key_t = key_fp32[token_idx, key_head]
                    query_t = query_fp32[token_idx, key_head] * float(scale_value)
                    delta = value_fp32[token_idx, value_head] - recurrent @ key_t
                    delta = delta * beta_fp32[token_idx, value_head]
                    recurrent = recurrent + torch.outer(delta, key_t)
                    result[token_idx, value_head] = recurrent @ query_t
                state_fp32[batch_idx, value_head] = recurrent
            seq_start = seq_end
        return result.to(value.dtype), state_fp32.to(state.dtype)

    @staticmethod
    def symbolic(graph, query, key, value, beta, state, actual_seq_lengths,
                 ssm_state_indices, g_decay, num_accepted_tokens):
        """Export RecurrentGatedDeltaRule as an ONNX Custom node."""
        scale_value = 1.0 / (128.0 ** 0.5)
        out, state_out = graph.op(
            "Custom",
            query, key, value, beta, state, actual_seq_lengths,
            ssm_state_indices, g_decay, num_accepted_tokens,
            type_s="RecurrentGatedDeltaRule",
            input_names_s=[
                "query", "key", "value", "beta", "state",
                "actual_seq_lengths", "ssm_state_indices", "g", "gk",
                "num_accepted_tokens",
            ],
            # Keep the exported GE IR signature aligned with the registered
            # 310P OpDef.  These optional inputs are still supplied below;
            # only gk is absent from this model adapter.
            optional_input_names_s=["g", "gk", "num_accepted_tokens"],
            input_index_i=[0, 1, 2, 3, 4, 5, 6, 7, 9],
            # Must match the names registered in recurrent_gated_delta_rule_def.cpp.
            output_names_s=["out", "state"],
            output_num_i=2,
            scale_value_f=scale_value,
            # MindSpore TypeId 42 is Float16.  Lite's Ascend Custom mapper
            # falls back to the first input dtype when a graph contains more
            # than one Custom type, so keep the public kernel boundary FP16.
            # The recurrent math remains FP32 inside the kernel and the model
            # casts the returned cache back to FP32 below.
            output_types_i=[42, 42],
            outputs_shape_s="3,-1,32,128,4,-1,32,128,128,",
            # ONNX TensorProto.FLOAT16.  Keep this attribute and the value
            # metadata aligned with the registered GE output contract.
            dtype_i=10,
            outputs=2,
        )
        # The registered kernel is RGDR<half, half>.  Its internal recurrent
        # arithmetic is FP32, but both graph outputs intentionally use FP16 to
        # match Lite's Custom-output allocation behavior.
        out.setType(value.type().with_dtype(torch.float16))
        state_out.setType(state.type().with_dtype(torch.float16))
        return out, state_out


def _recurrent_gated_delta_rule_custom(query, key, value, g, beta, initial_state,
                                       output_final_state=False,
                                       use_qk_l2norm_in_kernel=False):
    """Qwen3.5 adapter for the 310P RGDR Custom operator."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        # The current 310P kernel does not implement Q/K L2 normalization.
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)

    batch_size, sequence_length, num_key_heads, key_dim = query.shape
    num_value_heads = value.shape[2]
    value_dim = value.shape[3]
    token_count = batch_size * sequence_length

    # The registered 310P schema accepts FP16 for Q/K/V/beta/state even when
    # the surrounding model is exported with --dtype fp32.
    query_flat = query.reshape(token_count, num_key_heads, key_dim).to(torch.float16).contiguous()
    key_flat = key.reshape(token_count, num_key_heads, key_dim).to(torch.float16).contiguous()
    value_flat = value.reshape(token_count, num_value_heads, value_dim).to(torch.float16).contiguous()
    beta_flat = beta.reshape(token_count, num_value_heads).to(torch.float16).contiguous()
    g_flat = g.reshape(token_count, num_value_heads).float().contiguous()

    # Custom kernel layout is [B, NV, DV, DK] and its state input is FP16.
    state_custom = initial_state.transpose(-2, -1).contiguous().to(torch.float16)
    actual_seq_lengths = torch.full(
        (batch_size,), sequence_length, dtype=torch.int32, device=query.device
    )
    # The RGDR Decode export is fixed to batch one and a single token below,
    # so every token reads and updates state slot 0.  Keep this input constant:
    # exporting arange(...).repeat_interleave(...) produces an ONNX OneHot
    # subgraph that MindSpore Lite 2.9 cannot map for Ascend conversion.
    ssm_state_indices = torch.zeros(
        token_count, dtype=torch.int32, device=query.device
    )
    num_accepted_tokens = torch.full(
        (batch_size,), sequence_length, dtype=torch.int32, device=query.device
    )
    core_attn_out, state_out = RecurrentGatedDeltaRuleFunction.apply(
        query_flat, key_flat, value_flat, beta_flat, state_custom,
        actual_seq_lengths, ssm_state_indices, g_flat, num_accepted_tokens,
    )
    core_attn_out = core_attn_out.reshape(
        batch_size, sequence_length, num_value_heads, value_dim
    ).to(initial_dtype)
    # The model-facing recurrent cache remains FP32 even though the Custom
    # boundary writes FP16; make the conversion explicit in the outer graph.
    state_out = state_out.transpose(-2, -1).contiguous().float()
    if not output_final_state:
        state_out = None
    return core_attn_out, state_out


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

    if USE_CUSTOM_CGDR:
        core_attn_out, last_recurrent_state = _chunk_gated_delta_rule_custom(
            query, key, value, g=g, beta=beta,
            initial_state=None, output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
    else:
        if layer.num_v_heads // layer.num_k_heads > 1:
            repeat = layer.num_v_heads // layer.num_k_heads
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)
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

    if USE_CUSTOM_RGDR:
        core_attn_out, last_recurrent_state = _recurrent_gated_delta_rule_custom(
            query, key, value, g=g, beta=beta,
            initial_state=recurrent_state_in, output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
    else:
        if layer.num_v_heads // layer.num_k_heads > 1:
            repeat = layer.num_v_heads // layer.num_k_heads
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)
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


class IncreFlashAttentionFunction(torch.autograd.Function):
    """Export full-attention Decode as the CANN IncreFlashAttention op."""

    @staticmethod
    def forward(ctx, query, key, value, atten_mask, num_heads, scale_value,
                num_kv_heads):
        """Run a PyTorch reference implementation during ONNX tracing."""
        del ctx
        key_ref = key
        value_ref = value
        if 0 < num_kv_heads < num_heads:
            repeat = num_heads // num_kv_heads
            key_ref = key_ref.repeat_interleave(repeat, dim=1)
            value_ref = value_ref.repeat_interleave(repeat, dim=1)
        attn = torch.matmul(query, key_ref.transpose(2, 3)) * float(scale_value)
        if atten_mask is not None:
            mask = atten_mask.to(torch.bool)
            if mask.dim() == 4 and mask.shape[1] == 1:
                mask = mask.expand(attn.shape[0], attn.shape[1], mask.shape[2], mask.shape[3])
            attn = attn.masked_fill(mask, torch.finfo(attn.dtype).min)
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(query.dtype)
        output = torch.matmul(attn, value_ref)
        return output

    @staticmethod
    def symbolic(graph, query, key, value, atten_mask, num_heads, scale_value,
                 num_kv_heads):
        """Export an IncreFlashAttention Custom node."""
        inputs = [query, key, value]
        input_index = [0, 1, 2]
        if atten_mask is not None:
            inputs.append(atten_mask)
            input_index.append(3)
        output = graph.op(
            "Custom",
            *inputs,
            type_s="IncreFlashAttention",
            input_names_s=["query", "key", "value", "atten_mask"],
            optional_input_names_s=["atten_mask"],
            output_names_s=["attention_out"],
            output_num_i=1,
            input_index_i=input_index,
            num_heads_i=int(num_heads),
            scale_value_f=float(scale_value),
            input_layout_s="BNSD",
            num_key_value_heads_i=int(num_kv_heads),
            block_size_i=0,
            inner_precise_i=1,
        )
        output.setType(query.type())
        return output


def _incre_flash_attention(query, key, value, atten_mask, num_heads, scale_value,
                           num_kv_heads):
    """Call the fixed-cache incremental flash-attention Custom operator."""
    return IncreFlashAttentionFunction.apply(
        query, key, value, atten_mask, int(num_heads), float(scale_value), int(num_kv_heads),
    )


class ScatterUpdateFunction(torch.autograd.Function):
    """Export fixed KV-cache updates as the CANN Scatter Custom operator."""

    @staticmethod
    def forward(ctx, var, indices, updates, axis):
        """Run the cache update used while tracing the ONNX graph."""
        del ctx
        axis = int(axis)
        if axis != 2:
            raise ValueError(f"Only BNSD axis 2 is supported, but got axis={axis}")
        result = var.clone()
        positions = indices.to(torch.int64).reshape(-1)
        index = positions.reshape(-1, 1, 1, 1).expand(
            updates.size(0), updates.size(1), 1, updates.size(3)
        )
        result.scatter_(axis, index, updates.to(var.dtype))
        return result

    @staticmethod
    def symbolic(graph, var, indices, updates, axis):
        """Export a Scatter Custom node in update mode."""
        output = graph.op(
            "Custom",
            var,
            indices,
            updates,
            input_index_i=[0, 1, 2],
            input_names_s=["var", "indices", "updates"],
            optional_input_names_s=[],
            output_names_s=["var"],
            type_s="Scatter",
            reduce_s="update",
            axis_i=int(axis),
        )
        output.setType(var.type())
        return output


def _kv_cache_update(past, update, cache_pos):
    """Write the current key or value into a fixed-capacity BNSD cache."""
    indices = cache_pos.reshape(-1).to(torch.int64)
    return ScatterUpdateFunction.apply(past, indices, update.to(past.dtype), 2)


def _expand_kv_heads_for_ifa(cache, num_heads, num_kv_heads, head_dim):
    """Expand compact GQA cache heads to the MHA layout accepted by 310P IFA."""
    if num_heads == num_kv_heads:
        return cache
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads={num_heads} must be divisible by num_kv_heads={num_kv_heads}"
        )
    repeat = num_heads // num_kv_heads
    batch_size = cache.shape[0]
    cache_len = cache.shape[2]
    return cache.unsqueeze(2).expand(
        batch_size, num_kv_heads, repeat, cache_len, head_dim
    ).reshape(batch_size, num_heads, cache_len, head_dim)


def _full_attn_decode_fixed(layer, hidden_states, position_embeddings,
                            attention_mask, past_key, past_value, cache_pos):
    """Run one full-attention Decode layer with a fixed-capacity KV cache."""
    input_shape = hidden_states.shape[:-1]
    head_dim = layer.head_dim
    num_heads = int(layer.config.num_attention_heads)
    num_kv_heads = int(layer.config.num_key_value_heads)
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

    key_cache = _kv_cache_update(past_key, key_states, cache_pos)
    value_cache = _kv_cache_update(past_value, value_states, cache_pos)
    scaling = getattr(layer, "scaling", 1.0 / (head_dim ** 0.5))
    if num_kv_heads < num_heads:
        # The current 310P IFA tiler rejects GQA with head_dim=256.
        # Keep the persistent cache in compact GQA form and materialize an
        # equivalent MHA view only for the fused attention call.
        key_for_attn = _expand_kv_heads_for_ifa(
            key_cache, num_heads, num_kv_heads, head_dim
        )
        value_for_attn = _expand_kv_heads_for_ifa(
            value_cache, num_heads, num_kv_heads, head_dim
        )
        ifa_num_kv_heads = num_heads
    else:
        key_for_attn = key_cache
        value_for_attn = value_cache
        ifa_num_kv_heads = num_kv_heads
    attn_output = _incre_flash_attention(
        query_states.to(key_cache.dtype).contiguous(),
        key_for_attn.contiguous(), value_for_attn.contiguous(),
        attention_mask, num_heads, scaling, ifa_num_kv_heads,
    )
    attn_output = attn_output.to(hidden_states.dtype)
    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = attn_output * torch.sigmoid(gate)
    attn_output = layer.o_proj(attn_output)
    return attn_output, key_cache, value_cache


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
    def __init__(self, text_model, lm_head, image_token_id, max_seq_len=None):
        """
        Initialize the Qwen3.5 LLM prefill model.
        """
        super().__init__()
        self.text_model = text_model
        self.lm_head = lm_head
        self.image_token_id = int(image_token_id)
        self.config = text_model.config
        self.max_seq_len = None if max_seq_len is None else int(max_seq_len)

    def forward(self, input_ids, attention_mask, position_ids, image_embeds):
        """
        Forward function for the Qwen3.5 LLM prefill model.
        """
        inputs_embeds = self.text_model.embed_tokens(input_ids)
        image_mask = input_ids == self.image_token_id
        image_values = image_embeds.to(
            device=inputs_embeds.device, dtype=inputs_embeds.dtype
        )
        if self.max_seq_len is None:
            image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_values)
        else:
            image_indices = image_mask.long().cumsum(dim=1).clamp(min=1) - 1
            image_values = image_values[image_indices]
            inputs_embeds = torch.where(
                image_mask.unsqueeze(-1), image_values, inputs_embeds
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
        if self.max_seq_len is not None and present_kv_stack.dim() >= 4:
            if present_kv_stack.shape[3] > self.max_seq_len:
                raise ValueError(
                    f"Prefill sequence length exceeds max_seq_len={self.max_seq_len}"
                )
            if present_kv_stack.shape[3] < self.max_seq_len:
                pad_size = self.max_seq_len - present_kv_stack.shape[3]
                present_kv_stack = F.pad(present_kv_stack, (0, 0, 0, pad_size))

        return logits, present_conv_stack, present_recurrent_stack, present_kv_stack


class Qwen35LlmDecode(torch.nn.Module):
    """
    Qwen3.5 LLM decode model for the attention layer.
    """
    def __init__(self, text_model, lm_head, fixed_kv_cache=False):
        """
        Initialize the Qwen3.5 LLM decode model.
        """
        super().__init__()
        self.text_model = text_model
        self.lm_head = lm_head
        self.config = text_model.config
        self.fixed_kv_cache = bool(fixed_kv_cache)

    def forward(self, input_ids, attention_mask, position_ids,
                past_conv_states, past_recurrent_states, past_kv_cache):
        """
        Decode with either the original growing cache or a fixed-capacity cache.
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
        cache_pos = None
        if self.fixed_kv_cache:
            max_seq_len = past_kv_cache.shape[3]
            total_valid_tokens = attention_mask.sum(dim=1, keepdim=True)
            cache_pos = (total_valid_tokens - q_len).reshape(())
            kv_range = torch.arange(
                max_seq_len, device=inputs_embeds.device, dtype=torch.int64
            )
            allowed = kv_range.view(1, -1) <= cache_pos.view(-1, 1)
            allowed = allowed & attention_mask.to(torch.bool)
            attn_mask = (~allowed).view(-1, 1, 1, max_seq_len)
        else:
            past_len = past_kv_cache.shape[3]
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
                if self.fixed_kv_cache:
                    attn_out, pk, pv = _full_attn_decode_fixed(
                        layer.self_attn, hidden_states, position_embeddings, attn_mask,
                        pk_in, pv_in, cache_pos,
                    )
                else:
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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "qwen3_5_vision.onnx"
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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefill_path = output_dir / "qwen3_5_llm_prefill.onnx"
    if dummy_seq <= 0:
        raise ValueError(f"dummy_seq must be positive, but got {dummy_seq}")
    if prefill.max_seq_len is not None and dummy_seq > prefill.max_seq_len:
        raise ValueError(
            f"dummy_seq must be in [1, {prefill.max_seq_len}], but got {dummy_seq}"
        )
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
    if USE_CUSTOM_CGDR or prefill.max_seq_len is not None:
        # Both the fused Prefill and fixed-cache Decode paths target batch one.
        dynamic_axes = {
            "input_ids": {1: "seq_len"},
            "attention_mask": {1: "seq_len"},
            "position_ids": {2: "seq_len"},
            "image_embeds": {0: "num_image_tokens"},
            "logits": {1: "seq_len"},
        }
        if prefill.max_seq_len is None:
            dynamic_axes["present_kv_cache"] = {3: "seq_len"}
    else:
        dynamic_axes = {
            "input_ids": {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "position_ids": {1: "batch", 2: "seq_len"},
            "image_embeds": {0: "num_image_tokens"},
            "logits": {0: "batch", 1: "seq_len"},
            "present_conv_states": {1: "batch"},
            "present_recurrent_states": {1: "batch"},
            "present_kv_cache": {1: "batch"},
        }
        if prefill.max_seq_len is None:
            dynamic_axes["present_kv_cache"][3] = "seq_len"
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


def _export_llm_decode(decode, meta, output_dir, device, dummy_seq, max_seq_len=2048):
    """
    Export fixed-cache Custom Decode or the original dynamic fallback graph.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    decode_path = output_dir / "qwen3_5_llm_decode.onnx"
    dummy_step = 1
    dummy_past_len = dummy_seq
    fixed_kv_cache = decode.fixed_kv_cache
    if dummy_past_len <= 0:
        raise ValueError(f"dummy_seq must be positive, but got {dummy_past_len}")
    if fixed_kv_cache and dummy_past_len + dummy_step > max_seq_len:
        raise ValueError(
            f"dummy_seq must be in [1, {max_seq_len - dummy_step}], "
            f"but got {dummy_past_len}"
        )
    dummy_input_ids_step = torch.randint(
        0, 1000, (1, dummy_step), dtype=torch.int64, device=device
    )
    if fixed_kv_cache:
        dummy_attention_mask_step = torch.zeros(
            1, max_seq_len, dtype=torch.int64, device=device
        )
        dummy_attention_mask_step[:, :dummy_past_len + dummy_step] = 1
        kv_cache_len = max_seq_len
    else:
        dummy_attention_mask_step = torch.ones(
            1, dummy_past_len + dummy_step, dtype=torch.int64, device=device
        )
        kv_cache_len = dummy_past_len
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
        2 * num_full, 1, num_kv_heads, kv_cache_len, head_dim,
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
    shape_info = f"max_seq_len={max_seq_len}" if fixed_kv_cache else "dynamic KV cache"
    print(f"Exporting LLM decode to {decode_path} ({shape_info})...")
    export_options = {
        "input_names": input_names,
        "output_names": output_names,
        "opset_version": 14,
        "do_constant_folding": True,
    }
    if not fixed_kv_cache:
        export_options["dynamic_axes"] = dynamic_axes
    with torch.no_grad():
        onnx_utils.export(
            decode,
            (dummy_input_ids_step, dummy_attention_mask_step, dummy_position_ids_step,
             dummy_past_conv, dummy_past_recurrent, dummy_past_kv),
            str(decode_path),
            **export_options,
        )
    print("LLM decode exported successfully.")


def export_llm_prefill_decode(model, output_dir, device="cpu", dummy_seq=8,
                              dummy_num_img_tokens=16, dtype=torch.float16,
                              max_seq_len=2048):
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

    prefill = Qwen35LlmPrefill(
        text_model, lm_head, image_token_id,
        max_seq_len=max_seq_len if USE_CUSTOM_RGDR else None,
    ).to(device).eval()
    decode = Qwen35LlmDecode(
        text_model, lm_head, fixed_kv_cache=USE_CUSTOM_RGDR
    ).to(device).eval()

    # Independent component exports can assign the same graph-local external
    # data filename to different tensors.  Keep every graph self-contained so
    # that repeated or component-only exports cannot overwrite another graph.
    output_dir = Path(output_dir)
    _export_llm_prefill(
        prefill, output_dir / "prefill", device, dummy_seq,
        dummy_num_img_tokens, dtype=dtype,
    )
    _export_llm_decode(
        decode, meta, output_dir / "decode", device, dummy_seq,
        max_seq_len=max_seq_len,
    )


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
    parser.add_argument("--dummy-seq-len", type=int, default=8,
        help="Dummy sequence length for LLM export",
    )
    parser.add_argument("--max-seq-len", type=int, default=2048,
        help="Maximum sequence length for the fixed Decode mask and KV cache",
    )
    parser.add_argument(
        "--dtype", type=str, default="fp16", choices=["fp16", "fp32"],
        help="Export dtype",
    )
    parser.add_argument(
        "--component", type=str, default="all", choices=["all", "prefill", "decode"],
        help="Export all components, the LLM prefill graph, or the LLM decode graph",
    )
    parser.add_argument(
        "--enable-rgdr-custom", action="store_true",
        help="Enable RGDR plus fixed Decode KV cache with Scatter and IFA",
    )
    parser.add_argument(
        "--enable-cgdr-custom", action="store_true",
        help="Replace the expanded prefill chunk rule with the 310P Custom CGDR op",
    )
    args = parser.parse_args()

    global USE_CUSTOM_RGDR, USE_CUSTOM_CGDR
    USE_CUSTOM_RGDR = args.enable_rgdr_custom
    USE_CUSTOM_CGDR = args.enable_cgdr_custom

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

    if args.component == "all":
        export_vision_tower(
            model, output_dir / "vision", args.device, args.vision_image_size
        )
        export_llm_prefill_decode(model, output_dir, args.device, args.dummy_seq_len,
                                  dtype=torch_dtype, max_seq_len=args.max_seq_len)
    elif args.component == "prefill":
        meta = _get_model_meta(model)
        text_model = meta["text_model"].to(args.device).eval()
        lm_head = meta["lm_head"].to(args.device).eval()
        prefill = Qwen35LlmPrefill(
            text_model, lm_head, meta["image_token_id"],
            max_seq_len=args.max_seq_len if USE_CUSTOM_RGDR else None,
        ).to(args.device).eval()
        _export_llm_prefill(
            prefill, output_dir / "prefill", args.device, args.dummy_seq_len, 16,
            dtype=torch_dtype,
        )
    else:
        meta = _get_model_meta(model)
        text_model = meta["text_model"].to(args.device).eval()
        lm_head = meta["lm_head"].to(args.device).eval()
        decode = Qwen35LlmDecode(
            text_model, lm_head, fixed_kv_cache=USE_CUSTOM_RGDR
        ).to(args.device).eval()
        _export_llm_decode(
            decode, meta, output_dir / "decode", args.device, args.dummy_seq_len,
            max_seq_len=args.max_seq_len,
        )

    print("Clearing memory after export...")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nExport finished. Files saved in {args.output_dir}")


if __name__ == "__main__":
    main()
