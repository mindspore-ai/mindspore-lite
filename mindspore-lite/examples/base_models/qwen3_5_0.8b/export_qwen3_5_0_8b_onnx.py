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
Export Qwen3.5-0.8B model to ONNX format.

Qwen3.5-0.8B is a multimodal VL model with hybrid linear attention
(GatedDeltaNet) and full attention architecture. The export splits
the model into three ONNX files:
  - Vision Tower (fixed grid_thw)
  - LLM Prefill (with image_embeds input)
  - LLM Decode (with recurrent states + KV cache input)
"""

import os
import sys
import argparse
import gc
import importlib.util
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

# Disable torch_npu auto-load to avoid libhccl.so error
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

# Patch find_spec to hide torch_npu so transformers won't try to load it
# (torch_npu is installed but requires CANN/libhccl which is absent)
_orig_find_spec = importlib.util.find_spec

def _skip_torch_npu_find_spec(name, *args, **kwargs):
    if name == "torch_npu":
        return None
    return _orig_find_spec(name, *args, **kwargs)

importlib.util.find_spec = _skip_torch_npu_find_spec

try:
    getattr(torch, "_dynamo").disable()
except (ImportError, AttributeError):
    pass

try:
    try:
        from transformers import Qwen3_5ForConditionalGeneration as QwenForConditionalGeneration
    except ImportError:
        from transformers import Qwen3VLForConditionalGeneration as QwenForConditionalGeneration
except ImportError:
    print("Error: transformers package not found or version too low.")
    print("Please install: pip install --upgrade transformers")
    sys.exit(1)


_cgdr_config = {
    "enabled": False,
    "nk":16,
    "nv":16,
    "dk":128,
    "dv":128,
    "scale_value":1.0/(128**0.5)
}

def _set_cgdr_config(enabled = False, nk = 16, nv = 16, dk = 128, dv = 128):
    _cgdr_config["enabled"] = enabled
    _cgdr_config["nk"] = nk
    _cgdr_config["nv"] = nv
    _cgdr_config["dk"] = dk
    _cgdr_config["dv"] = dv
    _cgdr_config["scale_value"] = 1.0 / (dk**0.5)


class _CustomPromptFlashAttention(torch.autograd.Function):
    """CANN PromptFlashAttention for prefill full attention layers (with mask + GQA)."""
    @staticmethod
    def forward(ctx, query, key, value, attn_mask, num_heads, num_key_value_heads, scale_value):
        """Forward pass: CANN PromptFlashAttention simulation via torch."""
        del ctx
        if num_key_value_heads < num_heads:
            key_states_for_attn = key.repeat_interleave(num_heads // num_key_value_heads, dim=1)
            value_states_for_attn = value.repeat_interleave(num_heads // num_key_value_heads, dim=1)
        else:
            key_states_for_attn = key
            value_states_for_attn = value
        attn_weights = torch.matmul(query, key_states_for_attn.transpose(2, 3)) * scale_value
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        attn_weights = F.softmax(attn_weights, dim=-1,dtype=torch.float32).to(query.dtype)
        attn_output = torch.matmul(attn_weights, value_states_for_attn)
        attn_output = attn_output.transpose(1,2)
        return attn_output

    @staticmethod
    def symbolic(g,query,key,value,attn_mask,num_heads,num_key_value_heads,scale_value):
        """Define ONNX custom op for PromptFlashAttention."""
        y = g.op(
                "Custom",
                query,
                key,
                value,
                attn_mask,
                type_s="PromptFlashAttention",
                input_names_s=["query", "key", "value", "atten_mask"],
                optional_input_names_s=["atten_mask"],
                output_names_s= ["attention_out"],
                output_num_i=1,
                input_index_i=[0, 1, 2, 4],
                num_heads_i=int(num_heads),
                num_key_value_heads_i=int(num_key_value_heads),
                scale_value_f=float(scale_value),
                input_layout_s="BNSD_BSND",
                inner_precise_i=1,
            )
        y.setType(query.type())
        return y


def _cann_pfa(query,key,value,attn_mask,num_heads,num_key_value_heads,scale_value):
    return _CustomPromptFlashAttention.apply(query,key,value,attn_mask,num_heads,num_key_value_heads,scale_value)


class _ChunkGatedDeltaRuleOp(torch.autograd.Function):
    """CGDR custom op (prefill_cgdr version) using _cgdr_config for ONNX export."""
    @staticmethod
    def forward(ctx, query, key, value, beta, g, initial_state, actual_seq_lengths):
        """Forward pass for _ChunkGatedDeltaRuleOp."""
        del ctx, actual_seq_lengths
        core_attn_out, final_state = _chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta,
            initial_state=initial_state, output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        return core_attn_out, final_state.to(torch.float32)

    @staticmethod
    def symbolic(g, query, key, value, beta, g_in, initial_state, actual_seq_lengths):
        """Define ONNX custom op for ChunkGatedDeltaRule (prefill_cgdr)."""
        del initial_state
        nk = _cgdr_config["nk"]
        nv = _cgdr_config["nv"]
        dk = _cgdr_config["dk"]
        dv = _cgdr_config["dv"]
        scale_value = _cgdr_config["scale_value"]
        bfloat16_val = 16
        float16_val = 10
        float32_val = 1

        eps_const = g.op('Constant', value_t=torch.tensor(1e-6, dtype=torch.float16))
        one_const = g.op('Constant', value_t=torch.tensor(1.0, dtype=torch.float16))
        axes_const = g.op('Constant', value_t=torch.tensor([-1], dtype=torch.int64))

        q_sq = g.op("Mul", query, query)
        q_sq_sum = g.op("ReduceSum", q_sq, axes_const, keepdims_i=1)
        q_sq_sum_eps = g.op("Add", q_sq_sum, eps_const)
        q_norm = g.op("Sqrt", q_sq_sum_eps)
        q_inv_norm = g.op("Div", one_const, q_norm)
        q_normed = g.op("Mul", query, q_inv_norm)

        k_sq = g.op('Mul', key, key)
        k_sq_sum = g.op('ReduceSum', k_sq, axes_const, keepdims_i=1)
        k_sq_sum_eps = g.op('Add', k_sq_sum, eps_const)
        k_norm = g.op('Sqrt', k_sq_sum_eps)
        k_inv_norm = g.op('Div', one_const, k_norm)
        k_normed = g.op('Mul', key, k_inv_norm)

        q_bf16 = g.op("Cast", q_normed, to_i=bfloat16_val)
        k_bf16 = g.op("Cast", k_normed, to_i=bfloat16_val)
        v_bf16 = g.op("Cast", value, to_i=bfloat16_val)
        beta_bf16 = g.op("Cast", beta, to_i=bfloat16_val)

        shape_qk = g.op('Constant', value_t=torch.tensor([-1, nk, dk], dtype=torch.int64))
        shape_v = g.op('Constant', value_t=torch.tensor([-1, nv, dv], dtype=torch.int64))
        shape_bg = g.op('Constant', value_t=torch.tensor([-1, nv], dtype=torch.int64))

        q_tnd = g.op("Reshape", q_bf16, shape_qk)
        k_tnd = g.op("Reshape", k_bf16, shape_qk)
        v_tnd = g.op("Reshape", v_bf16, shape_v)
        beta_tnd = g.op("Reshape", beta_bf16, shape_bg)
        g_tnd = g.op("Reshape", g_in, shape_bg)

        init_fp16 = g.op("Constant", value_t=torch.zeros([1, nv, dv, dk], dtype=torch.float16))
        init_state_bf16 = g.op("Cast", init_fp16, to_i=bfloat16_val)

        out, final_state = g.op(
            "Custom",
            q_tnd, k_tnd, v_tnd, beta_tnd, init_state_bf16, actual_seq_lengths, g_tnd,
            outputs=2,
            type_s="ChunkGatedDeltaRule",
            input_names_s=["query", "key", "value", "beta", "initial_state", "actual_seq_lengths", "g"],
            optional_input_names_s=["g"],
            output_names_s=["out", "final_state"],
            output_num_i=2,
            scale_value_f=scale_value,
        )

        out_fp16 = g.op("Cast", out, to_i=float16_val)
        final_fp32 = g.op("Cast", final_state, to_i=float32_val)

        out_shape = g.op("Constant", value_t=torch.tensor([1, -1, nv, dv], dtype=torch.int64))
        out_bnds = g.op("Reshape", out_fp16, out_shape)
        final_transposed = g.op("Transpose", final_fp32, perm_i=[0, 1, 3, 2])
        return out_bnds, final_transposed


def _l2norm(x, dim=-1, eps=1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _make_additive_causal_mask(attention_mask, q_len, k_len, past_len):
    """Make boolean causal mask for CANN PromptFlashAttention ops.

    Returns a boolean mask (True = masked position) suitable for CANN
    PromptFlashAttention ops that accept boolean attention masks.
    """
    device = attention_mask.device
    ar_q = torch.arange(q_len, device=device)
    ar_k = torch.arange(k_len, device=device)
    causal = ar_k[None, :] > (past_len + ar_q[:, None])
    padding = ~attention_mask.to(torch.bool)
    mask = causal[None, None, :, :].expand(attention_mask.shape[0], 1, q_len, k_len) | padding[:, None, None, :]
    return mask


def _chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64, initial_state=None,
                            output_final_state=False, use_qk_l2norm_in_kernel=False):
    """Torch implementation of chunk gated delta rule for prefill."""
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
    """Torch implementation of recurrent gated delta rule for decode."""
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


class CustomRecurrentGatedDeltaRule(torch.autograd.Function):
    """RecurrentGatedDeltaRule: CANN custom op for decode-step linear attention.

    CANN interface (aclnnRecurrentGatedDeltaRule):
      Inputs:  query(T,Nk,Dk)BF16, key(T,Nk,Dk)BF16, value(T,Nv,Dv)BF16,
               beta(T,Nv)BF16, stateRef(BN,Nv,Dv,Dk)BF16 [in-out],
               actualSeqLengths(B,)INT32, ssmStateIndices(T,)INT32,
               g(T,Nv)FP32 [opt], gk(T,Nv,Dk)FP32 [opt],
               numAcceptedTokens(B,)INT32
      Outputs: out(T,Nv,Dv)BF16, stateRef(BN,Nv,Dv,Dk)BF16 (modified in-place)
    """

    @staticmethod
    def forward(ctx, query, key, value, beta, initial_state,
                cu_seqlens, ssm_state_indices, g, num_accepted_tokens):
        """Forward pass: invoke RecurrentGatedDeltaRule kernel via torch simulation."""
        del ctx, cu_seqlens, ssm_state_indices, num_accepted_tokens
        batch_size = initial_state.shape[0]
        total_tokens = query.shape[0]
        seq_len = total_tokens // batch_size
        query_4d = query.reshape(batch_size, seq_len, *query.shape[1:])
        key_4d = key.reshape(batch_size, seq_len, *key.shape[1:])
        value_4d = value.reshape(batch_size, seq_len, *value.shape[1:])
        beta_3d = beta.reshape(batch_size, seq_len, beta.shape[-1])
        g_3d = g.reshape(batch_size, seq_len, *g.shape[1:])
        out, final_state = _recurrent_gated_delta_rule(
            query_4d, key_4d, value_4d, g=g_3d, beta=beta_3d,
            initial_state=initial_state.to(dtype=torch.float32),
            output_final_state=True, use_qk_l2norm_in_kernel=False,
        )
        out = out.reshape(total_tokens, out.shape[2], out.shape[3]).contiguous()
        final_state = final_state.to(dtype=torch.float32)
        return out, final_state

    @staticmethod
    def symbolic(g, query, key, value, beta, initial_state,
                 cu_seqlens, ssm_state_indices, g_opt, num_accepted_tokens):
        """Define ONNX custom op matching CANN aclnnRecurrentGatedDeltaRule.

        Input order (matching CANN C API signature):
          0: query, 1: key, 2: value, 3: beta, 4: stateRef [in-out],
          5: actualSeqLengths, 6: ssmStateIndices, 7: g (opt),
          8: gk (opt, nullptr), 9: numAcceptedTokens
        """
        # Build inputs list: required + g + gk(nullptr) + numAcceptedTokens
        head_k_dim = (
            query.type().sizes()[-1]
            if query.type().sizes() and len(query.type().sizes()) >= 3
            else 128
        )
        scale_value = 1.0 / (float(head_k_dim) ** 0.5)

        # All inputs in CANN order: 0-6 required, 7 g, 8 gk (not provided),
        # 9 numAcceptedTokens
        can_inputs = [
            query, key, value, beta, initial_state,
            cu_seqlens, ssm_state_indices, g_opt,
        ]
        can_input_names = [
            "query", "key", "value", "beta", "state",
            "actual_seq_lengths", "ssm_state_indices", "g",
        ]

        # numAcceptedTokens is required (non-optional) per CANN spec
        can_inputs.append(num_accepted_tokens)
        can_input_names.append("num_accepted_tokens")

        # gk is optional — not provided (nullptr behavior)
        # Add gk to input_names_s / optional_input_names_s for schema completeness
        can_input_names.append("gk")

        # input_index_i: indices of CANN inputs that have non-null tensors
        # 0-6: required, 7: g (always provided), 9: numAcceptedTokens (always provided)
        # 8: gk (not provided, excluded from input_index_i)
        input_index = [0, 1, 2, 3, 4, 5, 6, 7, 9]

        out, final_state = g.op(
            "Custom", *can_inputs,
            type_s="RecurrentGatedDeltaRule",
            scale_value_f=float(scale_value),
            input_names_s=can_input_names,
            optional_input_names_s=["gk"],
            output_names_s=["out", "state"],
            output_num_i=2,
            input_index_i=input_index,
            outputs=2,
        )
        out.setType(value.type())
        final_state.setType(initial_state.type())
        return out, final_state


def _linear_attn_prefill(
    layer,
    hidden_states,
    attention_mask,
    actual_seq_lengths=None,
    ssm_state_indices=None,
):
    """Forward pass for GatedDeltaNet layer during prefill (CGDR custom op)."""
    del actual_seq_lengths, ssm_state_indices
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

    # _ChunkGatedDeltaRuleOp (prefill_cgdr version) does l2norm internally in symbolic,
    # so skip external l2norm here.
    state0 = torch.zeros(
        1, layer.num_v_heads, layer.head_v_dim, layer.head_k_dim,
        device=query.device, dtype=torch.float32,
    )
    shape = torch.onnx.operators.shape_as_tensor(hidden_states)
    cgdr_actual_seq_lengths = shape[1:2]
    core_attn_out, last_recurrent_state = _ChunkGatedDeltaRuleOp.apply(
        query, key, value, beta, g, state0, cgdr_actual_seq_lengths,
    )
    core_attn_out = core_attn_out.reshape(-1, layer.head_v_dim)
    z = z.reshape(-1, layer.head_v_dim)
    core_attn_out = layer.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

    output = layer.out_proj(core_attn_out)
    return output, conv_state, last_recurrent_state


def _linear_attn_decode(layer, hidden_states, conv_state_in, recurrent_state_in,
                        use_rgdr_custom: bool = True):
    """Forward pass for GatedDeltaNet layer during decode (single token).

    Args:
        use_rgdr_custom: If True, use RecurrentGatedDeltaRule custom op for ONNX export.
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
        # Save pre-repeated q/k for CANN custom op path (expects Nk heads)
        q_nk = query  # (B, seq, Nk, Dk)
        k_nk = key
        query = query.repeat_interleave(layer.num_v_heads // layer.num_k_heads, dim=2)
        key = key.repeat_interleave(layer.num_v_heads // layer.num_k_heads, dim=2)
    else:
        q_nk, k_nk = query, key

    if use_rgdr_custom:
        # CANN aclnnRecurrentGatedDeltaRule expects:
        #   query/key: (T, Nk, Dk) — original key heads, NOT expanded to Nv
        #   value:     (T, Nv, Dv)
        #   state:     (BlockNum, Nv, Dv, Dk) — Nv states
        # The op handles Nk→Nv expansion internally via ssmStateIndices.
        eps = 1e-6
        nk = layer.num_k_heads
        q_3d = q_nk[:, :, :nk, :].to(dtype=torch.float32).reshape(
            -1, nk, layer.head_k_dim)
        k_3d = k_nk[:, :, :nk, :].to(dtype=torch.float32).reshape(
            -1, nk, layer.head_k_dim)
        v_3d = value.to(dtype=torch.float32).reshape(
            -1, value.shape[2], value.shape[3])
        beta_2d = beta.to(dtype=torch.float32).reshape(-1, beta.shape[-1])
        g_2d = g.to(dtype=torch.float32).reshape(-1, g.shape[-1])
        # L2 normalize q/k to match non-custom path
        q_3d = q_3d / (q_3d.norm(dim=-1, keepdim=True) + eps)
        k_3d = k_3d / (k_3d.norm(dim=-1, keepdim=True) + eps)
        # actual_seq_lengths: (B,) — per CANN spec, each element = effective seq len
        # For single-token decode, each batch has seq_len=1
        actual_seq_lengths = torch.ones(
            batch_size, device=hidden_states.device, dtype=torch.int32
        )
        ssm_state_indices = torch.arange(
            batch_size, device=hidden_states.device, dtype=torch.int32
        )
        # num_accepted_tokens: (B,) — per CANN spec, required input
        num_accepted_tokens = torch.ones(
            batch_size, device=hidden_states.device, dtype=torch.int32
        )
        core_attn_out_3d, last_recurrent_state = CustomRecurrentGatedDeltaRule.apply(
            q_3d, k_3d, v_3d, beta_2d, recurrent_state_in.to(dtype=torch.float32),
            actual_seq_lengths, ssm_state_indices, g_2d, num_accepted_tokens,
        )
        core_attn_out = core_attn_out_3d.reshape(
            batch_size, seq_len, -1, layer.head_v_dim
        )
    else:
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
    """Forward pass for full attention layer using CANN PromptFlashAttention."""
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

    scaling = getattr(layer, "scaling", 1.0 / (head_dim ** 0.5))

    # CANN PromptFlashAttention handles GQA internally via num_key_value_heads
    attn_output = _cann_pfa(
        query_states, key_states, value_states,
        attention_mask, num_heads, num_kv_heads, scaling,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()

    attn_output = attn_output * torch.sigmoid(gate)
    attn_output = layer.o_proj(attn_output)
    return attn_output, key_states, value_states


class _IncreFlashAttentionCustom(torch.autograd.Function):
    """Custom IncreFlashAttention for ONNX export (decode step)."""

    @staticmethod
    # pylint: disable=unused-argument
    def forward(
        ctx,
        query,
        key,
        value,
        atten_mask,
        num_heads: int,
        scale_value: float,
        input_layout: str,
        num_key_value_heads: int,
        block_size: int,
        inner_precise: int,
    ):
        """Incremental flash attention forward (fallback to manual matmul)."""
        q = query
        k = key
        v = value
        layout = str(input_layout).upper()
        if layout in ("BSND", "SBND"):
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
        if 0 < num_key_value_heads < num_heads:
            rep = num_heads // num_key_value_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        attn = torch.matmul(q, k.transpose(2, 3)) * float(scale_value)
        if atten_mask is not None:
            m = atten_mask.to(torch.bool)
            if m.dim() == 4 and m.shape[1] == 1:
                m = m.expand(attn.shape[0], attn.shape[1], m.shape[2], m.shape[3])
            attn = attn.masked_fill(m, torch.finfo(attn.dtype).min)
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v)
        if layout in ("BSND", "SBND"):
            out = out.permute(0, 2, 1, 3)
        return out

    @staticmethod
    def symbolic(
        g,
        query,
        key,
        value,
        atten_mask,
        num_heads: int,
        scale_value: float,
        input_layout: str,
        num_key_value_heads: int,
        block_size: int,
        inner_precise: int,
    ):
        """ONNX symbolic for IncreFlashAttention custom op."""
        if atten_mask is None:
            y = g.op(
                "Custom",
                query, key, value,
                type_s="IncreFlashAttention",
                input_names_s=["query", "key", "value", "atten_mask"],
                optional_input_names_s=["atten_mask"],
                output_names_s=["attention_out"],
                output_num_i=1,
                input_index_i=[0, 1, 2],
                num_heads_i=int(num_heads),
                scale_value_f=float(scale_value),
                input_layout_s=str(input_layout),
                num_key_value_heads_i=int(num_key_value_heads),
                block_size_i=int(block_size),
                inner_precise_i=int(inner_precise),
            )
        else:
            y = g.op(
                "Custom",
                query, key, value, atten_mask,
                type_s="IncreFlashAttention",
                input_names_s=["query", "key", "value", "atten_mask"],
                optional_input_names_s=["atten_mask"],
                output_names_s=["attention_out"],
                output_num_i=1,
                input_index_i=[0, 1, 2, 3],
                num_heads_i=int(num_heads),
                scale_value_f=float(scale_value),
                input_layout_s=str(input_layout),
                num_key_value_heads_i=int(num_key_value_heads),
                block_size_i=int(block_size),
                inner_precise_i=int(inner_precise),
            )
        y.setType(query.type())
        return y


def incre_flash_attention(
    query,
    key,
    value,
    atten_mask,
    num_heads: int,
    scale_value: float,
    input_layout: str,
    num_key_value_heads: int,
    block_size: int = 0,
    inner_precise: int = 1,
):
    """Incremental flash attention using custom ONNX op."""
    return _IncreFlashAttentionCustom.apply(
        query, key, value, atten_mask,
        int(num_heads), float(scale_value), str(input_layout),
        int(num_key_value_heads), int(block_size), int(inner_precise),
    )


class _PromptFlashAttentionCustom(torch.autograd.Function):
    """Custom PromptFlashAttention for ONNX export (full bidir attention).

    Fuses: MatMul(Q, K^T) -> Scale -> Softmax -> MatMul(attn, V)
    into a single Custom node, matching CANN PromptFlashAttention op.

    Input layout: BNSD [B, N, S, D] for all of Q, K, V.
    No atten_mask needed for bidirectional vision attention.
    """

    @staticmethod
    def forward(ctx, query, key, value, num_heads, scale_value, input_layout):
        """Full attention forward (bidirectional, no causal mask)."""
        del ctx, num_heads
        q = query
        k = key
        v = value
        layout = str(input_layout).upper()
        # Support BNSD input, permute to BNSD if BSND
        if layout == "BSND":
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
        attn = torch.matmul(q, k.transpose(-2, -1)) * float(scale_value)
        attn = torch.softmax(attn.float(), dim=-1).to(q.dtype)
        out = torch.matmul(attn, v)
        if layout == "BSND":
            out = out.permute(0, 2, 1, 3)
        return out

    @staticmethod
    def symbolic(g, query, key, value, num_heads, scale_value, input_layout):
        """ONNX symbolic for PromptFlashAttention custom op."""
        y = g.op(
            "Custom", query, key, value,
            type_s="PromptFlashAttention",
            input_names_s=["query", "key", "value"],
            output_names_s=["attention_out"],
            output_num_i=1,
            input_index_i=[0, 1, 2],
            num_heads_i=int(num_heads),
            num_key_value_heads_i=int(num_heads),
            scale_value_f=float(scale_value),
            input_layout_s=str(input_layout),
            next_tokens_i=65536,
            inner_precise_i=1,
        )
        y.setType(query.type())
        return y


def prompt_flash_attention(query, key, value, num_heads, scale_value, input_layout="BNSD"):
    """Full bidirectional flash attention via CANN PromptFlashAttention.

    Q/K/V in BNSD layout [B, N, S, D].
    """
    return _PromptFlashAttentionCustom.apply(
        query, key, value,
        int(num_heads), float(scale_value), str(input_layout),
    )


class _CannRotaryMul(torch.autograd.Function):
    """CANN RotaryMul: fused RoPE computation.

    y = x * r1 + rotate_half(x) * r2
    rotate_half(x) = cat([-x[..., half:], x[..., :half]], dim=-1)
    """

    @staticmethod
    def forward(ctx, x, r1, r2):
        """Forward: reference RoPE implementation for tracing."""
        del ctx
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        rotated = torch.cat([-x2, x1], dim=-1)
        return (x * r1) + (rotated * r2)

    @staticmethod
    def symbolic(g, x, r1, r2):
        """ONNX symbolic for RotaryMul custom op."""
        return g.op(
            "Custom", x, r1, r2,
            input_names_s=["x", "r1", "r2"],
            output_names_s=["y"],
            type_s="RotaryMul",
        )


def rotary_mul(x, r1, r2):
    """Apply RoPE via CANN RotaryMul custom op."""
    return _CannRotaryMul.apply(x, r1, r2)



class _VisionFlashAttentionCustom(torch.autograd.Function):
    """Custom VisionFlashAttention for ONNX export (full bidir attention).

    Fuses: MatMul(Q, K^T) -> Scale -> Softmax -> MatMul(attn, V)
    into a single Custom node, matching CANN aclnnFlashAttention API.
    """

    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        num_heads: int,
        scale_value: float,
        input_layout: str,
    ):
        """Full attention forward (bidirectional, no causal mask)."""
        del ctx, num_heads
        q = query
        k = key
        v = value
        layout = str(input_layout).upper()
        # Always work in BNSD internally
        if layout == "BSND":
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
        attn = torch.matmul(q, k.transpose(-2, -1)) * float(scale_value)
        attn = torch.softmax(attn.float(), dim=-1).to(q.dtype)
        out = torch.matmul(attn, v)
        if layout == "BSND":
            out = out.permute(0, 2, 1, 3)
        return out

    @staticmethod
    def symbolic(
        g,
        query,
        key,
        value,
        num_heads: int,
        scale_value: float,
        input_layout: str,
    ):
        """ONNX symbolic for VisionFlashAttention custom op."""
        y = g.op(
            "Custom",
            query, key, value,
            type_s="VisionFlashAttention",
            input_names_s=["query", "key", "value"],
            output_names_s=["attention_out"],
            output_num_i=1,
            input_index_i=[0, 1, 2],
            num_heads_i=int(num_heads),
            scale_value_f=float(scale_value),
            input_layout_s=str(input_layout),
        )
        y.setType(query.type())
        return y


def vision_flash_attention(
    query,
    key,
    value,
    num_heads: int,
    scale_value: float,
    input_layout: str = "BNSD",
):
    """Full bidirectional flash attention using custom ONNX op.

    Fuses MatMul-Scale-Softmax-MatMul for vision transformer blocks.
    Q/K/V must be in BNSD layout [B, N, S, D].
    """
    return _VisionFlashAttentionCustom.apply(
        query, key, value,
        int(num_heads), float(scale_value), str(input_layout),
    )


class _VisionRoPEFlashAttentionCustom(torch.autograd.Function):
    """Custom op fusing: RoPE → BNSD transpose → MatMul(Q,K^T) → Scale → Softmax → MatMul(attn,V).

    Input layout (pre-RoPE):
        query:  [seq_len, num_heads, head_dim]
        key:    [seq_len, num_heads, head_dim]
        value:  [seq_len, num_heads, head_dim]
        cos:    [seq_len, 2*head_dim]
        sin:    [seq_len, 2*head_dim]
    Output layout:
        attention_out: [1, num_heads, seq_len, head_dim] (BNSD)
    """

    @staticmethod
    def forward(ctx, query, key, value, cos, sin, num_heads, scale_value):
        """Full attention with internal RoPE (bidirectional, no causal mask)."""
        del ctx, num_heads
        q_dtype, k_dtype, v_dtype = query.dtype, key.dtype, value.dtype

        # Step 1: Apply RoPE (cos/sin shape: [seq, head_dim] per empirical check)
        q = query.float()
        k = key.float()
        cos_2d = cos.unsqueeze(-2).float()   # [seq, 1, head_dim]
        sin_2d = sin.unsqueeze(-2).float()
        q_rope = (q * cos_2d) + (_rotate_half(q) * sin_2d)
        k_rope = (k * cos_2d) + (_rotate_half(k) * sin_2d)

        # Step 2: Transpose to BNSD [1, N, S, D]
        q_bnsd = q_rope.transpose(0, 1).unsqueeze(0).to(q_dtype)
        k_bnsd = k_rope.transpose(0, 1).unsqueeze(0).to(k_dtype)
        v_bnsd = value.transpose(0, 1).unsqueeze(0).to(v_dtype)

        # Step 3: Attention core (MatMul → Scale → Softmax → MatMul)
        scale = float(scale_value)
        attn = torch.matmul(q_bnsd, k_bnsd.transpose(-2, -1)) * scale
        attn = torch.softmax(attn.float(), dim=-1).to(q_bnsd.dtype)
        out = torch.matmul(attn, v_bnsd)
        return out

    @staticmethod
    def symbolic(g, query, key, value, cos, sin, num_heads, scale_value):
        """ONNX symbolic for VisionRoPEFlashAttention custom op."""
        y = g.op(
            "Custom",
            query, key, value, cos, sin,
            type_s="VisionRoPEFlashAttention",
            input_names_s=["query", "key", "value", "cos", "sin"],
            output_names_s=["attention_out"],
            output_num_i=1,
            input_index_i=[0, 1, 2, 3, 4],
            num_heads_i=int(num_heads),
            scale_value_f=float(scale_value),
        )
        y.setType(query.type())
        return y


def vision_rope_attention(query, key, value, cos, sin, num_heads, scale_value):
    """Full bidirectional attention with internal RoPE via custom ONNX op.

    Q/K/V in [seq, n_heads, head_dim] layout (pre-RoPE).
    Output in BNSD [1, n_heads, seq, head_dim].
    """
    return _VisionRoPEFlashAttentionCustom.apply(
        query, key, value, cos, sin,
        int(num_heads), float(scale_value),
    )


class CustomScatterUpdate(torch.autograd.Function):
    """Custom scatter update operator for KV cache update."""

    @staticmethod
    def forward(
        ctx: Any,
        var: torch.Tensor,
        indices: torch.Tensor,
        updates: torch.Tensor,
        axis: int = 0,
    ):
        """Update cache position(s) along the given axis.
        axis=2 → BNSD [B, n_kv, S, D];  axis=1 → BSND [B, S, n_kv, D].
        
        Uses scatter_() for exact numerical precision during tracing
        (symbolic generates Custom Scatter op for ONNX)."""
        del ctx
        ax = int(axis)
        result = var.clone()
        pos = indices.to(torch.int64).reshape(-1)
        if ax == 2:   # BNSD [B, n_kv, S, D]
            idx = pos.reshape(-1, 1, 1, 1).expand(
                updates.size(0), updates.size(1), 1, updates.size(3)
            )
            result.scatter_(ax, idx, updates.to(var.dtype))
        elif ax == 1:  # BSND [B, S, n_kv, D]
            idx = pos.reshape(-1, 1, 1, 1).expand(
                updates.size(0), 1, updates.size(2), updates.size(3)
            )
            result.scatter_(ax, idx, updates.to(var.dtype))
        else:
            raise ValueError(f"Unsupported axis={ax} for ScatterUpdate")
        return result

    @staticmethod
    def symbolic(g: torch.Graph, var, indices, updates, axis: int = 0):
        return g.op(
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


def _to_cache_indices(cache_pos: Any, device: Any) -> torch.Tensor:
    """Convert cache_pos to a 1D int64 tensor with batch-sized indices."""
    if torch.is_tensor(cache_pos):
        return cache_pos.reshape(-1).to(torch.int64)
    return torch.tensor([int(cache_pos)], dtype=torch.int64, device=device)


def _kv_cache_update(
    past: torch.Tensor,
    update: torch.Tensor,
    cache_pos: torch.Tensor,
    *,
    use_custom: bool,
) -> torch.Tensor:
    """Update KV cache at cache_pos and return the updated tensor."""
    if bool(use_custom):
        indices = _to_cache_indices(cache_pos, past.device)
        return CustomScatterUpdate.apply(past, indices, update, 2)

    pos = cache_pos.to(torch.int64).reshape(-1, 1, 1, 1)
    index = pos.expand(update.size(0), update.size(1), 1, update.size(3))
    result = past.clone()
    result.scatter_(2, index, update.to(past.dtype))
    return result


class _VisionMlpBlockCustom(torch.autograd.Function):
    """Custom op fusing: LayerNorm -> Linear(768->3072) -> GELUTanh -> Linear(3072->768) -> Add(residual).

    Takes hidden_states + all weight matrices as inputs, so the ONNX Custom node
    can be matched by CANN Norm+MLP fusion kernel.
    """

    @staticmethod
    def forward(ctx, hidden_states, norm_weight, norm_bias,
                fc1_weight, fc1_bias, fc2_weight, fc2_bias):
        """Forward: full MLP block with GELU(tanh) activation."""
        del ctx
        ndim = hidden_states.shape[-1]
        normed = torch.nn.functional.layer_norm(
            hidden_states, (ndim,), norm_weight, norm_bias, 1e-6
        )
        fc1 = torch.nn.functional.linear(normed, fc1_weight, fc1_bias)
        act = torch.nn.functional.gelu(fc1, approximate='tanh')
        fc2 = torch.nn.functional.linear(act, fc2_weight, fc2_bias)
        return hidden_states + fc2

    @staticmethod
    def symbolic(g, hidden_states, norm_weight, norm_bias,
                 fc1_weight, fc1_bias, fc2_weight, fc2_bias):
        """ONNX symbolic for VisionMlpBlock custom op."""
        y = g.op(
            "Custom", hidden_states, norm_weight, norm_bias,
            fc1_weight, fc1_bias, fc2_weight, fc2_bias,
            type_s="VisionMlpBlock",
            input_names_s=["hidden_states", "norm_weight", "norm_bias",
                           "fc1_weight", "fc1_bias", "fc2_weight", "fc2_bias"],
            output_names_s=["output"],
            output_num_i=1,
            input_index_i=[0, 1, 2, 3, 4, 5, 6],
        )
        y.setType(hidden_states.type())
        return y


def vision_mlp_block(hidden_states, norm_weight, norm_bias,
                     fc1_weight, fc1_bias, fc2_weight, fc2_bias):
    """Fused MLP block: LayerNorm -> Linear -> GELUTanh -> Linear -> ResidualAdd.

    Args:
        hidden_states: input tensor (also used as residual)
        norm_weight/bias: LayerNorm parameters
        fc1_weight/bias: first Linear (up-projection)
        fc2_weight/bias: second Linear (down-projection)
    Returns:
        hidden_states + fc2(gelu(fc1(norm(hidden_states))))
    """
    return _VisionMlpBlockCustom.apply(
        hidden_states, norm_weight, norm_bias,
        fc1_weight, fc1_bias, fc2_weight, fc2_bias,
    )


class VisionTowerWrapper(torch.nn.Module):
    """Wrapper for Qwen3.5 Vision Tower with runtime positional encodings.

    The upstream implementation converts ``grid_thw`` to Python lists and caches
    positional encodings, which makes the exported ONNX graph effectively static.
    For the current deployment path we only need a single square image input, so
    we rebuild the position / rotary embeddings from ``pixel_values.shape[0]``.
    """

    def __init__(self, vision_tower):
        super().__init__()
        self.vision_tower = vision_tower
        self.num_grid_per_side = int(vision_tower.num_grid_per_side)
        self.spatial_merge_size = int(vision_tower.spatial_merge_size)

    @staticmethod
    def _shape_as_tensor(x):
        from torch.onnx import operators as onnx_operators

        return onnx_operators.shape_as_tensor(x)

    def _seq_len_and_grid(self, pixel_values):
        seq_len = self._shape_as_tensor(pixel_values)[0].to(torch.int64)
        grid = torch.sqrt(seq_len.to(torch.float32))
        grid = torch.round(grid).to(torch.int64)
        return seq_len, grid

    def _interp_axis(self, grid, device):
        """Interpolate position embedding axis."""
        coord = torch.arange(grid, device=device, dtype=torch.float32)
        if self.num_grid_per_side <= 1:
            coord = torch.zeros_like(coord)
        else:
            denom = torch.clamp(grid.to(torch.float32) - 1.0, min=1.0)
            coord = coord * float(self.num_grid_per_side - 1) / denom
        floor = torch.floor(coord).to(torch.long)
        ceil = torch.clamp(floor + 1, max=self.num_grid_per_side - 1)
        frac = coord - floor.to(coord.dtype)
        return floor, ceil, frac

    def _pos_embeds(self, hidden_states, pixel_values):
        """Compute position embeddings for vision tower."""
        seq_len, grid = self._seq_len_and_grid(pixel_values)
        floor, ceil, frac = self._interp_axis(grid, hidden_states.device)
        pos_weight = self.vision_tower.pos_embed.weight.to(hidden_states.dtype)
        num_side = self.num_grid_per_side

        row_floor = floor[:, None]
        row_ceil = ceil[:, None]
        col_floor = floor[None, :]
        col_ceil = ceil[None, :]

        idx00 = (row_floor * num_side + col_floor).reshape(-1)
        idx01 = (row_floor * num_side + col_ceil).reshape(-1)
        idx10 = (row_ceil * num_side + col_floor).reshape(-1)
        idx11 = (row_ceil * num_side + col_ceil).reshape(-1)

        frac_row = frac[:, None]
        frac_col = frac[None, :]
        w00 = ((1.0 - frac_row) * (1.0 - frac_col)).reshape(-1, 1).to(hidden_states.dtype)
        w01 = ((1.0 - frac_row) * frac_col).reshape(-1, 1).to(hidden_states.dtype)
        w10 = (frac_row * (1.0 - frac_col)).reshape(-1, 1).to(hidden_states.dtype)
        w11 = (frac_row * frac_col).reshape(-1, 1).to(hidden_states.dtype)

        pos_embeds = (
            pos_weight.index_select(0, idx00) * w00
            + pos_weight.index_select(0, idx01) * w01
            + pos_weight.index_select(0, idx10) * w10
            + pos_weight.index_select(0, idx11) * w11
        )
        pos_embeds = pos_embeds.reshape(floor.shape[0], floor.shape[0], hidden_states.shape[1])

        merge = self.spatial_merge_size
        merged_grid = floor.shape[0] // merge
        pos_embeds = pos_embeds.reshape(
            1, merged_grid, merge, merged_grid, merge, hidden_states.shape[1]
        )
        pos_embeds = pos_embeds.permute(0, 1, 3, 2, 4, 5).reshape(seq_len, hidden_states.shape[1])
        return pos_embeds

    def _rotary_embeddings(self, hidden_states, pixel_values):
        """Compute rotary embeddings for vision tower."""
        _, grid = self._seq_len_and_grid(pixel_values)
        merge = self.spatial_merge_size
        merged_grid = grid // merge
        device = hidden_states.device
        rotary_dtype = self.vision_tower.rotary_pos_emb.inv_freq.dtype

        block_rows = torch.arange(merged_grid, device=device, dtype=torch.long)
        block_cols = torch.arange(merged_grid, device=device, dtype=torch.long)
        intra_row = torch.arange(merge, device=device, dtype=torch.long)
        intra_col = torch.arange(merge, device=device, dtype=torch.long)

        row_idx = block_rows[:, None, None, None] * merge + intra_row[None, None, :, None]
        col_idx = block_cols[None, :, None, None] * merge + intra_col[None, None, None, :]
        row_idx = row_idx.expand(merged_grid, merged_grid, merge, merge).reshape(-1)
        col_idx = col_idx.expand(merged_grid, merged_grid, merge, merge).reshape(-1)

        seq = torch.arange(grid, device=device, dtype=rotary_dtype)
        inv_freq = self.vision_tower.rotary_pos_emb.inv_freq.to(device=device, dtype=rotary_dtype)
        freq_table = torch.outer(seq, inv_freq)
        pos_ids = torch.stack((row_idx, col_idx), dim=-1).reshape(-1)
        rotary = freq_table.index_select(0, pos_ids).reshape(hidden_states.shape[0], 2, -1)
        rotary = rotary.reshape(hidden_states.shape[0], -1).to(hidden_states.dtype)
        rotary = torch.cat((rotary, rotary), dim=-1)
        return rotary.cos(), rotary.sin()

    def _vision_attention(self, attn, hidden_states, position_embeddings):
        """Run vision attention layer using CANN custom ops."""
        seq_length = hidden_states.shape[0]
        qkv = attn.qkv(hidden_states).reshape(seq_length, 3, attn.num_heads, attn.head_dim)
        query_states, key_states, value_states = qkv.permute(1, 0, 2, 3).unbind(0)

        # Transpose to BNSD [1, n_heads, seq, head_dim]
        q_bnsd = query_states.transpose(0, 1).unsqueeze(0)
        k_bnsd = key_states.transpose(0, 1).unsqueeze(0)
        v_bnsd = value_states.transpose(0, 1).unsqueeze(0)

        # RotaryMul: y = x * cos + rotate_half(x) * sin (4D BNSD)
        cos, sin = position_embeddings
        cos_4d = cos.unsqueeze(0).unsqueeze(1)  # [1, 1, seq, head_dim]
        sin_4d = sin.unsqueeze(0).unsqueeze(1)
        q_bnsd = rotary_mul(q_bnsd, cos_4d, sin_4d)
        k_bnsd = rotary_mul(k_bnsd, cos_4d, sin_4d)

        # Use CANN built-in PromptFlashAttention for full bidir attention
        attn_output = prompt_flash_attention(
            q_bnsd, k_bnsd, v_bnsd,
            num_heads=attn.num_heads,
            scale_value=float(attn.scaling),
            input_layout="BNSD",
        )
        attn_output = attn_output[0].transpose(0, 1).reshape(seq_length, -1).contiguous()
        return attn.proj(attn_output)

    def forward(self, pixel_values):
        """Encode pixel values to image embeddings for LLM prefill."""
        # Replace Conv3d(3→768, kernel=(2,16,16)) with equivalent Linear(1536→768)
        # Conv3d kernel covers full T×H×W patch extent, so it's mathematically equivalent
        pe = self.vision_tower.patch_embed
        weight = pe.proj.weight.reshape(pe.embed_dim, -1)
        hidden_states = torch.nn.functional.linear(pixel_values, weight, pe.proj.bias)
        hidden_states = hidden_states + self._pos_embeds(hidden_states, pixel_values)
        position_embeddings = self._rotary_embeddings(hidden_states, pixel_values)

        for blk in self.vision_tower.blocks:
            attn_input = blk.norm1(hidden_states)
            hidden_states = hidden_states + self._vision_attention(
                blk.attn, attn_input, position_embeddings
            )
            hidden_states = hidden_states + blk.mlp(blk.norm2(hidden_states))

        image_embeds = self.vision_tower.merger(hidden_states)
        if isinstance(image_embeds, (list, tuple)):
            image_embeds = torch.cat(image_embeds, dim=0)
        return image_embeds

class Qwen35LlmPrefill(torch.nn.Module):
    """Qwen3.5-0.8B LLM Prefill model for ONNX export."""

    def __init__(self, text_model, lm_head, image_token_id,
                 max_seq_len: int = 2048):
        super().__init__()
        self.text_model = text_model
        self.lm_head = lm_head
        self.image_token_id = int(image_token_id)
        self.config = text_model.config
        self.max_seq_len = int(max_seq_len)

    def forward(self, input_ids, attention_mask, position_ids, image_embeds):
        """Run prefill forward pass with image embeddings.

        Uses cumsum+gather+where instead of masked_scatter to avoid
        closed loops in the ONNX graph during dynamic shape export.

        Uses CANN PromptFlashAttention for full attention layers and the
        CGDR custom op for linear attention.
        """
        inputs_embeds = self.text_model.embed_tokens(input_ids)
        image_mask = input_ids == self.image_token_id  # (B, S)
        # Scatter image_embeds into the correct positions using traceable ops:
        # cumsum → gather → where (no masked_scatter which creates graph loops)
        img_idx = image_mask.long().cumsum(dim=1).clamp(min=1) - 1  # (B, S), indices into image_embeds
        image_embeds_4d = image_embeds[img_idx]  # (B, S, D) — gather from (N, D)
        inputs_embeds = torch.where(
            image_mask.unsqueeze(-1), image_embeds_4d, inputs_embeds
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

        # CANN PFA expects boolean mask (True = masked positions)
        attn_mask = _make_additive_causal_mask(attention_mask, q_len, k_len, 0)

        linear_attn_mask = attention_mask
        seq_lens = linear_attn_mask.to(dtype=torch.int32).sum(dim=1)
        actual_seq_lengths = torch.zeros((inputs_embeds.shape[0] + 1,), device=inputs_embeds.device, dtype=torch.int32)
        actual_seq_lengths[1:] = torch.cumsum(seq_lens, dim=0)
        ssm_state_indices = torch.arange(inputs_embeds.shape[0], device=inputs_embeds.device, dtype=torch.int32)

        hidden_states = inputs_embeds
        present_conv = []
        present_recurrent = []
        present_kv = []

        for layer in self.text_model.layers:
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            if layer.layer_type == "linear_attention":
                attn_out, conv_s, rec_s = _linear_attn_prefill(
                    layer.linear_attn,
                    hidden_states,
                    linear_attn_mask,
                    actual_seq_lengths=actual_seq_lengths,
                    ssm_state_indices=ssm_state_indices,
                )
                hidden_states = residual + attn_out
                present_conv.append(conv_s)
                present_recurrent.append(rec_s)
            else:
                attn_out, pk, pv = _full_attn_forward(
                    layer.self_attn, hidden_states, position_embeddings, attn_mask,
                    None, None,
                )
                hidden_states = residual + attn_out
                present_kv.append(pk)
                present_kv.append(pv)

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = residual + layer.mlp(hidden_states)

        hidden_states = self.text_model.norm(hidden_states)
        # Only compute lm_head for the last position to reduce matmul
        last_hidden = hidden_states[:, -1:, :]
        logits = self.lm_head(last_hidden)
        next_token_id = logits.argmax(dim=-1, keepdim=True).to(torch.int32)

        present_conv_stack = torch.stack(present_conv, dim=0) if present_conv else torch.zeros(0)
        present_recurrent_stack = torch.stack(present_recurrent, dim=0) if present_recurrent else torch.zeros(0)
        present_kv_stack = torch.stack(present_kv, dim=0) if present_kv else torch.zeros(0)
        # Pad present_kv_cache seq_len dimension to fixed max_seq_len
        if present_kv_stack.dim() >= 4 and present_kv_stack.shape[3] < self.max_seq_len:
            pad_size = self.max_seq_len - present_kv_stack.shape[3]
            present_kv_stack = F.pad(present_kv_stack, (0, 0, 0, pad_size))

        return next_token_id, present_conv_stack, present_recurrent_stack, present_kv_stack


class Qwen35LlmDecode(torch.nn.Module):
    """Qwen3.5-0.8B LLM Decode model for ONNX export."""

    def __init__(self, text_model, lm_head, use_rgdr_custom: bool = True):
        super().__init__()
        self.text_model = text_model
        self.lm_head = lm_head
        self.config = text_model.config
        self.use_rgdr_custom = bool(use_rgdr_custom)

    def forward(self, input_ids, attention_mask, position_ids,
                past_conv_states, past_recurrent_states, past_kv_cache):
        """Run decode forward pass with fixed-shape KV cache (scatter-based update).

        past_kv_cache: (12, 1, 2, MAX_SEQ_LEN, 256) — pre-allocated full-sized cache
        attention_mask: (1, MAX_SEQ_LEN) — 1 for valid tokens, 0 for padding
        The actual past length is derived from attention_mask sum.
        New key/value are scattered into the cache at position ``past_len``.
        """
        inputs_embeds = self.text_model.embed_tokens(input_ids)
        q_len = input_ids.shape[1]
        max_seq_len = past_kv_cache.shape[3]

        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            mm_position_ids = position_ids[1:]
        else:
            mm_position_ids = position_ids

        position_embeddings = self.text_model.rotary_emb(inputs_embeds, mm_position_ids)

        # ---- Fixed-shape: extract valid lengths ----
        # past_len = number of valid tokens in cache (= total_valid_1s - q_len)
        total_valid_1s = attention_mask.sum(dim=1, keepdim=True)  # (B, 1)
        past_len = (total_valid_1s - q_len).reshape(())  # scalar
        cache_pos = past_len  # position to write new KV into the cache

        # Build IFA boolean mask (covers full max_seq_len, True = masked position)
        kv_range = torch.arange(max_seq_len, device=inputs_embeds.device, dtype=torch.int64)
        allow = kv_range.view(1, max_seq_len) <= cache_pos.view(-1, 1)
        allow = allow & attention_mask.to(torch.bool)
        ifa_attn_mask = (~allow).view(-1, 1, 1, max_seq_len)

        hidden_states = inputs_embeds
        present_conv = []
        present_recurrent = []
        present_kv = []
        linear_idx = 0
        kv_idx = 0

        for _, layer in enumerate(self.text_model.layers):
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            if layer.layer_type == "linear_attention":
                conv_s_in = past_conv_states[linear_idx]
                rec_s_in = past_recurrent_states[linear_idx]
                attn_out, conv_s_out, rec_s_out = _linear_attn_decode(
                    layer.linear_attn, hidden_states, conv_s_in, rec_s_in,
                    use_rgdr_custom=self.use_rgdr_custom,
                )
                hidden_states = residual + attn_out
                present_conv.append(conv_s_out)
                present_recurrent.append(rec_s_out)
                linear_idx += 1
            else:
                # Fetch per-layer KV cache from the full-sized input
                pk_in = past_kv_cache[kv_idx]       # (1, 2, 2048, 256)
                pv_in = past_kv_cache[kv_idx + 1]   # (1, 2, 2048, 256)

                # ---- IFA fused path ----
                input_shape = hidden_states.shape[:-1]
                head_dim = layer.self_attn.head_dim
                hidden_shape = (*input_shape, -1, head_dim)
                num_heads = int(layer.self_attn.config.num_attention_heads)
                num_kv_heads = int(layer.self_attn.config.num_key_value_heads)
                scaling = getattr(layer.self_attn, "scaling", head_dim ** -0.5)

                qkv = layer.self_attn.q_proj(hidden_states).view(
                    *input_shape, -1, head_dim * 2)
                query_states, gate = torch.chunk(qkv, 2, dim=-1)
                gate = gate.reshape(*input_shape, -1)

                query_states = layer.self_attn.q_norm(
                    query_states.view(hidden_shape)).transpose(1, 2)
                key_states = layer.self_attn.k_norm(
                    layer.self_attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                value_states = layer.self_attn.v_proj(
                    hidden_states).view(hidden_shape).transpose(1, 2)

                cos, sin = position_embeddings
                from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin)

                k_full = _kv_cache_update(pk_in, key_states, cache_pos, use_custom=True)
                v_full = _kv_cache_update(pv_in, value_states, cache_pos, use_custom=True)

                q_bnsd = query_states.contiguous()
                k_bnsd = k_full.contiguous()
                v_bnsd = v_full.contiguous()

                attn_out = incre_flash_attention(
                    q_bnsd, k_bnsd, v_bnsd, ifa_attn_mask,
                    num_heads=num_heads, scale_value=float(scaling),
                    input_layout="BNSD", num_key_value_heads=num_kv_heads,
                    inner_precise=1,
                )
                if attn_out.dim() == 4:
                    attn_out = attn_out.transpose(1, 2).reshape(*input_shape, -1)

                attn_out = attn_out * torch.sigmoid(gate)
                attn_out = layer.self_attn.o_proj(attn_out)
                hidden_states = residual + attn_out

                present_kv.append(k_full)  # (1, 2, 2048, 256)
                present_kv.append(v_full)

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
    """Get metadata for Qwen3.5-0.8B export."""
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

    nk = linear_num_value_heads
    nv = linear_num_value_heads
    dk = linear_key_head_dim
    dv = linear_value_head_dim

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
        "hidden_size": config.hidden_size,
        "nk": nk,
        "nv": nv,
        "dk": dk,
        "dv": dv,
    }


def export_vision_tower(model, output_dir, device="cpu", vision_image_size=1024):
    """Export Qwen3.5 Vision Tower to ONNX with dynamic patch count."""
    output_path = Path(output_dir) / "qwen3_5_vision.onnx"
    print(f"Exporting Vision Tower to {output_path}...")

    vision_tower = model.model.visual
    vision_tower.eval()
    vision_tower.to(device)

    patch_size = model.config.vision_config.patch_size
    grid_h = int(vision_image_size) // int(patch_size)
    grid_w = int(vision_image_size) // int(patch_size)
    dummy_seq_len = int(grid_h * grid_w)
    in_channels = model.config.vision_config.in_channels
    temporal_patch_size = model.config.vision_config.temporal_patch_size
    patch_dim = in_channels * temporal_patch_size * patch_size * patch_size
    dummy_pixel_values = torch.randn(dummy_seq_len, patch_dim, device=device, dtype=torch.float32)

    wrapper = VisionTowerWrapper(vision_tower)
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
            dynamic_axes={"pixel_values": {0: "num_image_tokens"}},
        )
    print("Vision Tower exported successfully.")


def _export_llm_prefill(prefill, output_dir, device, dummy_seq, dummy_num_img_tokens):
    """Export LLM prefill model to ONNX."""
    output_name = "qwen3_5_llm_prefill.onnx"
    prefill_path = Path(output_dir) / "prefill" / output_name
    prefill_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input_ids = torch.randint(0, 1000, (1, dummy_seq), dtype=torch.int64, device=device)
    dummy_attention_mask = torch.ones(1, dummy_seq, dtype=torch.int64, device=device)
    base_pos = torch.arange(dummy_seq, device=device, dtype=torch.int64).view(1, -1)
    dummy_position_ids = base_pos.unsqueeze(0).expand(4, 1, dummy_seq)
    dummy_image_embeds = torch.randn(
        dummy_num_img_tokens, prefill.config.hidden_size,
        device=device, dtype=torch.float32,
    )
    input_names = ["input_ids", "attention_mask", "position_ids", "image_embeds"]
    output_names = ["next_token_id", "present_conv_states", "present_recurrent_states",
                    "present_kv_cache"]
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq_len"},
        "attention_mask": {0: "batch", 1: "seq_len"},
        "position_ids": {1: "batch", 2: "seq_len"},
        "image_embeds": {0: "num_image_tokens"},
        # next_token_id: (batch, 1) — fixed shape
        "present_conv_states": {1: "batch"},
        "present_recurrent_states": {1: "batch"},
        # present_kv_cache: seq_len fixed to max_seq_len (no dynamic axis)
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


def _export_llm_decode(decode, meta, output_dir, device, dummy_seq, max_seq_len=2048):
    """Export LLM decode model to ONNX with FIXED shape (padded to max_seq_len).

    All seq-related dimensions (attention_mask, past_kv_cache, present_kv_cache)
    are fixed to ``max_seq_len`` so converter_lite produces a static-shape MindIR.
    """
    decode_path = Path(output_dir) / "decoder" / "qwen3_5_llm_decode.onnx"
    decode_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_step = 1
    dummy_past_len = dummy_seq
    dummy_input_ids_step = torch.randint(
        0, 1000, (1, dummy_step), dtype=torch.int64, device=device
    )
    # attention_mask: fixed shape (1, max_seq_len), first (dummy_past_len + step) entries = 1
    dummy_attention_mask_step = torch.zeros(1, max_seq_len, dtype=torch.int64, device=device)
    dummy_attention_mask_step[:, :dummy_past_len + dummy_step] = 1
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
        dtype=torch.float32, device=device,
    )
    dummy_past_recurrent = torch.zeros(
        num_linear, 1, num_v_heads, k_head_dim, v_head_dim,
        dtype=torch.float32, device=device,
    )
    # past_kv_cache: fixed shape (2*num_full, 1, num_kv_heads, max_seq_len, head_dim)
    dummy_past_kv = torch.zeros(
        2 * num_full, 1, num_kv_heads, max_seq_len, head_dim,
        dtype=torch.float32, device=device,
    )

    input_names = ["input_ids", "attention_mask", "position_ids",
                   "past_conv_states", "past_recurrent_states", "past_kv_cache"]
    output_names = ["logits", "present_conv_states", "present_recurrent_states",
                    "present_kv_cache"]
    from torch.onnx import utils as onnx_utils
    print(f"Exporting LLM decode to {decode_path} (max_seq_len={max_seq_len})...")
    with torch.no_grad():
        onnx_utils.export(
            decode,
            (dummy_input_ids_step, dummy_attention_mask_step, dummy_position_ids_step,
             dummy_past_conv, dummy_past_recurrent, dummy_past_kv),
            str(decode_path),
            input_names=input_names, output_names=output_names,
            opset_version=14, do_constant_folding=True,
        )
    print("LLM decode exported successfully.")


def export_llm_prefill_decode(model, output_dir, device="cpu", dummy_seq=8,
                             dummy_num_img_tokens=16,
                             use_rgdr_custom: bool = True,
                             max_seq_len: int = 2048):
    """Export Qwen3.5-0.8B LLM prefill and decode models to ONNX."""
    meta = _get_model_meta(model)
    text_model = meta["text_model"]
    lm_head = meta["lm_head"]
    image_token_id = meta["image_token_id"]

    _set_cgdr_config(enabled=True, nk=meta.get('nk', meta['linear_num_value_heads']),
                     nv=meta.get('nv', meta['linear_num_value_heads']),
                     dk=meta.get('dk', meta['linear_key_head_dim']),
                     dv=meta.get('dv', meta['linear_value_head_dim']))
    print(f"[CGDR] CGDR enabled: NK={meta['linear_num_value_heads']}, "
          f"NV={meta['linear_num_value_heads']}, DK={meta['linear_key_head_dim']}, "
          f"DV={meta['linear_value_head_dim']}, "
          f"scale={1.0/(meta['linear_key_head_dim']**0.5):.6f}")

    text_model.eval()
    lm_head.eval()
    text_model.to(device)
    lm_head.to(device)

    prefill = Qwen35LlmPrefill(
        text_model, lm_head, image_token_id,
        max_seq_len=max_seq_len,
    ).to(device).eval()
    decode = Qwen35LlmDecode(
        text_model, lm_head, use_rgdr_custom=use_rgdr_custom,
    ).to(device).eval()

    _export_llm_prefill(prefill, output_dir, device, dummy_seq, dummy_num_img_tokens)
    _export_llm_decode(decode, meta, output_dir, device, dummy_seq)


def main():
    """Main function to export Qwen3.5-0.8B to ONNX."""
    parser = argparse.ArgumentParser(description="Export Qwen3.5-0.8B to ONNX")
    parser.add_argument(
        "--model-id", type=str,
        default="./Qwen3.5-0.8B",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./qwen3_5_onnx_static",
        help="Output directory",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device for export (cpu or cuda)",
    )
    parser.add_argument(
        "--vision-image-size", type=int, default=1024,
        help="Image size for vision tower export",
    )
    parser.add_argument(
        "--dummy-seq-len", type=int, default=8,
        help="Dummy sequence length for LLM export",
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=2048,
        help="Maximum sequence length for fixed-shape KV cache padding",
    )
    parser.add_argument(
        "--use-rgdr-custom",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable RecurrentGatedDeltaRule custom op in decode linear attention",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading model {args.model_id} in FP32 for export...")
    model = QwenForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.to(args.device)

    export_vision_tower(model, output_dir, args.device, args.vision_image_size)
    export_llm_prefill_decode(
        model,
        output_dir,
        args.device,
        args.dummy_seq_len,
        use_rgdr_custom=args.use_rgdr_custom,
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
