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
Export Qwen2.5-7B model to ONNX format as split prefill + decode subgraphs.

The export produces two ONNX models:
- Prefill: processes the full input prompt, outputs logits and KV cache.
- Decode: generates one token at a time using past KV cache.
"""

import argparse
import gc
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

KV_CACHE_LEN = 512

try:
    import torch._dynamo

    torch._dynamo.disable()
except Exception:
    pass

try:
    from transformers import AutoModelForCausalLM
except ImportError:
    print("Error: transformers package not found or version too low.")
    print("Please install the latest version: pip install transformers")
    sys.exit(1)

try:
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
except Exception:
    try:
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    except Exception:
        apply_rotary_pos_emb = None


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _as_list_str(items):
    """Convert a list of items to a list of strings for ONNX custom op attributes."""
    return [str(x) for x in items]


def _rotate_half(x):
    """Rotate the second half of the last dimension to the front, negated.

    Used by Rotary Position Embedding (RoPE).
    """
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _make_flash_attn_mask(attention_mask, q_len, k_len, past_len):
    """Build a boolean causal + padding mask for prefill attention.

    Returns shape (batch, 1, q_len, k_len).
    """
    ar_q = torch.arange(q_len, device=attention_mask.device)
    ar_k = torch.arange(k_len, device=attention_mask.device)
    causal = ar_k[None, :] > (past_len + ar_q[:, None])
    causal = causal[None, None, :, :].expand(attention_mask.shape[0], 1, q_len, k_len)
    padding = attention_mask[:, None, None, :].to(torch.bool).logical_not()
    return (causal | padding).to(torch.bool)


def _make_additive_causal_mask(attention_mask, q_len, k_len, past_len, dtype):
    """Build an additive causal + padding mask with -inf for masked positions."""
    mask_value = torch.finfo(dtype).min
    ar_q = torch.arange(q_len, device=attention_mask.device)
    ar_k = torch.arange(k_len, device=attention_mask.device)
    causal = ar_k[None, :] > (past_len + ar_q[:, None])
    causal = causal.to(dtype) * mask_value
    causal = causal[None, None, :, :].expand(attention_mask.shape[0], 1, q_len, k_len)
    padding = (1.0 - attention_mask.to(dtype)) * mask_value
    padding = padding[:, None, None, :]
    return causal + padding


# ---------------------------------------------------------------------------
# Custom ONNX ops – functional wrappers
# ---------------------------------------------------------------------------


class _RotaryMulCustom(torch.autograd.Function):
    """Custom RotaryMul op for ONNX export — applies rotary position embedding multiplication."""

    @staticmethod
    def forward(ctx, x, cos4, sin4):
        """Eager fallback: (x * cos) + (rotate_half(x) * sin)."""
        del ctx
        return (x * cos4) + (_rotate_half(x) * sin4)

    @staticmethod
    def symbolic(g, x, cos4, sin4):
        """ONNX symbolic for RotaryMul."""
        y = g.op(
            "Custom", x, cos4, sin4,
            type_s="RotaryMul",
            input_names_s=_as_list_str(["x", "r1", "r2"]),
            optional_input_names_s=_as_list_str([]),
            output_names_s=_as_list_str(["y"]),
            output_num_i=1,
            input_index_i=[0, 1, 2],
        )
        y.setType(x.type())
        return y


def rotary_mul(x, cos4, sin4):
    """Apply rotary position embedding multiplication via custom op."""
    return _RotaryMulCustom.apply(x, cos4, sin4)


class _ApplyRotaryPosEmbCustom(torch.autograd.Function):
    """Custom ApplyRotaryPosEmb op for ONNX export."""

    @staticmethod
    def forward(ctx, query, key, cos, sin, layout: int, rotary_mode: str):
        """Eager fallback: apply rotary position embedding to query and key."""
        del ctx, rotary_mode
        if apply_rotary_pos_emb is not None:
            if int(layout) == 1:
                q_bnsd = query.permute(0, 2, 1, 3)
                k_bnsd = key.permute(0, 2, 1, 3)
                q2, k2 = apply_rotary_pos_emb(q_bnsd, k_bnsd, cos, sin)
                return q2.permute(0, 2, 1, 3), k2.permute(0, 2, 1, 3)
            return apply_rotary_pos_emb(query, key, cos, sin)

        axis = 2 if int(layout) == 1 else 1
        cos4 = cos.unsqueeze(axis) if cos.dim() == 3 else cos
        sin4 = sin.unsqueeze(axis) if sin.dim() == 3 else sin
        q = (query * cos4) + (_rotate_half(query) * sin4)
        k = (key * cos4) + (_rotate_half(key) * sin4)
        return q, k

    @staticmethod
    def symbolic(g, query, key, cos, sin, layout: int, rotary_mode: str):
        """ONNX symbolic for rotary position embedding."""
        axis = 2 if int(layout) == 1 else 1
        cos4 = g.op("Unsqueeze", cos, axes_i=[axis])
        sin4 = g.op("Unsqueeze", sin, axes_i=[axis])
        q, k = g.op(
            "Custom", query, key, cos4, sin4,
            type_s="ApplyRotaryPosEmb",
            input_names_s=_as_list_str(["query", "key", "cos", "sin"]),
            optional_input_names_s=_as_list_str([]),
            output_names_s=_as_list_str(["query", "key"]),
            output_num_i=2,
            input_index_i=[0, 1, 2, 3],
            layout_i=int(layout),
            rotary_mode_s=str(rotary_mode),
            outputs=2,
        )
        q.setType(query.type())
        k.setType(key.type())
        return q, k


def apply_rotary_pos_emb_custom(query, key, cos, sin, layout: int = 3, rotary_mode: str = "half"):
    """Apply rotary position embedding via custom op."""
    return _ApplyRotaryPosEmbCustom.apply(query, key, cos, sin, int(layout), str(rotary_mode))


class _RmsNormCustom(torch.autograd.Function):
    """Custom RMSNorm op for ONNX export."""

    @staticmethod
    def forward(ctx, x, gamma, epsilon: float):
        """Eager fallback: RMS normalization."""
        del ctx
        x_fp32 = x.to(torch.float32)
        var = (x_fp32 * x_fp32).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + float(epsilon))
        y = (x_fp32 * rstd).to(x.dtype) * gamma
        return y, rstd

    @staticmethod
    def symbolic(g, x, gamma, epsilon: float):
        """ONNX symbolic for RMSNorm."""
        y, rstd = g.op(
            "Custom", x, gamma,
            type_s="RmsNorm",
            input_names_s=_as_list_str(["x", "gamma"]),
            optional_input_names_s=_as_list_str([]),
            output_names_s=_as_list_str(["y", "rstd"]),
            output_num_i=2,
            input_index_i=[0, 1],
            epsilon_f=float(epsilon),
            outputs=2,
        )
        y.setType(x.type())
        return y, rstd


def rms_norm(x, gamma, epsilon: float = 1e-6):
    """Apply RMS normalization via custom op."""
    return _RmsNormCustom.apply(x, gamma, float(epsilon))


def _rms_norm_layer(norm_mod, x):
    """Apply RMS normalization using a HuggingFace RMSNorm module's weight."""
    gamma = norm_mod.weight
    eps = getattr(norm_mod, "variance_epsilon", 1e-6)
    y, _ = rms_norm(x, gamma, eps)
    return y


# ---------------------------------------------------------------------------
# Flash-attention custom ops
# ---------------------------------------------------------------------------


def _eager_attn_forward(query, key, value, num_heads, num_kv_heads, scale, layout):
    """Run eager (non-optimized) multi-head attention with GQA support.

    Handles layout conversion (BSND↔BNSD) and key/value head repetition.
    Returns attention output in the original layout.
    """
    q, k, v = query, key, value
    if layout in ("BSND", "SBND"):
        q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
    if 0 < num_kv_heads < num_heads:
        rep = num_heads // num_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    attn = torch.matmul(q, k.transpose(2, 3)) * float(scale)
    return q, k, v, attn


def _eager_attn_finalize(q, attn, v, layout):
    """Compute softmax, weighted sum, and restore original layout."""
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(attn, v)
    if layout in ("BSND", "SBND"):
        out = out.permute(0, 2, 1, 3)
    return out


class _IncreFlashAttentionCustom(torch.autograd.Function):
    """Custom IncreFlashAttention op for ONNX export — single-token decode attention."""

    @staticmethod
    def forward(ctx, query, key, value, atten_mask, num_heads, scale_value,
                input_layout, num_key_value_heads, block_size, inner_precise):
        """Eager fallback for incremental flash attention."""
        del ctx, block_size, inner_precise
        layout = str(input_layout).upper()
        q, _k, v, attn = _eager_attn_forward(
            query, key, value, num_heads, num_key_value_heads, scale_value, layout)
        del _k
        if atten_mask is not None:
            m = atten_mask.to(torch.bool)
            if m.dim() == 4 and m.shape[1] == 1:
                m = m.expand(attn.shape[0], attn.shape[1], m.shape[2], m.shape[3])
            attn = attn.masked_fill(m, torch.finfo(attn.dtype).min)
        return _eager_attn_finalize(q, attn, v, layout)

    @staticmethod
    def symbolic(g, query, key, value, atten_mask, num_heads, scale_value,
                 input_layout, num_key_value_heads, block_size, inner_precise):
        """ONNX symbolic for incremental flash attention."""
        base_inputs = [query, key, value]
        base_index = [0, 1, 2]
        if atten_mask is not None:
            base_inputs.append(atten_mask)
            base_index.append(3)
        y = g.op(
            "Custom", *base_inputs,
            type_s="IncreFlashAttention",
            input_names_s=_as_list_str(["query", "key", "value", "atten_mask"]),
            optional_input_names_s=_as_list_str(["atten_mask"]),
            output_names_s=_as_list_str(["attention_out"]),
            output_num_i=1,
            input_index_i=base_index,
            num_heads_i=int(num_heads),
            scale_value_f=float(scale_value),
            input_layout_s=str(input_layout),
            num_key_value_heads_i=int(num_key_value_heads),
            block_size_i=int(block_size),
            inner_precise_i=int(inner_precise),
        )
        y.setType(query.type())
        return y


def incre_flash_attention(query, key, value, atten_mask, num_heads, scale_value,
                          input_layout, num_key_value_heads, block_size=0, inner_precise=1):
    """Functional wrapper for incremental flash attention."""
    return _IncreFlashAttentionCustom.apply(
        query, key, value, atten_mask,
        int(num_heads), float(scale_value), str(input_layout),
        int(num_key_value_heads), int(block_size), int(inner_precise))


class _PromptFlashAttentionCustom(torch.autograd.Function):
    """Custom PromptFlashAttention op for ONNX export — full-sequence prefill attention."""

    @staticmethod
    def forward(ctx, query, key, value, atten_mask, num_heads, scale_value,
                input_layout, num_key_value_heads, sparse_mode, inner_precise,
                pre_tokens, next_tokens):
        """Eager fallback for prompt flash attention."""
        del ctx, inner_precise, pre_tokens, next_tokens
        layout = str(input_layout).upper()
        q, _k, v, attn = _eager_attn_forward(
            query, key, value, num_heads, num_key_value_heads, scale_value, layout)
        del _k
        if atten_mask is not None:
            m = atten_mask.to(torch.bool)
            if m.dim() == 4 and m.shape[1] == 1:
                m = m.expand(attn.shape[0], attn.shape[1], m.shape[2], m.shape[3])
            attn = attn.masked_fill(m, torch.finfo(attn.dtype).min)
        elif int(sparse_mode) in (2, 3):
            attn = _apply_causal_mask(attn)
        return _eager_attn_finalize(q, attn, v, layout)

    @staticmethod
    def symbolic(g, query, key, value, atten_mask, num_heads, scale_value,
                 input_layout, num_key_value_heads, sparse_mode, inner_precise,
                 pre_tokens, next_tokens):
        """ONNX symbolic for prompt flash attention."""
        base_inputs = [query, key, value]
        base_index = [0, 1, 2]
        if atten_mask is not None:
            base_inputs.append(atten_mask)
            base_index.append(3)
        y = g.op(
            "Custom", *base_inputs,
            type_s="PromptFlashAttention",
            input_names_s=_as_list_str(["query", "key", "value", "atten_mask"]),
            optional_input_names_s=_as_list_str(["atten_mask"]),
            output_names_s=_as_list_str(["attention_out"]),
            output_num_i=1,
            input_index_i=base_index,
            num_heads_i=int(num_heads),
            scale_value_f=float(scale_value),
            pre_tokens_i=int(pre_tokens),
            next_tokens_i=int(next_tokens),
            input_layout_s=str(input_layout),
            num_key_value_heads_i=int(num_key_value_heads),
            sparse_mode_i=int(sparse_mode),
            inner_precise_i=int(inner_precise),
        )
        y.setType(query.type())
        return y


def _apply_causal_mask(attn):
    """Apply causal mask to attention scores for sparse modes 2/3."""
    q_len, k_len = attn.shape[-2], attn.shape[-1]
    ar_q = torch.arange(q_len, device=attn.device)
    ar_k = torch.arange(k_len, device=attn.device)
    causal = ar_k[None, :] > ar_q[:, None]
    causal = causal[None, None, :, :].expand(attn.shape[0], attn.shape[1], q_len, k_len)
    return attn.masked_fill(causal, torch.finfo(attn.dtype).min)


def prompt_flash_attention(query, key, value, atten_mask, num_heads, scale_value,
                           input_layout, num_key_value_heads, sparse_mode=0,
                           inner_precise=1, pre_tokens=214748647, next_tokens=0):
    """Functional wrapper for prompt flash attention."""
    return _PromptFlashAttentionCustom.apply(
        query, key, value, atten_mask,
        int(num_heads), float(scale_value), str(input_layout),
        int(num_key_value_heads), int(sparse_mode), int(inner_precise),
        int(pre_tokens), int(next_tokens))


# ---------------------------------------------------------------------------
# SwiGLU and Scatter custom ops
# ---------------------------------------------------------------------------


class _SwiGluCustom(torch.autograd.Function):
    """Custom SwiGLU op for ONNX export — fused SiLU-gate activation."""

    @staticmethod
    def forward(ctx, x, dim: int):
        """Eager fallback: silu(first_half) * second_half."""
        del ctx
        d = int(dim)
        if d < 0:
            d = x.dim() + d
        a, b = torch.chunk(x, 2, dim=d)
        return F.silu(a) * b

    @staticmethod
    def symbolic(g, x, dim: int):
        """ONNX symbolic for SwiGLU."""
        y = g.op(
            "Custom", x,
            type_s="SwiGlu",
            input_names_s=_as_list_str(["x"]),
            optional_input_names_s=_as_list_str([]),
            output_names_s=_as_list_str(["y"]),
            output_num_i=1,
            input_index_i=[0],
            dim_i=int(dim),
        )
        y.setType(x.type())
        return y


def swiglu(x, dim: int = -1):
    """Apply SwiGLU activation: silu(x[...,:d]) * x[...,d:] via custom op."""
    return _SwiGluCustom.apply(x, int(dim))


class _ScatterCustom(torch.autograd.Function):
    """Custom Scatter op for ONNX export — update KV cache at a specific position."""

    @staticmethod
    def forward(ctx, var, indices, updates, reduce: str, axis: int):
        """Eager fallback: scatter updates into var at indices along axis."""
        del ctx
        if str(reduce) != "update":
            raise RuntimeError("Only reduce='update' is supported.")
        ax = int(axis)
        if ax < 0:
            ax = var.dim() + ax
        if var.dim() != 4 or ax != 2:
            raise RuntimeError("Only 4D var with axis=-2/2 is supported.")
        out = var.clone()
        _scatter_update(out, indices, updates)
        return out

    @staticmethod
    def symbolic(g, var, indices, updates, reduce: str, axis: int):
        """ONNX symbolic for scatter."""
        y = g.op(
            "Custom", var, indices, updates,
            type_s="Scatter",
            input_names_s=_as_list_str(["var", "indices", "updates"]),
            optional_input_names_s=_as_list_str([]),
            output_names_s=_as_list_str(["var"]),
            output_num_i=1,
            input_index_i=[0, 1, 2],
            reduce_s=str(reduce),
            axis_i=int(axis),
        )
        y.setType(var.type())
        return y


def _scatter_update(var, indices, updates):
    """In-place scatter updates for 4D tensor at axis=2 using batch indices."""
    bsz, num_heads, _, _ = var.shape
    pos = indices
    if pos.dim() == 2 and pos.shape[-1] == 1:
        pos = pos.squeeze(-1)
    pos = pos.to(torch.long).view(bsz)
    upd = updates
    if upd.dim() == 4 and upd.shape[2] == 1:
        upd = upd[:, :, 0, :]
    b = torch.arange(bsz, device=var.device).view(bsz, 1).expand(bsz, num_heads)
    h = torch.arange(num_heads, device=var.device).view(1, num_heads).expand(bsz, num_heads)
    s = pos.view(bsz, 1).expand(bsz, num_heads)
    var[b, h, s, :] = upd


def scatter(var, indices, updates, reduce: str = "update", axis: int = -2):
    """Scatter updates into variable at given indices via custom op."""
    return _ScatterCustom.apply(var, indices, updates, str(reduce), int(axis))


# ---------------------------------------------------------------------------
# QKV projection and attention dispatch
# ---------------------------------------------------------------------------


def _compute_qkv(attn_mod, hidden_states):
    """Compute fused QKV projection and reshape to (batch, seq, heads, head_dim).

    Returns (query_states, key_states, value_states, input_shape).
    """
    input_shape = hidden_states.shape[:-1]
    head_dim = attn_mod.head_dim
    hidden_shape = (*input_shape, -1, head_dim)

    w = torch.cat([attn_mod.q_proj.weight, attn_mod.k_proj.weight, attn_mod.v_proj.weight], dim=0)
    q_b, k_b, v_b = attn_mod.q_proj.bias, attn_mod.k_proj.bias, attn_mod.v_proj.bias
    b = None if q_b is None else torch.cat([q_b, k_b, v_b], dim=0)

    q_out = int(attn_mod.q_proj.weight.shape[0])
    kv_out = int(attn_mod.k_proj.weight.shape[0])
    qkv = F.linear(hidden_states, w, b)

    query = qkv[..., :q_out].view(hidden_shape)
    key = qkv[..., q_out:q_out + kv_out].view(hidden_shape)
    value = qkv[..., q_out + kv_out:].view(hidden_shape)
    if hasattr(attn_mod, "q_norm"):
        query = _rms_norm_layer(attn_mod.q_norm, query)
    if hasattr(attn_mod, "k_norm"):
        key = _rms_norm_layer(attn_mod.k_norm, key)
    return query, key, value, input_shape


def _run_prefill_attention(query, key, value, attention_mask, num_heads, num_kv_heads, scaling):
    """Run prefill-path attention using standard matmul with causal mask.

    Returns (attn_output, key_states, value_states) with shapes for KV cache.
    """
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    if 0 < num_kv_heads < num_heads:
        rep = num_heads // num_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    attn = torch.matmul(q, k.transpose(2, 3)) * float(scaling)
    flash_mask = _make_flash_attn_mask(attention_mask, attn.shape[-2], attn.shape[-1], 0)
    if flash_mask.dim() == 4 and flash_mask.shape[1] == 1:
        flash_mask = flash_mask.expand(attn.shape[0], attn.shape[1], flash_mask.shape[2], flash_mask.shape[3])
    attn = attn.masked_fill(flash_mask.to(torch.bool), torch.finfo(attn.dtype).min)
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_output = torch.matmul(attn, v).permute(0, 2, 1, 3)
    return attn_output, k.transpose(1, 2), v.transpose(1, 2)


def _run_decode_attention(query, key, value, attention_mask, attn_mod, num_heads, num_kv_heads):
    """Run decode-path attention using IncreFlashAttention custom op.

    Returns attn_output in BNSD layout reshaped to (batch, 1, hidden).
    """
    scaling = getattr(attn_mod, "scaling", 1.0 / (attn_mod.head_dim ** 0.5))
    pad_mask = attention_mask[:, None, None, :].to(torch.bool).logical_not()
    return incre_flash_attention(
        query, key, value, pad_mask,
        num_heads=num_heads, scale_value=float(scaling),
        input_layout="BNSD", num_key_value_heads=num_kv_heads, inner_precise=1)


def _text_attn_forward(attn_mod, hidden_states, cos4, sin4, attention_mask,
                       cache_pos, past_key, past_value):
    """Dispatch attention computation to prefill or decode path.

    Prefill: standard matmul attention with causal mask.
    Decode: IncreFlashAttention with KV cache scatter update.
    """
    num_heads = attn_mod.config.num_attention_heads
    num_kv_heads = attn_mod.config.num_key_value_heads

    query, key, value, input_shape = _compute_qkv(attn_mod, hidden_states)

    if past_key is not None:
        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)

    query = rotary_mul(query, cos4, sin4)
    key = rotary_mul(key, cos4, sin4)

    if past_key is not None:
        pos = cache_pos
        if pos is None:
            raise RuntimeError("cache_pos is required when past_key_values is provided.")
        if pos.dim() == 2:
            pos = pos[:, -1]
        key = scatter(past_key, pos, key, reduce="update", axis=-2)
        value = scatter(past_value, pos, value, reduce="update", axis=-2)

    if past_key is None:
        attn_output, key, value = _run_prefill_attention(
            query, key, value, attention_mask, num_heads, num_kv_heads,
            getattr(attn_mod, "scaling", 1.0 / (attn_mod.head_dim ** 0.5)))
        out = attn_output.reshape(*input_shape, -1)
    else:
        attn_output = _run_decode_attention(
            query, key, value, attention_mask, attn_mod, num_heads, num_kv_heads)
        out = attn_output.transpose(1, 2).reshape(*input_shape, -1)

    return attn_mod.o_proj(out), key, value


# ---------------------------------------------------------------------------
# MLP helpers
# ---------------------------------------------------------------------------


def _mlp_gate_up_linear(mlp_mod, x):
    """Merge gate_proj and up_proj into a single linear, then split outputs."""
    gate_w = mlp_mod.gate_proj.weight
    up_w = mlp_mod.up_proj.weight
    gate_b = mlp_mod.gate_proj.bias
    up_b = mlp_mod.up_proj.bias
    w = torch.cat([gate_w, up_w], dim=0)
    b = None if gate_b is None else torch.cat([gate_b, up_b], dim=0)
    y = F.linear(x, w, b)
    gate_out = int(gate_w.shape[0])
    return y[..., :gate_out], y[..., gate_out:]


def _run_mlp(layer, hidden_states):
    """Run MLP forward pass with fused gate+up projection and SwiGLU activation."""
    mlp = layer.mlp
    if hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj") and hasattr(mlp, "down_proj"):
        gate, up = _mlp_gate_up_linear(mlp, hidden_states)
        return mlp.down_proj(swiglu(torch.cat([gate, up], dim=-1), dim=-1))
    return mlp(hidden_states)


def _pad_kv_to_cache_len(kv_tensor):
    """Pad a KV cache tensor (batch, heads, seq, dim) to KV_CACHE_LEN along dim 2."""
    pad_len = KV_CACHE_LEN - kv_tensor.shape[2]
    if pad_len <= 0:
        return kv_tensor[:, :, :KV_CACHE_LEN, :]
    zeros = kv_tensor.new_zeros(kv_tensor.shape[0], kv_tensor.shape[1], pad_len, kv_tensor.shape[3])
    return torch.cat([kv_tensor, zeros], dim=2)[:, :, :KV_CACHE_LEN, :]


# ---------------------------------------------------------------------------
# Prefill / Decode wrapper modules
# ---------------------------------------------------------------------------


class Qwen2_5LlmPrefill(torch.nn.Module):
    """Qwen2.5-7B LLM Prefill wrapper — processes full prompt and outputs padded KV cache."""

    def __init__(self, model, lm_head):
        """Initialize prefill wrapper with shared model and lm_head."""
        super().__init__()
        self.model = model.model
        self.lm_head = lm_head

    def forward(self, input_ids, attention_mask, position_ids):
        """Run prefill: embed tokens, process all layers, output logits + KV cache."""
        inputs_embeds = self.model.embed_tokens(input_ids)
        cos, sin = self.model.rotary_emb(inputs_embeds, position_ids)
        cos4 = cos.unsqueeze(2) if cos.dim() == 3 else cos
        sin4 = sin.unsqueeze(2) if sin.dim() == 3 else sin

        present_k, present_v = [], []
        hidden_states = inputs_embeds
        for layer in self.model.layers:
            residual = hidden_states
            hidden_states = _rms_norm_layer(layer.input_layernorm, hidden_states)
            attn_out, pk, pv = _text_attn_forward(
                layer.self_attn, hidden_states, cos4, sin4, attention_mask,
                None, None, None)
            pk, pv = _pad_kv_to_cache_len(pk), _pad_kv_to_cache_len(pv)
            hidden_states = residual + attn_out
            residual = hidden_states
            hidden_states = _rms_norm_layer(layer.post_attention_layernorm, hidden_states)
            hidden_states = residual + _run_mlp(layer, hidden_states)
            present_k.append(pk)
            present_v.append(pv)

        hidden_states = _rms_norm_layer(self.model.norm, hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, torch.stack(present_k, dim=0), torch.stack(present_v, dim=0)


class Qwen2_5LlmDecode(torch.nn.Module):
    """Qwen2.5-7B LLM Decode wrapper — single-token generation with KV cache update."""

    def __init__(self, model, lm_head):
        """Initialize decode wrapper with shared model and lm_head."""
        super().__init__()
        self.model = model.model
        self.lm_head = lm_head

    def forward(self, input_ids, attention_mask, position_ids, past_key_cache, past_value_cache):
        """Run decode: embed token, update KV cache per layer, output logits + new cache."""
        inputs_embeds = self.model.embed_tokens(input_ids)
        cos, sin = self.model.rotary_emb(inputs_embeds, position_ids)
        cos4 = cos.unsqueeze(1) if cos.dim() == 3 else cos
        sin4 = sin.unsqueeze(1) if sin.dim() == 3 else sin

        past_k_layers = past_key_cache.unbind(0)
        past_v_layers = past_value_cache.unbind(0)
        present_k, present_v = [], []
        hidden_states = inputs_embeds

        for i, layer in enumerate(self.model.layers):
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            attn_out, pk, pv = _text_attn_forward(
                layer.self_attn, hidden_states, cos4, sin4, attention_mask,
                position_ids, past_k_layers[i], past_v_layers[i])
            hidden_states = residual + attn_out
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = residual + _run_mlp(layer, hidden_states)
            present_k.append(pk)
            present_v.append(pv)

        hidden_states = _rms_norm_layer(self.model.norm, hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, torch.stack(present_k, dim=0), torch.stack(present_v, dim=0)


# ---------------------------------------------------------------------------
# Export orchestration
# ---------------------------------------------------------------------------


def _prepare_llm_modules(model, device: str):
    """Create prefill and decode wrappers, move to device, set eval mode."""
    lm_head = model.lm_head
    model.eval()
    lm_head.eval()
    model.to(device)
    lm_head.to(device)
    prefill = Qwen2_5LlmPrefill(model, lm_head).to(device).eval()
    decode = Qwen2_5LlmDecode(model, lm_head).to(device).eval()
    return prefill, decode, lm_head


def _get_kv_cache_config(model):
    """Extract (num_layers, num_kv_heads, head_dim) from model config."""
    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    head_dim = getattr(
        model.config, "head_dim",
        model.config.hidden_size // model.config.num_attention_heads)
    return num_layers, num_kv_heads, head_dim


def _prepare_output_paths(output_dir):
    """Create prefill/ and decode/ subdirectories and return ONNX output paths."""
    prefill_dir = Path(output_dir) / "prefill"
    decode_dir = Path(output_dir) / "decode"
    prefill_dir.mkdir(parents=True, exist_ok=True)
    decode_dir.mkdir(parents=True, exist_ok=True)
    return prefill_dir / "qwen2_5_7b_llm_prefill.onnx", decode_dir / "qwen2_5_7b_llm_decode.onnx"


def _create_prefill_dummy_inputs(device: str, dummy_seq_len: int):
    """Create random dummy inputs for prefill model export."""
    seq = int(dummy_seq_len)
    ids = torch.randint(0, 1000, (1, seq), dtype=torch.int64, device=device)
    mask = torch.ones(1, seq, dtype=torch.int64, device=device)
    pos = torch.arange(seq, device=device, dtype=torch.int64).view(1, -1)
    return seq, ids, mask, pos


def _export_prefill_onnx(prefill, prefill_path: Path, dummy_inputs, use_dynamo: bool):
    """Export prefill subgraph to ONNX with dynamic sequence length."""
    print(f"Exporting LLM prefill to {prefill_path}...")
    dynamic = {"input_ids": {0: "batch", 1: "seq"},
               "attention_mask": {0: "batch", 1: "seq"},
               "position_ids": {0: "batch", 1: "seq"},
               "logits": {0: "batch", 1: "seq"},
               "present_key_cache": {1: "batch"},
               "present_value_cache": {1: "batch"}}
    with torch.no_grad():
        torch.onnx.export(
            prefill, dummy_inputs, str(prefill_path),
            input_names=["input_ids", "attention_mask", "position_ids"],
            output_names=["logits", "present_key_cache", "present_value_cache"],
            opset_version=18, do_constant_folding=True, dynamo=use_dynamo,
            dynamic_axes=dynamic)
    print("LLM prefill exported successfully.")


def _create_decode_dummy_inputs(device, num_layers, num_kv_heads, head_dim, kv_dtype):
    """Create dummy inputs for decode model export with fixed KV cache length."""
    past_len = int(KV_CACHE_LEN)
    ids = torch.randint(0, 1000, (1, 1), dtype=torch.int64, device=device)
    mask = torch.ones(1, past_len, dtype=torch.int64, device=device)
    pos = torch.tensor([[past_len - 1]], dtype=torch.int64, device=device)
    k = torch.zeros(num_layers, 1, num_kv_heads, past_len, head_dim, dtype=kv_dtype, device=device)
    v = torch.zeros_like(k)
    return ids, mask, pos, k, v


def _export_decode_onnx(decode, decode_path: Path, dummy_inputs, use_dynamo: bool):
    """Export decode subgraph to ONNX with fixed shapes."""
    print(f"Exporting LLM decode to {decode_path}...")
    dynamic = {"input_ids": {0: "batch"}, "attention_mask": {0: "batch"},
               "position_ids": {0: "batch"}, "logits": {0: "batch"},
               "past_key_cache": {1: "batch"}, "past_value_cache": {1: "batch"},
               "present_key_cache": {1: "batch"}, "present_value_cache": {1: "batch"}}
    with torch.no_grad():
        torch.onnx.export(
            decode, dummy_inputs, str(decode_path),
            input_names=["input_ids", "attention_mask", "position_ids",
                          "past_key_cache", "past_value_cache"],
            output_names=["logits", "present_key_cache", "present_value_cache"],
            opset_version=18, do_constant_folding=True, dynamo=use_dynamo,
            dynamic_axes=dynamic)
    print("LLM decode exported successfully.")


def export_llm_prefill_decode(model, output_dir, device="cpu", dummy_seq_len=8, use_dynamo=False):
    """Export Qwen2.5-7B as two ONNX subgraphs (prefill + decode)."""
    prefill, decode, _ = _prepare_llm_modules(model, device=device)
    kv_dtype = next(model.parameters()).dtype
    num_layers, num_kv_heads, head_dim = _get_kv_cache_config(model)
    prefill_path, decode_path = _prepare_output_paths(output_dir)

    _, ids, mask, pos = _create_prefill_dummy_inputs(device=device, dummy_seq_len=dummy_seq_len)
    _export_prefill_onnx(prefill, prefill_path, (ids, mask, pos), use_dynamo)

    decode_inputs = _create_decode_dummy_inputs(device, num_layers, num_kv_heads, head_dim, kv_dtype)
    _export_decode_onnx(decode, decode_path, decode_inputs, use_dynamo)


def _parse_export_args():
    """Parse command-line arguments for the export script."""
    parser = argparse.ArgumentParser(description="Export Qwen2.5-7B to ONNX")
    parser.add_argument("--model-id", type=str, default="./Qwen2.5-7B-Instruct",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--output-dir", type=str, default="./qwen2_5_7b_onnx",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for export (cpu or cuda)")
    parser.add_argument("--dummy-seq-len", type=int, default=8,
                        help="Dummy sequence length for export")
    parser.add_argument("--dtype", type=str, default="fp16",
                        choices=["fp16", "bf16", "fp32"], help="Export dtype")
    parser.add_argument("--use-dynamo", action="store_true",
                        help="Use torch dynamo exporter path")
    return parser.parse_args()


def main():
    """Load Qwen2.5-7B model and export to ONNX prefill + decode subgraphs."""
    args = _parse_export_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    print(f"\nLoading model {args.model_id} for export (dtype={args.dtype})...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=False,
        attn_implementation="eager").to(device)

    export_llm_prefill_decode(model, output_dir, str(device), args.dummy_seq_len, args.use_dynamo)

    print("Clearing memory after export...")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"\nExport finished. Files saved in {args.output_dir}")


if __name__ == "__main__":
    main()
