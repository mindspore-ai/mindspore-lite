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
Export Qwen3-8B model to ONNX format.
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


def _as_list_str(items):
    """Convert items to a list of string representations."""
    return [str(x) for x in items]


def _rotate_half(x):
    """Rotate half the hidden dims of the input tensor for rotary embedding."""
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat([-x2, x1], dim=-1)


class _RotaryMulCustom(torch.autograd.Function):
    """Custom RotaryMul op for ONNX export."""

    @staticmethod
    def forward(ctx, x, cos4, sin4):
        """Eager forward for RotaryMul."""
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
    """Apply rotary position embedding multiplication (custom op wrapper)."""
    return _RotaryMulCustom.apply(x, cos4, sin4)


class _ApplyRotaryPosEmbCustom(torch.autograd.Function):
    """Custom ApplyRotaryPosEmb op for ONNX export."""

    @staticmethod
    def forward(ctx, query, key, cos, sin, layout: int, rotary_mode: str):
        """Eager forward for rotary position embedding."""
        del ctx, rotary_mode
        if apply_rotary_pos_emb is not None:
            if int(layout) == 1:
                q_bnsd = query.permute(0, 2, 1, 3)
                k_bnsd = key.permute(0, 2, 1, 3)
                q2, k2 = apply_rotary_pos_emb(q_bnsd, k_bnsd, cos, sin)
                return q2.permute(0, 2, 1, 3), k2.permute(0, 2, 1, 3)
            q, k = apply_rotary_pos_emb(query, key, cos, sin)
            return q, k

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
    """Apply rotary position embedding with custom op (wrapper)."""
    return _ApplyRotaryPosEmbCustom.apply(query, key, cos, sin, int(layout), str(rotary_mode))


class _RmsNormCustom(torch.autograd.Function):
    """Custom RMSNorm op for ONNX export."""

    @staticmethod
    def forward(ctx, x, gamma, epsilon: float):
        """Eager forward for RMSNorm."""
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
    """Apply RMS normalization (custom op wrapper)."""
    return _RmsNormCustom.apply(x, gamma, float(epsilon))


def _make_flash_attn_mask(attention_mask, q_len, k_len, past_len):
    """Create boolean causal + padding mask for flash attention."""
    ar_q = torch.arange(q_len, device=attention_mask.device)
    ar_k = torch.arange(k_len, device=attention_mask.device)
    causal = ar_k[None, :] > (past_len + ar_q[:, None])
    causal = causal[None, None, :, :].expand(attention_mask.shape[0], 1, q_len, k_len)
    padding = attention_mask[:, None, None, :].to(torch.bool).logical_not()
    return (causal | padding).to(torch.bool)


def _expand_gqa_kv(k, v, num_heads, num_kv_heads):
    """Expand GQA key/value tensors to match num_heads via repeat_interleave."""
    if 0 < num_kv_heads < num_heads:
        rep = num_heads // num_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    return k, v


def _apply_attn_mask(attn, atten_mask):
    """Apply boolean attention mask to attention scores."""
    if atten_mask is not None:
        m = atten_mask.to(torch.bool)
        if m.dim() == 4 and m.shape[1] == 1:
            m = m.expand(attn.shape[0], attn.shape[1], m.shape[2], m.shape[3])
        attn = attn.masked_fill(m, torch.finfo(attn.dtype).min)
    return attn


def _maybe_transpose_bsnd(tensor, layout):
    """Permute tensor from BNSD to BSND layout if needed."""
    if str(layout).upper() in ("BSND", "SBND"):
        return tensor.permute(0, 2, 1, 3)
    return tensor


class _IncreFlashAttentionCustom(torch.autograd.Function):
    """Custom IncreFlashAttention op for ONNX export."""

    @staticmethod
    def forward(ctx, query, key, value, atten_mask, num_heads, scale_value,
                input_layout, num_key_value_heads, block_size, inner_precise):
        """Eager forward for incremental flash attention."""
        del ctx, block_size, inner_precise
        q = _maybe_transpose_bsnd(query, input_layout)
        k = _maybe_transpose_bsnd(key, input_layout)
        v = _maybe_transpose_bsnd(value, input_layout)
        k, v = _expand_gqa_kv(k, v, num_heads, num_key_value_heads)
        attn = torch.matmul(q, k.transpose(2, 3)) * float(scale_value)
        attn = _apply_attn_mask(attn, atten_mask)
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v)
        return _maybe_transpose_bsnd(out, input_layout)

    @staticmethod
    def symbolic(g, query, key, value, atten_mask, num_heads, scale_value,
                 input_layout, num_key_value_heads, block_size, inner_precise):
        """ONNX symbolic for incremental flash attention."""
        base_attrs = {
            "type_s": "IncreFlashAttention",
            "input_names_s": _as_list_str(["query", "key", "value", "atten_mask"]),
            "optional_input_names_s": _as_list_str(["atten_mask"]),
            "output_names_s": _as_list_str(["attention_out"]),
            "output_num_i": 1,
            "num_heads_i": int(num_heads),
            "scale_value_f": float(scale_value),
            "input_layout_s": str(input_layout),
            "num_key_value_heads_i": int(num_key_value_heads),
            "block_size_i": int(block_size),
            "inner_precise_i": int(inner_precise),
        }
        if atten_mask is None:
            y = g.op("Custom", query, key, value, input_index_i=[0, 1, 2], **base_attrs)
        else:
            y = g.op("Custom", query, key, value, atten_mask,
                      input_index_i=[0, 1, 2, 3], **base_attrs)
        y.setType(query.type())
        return y


def incre_flash_attention(query, key, value, atten_mask, num_heads: int,
                          scale_value: float, input_layout: str,
                          num_key_value_heads: int, block_size: int = 0,
                          inner_precise: int = 1):
    """Apply incremental flash attention (custom op wrapper)."""
    return _IncreFlashAttentionCustom.apply(
        query, key, value, atten_mask,
        int(num_heads), float(scale_value), str(input_layout),
        int(num_key_value_heads), int(block_size), int(inner_precise),
    )


class _PromptFlashAttentionCustom(torch.autograd.Function):
    """Custom prompt flash attention op for ONNX export."""

    @staticmethod
    def forward(ctx, query, key, value, atten_mask, num_heads, scale_value,
                input_layout, num_key_value_heads, sparse_mode,
                inner_precise, pre_tokens, next_tokens):
        """Eager forward for prompt flash attention."""
        del ctx, inner_precise, pre_tokens, next_tokens
        q = _maybe_transpose_bsnd(query, input_layout)
        k = _maybe_transpose_bsnd(key, input_layout)
        v = _maybe_transpose_bsnd(value, input_layout)
        k, v = _expand_gqa_kv(k, v, num_heads, num_key_value_heads)
        attn = torch.matmul(q, k.transpose(2, 3)) * float(scale_value)
        attn = _apply_attn_mask(attn, atten_mask)
        if atten_mask is None and int(sparse_mode) in (2, 3):
            q_len, k_len = attn.shape[-2], attn.shape[-1]
            causal = torch.arange(k_len, device=attn.device)[None, :] > \
                     torch.arange(q_len, device=attn.device)[:, None]
            causal = causal[None, None].expand(attn.shape[0], attn.shape[1], q_len, k_len)
            attn = attn.masked_fill(causal, torch.finfo(attn.dtype).min)
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v)
        return _maybe_transpose_bsnd(out, input_layout)

    @staticmethod
    def symbolic(g, query, key, value, atten_mask, num_heads, scale_value,
                 input_layout, num_key_value_heads, sparse_mode,
                 inner_precise, pre_tokens, next_tokens):
        """ONNX symbolic for prompt flash attention."""
        base_attrs = {
            "type_s": "PromptFlashAttention",
            "input_names_s": _as_list_str(["query", "key", "value", "atten_mask"]),
            "optional_input_names_s": _as_list_str(["atten_mask"]),
            "output_names_s": _as_list_str(["attention_out"]),
            "output_num_i": 1,
            "num_heads_i": int(num_heads),
            "scale_value_f": float(scale_value),
            "pre_tokens_i": int(pre_tokens),
            "next_tokens_i": int(next_tokens),
            "input_layout_s": str(input_layout),
            "num_key_value_heads_i": int(num_key_value_heads),
            "sparse_mode_i": int(sparse_mode),
            "inner_precise_i": int(inner_precise),
        }
        if atten_mask is None:
            y = g.op("Custom", query, key, value, input_index_i=[0, 1, 2], **base_attrs)
        else:
            y = g.op("Custom", query, key, value, atten_mask,
                      input_index_i=[0, 1, 2, 3], **base_attrs)
        y.setType(query.type())
        return y


def prompt_flash_attention(query, key, value, atten_mask, num_heads: int,
                           scale_value: float, input_layout: str,
                           num_key_value_heads: int, sparse_mode: int = 0,
                           inner_precise: int = 1, pre_tokens: int = 214748647,
                           next_tokens: int = 0):
    """Apply prompt flash attention (custom op wrapper)."""
    return _PromptFlashAttentionCustom.apply(
        query, key, value, atten_mask,
        int(num_heads), float(scale_value), str(input_layout),
        int(num_key_value_heads), int(sparse_mode), int(inner_precise),
        int(pre_tokens), int(next_tokens),
    )


class _SwiGluCustom(torch.autograd.Function):
    """Custom SwiGLU op for ONNX export."""

    @staticmethod
    def forward(ctx, x, dim: int):
        """Eager forward for SwiGLU."""
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
    """Apply SwiGLU activation (custom op wrapper)."""
    return _SwiGluCustom.apply(x, int(dim))


class _ScatterCustom(torch.autograd.Function):
    """Custom scatter op for ONNX export."""

    @staticmethod
    def forward(ctx, var, indices, updates, reduce: str, axis: int):
        """Eager forward for scatter update on 4D tensor at axis=2."""
        del ctx
        if str(reduce) != "update":
            raise RuntimeError("Only reduce='update' is supported.")
        ax = int(axis)
        if ax < 0:
            ax = var.dim() + ax
        if var.dim() != 4 or ax != 2:
            raise RuntimeError("Only 4D var with axis=-2/2 is supported.")
        out, _ = _scatter_fwd_impl(var, indices, updates)
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


def _scatter_fwd_impl(var, indices, updates):
    """Perform scatter update on 4D var at axis=2, return (output, squeezed_updates)."""
    bsz, num_heads, _, _ = var.shape
    pos = indices
    if pos.dim() == 2 and pos.shape[-1] == 1:
        pos = pos.squeeze(-1)
    pos = pos.to(torch.long).view(bsz)
    upd = updates
    if upd.dim() == 4 and upd.shape[2] == 1:
        upd = upd[:, :, 0, :]
    out = var.clone()
    b = torch.arange(bsz, device=out.device).view(bsz, 1).expand(bsz, num_heads)
    h = torch.arange(num_heads, device=out.device).view(1, num_heads).expand(bsz, num_heads)
    s = pos.view(bsz, 1).expand(bsz, num_heads)
    out[b, h, s, :] = upd
    return out, upd


def scatter(var, indices, updates, reduce: str = "update", axis: int = -2):
    """Apply scatter update (custom op wrapper)."""
    return _ScatterCustom.apply(var, indices, updates, str(reduce), int(axis))


def _make_additive_causal_mask(attention_mask, q_len, k_len, past_len, dtype):
    """Create additive causal + padding mask with dtype-appropriate min value."""
    mask_value = torch.finfo(dtype).min
    ar_q = torch.arange(q_len, device=attention_mask.device)
    ar_k = torch.arange(k_len, device=attention_mask.device)
    causal = ar_k[None, :] > (past_len + ar_q[:, None])
    causal = causal.to(dtype) * mask_value
    causal = causal[None, None, :, :].expand(attention_mask.shape[0], 1, q_len, k_len)
    padding = (1.0 - attention_mask.to(dtype)) * mask_value
    padding = padding[:, None, None, :]
    return causal + padding


def _compute_qkv_linear(hidden_states, attn_mod):
    """Compute fused QKV linear projection and return (query, key, value) states."""
    q_w, k_w, v_w = attn_mod.q_proj.weight, attn_mod.k_proj.weight, attn_mod.v_proj.weight
    q_b, k_b, v_b = attn_mod.q_proj.bias, attn_mod.k_proj.bias, attn_mod.v_proj.bias
    w = torch.cat([q_w, k_w, v_w], dim=0)
    b = None if q_b is None else torch.cat([q_b, k_b, v_b], dim=0)
    q_out = int(q_w.shape[0])
    kv_out = int(k_w.shape[0])
    qkv = F.linear(hidden_states, w, b)
    return qkv[..., :q_out], qkv[..., q_out:q_out + kv_out], qkv[..., q_out + kv_out:]


def _apply_rotary_and_cache(cos4, sin4, query_states, key_states, value_states,
                            cache_pos, past_key, past_value):
    """Apply rotary embedding and scatter KV cache updates if past_key is provided."""
    query_states = rotary_mul(query_states, cos4, sin4)
    key_states = rotary_mul(key_states, cos4, sin4)
    if past_key is not None:
        pos = cache_pos
        if pos is None:
            raise RuntimeError("cache_pos is required when past_key_values is provided.")
        if pos.dim() == 2:
            pos = pos[:, -1]
        key_states = scatter(past_key, pos, key_states, reduce="update", axis=-2)
        value_states = scatter(past_value, pos, value_states, reduce="update", axis=-2)
    return query_states, key_states, value_states


def _prefill_attn(query_states, key_states, value_states, attention_mask, scaling,
                  num_heads, num_kv_heads):
    """Compute prefill-phase attention using PromptFlashAttention custom op."""
    q = query_states.permute(0, 2, 1, 3)
    k = key_states.permute(0, 2, 1, 3)
    v = value_states.permute(0, 2, 1, 3)
    # Build an explicit causal + padding mask (True == masked out). PromptFlash
    # Attention runs with sparse_mode=0 (apply ONLY the supplied mask), so the
    # mask must encode BOTH:
    #   * causal triangle -- future keys (j > i) masked, else the prefill leaks
    #     future tokens and degrades to near-immediate EOS;
    #   * padding keys -- attention_mask==0 positions masked.
    # The mask's Q axis must equal sQ too, or the ascend op's tiling rejects it
    # (CheckAttenMaskShape: attenMask Q_S must be >= sQ).
    q_len, kv_len = q.shape[2], k.shape[2]
    ar_q = torch.arange(q_len, device=q.device)
    ar_k = torch.arange(kv_len, device=k.device)
    causal = ar_k[None, :] > ar_q[:, None]                       # (q, k) future
    pad = attention_mask.to(torch.bool).logical_not()            # (b, k) padding
    full_mask = causal[None, None, :, :] | pad[:, None, None, :]
    attn_output = prompt_flash_attention(
        q, k, v, full_mask,
        num_heads=num_heads, scale_value=float(scaling),
        input_layout="BNSD", num_key_value_heads=num_kv_heads,
        sparse_mode=0, inner_precise=1,
    )
    return attn_output.permute(0, 2, 1, 3)


def _text_attn_forward(attn_mod, hidden_states, cos4, sin4, attention_mask,
                       cache_pos, past_key, past_value):
    """Run text attention: QKV projection, rotary embedding, attention, output projection."""
    input_shape = hidden_states.shape[:-1]
    head_dim = attn_mod.head_dim
    num_heads = attn_mod.config.num_attention_heads
    num_kv_heads = attn_mod.config.num_key_value_heads
    hidden_shape = (*input_shape, -1, head_dim)

    q_lin, k_lin, v_lin = _compute_qkv_linear(hidden_states, attn_mod)
    query_states = q_lin.view(hidden_shape)
    key_states = k_lin.view(hidden_shape)
    if hasattr(attn_mod, "q_norm"):
        query_states = _rms_norm_layer(attn_mod.q_norm, query_states)
    if hasattr(attn_mod, "k_norm"):
        key_states = _rms_norm_layer(attn_mod.k_norm, key_states)
    value_states = v_lin.view(hidden_shape)
    if past_key is not None:
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

    query_states, key_states, value_states = _apply_rotary_and_cache(
        cos4, sin4, query_states, key_states, value_states, cache_pos, past_key, past_value)

    scaling = getattr(attn_mod, "scaling", 1.0 / (head_dim ** 0.5))
    if past_key is None:
        attn_output = _prefill_attn(query_states, key_states, value_states, attention_mask, scaling,
                                   num_heads, num_kv_heads)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
    else:
        pad_mask = attention_mask[:, None, None, :].to(torch.bool).logical_not()
        attn_output = incre_flash_attention(
            query_states, key_states, value_states, pad_mask,
            num_heads=num_heads, scale_value=float(scaling),
            input_layout="BNSD", num_key_value_heads=num_kv_heads, inner_precise=1)

    if past_key is None:
        attn_output = attn_output.reshape(*input_shape, -1)
    else:
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1)
    attn_output = attn_mod.o_proj(attn_output)
    return attn_output, key_states, value_states


def _rms_norm_layer(norm_mod, x):
    """Apply RMS normalization using the custom RmsNorm op."""
    gamma = norm_mod.weight
    eps = getattr(norm_mod, "variance_epsilon", 1e-6)
    y, _ = rms_norm(x, gamma, eps)
    return y


def _mlp_gate_up_linear(mlp_mod, x):
    """Merge gate_proj and up_proj into a single linear, return (gate, up)."""
    gate_w = mlp_mod.gate_proj.weight
    up_w = mlp_mod.up_proj.weight
    gate_b = mlp_mod.gate_proj.bias
    up_b = mlp_mod.up_proj.bias
    w = torch.cat([gate_w, up_w], dim=0)
    b = None if gate_b is None else torch.cat([gate_b, up_b], dim=0)
    y = F.linear(x, w, b)
    gate_out = int(gate_w.shape[0])
    return y[..., :gate_out], y[..., gate_out:]


def _run_layer_forward(layer, hidden_states, cos4, sin4, attention_mask,
                       cache_pos, past_key, past_value):
    """Run a single transformer layer: attention + residual + MLP + residual."""
    residual = hidden_states
    hidden_states = _rms_norm_layer(layer.input_layernorm, hidden_states)
    if past_key is None:
        normed = hidden_states
    else:
        normed = layer.input_layernorm(residual)
        hidden_states = normed
    attn_out, pk, pv = _text_attn_forward(
        layer.self_attn, hidden_states, cos4, sin4, attention_mask,
        cache_pos, past_key, past_value)
    hidden_states = residual + attn_out
    residual = hidden_states
    hidden_states = _rms_norm_layer(layer.post_attention_layernorm, hidden_states)
    mlp = layer.mlp
    if hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj") and hasattr(mlp, "down_proj"):
        gate, up = _mlp_gate_up_linear(mlp, hidden_states)
        mlp_out = mlp.down_proj(swiglu(torch.cat([gate, up], dim=-1), dim=-1))
        hidden_states = residual + mlp_out
    else:
        hidden_states = residual + mlp(hidden_states)
    return hidden_states, pk, pv


def _run_mlp(layer, hidden_states, residual):
    """Run MLP sub-layer with SwiGlu activation, return output hidden states."""
    mlp = layer.mlp
    if hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj") and hasattr(mlp, "down_proj"):
        gate, up = _mlp_gate_up_linear(mlp, hidden_states)
        mlp_out = mlp.down_proj(swiglu(torch.cat([gate, up], dim=-1), dim=-1))
        return residual + mlp_out
    return residual + mlp(hidden_states)


class Qwen3LlmPrefill(torch.nn.Module):
    """Qwen3-8B LLM Prefill wrapper: processes full prompt sequence."""

    def __init__(self, model, lm_head):
        """Initialize prefill wrapper with backbone model and lm_head."""
        super().__init__()
        self.model = model.model
        self.lm_head = lm_head

    def forward(self, input_ids, attention_mask, position_ids):
        """Run prefill: embed → rotary → per-layer attention+MLP → norm → logits."""
        inputs_embeds = self.model.embed_tokens(input_ids)
        cos, sin = self.model.rotary_emb(inputs_embeds, position_ids)
        cos4 = cos.unsqueeze(2) if cos.dim() == 3 else cos
        sin4 = sin.unsqueeze(2) if sin.dim() == 3 else sin
        hidden_states = inputs_embeds
        present_k, present_v = [], []

        for layer in self.model.layers:
            residual = hidden_states
            hidden_states = _rms_norm_layer(layer.input_layernorm, hidden_states)
            attn_out, pk, pv = _text_attn_forward(
                layer.self_attn, hidden_states, cos4, sin4, attention_mask, None, None, None)
            pk = _pad_kv_cache(pk)
            pv = _pad_kv_cache(pv)
            hidden_states = residual + attn_out
            residual = hidden_states
            hidden_states = _rms_norm_layer(layer.post_attention_layernorm, hidden_states)
            hidden_states = _run_mlp(layer, hidden_states, residual)
            present_k.append(pk)
            present_v.append(pv)

        hidden_states = _rms_norm_layer(self.model.norm, hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, torch.stack(present_k, dim=0), torch.stack(present_v, dim=0)


def _pad_kv_cache(kv_tensor):
    """Pad KV cache tensor along dim=2 to KV_CACHE_LEN and truncate."""
    pad = kv_tensor.new_zeros(kv_tensor.shape[0], kv_tensor.shape[1], KV_CACHE_LEN, kv_tensor.shape[3])
    return torch.cat([kv_tensor, pad], dim=2)[:, :, :KV_CACHE_LEN, :]


class Qwen3LlmDecode(torch.nn.Module):
    """Qwen3-8B LLM Decode wrapper: generates one token per step using KV cache."""

    def __init__(self, model, lm_head):
        """Initialize decode wrapper with backbone model and lm_head."""
        super().__init__()
        self.model = model.model
        self.lm_head = lm_head

    def forward(self, input_ids, attention_mask, position_ids,
                past_key_cache, past_value_cache):
        """Run decode: embed → rotary → per-layer attention+MLP → norm → logits."""
        inputs_embeds = self.model.embed_tokens(input_ids)
        cos, sin = self.model.rotary_emb(inputs_embeds, position_ids)
        cos4 = cos.unsqueeze(1) if cos.dim() == 3 else cos
        sin4 = sin.unsqueeze(1) if sin.dim() == 3 else sin
        hidden_states = inputs_embeds
        present_k, present_v = [], []
        past_k_layers = past_key_cache.unbind(0)
        past_v_layers = past_value_cache.unbind(0)

        for i, layer in enumerate(self.model.layers):
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            attn_out, pk, pv = _text_attn_forward(
                layer.self_attn, hidden_states, cos4, sin4,
                attention_mask, position_ids, past_k_layers[i], past_v_layers[i])
            hidden_states = residual + attn_out
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = _run_mlp(layer, hidden_states, residual)
            present_k.append(pk)
            present_v.append(pv)

        hidden_states = _rms_norm_layer(self.model.norm, hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, torch.stack(present_k, dim=0), torch.stack(present_v, dim=0)


def _prepare_llm_modules(model, device: str):
    """Build and return (prefill, decode, lm_head) modules ready for export."""
    lm_head = model.lm_head
    model.eval()
    lm_head.eval()
    model.to(device)
    lm_head.to(device)
    prefill = Qwen3LlmPrefill(model, lm_head).to(device).eval()
    decode = Qwen3LlmDecode(model, lm_head).to(device).eval()
    return prefill, decode, lm_head


def _get_kv_cache_config(model):
    """Return (num_layers, num_kv_heads, head_dim) from model config."""
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
    return prefill_dir / "qwen3_8b_llm_prefill.onnx", decode_dir / "qwen3_8b_llm_decode.onnx"


def _create_prefill_dummy_inputs(device: str, dummy_seq_len: int):
    """Create dummy (input_ids, attention_mask, position_ids) for prefill export."""
    seq = int(dummy_seq_len)
    ids = torch.randint(0, 1000, (1, seq), dtype=torch.int64, device=device)
    mask = torch.ones(1, seq, dtype=torch.int64, device=device)
    pos = torch.arange(seq, device=device, dtype=torch.int64).view(1, -1)
    return seq, ids, mask, pos


def _export_prefill_onnx(prefill, prefill_path: Path, dummy_inputs, use_dynamo: bool):
    """Export prefill sub-graph to ONNX with dynamic sequence axis."""
    print(f"Exporting LLM prefill to {prefill_path}...")
    with torch.no_grad():
        torch.onnx.export(
            prefill, dummy_inputs, str(prefill_path),
            input_names=["input_ids", "attention_mask", "position_ids"],
            output_names=["logits", "present_key_cache", "present_value_cache"],
            opset_version=18, do_constant_folding=True, dynamo=use_dynamo,
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "position_ids": {0: "batch", 1: "seq"},
                "logits": {0: "batch", 1: "seq"},
                "present_key_cache": {1: "batch"},
                "present_value_cache": {1: "batch"},
            })
    print("LLM prefill exported successfully.")


def _create_decode_dummy_inputs(device, dummy_seq, num_layers, num_kv_heads,
                                head_dim, kv_dtype):
    """Create dummy inputs for decode export including zero KV caches."""
    del dummy_seq
    step = 1
    past_len = int(KV_CACHE_LEN)
    ids = torch.randint(0, 1000, (1, step), dtype=torch.int64, device=device)
    mask = torch.ones(1, past_len, dtype=torch.int64, device=device)
    pos = torch.tensor([[past_len - 1]], dtype=torch.int64, device=device)
    k = torch.zeros(num_layers, 1, num_kv_heads, past_len, head_dim, dtype=kv_dtype, device=device)
    v = torch.zeros(num_layers, 1, num_kv_heads, past_len, head_dim, dtype=kv_dtype, device=device)
    return ids, mask, pos, k, v


def _export_decode_onnx(decode, decode_path: Path, dummy_inputs, use_dynamo: bool):
    """Export decode sub-graph to ONNX with fixed KV cache shape."""
    print(f"Exporting LLM decode to {decode_path}...")
    with torch.no_grad():
        torch.onnx.export(
            decode, dummy_inputs, str(decode_path),
            input_names=["input_ids", "attention_mask", "position_ids",
                         "past_key_cache", "past_value_cache"],
            output_names=["logits", "present_key_cache", "present_value_cache"],
            opset_version=18, do_constant_folding=True, dynamo=use_dynamo,
            dynamic_axes={
                "input_ids": {0: "batch"}, "attention_mask": {0: "batch"},
                "position_ids": {0: "batch"}, "logits": {0: "batch"},
                "past_key_cache": {1: "batch"}, "past_value_cache": {1: "batch"},
                "present_key_cache": {1: "batch"}, "present_value_cache": {1: "batch"},
            })
    print("LLM decode exported successfully.")


def export_llm_prefill_decode(model, output_dir, device="cpu", dummy_seq_len=8,
                              use_dynamo=False):
    """Export Qwen3-8B as separate prefill and decode ONNX sub-graphs."""
    prefill, decode, _ = _prepare_llm_modules(model, device=device)
    kv_dtype = next(model.parameters()).dtype
    num_layers, num_kv_heads, head_dim = _get_kv_cache_config(model)
    prefill_path, decode_path = _prepare_output_paths(output_dir)

    dummy_seq, dummy_ids, dummy_mask, dummy_pos = _create_prefill_dummy_inputs(
        device=device, dummy_seq_len=dummy_seq_len)
    _export_prefill_onnx(prefill, prefill_path,
                         (dummy_ids, dummy_mask, dummy_pos), use_dynamo)

    decode_inputs = _create_decode_dummy_inputs(
        device, dummy_seq, num_layers, num_kv_heads, head_dim, kv_dtype)
    _export_decode_onnx(decode, decode_path, decode_inputs, use_dynamo)


def _parse_export_args():
    """Parse command-line arguments for the export script."""
    parser = argparse.ArgumentParser(description="Export Qwen3-8B to ONNX")
    parser.add_argument("--model-id", type=str, default="./Qwen3-8B",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--output-dir", type=str, default="./qwen3_8b_onnx",
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


def _load_model_for_export(model_id, device, dtype_str):
    """Load AutoModelForCausalLM on the target device with specified dtype."""
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map[dtype_str]
    device_obj = torch.device(device)
    return AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch_dtype,
        low_cpu_mem_usage=False, attn_implementation="eager").to(device_obj)


def main():
    """Parse args, load model, export prefill+decode ONNX, clean up."""
    args = _parse_export_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading model {args.model_id} for export (dtype={args.dtype})...")
    model = _load_model_for_export(args.model_id, args.device, args.dtype)
    export_llm_prefill_decode(model, output_dir, args.device, args.dummy_seq_len, args.use_dynamo)

    print("Clearing memory after export...")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"\nExport finished. Files saved in {args.output_dir}")


if __name__ == "__main__":
    main()
