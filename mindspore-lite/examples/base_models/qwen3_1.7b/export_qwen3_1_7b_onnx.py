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
Export Qwen3-1.7B model to ONNX format.
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
    return [str(x) for x in items]


def _rotate_half(x):
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat([-x2, x1], dim=-1)


class _RotaryMulCustom(torch.autograd.Function):
    """Custom RotaryMul op for ONNX export."""

    @staticmethod
    def forward(ctx, x, cos4, sin4):
        """Forward for RotaryMul (eager fallback)."""
        del ctx
        return (x * cos4) + (_rotate_half(x) * sin4)

    @staticmethod
    def symbolic(g, x, cos4, sin4):
        """ONNX symbolic for RotaryMul."""
        y = g.op(
            "Custom",
            x,
            cos4,
            sin4,
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
    return _RotaryMulCustom.apply(x, cos4, sin4)



class _ApplyRotaryPosEmbCustom(torch.autograd.Function):
    """Custom ApplyRotaryPosEmb op for ONNX export."""

    @staticmethod
    def forward(ctx, query, key, cos, sin, layout: int, rotary_mode: str):
        """Forward for rotary position embedding (eager fallback)."""
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
            "Custom",
            query,
            key,
            cos4,
            sin4,
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
    return _ApplyRotaryPosEmbCustom.apply(query, key, cos, sin, int(layout), str(rotary_mode))


class _RmsNormCustom(torch.autograd.Function):
    """Custom RMSNorm op for ONNX export."""

    @staticmethod
    def forward(ctx, x, gamma, epsilon: float):
        """Forward for RMSNorm (eager fallback)."""
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
            "Custom",
            x,
            gamma,
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
    return _RmsNormCustom.apply(x, gamma, float(epsilon))


def _make_flash_attn_mask(attention_mask, q_len, k_len, past_len):
    ar_q = torch.arange(q_len, device=attention_mask.device)
    ar_k = torch.arange(k_len, device=attention_mask.device)
    causal = ar_k[None, :] > (past_len + ar_q[:, None])
    causal = causal[None, None, :, :].expand(attention_mask.shape[0], 1, q_len, k_len)
    padding = attention_mask[:, None, None, :].to(torch.bool).logical_not()
    return (causal | padding).to(torch.bool)


class _IncreFlashAttentionCustom(torch.autograd.Function):
    """Custom IncreFlashAttention op for ONNX export."""

    @staticmethod
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
        """Forward for incremental flash attention (eager fallback)."""
        del ctx, block_size, inner_precise
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
        """ONNX symbolic for incremental flash attention."""
        if atten_mask is None:
            y = g.op(
                "Custom",
                query,
                key,
                value,
                type_s="IncreFlashAttention",
                input_names_s=_as_list_str(["query", "key", "value", "atten_mask"]),
                optional_input_names_s=_as_list_str(["atten_mask"]),
                output_names_s=_as_list_str(["attention_out"]),
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
                query,
                key,
                value,
                atten_mask,
                type_s="IncreFlashAttention",
                input_names_s=_as_list_str(["query", "key", "value", "atten_mask"]),
                optional_input_names_s=_as_list_str(["atten_mask"]),
                output_names_s=_as_list_str(["attention_out"]),
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
    """Functional wrapper for incremental flash attention."""
    return _IncreFlashAttentionCustom.apply(
        query,
        key,
        value,
        atten_mask,
        int(num_heads),
        float(scale_value),
        str(input_layout),
        int(num_key_value_heads),
        int(block_size),
        int(inner_precise),
    )



class _PromptFlashAttentionCustom(torch.autograd.Function):
    """Custom prompt flash attention op for ONNX export."""

    @staticmethod
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
        sparse_mode: int,
        inner_precise: int,
        pre_tokens: int,
        next_tokens: int,
    ):
        """Forward for prompt flash attention (eager fallback)."""
        del ctx, inner_precise, pre_tokens, next_tokens
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
        elif int(sparse_mode) in (2, 3):
            q_len = attn.shape[-2]
            k_len = attn.shape[-1]
            ar_q = torch.arange(q_len, device=attn.device)
            ar_k = torch.arange(k_len, device=attn.device)
            causal = ar_k[None, :] > ar_q[:, None]
            causal = causal[None, None, :, :].expand(attn.shape[0], attn.shape[1], q_len, k_len)
            attn = attn.masked_fill(causal, torch.finfo(attn.dtype).min)
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
        sparse_mode: int,
        inner_precise: int,
        pre_tokens: int,
        next_tokens: int,
    ):
        """ONNX symbolic for prompt flash attention."""
        if atten_mask is None:
            y = g.op(
                "Custom",
                query,
                key,
                value,
                type_s="PromptFlashAttention",
                input_names_s=_as_list_str(["query", "key", "value", "atten_mask"]),
                optional_input_names_s=_as_list_str(["atten_mask"]),
                output_names_s=_as_list_str(["attention_out"]),
                output_num_i=1,
                input_index_i=[0, 1, 2],
                num_heads_i=int(num_heads),
                scale_value_f=float(scale_value),
                pre_tokens_i=int(pre_tokens),
                next_tokens_i=int(next_tokens),
                input_layout_s=str(input_layout),
                num_key_value_heads_i=int(num_key_value_heads),
                sparse_mode_i=int(sparse_mode),
                inner_precise_i=int(inner_precise),
            )
        else:
            y = g.op(
                "Custom",
                query,
                key,
                value,
                atten_mask,
                type_s="PromptFlashAttention",
                input_names_s=_as_list_str(["query", "key", "value", "atten_mask"]),
                optional_input_names_s=_as_list_str(["atten_mask"]),
                output_names_s=_as_list_str(["attention_out"]),
                output_num_i=1,
                input_index_i=[0, 1, 2, 3],
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


def prompt_flash_attention(
    query,
    key,
    value,
    atten_mask,
    num_heads: int,
    scale_value: float,
    input_layout: str,
    num_key_value_heads: int,
    sparse_mode: int = 0,
    inner_precise: int = 1,
    pre_tokens: int = 214748647,
    next_tokens: int = 0,
):
    """Functional wrapper for prompt flash attention."""
    return _PromptFlashAttentionCustom.apply(
        query,
        key,
        value,
        atten_mask,
        int(num_heads),
        float(scale_value),
        str(input_layout),
        int(num_key_value_heads),
        int(sparse_mode),
        int(inner_precise),
        int(pre_tokens),
        int(next_tokens),
    )


class _SwiGluCustom(torch.autograd.Function):
    """Custom SwiGLU op for ONNX export."""

    @staticmethod
    def forward(ctx, x, dim: int):
        """Forward for SwiGLU (eager fallback)."""
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
            "Custom",
            x,
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
    return _SwiGluCustom.apply(x, int(dim))


class _ScatterCustom(torch.autograd.Function):
    """Custom scatter op for ONNX export."""

    @staticmethod
    def forward(ctx, var, indices, updates, reduce: str, axis: int):
        """Forward for scatter (eager fallback)."""
        del ctx
        if str(reduce) != "update":
            raise RuntimeError("Only reduce='update' is supported.")
        ax = int(axis)
        if ax < 0:
            ax = var.dim() + ax
        if var.dim() != 4 or ax != 2:
            raise RuntimeError("Only 4D var with axis=-2/2 is supported.")
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
        h = torch.arange(num_heads, device=out.device).view(1, num_heads).expand(
            bsz, num_heads
        )
        s = pos.view(bsz, 1).expand(bsz, num_heads)
        out[b, h, s, :] = upd
        return out

    @staticmethod
    def symbolic(g, var, indices, updates, reduce: str, axis: int):
        """ONNX symbolic for scatter."""
        y = g.op(
            "Custom",
            var,
            indices,
            updates,
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


def scatter(var, indices, updates, reduce: str = "update", axis: int = -2):
    return _ScatterCustom.apply(var, indices, updates, str(reduce), int(axis))



def _make_additive_causal_mask(attention_mask, q_len, k_len, past_len, dtype):
    """
    Make additive causal mask for Qwen3-1.7B inference.
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


def _text_attn_forward(
    attn_mod, hidden_states, cos4, sin4, attention_mask, cache_pos, past_key, past_value
):
    """
    Text attention forward function for Qwen3-1.7B inference.
    """
    input_shape = hidden_states.shape[:-1]
    head_dim = attn_mod.head_dim
    num_heads = attn_mod.config.num_attention_heads
    num_kv_heads = attn_mod.config.num_key_value_heads
    hidden_shape = (*input_shape, -1, head_dim)

    q_w = attn_mod.q_proj.weight
    k_w = attn_mod.k_proj.weight
    v_w = attn_mod.v_proj.weight
    q_b = attn_mod.q_proj.bias
    k_b = attn_mod.k_proj.bias
    v_b = attn_mod.v_proj.bias

    w = torch.cat([q_w, k_w, v_w], dim=0)
    if q_b is None:
        b = None
    else:
        b = torch.cat([q_b, k_b, v_b], dim=0)

    q_out_features = int(q_w.shape[0])
    kv_out_features = int(k_w.shape[0])
    qkv = F.linear(hidden_states, w, b)
    q_lin = qkv[..., :q_out_features]
    k_lin = qkv[..., q_out_features : q_out_features + kv_out_features]
    v_lin = qkv[..., q_out_features + kv_out_features :]

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

    if past_key is None:
        query_states = rotary_mul(query_states, cos4, sin4)
        key_states = rotary_mul(key_states, cos4, sin4)
    else:
        query_states = rotary_mul(query_states, cos4, sin4)
        key_states = rotary_mul(key_states, cos4, sin4)

    if past_key is not None:
        pos = cache_pos
        if pos is None:
            raise RuntimeError("cache_pos is required when past_key_values is provided.")
        if pos.dim() == 2:
            pos = pos[:, -1]
        key_cache = scatter(past_key, pos, key_states, reduce="update", axis=-2)
        value_cache = scatter(past_value, pos, value_states, reduce="update", axis=-2)
        key_states = key_cache
        value_states = value_cache

    scaling = getattr(attn_mod, "scaling", 1.0 / (head_dim**0.5))
    if past_key is None:
        q = query_states
        k = key_states
        v = value_states
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        if 0 < num_kv_heads < num_heads:
            rep = num_heads // num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        attn = torch.matmul(q, k.transpose(2, 3)) * float(scaling)
        q_len = attn.shape[-2]
        k_len = attn.shape[-1]
        flash_mask = _make_flash_attn_mask(attention_mask, q_len, k_len, 0)
        if flash_mask.dim() == 4 and flash_mask.shape[1] == 1:
            flash_mask = flash_mask.expand(attn.shape[0], attn.shape[1], flash_mask.shape[2], flash_mask.shape[3])
        attn = attn.masked_fill(flash_mask.to(torch.bool), torch.finfo(attn.dtype).min)
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn, v).permute(0, 2, 1, 3)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
    else:
        pad_mask = attention_mask[:, None, None, :].to(torch.bool).logical_not()
        attn_output = incre_flash_attention(
            query_states,
            key_states,
            value_states,
            pad_mask,
            num_heads=num_heads,
            scale_value=float(scaling),
            input_layout="BNSD",
            num_key_value_heads=num_kv_heads,
            inner_precise=1,
        )
    if past_key is None:
        attn_output = attn_output.reshape(*input_shape, -1)
    else:
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1)
    attn_output = attn_mod.o_proj(attn_output)
    return attn_output, key_states, value_states


def _rms_norm_layer(norm_mod, x):
    gamma = norm_mod.weight
    eps = getattr(norm_mod, "variance_epsilon", 1e-6)
    y, _ = rms_norm(x, gamma, eps)
    return y


def _mlp_gate_up_linear(mlp_mod, x):
    """
    Merge gate_proj and up_proj into a single linear and split outputs.
    """
    gate_w = mlp_mod.gate_proj.weight
    up_w = mlp_mod.up_proj.weight
    gate_b = mlp_mod.gate_proj.bias
    up_b = mlp_mod.up_proj.bias
    w = torch.cat([gate_w, up_w], dim=0)
    if gate_b is None:
        b = None
    else:
        b = torch.cat([gate_b, up_b], dim=0)
    y = F.linear(x, w, b)
    gate_out_features = int(gate_w.shape[0])
    gate = y[..., :gate_out_features]
    up = y[..., gate_out_features:]
    return gate, up


class Qwen3LlmPrefill(torch.nn.Module):
    """Qwen3-1.7B LLM Prefill wrapper."""

    def __init__(self, model, lm_head):
        """
        Qwen3-1.7B LLM Prefill wrapper.
        """
        super().__init__()
        self.model = model.model
        self.lm_head = lm_head

    def forward(self, input_ids, attention_mask, position_ids):
        """
        Prefill forward function for Qwen3-1.7B inference.
        """
        inputs_embeds = self.model.embed_tokens(input_ids)

        cos, sin = self.model.rotary_emb(inputs_embeds, position_ids)
        cos4 = cos.unsqueeze(2) if cos.dim() == 3 else cos
        sin4 = sin.unsqueeze(2) if sin.dim() == 3 else sin
        hidden_states = inputs_embeds
        present_k = []
        present_v = []

        for layer in self.model.layers:
            residual = hidden_states
            hidden_states = _rms_norm_layer(layer.input_layernorm, hidden_states)
            attn_out, pk, pv = _text_attn_forward(
                layer.self_attn,
                hidden_states,
                cos4,
                sin4,
                attention_mask,
                None,
                None,
                None,
            )
            pk = torch.cat(
                [pk, pk.new_zeros(pk.shape[0], pk.shape[1], KV_CACHE_LEN, pk.shape[3])],
                dim=2,
            )[:, :, :KV_CACHE_LEN, :]
            pv = torch.cat(
                [pv, pv.new_zeros(pv.shape[0], pv.shape[1], KV_CACHE_LEN, pv.shape[3])],
                dim=2,
            )[:, :, :KV_CACHE_LEN, :]
            hidden_states = residual + attn_out
            residual = hidden_states
            hidden_states = _rms_norm_layer(layer.post_attention_layernorm, hidden_states)
            mlp = layer.mlp
            if hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj") and hasattr(mlp, "down_proj"):
                gate, up = _mlp_gate_up_linear(mlp, hidden_states)
                x = torch.cat([gate, up], dim=-1)
                mlp_out = mlp.down_proj(swiglu(x, dim=-1))
                hidden_states = residual + mlp_out
            else:
                hidden_states = residual + mlp(hidden_states)
            present_k.append(pk)
            present_v.append(pv)

        hidden_states = _rms_norm_layer(self.model.norm, hidden_states)
        logits = self.lm_head(hidden_states)
        present_k = torch.stack(present_k, dim=0)
        present_v = torch.stack(present_v, dim=0)
        return logits, present_k, present_v


class Qwen3LlmDecode(torch.nn.Module):
    """Qwen3-1.7B LLM Decode wrapper."""

    def __init__(self, model, lm_head):
        """
        Qwen3-1.7B LLM Decode wrapper.
        """
        super().__init__()
        self.model = model.model
        self.lm_head = lm_head

    def forward(self, input_ids, attention_mask, position_ids, past_key_cache, past_value_cache):
        """
        Decode forward function for Qwen3-1.7B inference.
        """
        inputs_embeds = self.model.embed_tokens(input_ids)

        cos, sin = self.model.rotary_emb(inputs_embeds, position_ids)
        cos4 = cos.unsqueeze(1) if cos.dim() == 3 else cos
        sin4 = sin.unsqueeze(1) if sin.dim() == 3 else sin
        hidden_states = inputs_embeds
        present_k = []
        present_v = []

        past_k_layers = past_key_cache.unbind(0)
        past_v_layers = past_value_cache.unbind(0)

        for i, layer in enumerate(self.model.layers):
            pk_in = past_k_layers[i]
            pv_in = past_v_layers[i]
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            attn_out, pk, pv = _text_attn_forward(
                layer.self_attn,
                hidden_states,
                cos4,
                sin4,
                attention_mask,
                position_ids,
                pk_in,
                pv_in,
            )
            hidden_states = residual + attn_out
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            mlp = layer.mlp
            if hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj") and hasattr(mlp, "down_proj"):
                gate, up = _mlp_gate_up_linear(mlp, hidden_states)
                x = torch.cat([gate, up], dim=-1)
                mlp_out = mlp.down_proj(swiglu(x, dim=-1))
                hidden_states = residual + mlp_out
            else:
                hidden_states = residual + mlp(hidden_states)
            present_k.append(pk)
            present_v.append(pv)

        hidden_states = _rms_norm_layer(self.model.norm, hidden_states)
        logits = self.lm_head(hidden_states)
        present_k = torch.stack(present_k, dim=0)
        present_v = torch.stack(present_v, dim=0)
        return logits, present_k, present_v


def _prepare_llm_modules(model, device: str):
    """
    Prepare LLM modules for Qwen3-1.7B inference.
    """
    lm_head = model.lm_head
    model.eval()
    lm_head.eval()
    model.to(device)
    lm_head.to(device)
    prefill = Qwen3LlmPrefill(model, lm_head).to(device).eval()
    decode = Qwen3LlmDecode(model, lm_head).to(device).eval()
    return prefill, decode, lm_head


def _get_kv_cache_config(model):
    """
    Get KV cache configuration for Qwen3-1.7B inference.
    """
    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    head_dim = getattr(
        model.config,
        "head_dim",
        model.config.hidden_size // model.config.num_attention_heads,
    )
    return num_layers, num_kv_heads, head_dim


def _prepare_output_paths(output_dir):
    """
    Prepare output paths for Qwen3-1.7B inference.
    """
    prefill_dir = Path(output_dir) / "prefill"
    decode_dir = Path(output_dir) / "decode"
    prefill_dir.mkdir(parents=True, exist_ok=True)
    decode_dir.mkdir(parents=True, exist_ok=True)
    prefill_path = prefill_dir / "qwen3_1_7b_llm_prefill.onnx"
    decode_path = decode_dir / "qwen3_1_7b_llm_decode.onnx"
    return prefill_path, decode_path


def _create_prefill_dummy_inputs(device: str, dummy_seq_len: int):
    """
    Create dummy inputs for Qwen3-1.7B prefill.
    """
    dummy_seq = int(dummy_seq_len)
    dummy_input_ids = torch.randint(
        0, 1000, (1, dummy_seq), dtype=torch.int64, device=device
    )
    dummy_attention_mask = torch.ones(1, dummy_seq, dtype=torch.int64, device=device)
    dummy_position_ids = torch.arange(dummy_seq, device=device, dtype=torch.int64).view(
        1, -1
    )
    return dummy_seq, dummy_input_ids, dummy_attention_mask, dummy_position_ids


def _export_prefill_onnx(prefill, prefill_path: Path, dummy_inputs, use_dynamo: bool):
    """
    Export LLM prefill to ONNX format.
    """
    print(f"Exporting LLM prefill to {prefill_path}...")
    with torch.no_grad():
        torch.onnx.export(
            prefill,
            dummy_inputs,
            str(prefill_path),
            input_names=["input_ids", "attention_mask", "position_ids"],
            output_names=["logits", "present_key_cache", "present_value_cache"],
            opset_version=18,
            do_constant_folding=True,
            dynamo=use_dynamo,
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "position_ids": {0: "batch", 1: "seq"},
                "logits": {0: "batch", 1: "seq"},
                "present_key_cache": {1: "batch"},
                "present_value_cache": {1: "batch"},
            },
        )
    print("LLM prefill exported successfully.")


def _create_decode_dummy_inputs(
    device: str,
    dummy_seq: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    kv_dtype: torch.dtype,
):
    """
    Create dummy inputs for Qwen3-1.7B decode.
    """
    del dummy_seq
    dummy_step = 1
    dummy_past_len = int(KV_CACHE_LEN)
    dummy_input_ids_step = torch.randint(
        0, 1000, (1, dummy_step), dtype=torch.int64, device=device
    )
    dummy_attention_mask_step = torch.ones(1, dummy_past_len, dtype=torch.int64, device=device)
    dummy_position_ids_step = torch.tensor(
        [[dummy_past_len - 1]], dtype=torch.int64, device=device
    )
    dummy_k = torch.zeros(
        num_layers, 1, num_kv_heads, dummy_past_len, head_dim, dtype=kv_dtype, device=device
    )
    dummy_v = torch.zeros(
        num_layers, 1, num_kv_heads, dummy_past_len, head_dim, dtype=kv_dtype, device=device
    )
    return (
        dummy_input_ids_step,
        dummy_attention_mask_step,
        dummy_position_ids_step,
        dummy_k,
        dummy_v,
    )


def _export_decode_onnx(decode, decode_path: Path, dummy_inputs, use_dynamo: bool):
    """
    Export LLM decode to ONNX format.
    """
    print(f"Exporting LLM decode to {decode_path}...")
    with torch.no_grad():
        torch.onnx.export(
            decode,
            dummy_inputs,
            str(decode_path),
            input_names=[
                "input_ids",
                "attention_mask",
                "position_ids",
                "past_key_cache",
                "past_value_cache",
            ],
            output_names=["logits", "present_key_cache", "present_value_cache"],
            opset_version=18,
            do_constant_folding=True,
            dynamo=use_dynamo,
            dynamic_axes={
                "input_ids": {0: "batch"},
                "attention_mask": {0: "batch"},
                "position_ids": {0: "batch"},
                "logits": {0: "batch"},
                "past_key_cache": {1: "batch"},
                "past_value_cache": {1: "batch"},
                "present_key_cache": {1: "batch"},
                "present_value_cache": {1: "batch"},
            },
        )
    print("LLM decode exported successfully.")


def export_llm_prefill_decode(
    model, output_dir, device="cpu", dummy_seq_len=8, use_dynamo=False
):
    """
    Export Qwen3-1.7B model to ONNX format.
    """
    prefill, decode, _ = _prepare_llm_modules(model, device=device)
    kv_dtype = next(model.parameters()).dtype
    num_layers, num_kv_heads, head_dim = _get_kv_cache_config(model)
    prefill_path, decode_path = _prepare_output_paths(output_dir)

    dummy_seq, dummy_input_ids, dummy_attention_mask, dummy_position_ids = (
        _create_prefill_dummy_inputs(device=device, dummy_seq_len=dummy_seq_len)
    )
    _export_prefill_onnx(
        prefill=prefill,
        prefill_path=prefill_path,
        dummy_inputs=(dummy_input_ids, dummy_attention_mask, dummy_position_ids),
        use_dynamo=use_dynamo,
    )

    decode_dummy_inputs = _create_decode_dummy_inputs(
        device=device,
        dummy_seq=dummy_seq,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        kv_dtype=kv_dtype,
    )
    _export_decode_onnx(
        decode=decode,
        decode_path=decode_path,
        dummy_inputs=decode_dummy_inputs,
        use_dynamo=use_dynamo,
    )


def main():
    """
    Main function to export Qwen3-1.7B model to ONNX format.
    """
    parser = argparse.ArgumentParser(description="Export Qwen3-1.7B to ONNX")
    parser.add_argument(
        "--model-id",
        type=str,
        default="./Qwen3-1.7B",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./qwen3_1_7b_onnx", help="Output directory"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device for export (cpu or cuda)"
    )
    parser.add_argument(
        "--dummy-seq-len", type=int, default=8, help="Dummy sequence length for export"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Export dtype",
    )
    parser.add_argument(
        "--use-dynamo", action="store_true", help="Use torch dynamo exporter path"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading model {args.model_id} for export (dtype={args.dtype})...")
    if args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=False,
        attn_implementation="eager",
    ).to(device)

    export_llm_prefill_decode(
        model,
        output_dir,
        str(device),
        args.dummy_seq_len,
        args.use_dynamo,
    )

    print("Clearing memory after export...")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nExport finished. Files saved in {args.output_dir}")


if __name__ == "__main__":
    main()
