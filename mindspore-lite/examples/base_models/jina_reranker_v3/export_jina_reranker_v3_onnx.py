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
Export Jina Reranker V3 to ONNX format (Listwise + Pointwise).

This script exports the JinaForRanking model (built on Qwen3-0.6B) with its
native "last but not late interaction" architecture. The ONNX model supports
both listwise (multiple docs per forward) and pointwise (one doc per forward)
scoring modes via the same graph.

Key design: instead of boolean indexing to find special token positions (which
is not ONNX-exportable with dynamic shapes), we pre-compute the positions of
<|embed_token|> and <|rerank_token|> and pass them as explicit inputs. The
model uses torch.gather to extract hidden states at those positions, then
projects them through the MLP projector and computes cosine similarity scores.

By default, the export replaces key subgraphs with CANN fused Custom operators
for Ascend backend performance. Use --disable-fusion-opt to export a pure ONNX
model for ONNX Runtime inference.

The fused Custom operators include:
  - RotaryMul:   rotate_half + cos/sin multiply -> Custom(RotaryMul)
  - PromptFlashAttention: QK^T+softmax+V        -> Custom(PromptFlashAttention)
  - SwiGlu:      SiLU(gate)*up                  -> Custom(SwiGlu)
"""

import argparse
import os
from collections import Counter

import torch
import torch.nn.functional as F
import onnx
from transformers import AutoModel, AutoTokenizer


DOC_EMBED_TOKEN_ID = 151670
QUERY_EMBED_TOKEN_ID = 151671
MAX_DOCS = 64

_ONNX_DYNAMIC_EXPORT = False

THINK_OPEN = "\u003cthink\u003e"
THINK_CLOSE = "\u003c/think\u003e"
NO_THINK_SUFFIX = THINK_OPEN + "\n\n" + THINK_CLOSE + "\n\n"


# ---------------------------------------------------------------------------
# CANN Custom Op implementations for ONNX export
# ---------------------------------------------------------------------------


class _CannRmsNorm(torch.autograd.Function):
    """torch.autograd.Function for exporting a CANN RmsNorm Custom op to ONNX."""

    @staticmethod
    def forward(ctx, x, gamma, epsilon):
        del ctx
        eps = float(epsilon)
        x_fp32 = x.to(torch.float32)
        rstd = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)
        y = (x_fp32 * rstd) * gamma.to(torch.float32)
        return y.to(x.dtype), rstd

    @staticmethod
    def symbolic(g, x, gamma, epsilon):
        """Export the fused RmsNorm Custom op node."""
        sizes = x.type().sizes()
        if sizes is None:
            out_shapes = ""
        else:
            dims = [int(d) if d is not None else -1 for d in list(sizes)]
            if len(dims) == 3:
                dims[0] = -1
                dims[1] = -1
            elif len(dims) == 4:
                dims[0] = -1
                dims[2] = -1
            out_shapes = ",".join([str(len(dims))] + [str(i) for i in dims])

        y, rstd = g.op(
            "Custom",
            x,
            gamma,
            type_s="RmsNorm",
            input_names_s=["x", "gamma"],
            optional_input_names_s=[],
            output_names_s=["y", "rstd"],
            output_num_i=2,
            input_index_i=[0, 1],
            epsilon_f=float(epsilon),
            output_shapes_s=out_shapes,
            outputs=2,
        )
        y.setType(x.type())
        rstd.setType(x.type())
        return y, rstd


class CannRmsNorm(torch.nn.Module):
    """Module wrapper that applies the CANN RmsNorm Custom op."""

    def __init__(self, weight, epsilon):
        super().__init__()
        self.weight = weight
        self.epsilon = float(epsilon)

    def forward(self, x):
        """Apply fused RmsNorm to the input tensor."""
        y, _ = _CannRmsNorm.apply(x, self.weight, self.epsilon)
        return y


class _CannAddRmsNorm(torch.autograd.Function):
    """torch.autograd.Function for exporting a CANN AddRmsNorm Custom op to ONNX."""

    @staticmethod
    def forward(ctx, x1, x2, gamma, epsilon):
        del ctx
        eps = float(epsilon)
        x = x1 + x2
        x_fp32 = x.to(torch.float32)
        rstd = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)
        y = (x_fp32 * rstd) * gamma.to(torch.float32)
        return y.to(x.dtype), rstd, x

    @staticmethod
    def symbolic(g, x1, x2, gamma, epsilon):
        """Export the fused AddRmsNorm Custom op node."""
        sizes = x1.type().sizes()
        if sizes is None:
            out_shapes = ""
        else:
            dims = [int(d) if d is not None else -1 for d in list(sizes)]
            if len(dims) == 3:
                dims[0] = -1
                dims[1] = -1
            elif len(dims) == 4:
                dims[0] = -1
                dims[2] = -1
            out_shapes = ",".join([str(len(dims))] + [str(i) for i in dims])

        y, rstd, x = g.op(
            "Custom",
            x1,
            x2,
            gamma,
            type_s="AddRmsNorm",
            input_names_s=["x1", "x2", "gamma"],
            optional_input_names_s=[],
            output_names_s=["y", "rstd", "x"],
            output_num_i=3,
            input_index_i=[0, 1, 2],
            epsilon_f=float(epsilon),
            output_shapes_s=out_shapes,
            outputs=3,
        )
        y.setType(x1.type())
        rstd.setType(x1.type())
        x.setType(x1.type())
        return y, rstd, x


class _CannRotaryMul(torch.autograd.Function):
    """torch.autograd.Function for exporting a CANN RotaryMul Custom op to ONNX."""

    @staticmethod
    def forward(ctx, x, r1, r2):
        """Run RotaryMul reference implementation for tracing."""
        del ctx
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        rotated = torch.cat([-x2, x1], dim=-1)
        y = x * r1 + rotated * r2
        return y

    @staticmethod
    def symbolic(g, x, r1, r2):
        """Export the fused RotaryMul Custom op node."""
        sizes = x.type().sizes()
        if sizes is None:
            out_shapes = ""
        else:
            dims = [int(d) if d is not None else -1 for d in list(sizes)]
            if len(dims) == 4:
                dims[0] = -1
                dims[2] = -1
            out_shapes = ",".join([str(len(dims))] + [str(i) for i in dims])

        y = g.op(
            "Custom",
            x,
            r1,
            r2,
            type_s="RotaryMul",
            input_names_s=["x", "r1", "r2"],
            optional_input_names_s=[],
            output_names_s=["y"],
            output_num_i=1,
            input_index_i=[0, 1, 2],
            output_shapes_s=out_shapes,
        )
        y.setType(x.type())
        return y


class _CannPromptFlashAttention(torch.autograd.Function):
    """torch.autograd.Function for exporting a CANN PromptFlashAttention Custom op to ONNX."""

    @staticmethod
    def forward(ctx, query, key, value, atten_mask, num_heads, num_key_value_heads, scale_value):
        """Run PromptFlashAttention reference implementation for tracing."""
        del ctx
        if int(num_key_value_heads) != int(num_heads):
            repeat = int(num_heads) // int(num_key_value_heads)
            key = key.repeat_interleave(repeat, dim=1)
            value = value.repeat_interleave(repeat, dim=1)
        scale = float(scale_value)
        attn = torch.matmul(query, key.transpose(2, 3)) * scale
        if atten_mask is not None:
            if atten_mask.dtype == torch.bool:
                mask_value = torch.finfo(query.dtype).min
                attn = torch.where(atten_mask, torch.full_like(attn, mask_value), attn)
            else:
                attn = attn + atten_mask
        attn = torch.nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(query.dtype)
        return torch.matmul(attn, value)

    @staticmethod
    def symbolic(g, query, key, value, atten_mask, num_heads, num_key_value_heads, scale_value):
        """Export the fused PromptFlashAttention Custom op node."""
        y = g.op(
            "Custom",
            query,
            key,
            value,
            atten_mask,
            type_s="PromptFlashAttention",
            input_names_s=["query", "key", "value", "atten_mask"],
            output_names_s=["attention_out"],
            num_heads_i=int(num_heads),
            num_key_value_heads_i=int(num_key_value_heads),
            scale_value_f=float(scale_value),
            input_layout_s="BNSD_BSND",
            inner_precise_i=1,
        )
        y.setType(query.type())
        return y


class _CannSwiGlu(torch.autograd.Function):
    """torch.autograd.Function for exporting a CANN SwiGlu Custom op to ONNX."""

    @staticmethod
    def forward(ctx, x, dim):
        """Run SwiGlu reference implementation for tracing."""
        del ctx
        d = int(dim)
        split = x.shape[d] // 2
        a, b = torch.split(x, [split, split], dim=d)
        return torch.nn.functional.silu(a) * b

    @staticmethod
    def symbolic(g, x, dim):
        """Export the fused SwiGlu Custom op node."""
        sizes = x.type().sizes()
        if sizes is None:
            out_shapes = ""
        else:
            dims = [int(d) if d is not None else -1 for d in list(sizes)]
            if len(dims) == 3:
                dims[0] = -1
                dims[1] = -1
            out_shapes = ",".join([str(len(dims))] + [str(i) for i in dims])

        y = g.op(
            "Custom",
            x,
            type_s="SwiGlu",
            input_names_s=["x"],
            optional_input_names_s=[],
            output_names_s=["y"],
            output_num_i=1,
            input_index_i=[0],
            dim_i=int(dim),
            output_shapes_s=out_shapes,
        )
        y.setType(x.type())
        return y


class _CannMatMulV2(torch.autograd.Function):
    """torch.autograd.Function for exporting a CANN MatMulV2 Custom op to ONNX."""

    @staticmethod
    def forward(ctx, x1, x2):
        del ctx
        return torch.matmul(x1, x2)

    @staticmethod
    def symbolic(g, x1, x2):
        """Export the fused MatMulV2 Custom op node."""
        x1_sizes = x1.type().sizes()
        x2_sizes = x2.type().sizes()
        if x1_sizes is None or x2_sizes is None:
            out_shapes = ""
        else:
            m = x1_sizes[0]
            n = x2_sizes[1] if len(x2_sizes) >= 2 else None
            dims = [
                int(m) if m is not None else -1,
                int(n) if n is not None else -1,
            ]
            out_shapes = ",".join(["2"] + [str(i) for i in dims])

        y = g.op(
            "Custom",
            x1,
            x2,
            type_s="MatMulV2",
            input_names_s=["x1", "x2"],
            optional_input_names_s=[],
            output_names_s=["y"],
            output_num_i=1,
            input_index_i=[0, 1],
            output_shapes_s=out_shapes,
        )
        y.setType(x1.type())
        return y


# ---------------------------------------------------------------------------
# CANN-fused forward helpers
# ---------------------------------------------------------------------------

def _cann_rotary_mul(x, cos, sin):
    return _CannRotaryMul.apply(x, cos, sin)


def _expand_rotary_cos_sin(cos, sin, target_dim):
    if cos.dim() != int(target_dim):
        while cos.dim() < int(target_dim):
            cos = cos.unsqueeze(1)
        while sin.dim() < int(target_dim):
            sin = sin.unsqueeze(1)
    return cos, sin


def _cann_apply_rotary_pos_emb(query, key, cos, sin):
    query_out = _cann_rotary_mul(query, cos, sin)
    key_out = _cann_rotary_mul(key, cos, sin)
    return query_out, key_out


def _linear_2d(linear_mod, x):
    """
    Apply a Linear module with an explicit 2D matmul to avoid BatchMatMul lowering.

    Converts input (..., in_features) to (-1, in_features), applies MatMul + bias,
    then reshapes back to (..., out_features).
    """
    if x.dim() == 2:
        out = _CannMatMulV2.apply(x, linear_mod.weight.t())
        if linear_mod.bias is not None:
            out = out + linear_mod.bias
        return out

    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1])
    out2d = _CannMatMulV2.apply(x2d, linear_mod.weight.t())
    if linear_mod.bias is not None:
        out2d = out2d + linear_mod.bias
    return out2d.reshape(*orig_shape[:-1], -1)


def _linear(linear_mod, x, enable_bmm2mm_fusion):
    if enable_bmm2mm_fusion:
        return _linear_2d(linear_mod, x)
    x16 = x.to(torch.float16)
    w16 = linear_mod.weight.to(torch.float16)
    out = torch.matmul(x16, w16.t())
    if linear_mod.bias is not None:
        out = out + linear_mod.bias.to(torch.float16)
    return out.to(x.dtype)


def _apply_projector(projector, x, enable_bmm2mm_fusion):
    """
    Apply projector to x while routing Linear layers through _linear_2d.
    """
    if isinstance(projector, torch.nn.Sequential):
        seq = projector
    elif hasattr(projector, "projector") and isinstance(projector.projector, torch.nn.Sequential):
        seq = projector.projector
    else:
        return projector(x)

    out = x
    for mod in seq:
        if isinstance(mod, torch.nn.Linear):
            out = _linear(mod, out, enable_bmm2mm_fusion)
        else:
            out = mod(out)
    return out


def _qwen3_rotary_emb_matmul2d(rotary_emb, x, position_ids):
    """Compute Qwen3 rotary embeddings with matmul-friendly tensor shapes."""
    if "dynamic" in getattr(rotary_emb, "rope_type", ""):
        seq_len = torch.max(position_ids) + 1
        if seq_len > rotary_emb.max_seq_len_cached:
            inv_freq, rotary_emb.attention_scaling = rotary_emb.rope_init_fn(
                rotary_emb.config, x.device, seq_len=seq_len
            )
            rotary_emb.register_buffer("inv_freq", inv_freq, persistent=False)
            rotary_emb.max_seq_len_cached = seq_len
        if (
            seq_len < rotary_emb.original_max_seq_len
            and rotary_emb.max_seq_len_cached > rotary_emb.original_max_seq_len
        ):
            rotary_emb.original_inv_freq = rotary_emb.original_inv_freq.to(x.device)
            rotary_emb.register_buffer("inv_freq", rotary_emb.original_inv_freq, persistent=False)
            rotary_emb.max_seq_len_cached = rotary_emb.original_max_seq_len
    elif getattr(rotary_emb, "rope_type", "") == "longrope":
        seq_len = torch.max(position_ids) + 1
        if hasattr(rotary_emb.config, "original_max_position_embeddings"):
            original_max_position_embeddings = rotary_emb.config.original_max_position_embeddings
        else:
            original_max_position_embeddings = rotary_emb.config.max_position_embeddings
        if seq_len > original_max_position_embeddings:
            if not hasattr(rotary_emb, "long_inv_freq"):
                rotary_emb.long_inv_freq, _ = rotary_emb.rope_init_fn(
                    rotary_emb.config, x.device, seq_len=original_max_position_embeddings + 1
                )
            rotary_emb.register_buffer("inv_freq", rotary_emb.long_inv_freq, persistent=False)
        else:
            rotary_emb.original_inv_freq = rotary_emb.original_inv_freq.to(x.device)
            rotary_emb.register_buffer("inv_freq", rotary_emb.original_inv_freq, persistent=False)

    inv_freq = rotary_emb.inv_freq.to(device=x.device, dtype=torch.float32)
    position_ids_f = position_ids.to(dtype=torch.float32)
    bsz, seq_len = position_ids_f.shape
    freqs = (position_ids_f.reshape(-1, 1) @ inv_freq.reshape(1, -1)).reshape(bsz, seq_len, -1)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * rotary_emb.attention_scaling
    sin = emb.sin() * rotary_emb.attention_scaling
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def _make_bool_causal_mask(attention_mask, q_len, k_len, past_len):
    """Build a boolean causal mask (True=allowed) with padding mask applied."""
    batch_size = attention_mask.shape[0]
    ar_q = torch.arange(q_len, device=attention_mask.device)
    ar_k = torch.arange(k_len, device=attention_mask.device)
    causal_mask = ar_k[None, :] <= (past_len + ar_q[:, None])
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    causal_mask = causal_mask.repeat(batch_size, 1, 1, 1)
    padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    padding_mask = padding_mask.expand(batch_size, 1, q_len, k_len)
    return causal_mask & padding_mask.to(torch.bool)


def _make_bool_causal_mask_padded(attention_mask, seq_len, padded_len):
    """Build a boolean causal mask (True=masked) with internal padding to padded_len."""
    pad_len = int(padded_len - seq_len)
    if pad_len > 0:
        attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_len), value=0)
    batch_size = attention_mask.shape[0]
    ar = torch.arange(int(padded_len), device=attention_mask.device)
    causal = ar[None, :] <= ar[:, None]
    causal = causal.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    k_valid = attention_mask[:, None, None, :].to(torch.bool)
    q_valid = attention_mask[:, None, :, None].to(torch.bool)
    return ~(causal & k_valid & q_valid)


def _make_bool_causal_mask_dynamic(attention_mask):
    """Build a boolean causal mask (True=masked) without relying on static lengths."""
    batch_size = attention_mask.shape[0]
    ar = torch.ones_like(attention_mask, dtype=torch.int32).cumsum(dim=1) - 1
    ar = ar[0]
    causal = ar[None, :] <= ar[:, None]
    causal = causal.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    k_valid = attention_mask[:, None, None, :].to(torch.bool)
    q_valid = attention_mask[:, None, :, None].to(torch.bool)
    return ~(causal & k_valid & q_valid)



def _make_additive_causal_mask(attention_mask, q_len, k_len, past_len, dtype):
    """Build an additive causal mask (dtype) compatible with non-fused listwise export."""
    mask_value = torch.finfo(dtype).min
    ar_q = torch.arange(q_len, device=attention_mask.device)
    ar_k = torch.arange(k_len, device=attention_mask.device)
    causal = ar_k[None, :] > (past_len + ar_q[:, None])
    causal = causal.to(dtype) * mask_value
    causal = causal[None, None, :, :].expand(attention_mask.shape[0], 1, q_len, k_len)
    padding = (1.0 - attention_mask.to(dtype)) * mask_value
    padding = padding[:, None, None, :]
    return causal + padding


def _prepare_pfa_mask(attention_mask, seq_len):
    if _ONNX_DYNAMIC_EXPORT:
        return _make_bool_causal_mask_dynamic(attention_mask).to(torch.bool), 0, int(seq_len)
    padded_len = ((int(seq_len) + 127) // 128) * 128
    pad_len = int(padded_len - seq_len)
    bool_mask = _make_bool_causal_mask_padded(attention_mask, seq_len, padded_len).to(torch.bool)
    return bool_mask, pad_len, padded_len


def _cann_attn_forward(
    attn_mod,
    hidden_states,
    position_embeddings,
    seq_len,
    enable_bmm2mm_fusion,
    bool_mask,
    pad_len,
    padded_len,
):
    """Attention forward that routes the core softmax to CANN PromptFlashAttention Custom op."""
    del padded_len
    input_shape = hidden_states.shape[:-1]
    orig_dtype = hidden_states.dtype
    head_dim = attn_mod.head_dim
    num_heads = attn_mod.config.num_attention_heads
    num_kv_heads = attn_mod.config.num_key_value_heads
    hidden_shape = (*input_shape, -1, head_dim)
    hidden_states_fp16 = hidden_states.to(torch.float16)

    value_states = _linear(attn_mod.v_proj, hidden_states_fp16, enable_bmm2mm_fusion).view(hidden_shape)
    if hasattr(attn_mod, "qk_proj"):
        qk_states = _linear(attn_mod.qk_proj, hidden_states_fp16, enable_bmm2mm_fusion)
        q_size = int(num_heads) * int(head_dim)
        k_size = int(num_kv_heads) * int(head_dim)
        query_states, key_states = torch.split(qk_states, [q_size, k_size], dim=-1)
        query_states = query_states.view(hidden_shape)
        key_states = key_states.view(hidden_shape)
    else:
        query_states = _linear(attn_mod.q_proj, hidden_states_fp16, enable_bmm2mm_fusion).view(hidden_shape)
        key_states = _linear(attn_mod.k_proj, hidden_states_fp16, enable_bmm2mm_fusion).view(hidden_shape)
    if hasattr(attn_mod, "q_norm"):
        query_states = attn_mod.q_norm(query_states)
    if hasattr(attn_mod, "k_norm"):
        key_states = attn_mod.k_norm(key_states)

    query_states = query_states.to(torch.float16)
    key_states = key_states.to(torch.float16)
    value_states = value_states.to(torch.float16)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = _cann_apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if not _ONNX_DYNAMIC_EXPORT and int(pad_len) > 0:
        pad_4d = (0, 0, 0, int(pad_len), 0, 0, 0, 0)
        query_states = torch.nn.functional.pad(query_states, pad_4d, value=0)
        key_states = torch.nn.functional.pad(key_states, pad_4d, value=0)
        value_states = torch.nn.functional.pad(value_states, pad_4d, value=0)

    scaling = getattr(attn_mod, "scaling", 1.0 / (head_dim ** 0.5))

    attn_output = _CannPromptFlashAttention.apply(
        query_states,
        key_states,
        value_states,
        bool_mask,
        int(num_heads),
        int(num_kv_heads),
        float(scaling),
    )
    if not _ONNX_DYNAMIC_EXPORT and int(pad_len) > 0:
        attn_output = attn_output[:, :seq_len, :, :]
    attn_output = attn_output.to(orig_dtype)
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = _linear(attn_mod.o_proj, attn_output, enable_bmm2mm_fusion)
    return attn_output


def _cann_mlp_forward(mlp_mod, hidden_states, enable_bmm2mm_fusion):
    """MLP forward that routes the SwiGlu activation to CANN SwiGlu Custom op."""
    gate_up = torch.cat(
        [
            _linear(mlp_mod.gate_proj, hidden_states, enable_bmm2mm_fusion),
            _linear(mlp_mod.up_proj, hidden_states, enable_bmm2mm_fusion),
        ],
        dim=-1,
    )
    gate_up = _CannSwiGlu.apply(gate_up, -1)
    return _linear(mlp_mod.down_proj, gate_up, enable_bmm2mm_fusion)


def _get_rmsnorm_epsilon(norm_mod):
    for attr in ("variance_epsilon", "eps", "epsilon"):
        val = getattr(norm_mod, attr, None)
        if val is not None:
            return float(val)
    return 1e-6


def _replace_rmsnorm_with_cann(model):
    """Replace Qwen3 RMSNorm modules with CANN-exportable wrappers."""
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
    except Exception:
        return

    for layer in model.model.layers:
        for name in ("input_layernorm", "post_attention_layernorm"):
            mod = getattr(layer, name, None)
            if isinstance(mod, Qwen3RMSNorm) and hasattr(mod, "weight"):
                setattr(layer, name, CannRmsNorm(mod.weight, _get_rmsnorm_epsilon(mod)))
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            for name in ("q_norm", "k_norm"):
                mod = getattr(attn, name, None)
                if isinstance(mod, Qwen3RMSNorm) and hasattr(mod, "weight"):
                    setattr(attn, name, CannRmsNorm(mod.weight, _get_rmsnorm_epsilon(mod)))

    mod = getattr(model.model, "norm", None)
    if isinstance(mod, Qwen3RMSNorm) and hasattr(mod, "weight"):
        model.model.norm = CannRmsNorm(mod.weight, _get_rmsnorm_epsilon(mod))


def _cann_add_rms_norm(residual, hidden_states, norm_mod, enable_rmsnorm_fusion):
    if enable_rmsnorm_fusion:
        eps = _get_rmsnorm_epsilon(norm_mod)
        y, _, x = _CannAddRmsNorm.apply(residual, hidden_states, norm_mod.weight, eps)
        return y, x
    x = residual + hidden_states
    y = norm_mod(x)
    return y, x


# ---------------------------------------------------------------------------
# Wrapper models
# ---------------------------------------------------------------------------

class JinaRerankerV3ListwiseWrapper(torch.nn.Module):
    """
    Wrapper for Jina Reranker V3 listwise ONNX export (no fusion).
    """

    def __init__(self, model):
        super().__init__()
        self.backbone = model.model
        self.projector = model.projector

    def forward(self, input_ids, attention_mask, doc_token_indices, query_token_index):
        """Compute listwise scores with the original (non-fused) backbone."""
        seq_len = input_ids.shape[1]
        cache_position = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)

        attn_mask = _make_additive_causal_mask(
            attention_mask, seq_len, seq_len, 0, torch.float32
        )

        causal_mask_mapping = {"full_attention": attn_mask}

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=causal_mask_mapping,
            cache_position=cache_position,
            use_cache=False,
        )
        hidden_states = outputs.last_hidden_state
        _, _, dim = hidden_states.shape

        doc_idx = doc_token_indices.unsqueeze(-1).expand(-1, -1, dim)
        doc_embeds = torch.gather(hidden_states, 1, doc_idx)

        query_idx = query_token_index.unsqueeze(-1).expand(-1, -1, dim)
        query_embeds = torch.gather(hidden_states, 1, query_idx)

        doc_embeds = self.projector(doc_embeds)
        query_embeds = self.projector(query_embeds)

        query_embeds_expanded = query_embeds.expand_as(doc_embeds)
        scores = F.cosine_similarity(doc_embeds, query_embeds_expanded, dim=-1)

        return scores


class JinaRerankerV3FusedWrapper(torch.nn.Module):
    """
    Wrapper for Jina Reranker V3 listwise ONNX export with CANN fused ops.

    Manually unrolls the Qwen3Model forward pass so that each subgraph
    (RotaryMul, PromptFlashAttention, SwiGlu) is replaced by the corresponding
    CANN Custom operator.
    """

    def __init__(
        self,
        model,
        enable_bmm2mm_fusion,
        enable_rmsnorm_fusion,
        enable_qk_merge,
    ):
        super().__init__()
        self.embed_tokens = model.model.embed_tokens
        self.layers = model.model.layers
        self.norm = model.model.norm
        self.rotary_emb = model.model.rotary_emb
        self.projector = model.projector
        self.enable_bmm2mm_fusion = bool(enable_bmm2mm_fusion)
        self.enable_rmsnorm_fusion = bool(enable_rmsnorm_fusion)
        self.enable_qk_merge = bool(enable_qk_merge)
        if self.enable_qk_merge:
            self._enable_qk_merge()

    def _enable_qk_merge(self):
        """Merge query and key projection weights for fused export."""
        for layer in self.layers:
            attn = layer.self_attn
            if not (hasattr(attn, "q_proj") and hasattr(attn, "k_proj")):
                continue
            q_weight = attn.q_proj.weight
            k_weight = attn.k_proj.weight
            out_features = int(q_weight.shape[0] + k_weight.shape[0])
            in_features = int(q_weight.shape[1])

            bias_tensors = [attn.q_proj.bias, attn.k_proj.bias]
            use_bias = any(b is not None for b in bias_tensors)
            qk_proj = torch.nn.Linear(in_features, out_features, bias=use_bias).to(
                device=q_weight.device, dtype=q_weight.dtype
            )

            with torch.no_grad():
                qk_proj.weight.copy_(torch.cat([q_weight, k_weight], dim=0))
                if use_bias:
                    merged_bias = []
                    for w, b in zip([q_weight, k_weight], bias_tensors):
                        if b is None:
                            merged_bias.append(
                                torch.zeros(w.shape[0], device=w.device, dtype=w.dtype)
                            )
                        else:
                            merged_bias.append(b.to(dtype=w.dtype, device=w.device))
                    qk_proj.bias.copy_(torch.cat(merged_bias, dim=0))
            attn.qk_proj = qk_proj

    def forward(self, input_ids, attention_mask, doc_token_indices, query_token_index):
        """Compute listwise scores with CANN fused Custom ops."""
        seq_len = input_ids.shape[1]
        position_ids = attention_mask.to(torch.long).cumsum(dim=1) - 1
        position_ids = torch.where(attention_mask.to(torch.bool), position_ids, torch.zeros_like(position_ids))

        inputs_embeds = self.embed_tokens(input_ids)
        position_embeddings = _qwen3_rotary_emb_matmul2d(self.rotary_emb, inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        residual = hidden_states
        hidden_states = self.layers[0].input_layernorm(hidden_states)

        num_layers = len(self.layers)
        cos, sin = position_embeddings
        cos, sin = _expand_rotary_cos_sin(cos, sin, 4)
        cos = cos.to(torch.float16)
        sin = sin.to(torch.float16)
        position_embeddings = (cos, sin)
        bool_mask, pad_len, padded_len = _prepare_pfa_mask(attention_mask, seq_len)
        for i, layer in enumerate(self.layers):
            attn_out = _cann_attn_forward(
                layer.self_attn,
                hidden_states,
                position_embeddings,
                seq_len,
                self.enable_bmm2mm_fusion,
                bool_mask,
                pad_len,
                padded_len,
            )
            hidden_states, residual = _cann_add_rms_norm(
                residual, attn_out, layer.post_attention_layernorm, self.enable_rmsnorm_fusion
            )

            mlp_out = _cann_mlp_forward(layer.mlp, hidden_states, self.enable_bmm2mm_fusion)
            if i < num_layers - 1:
                hidden_states, residual = _cann_add_rms_norm(
                    residual,
                    mlp_out,
                    self.layers[i + 1].input_layernorm,
                    self.enable_rmsnorm_fusion,
                )
            else:
                hidden_states, _ = _cann_add_rms_norm(
                    residual, mlp_out, self.norm, self.enable_rmsnorm_fusion
                )

        _, _, dim = hidden_states.shape

        doc_idx = doc_token_indices.unsqueeze(-1).expand(-1, -1, dim)
        doc_embeds = torch.gather(hidden_states, 1, doc_idx)

        query_idx = query_token_index.unsqueeze(-1).expand(-1, -1, dim)
        query_embeds = torch.gather(hidden_states, 1, query_idx)

        doc_embeds = _apply_projector(self.projector, doc_embeds, self.enable_bmm2mm_fusion)
        query_embeds = _apply_projector(self.projector, query_embeds, self.enable_bmm2mm_fusion)

        query_embeds_expanded = query_embeds.expand_as(doc_embeds)
        scores = F.cosine_similarity(doc_embeds, query_embeds_expanded, dim=-1)
        return scores


# ---------------------------------------------------------------------------
# CLI, loading, export helpers
# ---------------------------------------------------------------------------

def _parse_args():
    """Parse CLI arguments for ONNX export."""
    parser = argparse.ArgumentParser(
        description="Export Jina Reranker V3 (Listwise) to ONNX"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="jinaai/jina-reranker-v3",
        help="Model ID or local path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./onnx",
        help="Output directory for ONNX model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu/cuda)",
    )
    parser.add_argument(
        "--disable-fusion-opt",
        "--disable_fusion_opt",
        dest="enable_fusion_opt",
        action="store_false",
        help="Disable fused Custom ops and export a pure ONNX runnable by ONNX Runtime.",
    )
    parser.set_defaults(enable_fusion_opt=True)
    parser.add_argument(
        "--enable_bmm2mm_fusion",
        action="store_true",
        help="Enable BMM->MMv2 optimization: route Linear through 2D MatMulV2 Custom op. Default: disabled.",
    )
    parser.add_argument(
        "--enable_rmsnorm_fusion",
        action="store_true",
        help="Enable RmsNorm/AddRmsNorm fusion Custom ops. Default: disabled.",
    )
    parser.add_argument(
        "--enable_qk_merge",
        action="store_true",
        help=(
            "Enable QK merge optimization: merge q_proj+k_proj into a single Linear "
            "(fused export only). Default: disabled."
        ),
    )
    return parser.parse_args()


def _load_model_and_tokenizer(args):
    """Load model and tokenizer from HuggingFace ID or local path."""
    print(f"Loading model from {args.model_id}")
    model = AutoModel.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    )
    return model, tokenizer


def _format_listwise_prompt(query, docs):
    """Format a listwise prompt in the model's native chat template."""
    prefix = (
        "<|im_start|>system\n"
        "You are a search relevance expert who can determine a ranking of "
        "the passages based on how relevant they are to the query. "
        "If the query is a question, how relevant a passage is depends on "
        "how well it answers the question. If not, try to analyze the intent "
        "of the query and assess how well each passage satisfies the intent. "
        "If an instruction is provided, you should follow the instruction "
        "when determining the ranking."
        "<|im_end|>\n<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n" + NO_THINK_SUFFIX

    prompt = (
        f"I will provide you with {len(docs)} passages, each indicated by "
        f"a numerical identifier. Rank the passages based on their relevance "
        f"to query: {query}\n"
    )

    doc_prompts = [
        f'<passage id="{i}">\n{doc}<|embed_token|>\n</passage>'
        for i, doc in enumerate(docs)
    ]
    prompt += "\n".join(doc_prompts) + "\n"
    prompt += f"<query>\n{query}<|rerank_token|>\n</query>"

    return prefix + prompt + suffix


def _create_dummy_inputs(tokenizer):
    """Create a small listwise prompt and its tensor inputs for ONNX export."""
    dummy_max_length = 1280
    query = "What is the capital of China?"
    docs = [
        "The capital of China is Beijing.",
        "China has many large cities.",
        "Beijing is the political center of China.",
    ]

    prompt = _format_listwise_prompt(query, docs)
    encoded = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=int(dummy_max_length),
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    doc_token_indices = _find_token_positions(input_ids, DOC_EMBED_TOKEN_ID)
    query_token_index = _find_token_positions(input_ids, QUERY_EMBED_TOKEN_ID)

    return input_ids, attention_mask, doc_token_indices, query_token_index


def _find_token_positions(input_ids, token_id):
    """Find positions of a token ID and return them as a dense tensor."""
    positions = (input_ids == token_id).nonzero(as_tuple=True)
    batch_indices = positions[0]
    seq_indices = positions[1]

    result = torch.zeros(
        1, MAX_DOCS if token_id == DOC_EMBED_TOKEN_ID else 1,
        dtype=torch.long,
    )
    count = 0
    for i, batch_idx in enumerate(batch_indices):
        if batch_idx == 0:
            col_idx = count if token_id == DOC_EMBED_TOKEN_ID else 0
            if col_idx < result.shape[1]:
                result[0, col_idx] = seq_indices[i]
                count += 1

    return result


def _export_to_onnx(model, output_path, dummy_inputs, use_fusion):
    """Export the wrapper model to ONNX with stable masking behavior during tracing."""
    import transformers.masking_utils as _mu

    _orig_preprocess = _mu._preprocess_mask_arguments

    def _patched_preprocess(*args, **kwargs):
        result = _orig_preprocess(*args, **kwargs)
        if not isinstance(result, tuple):
            return result
        result = list(result)
        for idx, val in enumerate(result):
            if isinstance(val, torch.Tensor) and val.dim() == 0:
                result[idx] = int(val.item())
        return tuple(result)

    _mu._preprocess_mask_arguments = _patched_preprocess

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dummy_input_ids, dummy_attention_mask, dummy_doc_indices, dummy_query_idx = dummy_inputs

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence"},
        "attention_mask": {0: "batch_size", 1: "sequence"},
        "doc_token_indices": {0: "batch_size", 1: "num_docs"},
        "query_token_index": {0: "batch_size"},
    }
    input_names = ["input_ids", "attention_mask", "doc_token_indices", "query_token_index"]
    dynamic_axes["scores"] = {0: "batch_size", 1: "num_docs"}
    output_names = ["scores"]

    print(f"Exporting model to {output_path}")
    try:
        global _ONNX_DYNAMIC_EXPORT
        with torch.no_grad():
            enable_dynamic = bool(use_fusion) and (int(dummy_input_ids.shape[1]) % 128 == 0)
            _ONNX_DYNAMIC_EXPORT = bool(enable_dynamic)
            torch.onnx.export(
                model,
                (dummy_input_ids, dummy_attention_mask, dummy_doc_indices, dummy_query_idx),
                output_path,
                export_params=True,
                opset_version=17,
                do_constant_folding=not use_fusion,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                dynamo=False,
            )
    finally:
        _ONNX_DYNAMIC_EXPORT = False
        _mu._preprocess_mask_arguments = _orig_preprocess

    print(f"Model exported successfully to {output_path}")


def _optimize_onnx_model(onnx_model):
    """Optimize the ONNX graph by removing IsNaN/Where patterns for deployment."""
    isnan_nodes = [node for node in onnx_model.graph.node if node.op_type == "IsNaN"]

    if not isnan_nodes:
        print("No IsNaN nodes found in the model")
        return onnx_model

    print(
        f"Found {len(isnan_nodes)} IsNaN nodes, replacing "
        "Where(IsNaN(x), default, x) patterns with Identity(x)..."
    )

    nodes_to_remove_names = set()
    nodes_to_add = []

    for isnan_node in isnan_nodes:
        isnan_output = isnan_node.output[0]
        where_nodes = [
            node
            for node in onnx_model.graph.node
            if node.op_type == "Where" and isnan_output in node.input
        ]
        print(f"  IsNaN node {isnan_node.name}: found {len(where_nodes)} Where consumers")

        for where_node in where_nodes:
            where_inputs = where_node.input
            if len(where_inputs) == 3 and where_inputs[0] == isnan_output:
                where_output = where_node.output[0]
                value_if_false = where_inputs[2]
                identity_node = onnx.helper.make_node(
                    "Identity",
                    inputs=[value_if_false],
                    outputs=[where_output],
                    name=where_node.name + "_identity",
                )
                nodes_to_add.append(identity_node)
                nodes_to_remove_names.add(where_node.name)
                print(f"    Will replace Where node {where_node.name} with Identity")

        nodes_to_remove_names.add(isnan_node.name)
        print(f"  Will remove IsNaN node {isnan_node.name}")

    print(
        f"Removing {len(nodes_to_remove_names)} nodes and "
        f"adding {len(nodes_to_add)} nodes..."
    )

    new_nodes = [
        node for node in onnx_model.graph.node if node.name not in nodes_to_remove_names
    ]
    new_nodes.extend(nodes_to_add)

    onnx_model.graph.ClearField("node")
    onnx_model.graph.node.extend(new_nodes)

    print(f"Successfully replaced {len(isnan_nodes)} IsNaN nodes")
    remaining_isnan = sum(1 for node in onnx_model.graph.node if node.op_type == "IsNaN")
    print(f"Remaining IsNaN nodes after removal: {remaining_isnan}")

    return onnx_model


def _print_onnx_custom_stats(onnx_path, label=""):
    """Print counts of Custom ops grouped by their type attribute."""
    try:
        m = onnx.load(onnx_path, load_external_data=False)
        ops = Counter(n.op_type for n in m.graph.node)
        custom_cnt = ops.get("Custom", 0)
        if custom_cnt:
            type_cnts = Counter()
            for n in m.graph.node:
                if n.op_type != "Custom":
                    continue
                for a in n.attribute:
                    if a.name == "type":
                        type_cnts[a.s.decode("utf-8")] += 1
                        break
            print(f"{label}Custom op counts: {dict(type_cnts)}")
        else:
            print(f"{label}No Custom ops found in ONNX model")
        del m
    except Exception:
        pass


def main():
    """Run model loading, wrapping, and ONNX export."""
    args = _parse_args()
    model, tokenizer = _load_model_and_tokenizer(args)

    use_fusion = args.enable_fusion_opt

    if use_fusion:
        print("Enabling CANN fusion optimizations...")
        enable_rmsnorm_fusion = bool(getattr(args, "enable_rmsnorm_fusion", False))
        if enable_rmsnorm_fusion:
            _replace_rmsnorm_with_cann(model)
        print("Preparing fused model for export...")
        wrapper = JinaRerankerV3FusedWrapper(
            model,
            enable_bmm2mm_fusion=bool(args.enable_bmm2mm_fusion),
            enable_rmsnorm_fusion=enable_rmsnorm_fusion,
            enable_qk_merge=bool(getattr(args, "enable_qk_merge", False)),
        ).to(args.device).eval()
    else:
        print("Preparing model for export (no fusion)...")
        wrapper = JinaRerankerV3ListwiseWrapper(model).to(args.device).eval()

    print("Creating dummy inputs with listwise prompt format...")
    dummy_inputs = _create_dummy_inputs(tokenizer)
    dummy_input_ids = dummy_inputs[0].to(args.device)
    dummy_attention_mask = dummy_inputs[1].to(args.device)
    dummy_doc_indices = dummy_inputs[2].to(args.device)
    dummy_query_idx = dummy_inputs[3].to(args.device)
    dummy_inputs_device = (dummy_input_ids, dummy_attention_mask, dummy_doc_indices, dummy_query_idx)

    output_path = os.path.join(args.output_dir, "jina_reranker_v3.onnx")
    _export_to_onnx(wrapper, output_path, dummy_inputs_device, use_fusion)

    print("Optimizing ONNX model...")
    onnx_model = onnx.load(output_path, load_external_data=True)
    onnx_model = _optimize_onnx_model(onnx_model)

    onnx_path = os.path.join(args.output_dir, "jina_reranker_v3.onnx")
    data_path = onnx_path + ".data"
    if os.path.exists(data_path):
        os.remove(data_path)
    onnx.save_model(
        onnx_model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="jina_reranker_v3.onnx.data",
        size_threshold=1024,
        convert_attribute=True,
    )
    print(f"Optimized model saved to {output_path}")

    _print_onnx_custom_stats(onnx_path, label="Export result: ")

    print("\nExport completed successfully!")
    print(f"ONNX model saved to: {output_path}")
    print(f"Max documents per query: {MAX_DOCS}")
    print(f"Fusion optimizations: {'ENABLED' if use_fusion else 'DISABLED'}")


if __name__ == "__main__":
    main()
