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

# When True, always use matmul attention for decode (skip IncreFlashAttention).
# For testing: verify matmul decode correctness without needing TP=4 (which requires
# num_kv_heads_local < 2). Set to False for production (IncreFlashAttention is faster).
FORCE_MATMUL_ATTN = False

# Tensor-parallel config (set from --tp-size / --rank before exporting the decode
# subgraph). TP_SIZE=1 reproduces the original single-shard export.
TP_SIZE = 1
TP_RANK = 0
# Debug: when True, the decode wrapper emits layer-0 intermediate hidden states
# (post-attention, post-mlp) as extra outputs to localize graph-miscompilation.
DEBUG_TAP = False
_TAP_RAW_OUT = []  # collected raw attention outputs (before o_proj) per layer when DEBUG_TAP
# AllReduce fusion id: 0 = no fusion (default, each AR its own collective); >0 = all
# Custom(AllReduce) nodes share this fusion_id, activating GE's fusion channel so
# HcomAllReduce ops are batched into one fused communication stream (fewer per-AR
# launch/sync overhead, potential AR/compute overlap). 2p AR is correct under both.
AR_FUSION_ID = 0

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
    """Apply RMS normalization using a HuggingFace RMSNorm module's weight.

    For per-head norms (q_norm/k_norm, gamma shape [heads, head_dim]) under TP,
    slice the head axis to this rank. Regular layernorms (gamma shape [hidden])
    are replicated.
    """
    gamma = norm_mod.weight
    eps = getattr(norm_mod, "variance_epsilon", 1e-6)
    if TP_SIZE > 1 and gamma.dim() > 1:
        g_per = int(gamma.shape[0]) // TP_SIZE
        gamma = gamma[TP_RANK * g_per:(TP_RANK + 1) * g_per]
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


def scatter(var, indices, updates):
    """Scatter updates into KV cache at a given position.

    Uses native PyTorch torch.scatter (exports as ONNX ScatterElements) instead of
    a Custom op — the Custom Scatter op in the 28-layer TP=4 decode graph triggers a
    GE graph-optimization miscompile of the downstream o_proj AllReduce, producing
    garbage. Native ScatterElements avoids this.
    """
    out = var.clone()
    pos = indices
    if pos.dim() == 2 and pos.shape[-1] == 1:
        pos = pos.squeeze(-1)
    pos = pos.to(torch.long)
    upd = updates
    if upd.dim() == 4 and upd.shape[2] == 1:
        upd = upd[:, :, 0, :]  # (B, H, D)
    bsz, num_heads, _, head_dim = out.shape
    # Build index: (B, H, 1, D) filled with pos[b] for each batch
    idx = pos.view(bsz, 1, 1, 1).expand(bsz, num_heads, 1, head_dim)
    src = upd.unsqueeze(2)  # (B, H, 1, D)
    out = torch.scatter(out, dim=2, index=idx, src=src)
    return out


class _AllReduceCustom(torch.autograd.Function):
    """Custom AllReduce(sum) op for ONNX export.

    Eager fallback is identity (shape-preserving) -- only used during trace; the
    real cross-rank sum happens at runtime where the plugin lowers the Custom
    (type=AllReduce, op=sum, group=hccl_world_group, rank_size, fusion=0) node to a
    GE HcomAllReduce. Requires the convert.cc group-injection fix + fusion attr set.
    """

    @staticmethod
    def forward(ctx, x):
        del ctx
        return x

    @staticmethod
    def symbolic(g, x):
        """Emit ONNX Custom(AllReduce) node; GE lowers to HcomAllReduce."""
        y = g.op(
            "Custom", x,
            type_s="AllReduce",
            input_names_s=_as_list_str(["x"]),
            optional_input_names_s=_as_list_str([]),
            output_names_s=_as_list_str(["y"]),
            output_num_i=1,
            input_index_i=[0],
            op_s="sum",
            group_s="hccl_world_group",
            rank_size_i=int(TP_SIZE),
            fusion_i=int(AR_FUSION_ID),
        )
        y.setType(x.type())
        return y


def allreduce_sum(x):
    """AllReduce-sum across the TP group (no-op when TP_SIZE == 1)."""
    if TP_SIZE <= 1:
        return x
    return _AllReduceCustom.apply(x)


class _MatmulAllReduceCustom(torch.autograd.Function):
    """Custom MatmulAllReduce op: fused MatMul + AllReduce(sum) for TP row-parallel lines.

    Replaces the ``allreduce_sum(F.linear(x, w, b))`` pattern (MatMul + a separate
    HcomAllReduce) with a single aclnnMatmulAllReduce node, cutting the per-step op
    count by ~56 (28 layers x 2 row-parallel projections + lm_head). Eager fallback
    is plain F.linear (AR is identity during trace); the real cross-rank sum happens
    at runtime where the plugin lowers Custom(type=MatmulAllReduce, group, reduce_op,
    comm_turn) to aclnnMatmulAllReduce. mslite's convert.cc routes any Custom op
    carrying a ``group`` attr through the collective handler, injecting group/fusion
    onto the GE node (same path as AllReduce). Requires 300I Duo transformer op set.
    """

    @staticmethod
    def forward(ctx, x, weight, bias, group, reduce_op):
        del ctx, group, reduce_op
        return F.linear(x, weight, bias)

    @staticmethod
    def symbolic(g, x, weight, bias, group, reduce_op):
        """Emit ONNX Custom(MatmulAllReduce) node; GE lowers to aclnnMatmulAllReduce."""
        # nn.Linear weight is (out, in); set is_trans_b=true so MatmulAllReduce
        # computes x @ weight.T (== F.linear). mslite's onnx_custom_parser maps
        # string "true"/"false" to a bool attr (SetAttrString), so BOOL-typed GE
        # op attributes are expressed via the _s suffix.
        base_inputs = [x, weight]
        base_index = [0, 1]
        if bias is not None:
            base_inputs.append(bias)
            base_index.append(2)
        y = g.op(
            "Custom", *base_inputs,
            type_s="MatmulAllReduce",
            input_names_s=_as_list_str(["x1", "x2", "bias"]),
            optional_input_names_s=_as_list_str(["bias"]),
            output_names_s=_as_list_str(["y"]),
            output_num_i=1,
            input_index_i=base_index,
            group_s=str(group),
            reduce_op_s=str(reduce_op),
            comm_turn_i=0,
            fusion_i=0,
            is_trans_a_s="false",
            is_trans_b_s="true",
        )
        y.setType(x.type())
        return y


def matmul_all_reduce(x, weight, bias=None, group="hccl_world_group", reduce_op="sum"):
    """Fused MatMul + AllReduce-sum for TP row-parallel (no-op when TP_SIZE == 1)."""
    if TP_SIZE <= 1:
        return F.linear(x, weight, bias)
    return _MatmulAllReduceCustom.apply(x, weight, bias, group, reduce_op)


# Module-level counter giving each AllReduce a unique fusion id. With fusion_i=0
# all AllReduces share GE's "no-fusion" path; in large 28-layer TP=4 decode graphs
# that path appears to mis-batch/mis-route the o_proj AllReduce (silent precision
# corruption). Unique fusion ids (>0) route through GE's "fusion-by-id" path where
# each lives in its own group, which sidesteps the bug.
_ALLREDUCE_COUNTER = [0]


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

    # TP: column-parallel QKV -- each rank holds num_heads/TP q-heads and
    # num_kv_heads/TP kv-heads (slice the output/row axis of each projection).
    q_w = attn_mod.q_proj.weight
    k_w = attn_mod.k_proj.weight
    v_w = attn_mod.v_proj.weight
    q_per = int(q_w.shape[0]) // TP_SIZE
    kv_per = int(k_w.shape[0]) // TP_SIZE
    qs, qe = TP_RANK * q_per, (TP_RANK + 1) * q_per
    ks, ke = TP_RANK * kv_per, (TP_RANK + 1) * kv_per
    w = torch.cat([q_w[qs:qe], k_w[ks:ke], v_w[ks:ke]], dim=0)
    q_b, k_b, v_b = attn_mod.q_proj.bias, attn_mod.k_proj.bias, attn_mod.v_proj.bias
    if q_b is None:
        b = None
    else:
        b = torch.cat([q_b[qs:qe], k_b[ks:ke], v_b[ks:ke]], dim=0)

    q_out = q_per
    kv_out = kv_per
    qkv = F.linear(hidden_states, w, b)

    query = qkv[..., :q_out].view(hidden_shape)
    key = qkv[..., q_out:q_out + kv_out].view(hidden_shape)
    value = qkv[..., q_out + kv_out:].view(hidden_shape)
    if hasattr(attn_mod, "q_norm"):
        query = _rms_norm_layer(attn_mod.q_norm, query)
    if hasattr(attn_mod, "k_norm"):
        key = _rms_norm_layer(attn_mod.k_norm, key)
    return query, key, value, input_shape


def _run_decode_attention_matmul(query, key, value, attention_mask, num_heads, num_kv_heads):
    """Standard matmul attention for decode (no IncreFlashAttention).

    Used when num_kv_heads_local == 1 (e.g. TP=4) where the IncreFlashAttention
    plugin may not support single KV head. q is (1, H, 1, D), k/v are (1, kv_H, L, D).
    """
    scaling = 1.0 / (query.shape[-1] ** 0.5)
    q = query  # (1, H, 1, D)
    k = key    # (1, kv_H, L, D)
    v = value
    if 0 < num_kv_heads < num_heads:
        rep = num_heads // num_kv_heads
        # Use expand+reshape instead of repeat_interleave (which exports as Split op that GE can't compile)
        b, _, l, d = k.shape
        k = k.unsqueeze(2).expand(b, num_kv_heads, rep, l, d).reshape(b, num_heads, l, d)
        v = v.unsqueeze(2).expand(b, num_kv_heads, rep, l, d).reshape(b, num_heads, l, d)
    attn = torch.matmul(q, k.transpose(2, 3)) * float(scaling)  # (1, H, 1, L)
    # Padding mask: attention_mask is (batch, L), 1=valid, 0=pad
    pad_mask = attention_mask[:, None, None, :].to(torch.bool)  # (B, 1, 1, L)
    pad_mask = pad_mask.expand(attn.shape[0], attn.shape[1], 1, attn.shape[3])
    attn = attn.masked_fill(~pad_mask, torch.finfo(attn.dtype).min)
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    output = torch.matmul(attn, v)  # (1, H, 1, D)
    return output


def _run_prefill_attention(query, key, value, attention_mask, num_heads, num_kv_heads, scaling):
    """Run prefill-path attention via fused PromptFlashAttention (BSND, GQA fused).

    query/key/value arrive as (batch, seq, heads, head_dim) = BSND. PFA handles
    GQA internally (num_key_value_heads < num_heads), removing the manual
    repeat_interleave + matmul + softmax split. Returns (attn_output, k, v) with
    KV in BNSD fp16 to match decode past_key_cache.
    """
    q_len, k_len = query.shape[1], key.shape[1]
    flash_mask = _make_flash_attn_mask(attention_mask, q_len, k_len, 0)
    attn_output = prompt_flash_attention(
        query, key, value, atten_mask=flash_mask,
        num_heads=num_heads, scale_value=float(scaling),
        input_layout="BSND", num_key_value_heads=num_kv_heads,
        sparse_mode=0, inner_precise=1)
    # KV fp16 BNSD (matches decode past_key_cache)
    return attn_output, key.permute(0, 2, 1, 3), value.permute(0, 2, 1, 3)


def _run_decode_attention(query, key, value, attention_mask, attn_mod, num_heads, num_kv_heads):
    """Run decode-path attention using IncreFlashAttention custom op (MHA mode).

    Manually repeats kv heads to num_heads first (expand+reshape), then calls
    IncreFlash with num_key_value_heads == num_heads (pure MHA). This bypasses
    the kernel's GQA path, which mis-computes when num_key_value_heads == 1
    (the TP=4 case: 4 KV heads / 4 ranks = 1 KV head/rank). Returns attn_output
    in BNSD layout.
    """
    scaling = getattr(attn_mod, "scaling", 1.0 / (attn_mod.head_dim ** 0.5))
    if 0 < num_kv_heads < num_heads:
        rep = num_heads // num_kv_heads
        b, _, l, d = key.shape
        # expand+reshape (not repeat_interleave) -- repeat_interleave exports as Split that GE can't compile
        key = key.unsqueeze(2).expand(b, num_kv_heads, rep, l, d).reshape(b, num_heads, l, d)
        value = value.unsqueeze(2).expand(b, num_kv_heads, rep, l, d).reshape(b, num_heads, l, d)
        num_kv_heads = num_heads  # now MHA: every q head has its own (repeated) kv head
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
        key = scatter(past_key, pos, key)
        value = scatter(past_value, pos, value)

    num_heads_local = num_heads // TP_SIZE
    num_kv_heads_local = num_kv_heads // TP_SIZE
    if past_key is None:
        attn_output, key, value = _run_prefill_attention(
            query, key, value, attention_mask, num_heads_local, num_kv_heads_local,
            getattr(attn_mod, "scaling", 1.0 / (attn_mod.head_dim ** 0.5)))
        out = attn_output.reshape(*input_shape, -1)
    elif FORCE_MATMUL_ATTN:
        # Escape-hatch fallback: plain matmul attention (has a known unbind(0)→GE Split
        # compile issue in the multi-layer graph; normally unused).
        attn_output = _run_decode_attention_matmul(
            query, key, value, attention_mask, num_heads_local, num_kv_heads_local)
        out = attn_output.transpose(1, 2).reshape(*input_shape, -1)
    else:
        # Default: IncreFlash in MHA mode after manual GQA repeat (handles
        # num_kv_heads_local==1 for TP=4 where the kernel's own GQA path is wrong).
        attn_output = _run_decode_attention(
            query, key, value, attention_mask, attn_mod, num_heads_local, num_kv_heads_local)
        out = attn_output.transpose(1, 2).reshape(*input_shape, -1)

    if DEBUG_TAP:
        _TAP_RAW_OUT.append(out)  # raw attention output (heads concatenated), before o_proj

    # o_proj: row-parallel under TP (input dim = local q-heads * head_dim), then AllReduce.
    if TP_SIZE > 1:
        q_dim_local = num_heads_local * attn_mod.head_dim
        o_w = attn_mod.o_proj.weight[:, TP_RANK * q_dim_local:(TP_RANK + 1) * q_dim_local]
        out_proj = allreduce_sum(F.linear(out, o_w, attn_mod.o_proj.bias))
    else:
        out_proj = attn_mod.o_proj(out)
    return out_proj, key, value


# ---------------------------------------------------------------------------
# MLP helpers
# ---------------------------------------------------------------------------


def _mlp_gate_up_linear(mlp_mod, x):
    """Merge gate_proj and up_proj into a single linear, then split outputs.

    TP: column-parallel -- each rank holds intermediate/TP rows of gate and up.
    """
    gate_w = mlp_mod.gate_proj.weight
    up_w = mlp_mod.up_proj.weight
    gate_b = mlp_mod.gate_proj.bias
    up_b = mlp_mod.up_proj.bias
    g_per = int(gate_w.shape[0]) // TP_SIZE
    gs, ge = TP_RANK * g_per, (TP_RANK + 1) * g_per
    w = torch.cat([gate_w[gs:ge], up_w[gs:ge]], dim=0)
    b = None if gate_b is None else torch.cat([gate_b[gs:ge], up_b[gs:ge]], dim=0)
    y = F.linear(x, w, b)
    gate_out = g_per
    return y[..., :gate_out], y[..., gate_out:]


def _run_mlp(layer, hidden_states):
    """Run MLP forward pass with fused gate+up projection and SwiGLU activation."""
    mlp = layer.mlp
    if hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj") and hasattr(mlp, "down_proj"):
        gate, up = _mlp_gate_up_linear(mlp, hidden_states)
        act = swiglu(torch.cat([gate, up], dim=-1), dim=-1)
        if TP_SIZE > 1:
            # down_proj row-parallel: input dim = intermediate/TP, then AllReduce.
            d_w = mlp.down_proj.weight
            in_per = int(d_w.shape[1]) // TP_SIZE
            d_w_local = d_w[:, TP_RANK * in_per:(TP_RANK + 1) * in_per]
            return allreduce_sum(F.linear(act, d_w_local, mlp.down_proj.bias))
        return mlp.down_proj(act)
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
            # Prefill uses Custom RmsNorm (_rms_norm_layer): in the 28-layer TP=4 PREFILL
            # graph the native HF RMSNorm gets miscompiled (wrong first token), while the
            # Custom RmsNorm is correct. (Decode is the opposite — it uses native RMSNorm.)
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
        if TP_SIZE > 1:
            h_per = hidden_states.shape[-1] // TP_SIZE
            hs, he = TP_RANK * h_per, (TP_RANK + 1) * h_per
            lh_w = self.lm_head.weight[:, hs:he]
            logits = allreduce_sum(F.linear(hidden_states[..., hs:he], lh_w, self.lm_head.bias))
            logits = logits.float()
        else:
            logits = self.lm_head(hidden_states)
        return logits[:, -1:, :], torch.stack(present_k, dim=0), torch.stack(present_v, dim=0)


class Qwen2_5LlmDecode(torch.nn.Module):
    """Qwen2.5-7B LLM Decode wrapper — single-token generation with KV cache update.
    Used for TP=1/2 (single model, no chunking). TP=4 uses Qwen2_5LlmDecodeChunk."""

    def __init__(self, model, lm_head):
        super().__init__()
        self.model = model.model
        self.lm_head = lm_head

    def forward(self, input_ids, attention_mask, position_ids, past_key_cache, past_value_cache):
        """Run single-token decode over all layers; return logits + stacked KV."""
        inputs_embeds = self.model.embed_tokens(input_ids)
        cos, sin = self.model.rotary_emb(inputs_embeds, position_ids)
        cos4 = cos.unsqueeze(1) if cos.dim() == 3 else cos
        sin4 = sin.unsqueeze(1) if sin.dim() == 3 else sin

        num_layers = len(self.model.layers)
        past_k_layers = [past_key_cache[i] for i in range(num_layers)]
        past_v_layers = [past_value_cache[i] for i in range(num_layers)]
        present_k, present_v = [], []
        hidden_states = inputs_embeds
        barriers = []
        BARRIER_INTERVAL = 4

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
            if TP_SIZE >= 4 and (i + 1) % BARRIER_INTERVAL == 0:
                barriers.append(hidden_states)

        hidden_states = _rms_norm_layer(self.model.norm, hidden_states)
        if TP_SIZE > 1:
            h_per = hidden_states.shape[-1] // TP_SIZE
            hs, he = TP_RANK * h_per, (TP_RANK + 1) * h_per
            lh_w = self.lm_head.weight[:, hs:he]
            logits = allreduce_sum(F.linear(hidden_states[..., hs:he], lh_w, self.lm_head.bias))
            logits = logits.float()
        else:
            logits = self.lm_head(hidden_states)
        outputs = [logits[:, -1:, :], torch.stack(present_k, dim=0), torch.stack(present_v, dim=0)]
        if barriers:
            outputs.append(torch.stack(barriers, dim=0))
        return tuple(outputs)


class Qwen2_5LlmDecodeChunk(torch.nn.Module):
    """Decode sub-model for a chunk of layers (split-decode for TP=4).

    Processes a contiguous range of transformer layers. The first chunk also
    does token embedding; the last chunk also applies final norm + lm_head.
    Each chunk is compiled as a separate MindIR so GE's graph optimizer
    never sees more than CHUNK_SIZE layers worth of AllReduce nodes at once
    (threshold: >4 layers → miscompile for TP=4).
    """

    def __init__(self, model, lm_head, layer_start, layer_end, is_first, is_last):
        super().__init__()
        self.model = model.model
        self.lm_head = lm_head
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.is_first = is_first
        self.is_last = is_last
        self.num_chunk_layers = layer_end - layer_start

    def forward(self, input_ids_or_hidden, attention_mask, position_ids,
                past_key_cache, past_value_cache):
        """Run decode over a chunk of layers; first chunk embeds, last emits logits."""
        if self.is_first:
            hidden_states = self.model.embed_tokens(input_ids_or_hidden)
        else:
            hidden_states = input_ids_or_hidden
        cos, sin = self.model.rotary_emb(hidden_states, position_ids)
        cos4 = cos.unsqueeze(1) if cos.dim() == 3 else cos
        sin4 = sin.unsqueeze(1) if sin.dim() == 3 else sin

        present_k, present_v = [], []
        for idx in range(self.num_chunk_layers):
            i = self.layer_start + idx
            residual = hidden_states
            hidden_states = self.model.layers[i].input_layernorm(hidden_states)
            attn_out, pk, pv = _text_attn_forward(
                self.model.layers[i].self_attn, hidden_states, cos4, sin4,
                attention_mask, position_ids,
                past_key_cache[idx], past_value_cache[idx])
            hidden_states = residual + attn_out
            residual = hidden_states
            hidden_states = self.model.layers[i].post_attention_layernorm(hidden_states)
            hidden_states = residual + _run_mlp(self.model.layers[i], hidden_states)
            present_k.append(pk)
            present_v.append(pv)

        if self.is_last:
            hidden_states = _rms_norm_layer(self.model.norm, hidden_states)
            if TP_SIZE > 1:
                h_per = hidden_states.shape[-1] // TP_SIZE
                hs, he = TP_RANK * h_per, (TP_RANK + 1) * h_per
                lh_w = self.lm_head.weight[:, hs:he]
                logits = allreduce_sum(F.linear(hidden_states[..., hs:he], lh_w, self.lm_head.bias))
                return logits.float(), torch.stack(present_k, dim=0), torch.stack(present_v, dim=0)
            return self.lm_head(hidden_states), torch.stack(present_k, dim=0), torch.stack(present_v, dim=0)
        # Intermediate chunk: output hidden state (not logits) + KV cache.
        # No explicit Cast — let PyTorch/ONNX use the model's native fp16 dtype.
        # GE's HcomAllReduce outputs fp32 internally but GE inserts Cast(fp32→fp16)
        # before the output node to match the MindIR-declared fp16 output type.
        return hidden_states, torch.stack(present_k, dim=0), torch.stack(present_v, dim=0)


# Chunk size for split-decode. Historically <=4 to work around a *suspected* GE
# hccl_graph_optimizer miscompile for >4-layer 4-rank decode graphs; REPORT.md §4.2
# has since traced the 4p accuracy breakage to 4-rank AllReduce itself (not a
# miscompile), so the chunk workaround is no longer needed. 999 = full decode.
DECODE_CHUNK_SIZE = 999


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


def _export_prefill_onnx(prefill, prefill_path: Path, dummy_inputs, use_dynamo: bool, static: bool = False):
    """Export prefill subgraph to ONNX.

    static=True -> no dynamic_axes (fixed seq + batch=1). Required for the TP
    prefill (online path doesn't resolve dynamic dims).
    """
    print(f"Exporting LLM prefill to {prefill_path}...")
    dynamic = None
    if not static:
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
    # TP: each rank holds num_kv_heads/TP KV heads.
    num_kv_heads = num_kv_heads // TP_SIZE
    k = torch.zeros(num_layers, 1, num_kv_heads, past_len, head_dim, dtype=kv_dtype, device=device)
    v = torch.zeros_like(k)
    return ids, mask, pos, k, v


def _export_decode_onnx(decode, decode_path: Path, dummy_inputs, use_dynamo: bool, static: bool = False):
    """Export decode subgraph to ONNX.

    static=True -> no dynamic_axes (fixed batch=1). Required for the TP decode which
    is converted with --optimize=none (online): the online path does not resolve a
    dynamic batch dim, leaving the output shape as -1 and breaking output size checks.
    """
    print(f"Exporting LLM decode to {decode_path}...")
    dynamic = None
    if not static:
        dynamic = {"input_ids": {0: "batch"}, "attention_mask": {0: "batch"},
                   "position_ids": {0: "batch"}, "logits": {0: "batch"},
                   "past_key_cache": {1: "batch"}, "past_value_cache": {1: "batch"},
                   "present_key_cache": {1: "batch"}, "present_value_cache": {1: "batch"}}
    out_names = ["output", "present_key_cache", "present_value_cache"]
    with torch.no_grad():
        torch.onnx.export(
            decode, dummy_inputs, str(decode_path),
            input_names=["input_ids", "attention_mask", "position_ids",
                          "past_key_cache", "past_value_cache"],
            output_names=out_names,
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


def export_prefill_only(model, output_dir, device="cpu", dummy_seq_len=8, use_dynamo=False):
    """Export only the single-shard prefill subgraph (TP decode reuses it on dev0)."""
    global TP_SIZE, TP_RANK
    TP_SIZE, TP_RANK = 1, 0
    prefill, _, _ = _prepare_llm_modules(model, device=device)
    prefill_path, _ = _prepare_output_paths(output_dir)
    _, ids, mask, pos = _create_prefill_dummy_inputs(device=device, dummy_seq_len=dummy_seq_len)
    _export_prefill_onnx(prefill, prefill_path, (ids, mask, pos), use_dynamo)


def export_tp_decode(model, output_dir, rank, tp_size, device="cpu", use_dynamo=False, static=True):
    """Export the TP-sharded decode subgraph for one rank.

    For TP>=4: exports as split-decode chunks (each <= DECODE_CHUNK_SIZE layers).
    For TP<=2: exports as a single decode model (no miscompile at this scale).
    """
    global TP_SIZE, TP_RANK
    TP_SIZE = int(tp_size)
    TP_RANK = int(rank)
    num_layers, num_kv_heads, head_dim = _get_kv_cache_config(model)
    kv_dtype = next(model.parameters()).dtype
    lm_head = model.lm_head

    if tp_size >= 4 and num_layers > DECODE_CHUNK_SIZE:
        _export_split_decode(model, lm_head, output_dir, rank, tp_size, num_layers,
                             num_kv_heads, head_dim, kv_dtype, device, use_dynamo)
    else:
        print(f"Exporting decode rank={rank}/{tp_size} (single, static={static})...")
        _, decode, _ = _prepare_llm_modules(model, device=device)
        decode_dir = Path(output_dir) / "decode"
        decode_dir.mkdir(parents=True, exist_ok=True)
        decode_path = decode_dir / f"qwen2_5_7b_llm_decode_rank{rank}.onnx"
        decode_inputs = _create_decode_dummy_inputs(str(device), num_layers, num_kv_heads, head_dim, kv_dtype)
        _export_decode_onnx(decode, decode_path, decode_inputs, use_dynamo, static=static)
    TP_SIZE, TP_RANK = 1, 0


def _export_split_decode(model, lm_head, output_dir, rank, tp_size, num_layers,
                         num_kv_heads, head_dim, kv_dtype, device, use_dynamo):
    """Export decode as multiple chunk sub-models for TP>=4 split-decode."""
    chunk = DECODE_CHUNK_SIZE
    num_chunks = (num_layers + chunk - 1) // chunk
    kv_per = num_kv_heads // tp_size
    print(f"Exporting split-decode rank={rank}/{tp_size}: {num_chunks} chunks × {chunk} layers...")

    for ci in range(num_chunks):
        ls = ci * chunk
        le = min(ls + chunk, num_layers)
        is_first = ci == 0
        is_last = ci == num_chunks - 1
        n_chunk_layers = le - ls

        chunk_module = Qwen2_5LlmDecodeChunk(model, lm_head, ls, le, is_first, is_last)
        chunk_module = chunk_module.to(device).eval()
        chunk_dir = Path(output_dir) / "decode"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_path = chunk_dir / f"qwen2_5_7b_llm_decode_rank{rank}_chunk{ci}.onnx"

        # Dummy inputs for this chunk
        if is_first:
            d_ids = torch.randint(0, 1000, (1, 1), dtype=torch.int64, device=device)
        else:
            d_ids = torch.randn(1, 1, 3584, dtype=kv_dtype, device=device)
        d_mask = torch.ones(1, KV_CACHE_LEN, dtype=torch.int64, device=device)
        d_pos = torch.tensor([[KV_CACHE_LEN - 1]], dtype=torch.int64, device=device)
        d_k = torch.zeros(n_chunk_layers, 1, kv_per, KV_CACHE_LEN, head_dim, dtype=kv_dtype, device=device)
        d_v = torch.zeros_like(d_k)

        out_names = ["output", "present_key_cache", "present_value_cache"]
        with torch.no_grad():
            torch.onnx.export(
                chunk_module, (d_ids, d_mask, d_pos, d_k, d_v), str(chunk_path),
                input_names=["input_ids_or_hidden", "attention_mask", "position_ids",
                              "past_key_cache", "past_value_cache"],
                output_names=out_names,
                opset_version=18, do_constant_folding=True, dynamo=use_dynamo,
                dynamic_axes=None)
        print(f"  chunk{ci} (layers {ls}-{le-1}, {'first' if is_first else 'mid'}, "
              f"{'last' if is_last else 'mid'}) → {chunk_path.name}")



def export_tp_prefill(model, output_dir, rank, tp_size, device="cpu", dummy_seq_len=64, use_dynamo=False, static=True):
    """Export the TP-sharded prefill subgraph for one rank.

    static=True (TP>=2): no dynamic axes, for online (optimize=none) convert.
    static=False (TP=1): dynamic batch/seq axes, for offline (ascend_oriented) convert.
    """
    global TP_SIZE, TP_RANK
    TP_SIZE = int(tp_size)
    TP_RANK = int(rank)
    print(f"Exporting prefill rank={rank}/{tp_size} (static={static})...")
    prefill, _, _ = _prepare_llm_modules(model, device=device)
    prefill_dir = Path(output_dir) / "prefill"
    prefill_dir.mkdir(parents=True, exist_ok=True)
    prefill_path = prefill_dir / f"qwen2_5_7b_llm_prefill_rank{rank}.onnx"
    _, ids, mask, pos = _create_prefill_dummy_inputs(device=device, dummy_seq_len=dummy_seq_len)
    _export_prefill_onnx(prefill, prefill_path, (ids, mask, pos), use_dynamo, static=static)
    TP_SIZE, TP_RANK = 1, 0


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
    parser.add_argument("--tp-size", type=int, default=1,
                        help="Tensor-parallel size; >1 exports a TP-sharded decode per rank "
                             "(+ single-shard prefill). Custom(AllReduce) is inserted after o_proj/"
                             "down_proj/lm_head; convert with --optimize=none, run with provider=ge.")
    parser.add_argument("--num-layers", type=int, default=0,
                        help="If >0, slice the model to N layers (debugging; 0 = all 28).")
    parser.add_argument("--kv-cache-len", type=int, default=0,
                        help="Override KV_CACHE_LEN (default 512). Set >=1024 to enable prefill "
                             "seq > 512 for long-prompt perf sweep. 0 = keep default 512.")
    parser.add_argument("--ar-fusion-id", type=int, default=0,
                        help="AllReduce fusion id (0=no fusion default; >0 activates GE fusion "
                             "channel to batch HcomAllReduce ops — 2p decode optimization).")
    parser.add_argument("--seq-list", type=str, default="",
                        help="Comma-sep prefill seq lengths for per-seq static export (TP>=2 only; "
                             "online GE dynamic prefill fails with aicore error, so per-seq static "
                             "is required). Each seq → separate OUT_DIR_seqN. E.g. '32,64,128,512,1024'. "
                             "Empty (default) = single export at --dummy-seq-len.")
    return parser.parse_args()


def main():
    """Load Qwen2.5-7B model and export to ONNX prefill + decode subgraphs."""
    args = _parse_export_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.kv_cache_len and args.kv_cache_len > 0:
        global KV_CACHE_LEN
        KV_CACHE_LEN = int(args.kv_cache_len)
        print(f"[kv-cache-len] override KV_CACHE_LEN = {KV_CACHE_LEN}")
    if args.ar_fusion_id and args.ar_fusion_id > 0:
        global AR_FUSION_ID
        AR_FUSION_ID = int(args.ar_fusion_id)
        print(f"[ar-fusion-id] AR_FUSION_ID = {AR_FUSION_ID} (GE fusion channel on)")

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    print(f"\nLoading model {args.model_id} for export (dtype={args.dtype})...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=False,
        attn_implementation="eager").to(device)

    if args.num_layers and args.num_layers > 0:
        model.model.layers = model.model.layers[:args.num_layers]  # type: ignore[attr-defined]
        model.config.num_hidden_layers = args.num_layers  # type: ignore[attr-defined]
        print(f"Sliced model to {args.num_layers} layers (debug)")

    # TP>=2 prefill is static-per-seq (online GE dynamic prefill fails with
    # aicore error). --seq-list exports one model per seq to OUT_DIR_seqN.
    # TP=1 prefill is dynamic (ascend_oriented offline supports dynamicDims).
    if args.tp_size > 1:
        if args.seq_list:
            seq_list = [int(x) for x in args.seq_list.split(",") if x.strip()]
        else:
            seq_list = [args.dummy_seq_len]
        for seq in seq_list:
            seq_out = output_dir if len(seq_list) == 1 else output_dir.parent / f"{output_dir.name}_seq{seq}"
            seq_out = Path(seq_out)
            seq_out.mkdir(parents=True, exist_ok=True)
            print(f"\n--- TP={args.tp_size} seq={seq} -> {seq_out} ---")
            for rank in range(args.tp_size):
                export_tp_prefill(model, seq_out, rank, args.tp_size, str(device),
                                  seq, args.use_dynamo, static=True)
                export_tp_decode(model, seq_out, rank, args.tp_size, str(device),
                                 args.use_dynamo, static=True)
    else:
        # tp_size=1: single-shard (rank0), dynamic axes for offline ascend_oriented convert
        export_tp_prefill(model, output_dir, 0, 1, str(device), args.dummy_seq_len,
                          args.use_dynamo, static=False)
        export_tp_decode(model, output_dir, 0, 1, str(device), args.use_dynamo, static=False)

    print("Clearing memory after export...")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"\nExport finished. Files saved in {args.output_dir}")


if __name__ == "__main__":
    main()
