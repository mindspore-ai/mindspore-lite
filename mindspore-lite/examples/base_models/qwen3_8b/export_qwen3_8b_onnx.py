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
Export Qwen3-8B to ONNX as split prefill + decode subgraphs, with optional
tensor-parallel (TP) sharding for 1p / 2p / 4p inference.

The export produces two ONNX models per rank:
- Prefill: processes the full input prompt, outputs logits + decode-compatible KV cache.
- Decode: generates one token at a time using the past KV cache.

TP (--tp-size > 1) inserts Custom(AllReduce) after o_proj / down_proj / lm_head
(column-parallel QKV, row-parallel o_proj/down_proj/lm_head). Each rank exports
its own shard; convert with --optimize=none and run with provider=ge + HCCL.

Qwen3-8B has 8 KV heads, so under TP=4 each rank holds 2 KV heads
(num_kv_heads_local >= 2 for every TP size). IncreFlashAttention's native GQA
path handles this directly -- unlike Qwen2.5-7B TP=4 (1 KV head/rank), no manual
kv-head repeat-to-MHA workaround is needed.
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn.functional as F

KV_CACHE_LEN = 256
MAX_OUTPUT_TOKENS = 512
COMMON_PREFIX_LEN = 768
COMMON_SUFFIX_LEN = 16

# Tensor-parallel config (set from --tp-size / --rank before exporting each
# subgraph). TP_SIZE=1 reproduces the single-shard export.
TP_SIZE = 1
TP_RANK = 0
# Debug: when True, the decode wrapper emits layer-0 intermediate hidden states
# (post-attention, post-mlp) as extra outputs. TP=4 (36 layers) can hit a GE
# graph-optimization miscompile that dampens decode logits; emitting these taps
# forces GE to materialize the intermediates, which blocks the bad optimization
# and restores correct precision. Taps alias existing intermediates -> negligible
# overhead; the infer script allocates the extra output buffers and ignores them.
DEBUG_TAP = False
# Replicate lm_head (full logits per rank, no AllReduce) instead of row-parallel
# shard+AllReduce. Needed at TP=4: the large-vocab (151936) AllReduce corrupts
# (manual partial-sum != AllReduce, diff 1.8M) on the 2-ranks-per-card x 2-card
# topology. Hidden is already full+identical on every rank (layer AllReduces
# work), so replicating lm_head is correct. Enabled for tp_size>=4 in main().
REPLICATE_LM_HEAD = False
_TAP_RAW_OUT = []  # collected raw attention outputs (before o_proj) per layer when DEBUG_TAP

try:
    torch_dynamo = getattr(torch, "_dynamo", None)
    if torch_dynamo is not None:
        torch_dynamo.disable()
except (AttributeError, RuntimeError):
    pass

try:
    from transformers import AutoModelForCausalLM
except ImportError:
    print("Error: transformers package not found or version too low.")
    print("Please install the latest version: pip install transformers")
    sys.exit(1)

try:
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
except ImportError:
    try:
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    except ImportError:
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

    Returns shape (batch, 1, q_len, k_len). Avoids redundant Cast ops:
    uses == 0 instead of .to(torch.bool).logical_not() for the padding mask,
    and skips the final .to(torch.bool) since | on bool tensors already yields bool.
    """
    ar_q = torch.arange(q_len, device=attention_mask.device)
    ar_k = torch.arange(k_len, device=attention_mask.device)
    causal = (ar_k[None, :] > (past_len + ar_q[:, None])).to(torch.float32)
    causal = causal[None, None, :, :].expand(attention_mask.shape[0], 1, q_len, k_len)
    # Keep Not/Or out of the exported graph: those logical ops prevent the
    # ONNX -> Ascend graph from being fully lowered on 310P.
    padding = 1.0 - attention_mask[:, None, None, :].to(torch.float32)
    return (causal + padding) > 0.5


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
    slice the head axis to this rank. Regular norms (gamma shape [hidden]) are
    replicated.
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

    Handles layout conversion (BSND<->BNSD) and key/value head repetition.
    Returns (q, k, v, raw attention scores).
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
        q, k_val, v, attn = _eager_attn_forward(
            query, key, value, num_heads, num_key_value_heads, scale_value, layout)
        del k_val
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
        q, k_val, v, attn = _eager_attn_forward(
            query, key, value, num_heads, num_key_value_heads, scale_value, layout)
        del k_val
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
            base_index.append(4)
        y = g.op(
            "Custom", *base_inputs,
            type_s="InnerPromptFlashAttention",
            input_names_s=_as_list_str(["query", "key", "value","pse_shift", "atten_mask"]),
            optional_input_names_s=_as_list_str(["pse_shift","atten_mask"]),
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
    """Custom SwiGLU op for ONNX export — fused SiLU-gate activation.

    Output shape = input shape with the target dim halved (chunk(2, dim) then
    element-wise silu(gate) * up). The symbolic MUST emit the correct halved
    dim explicitly (not `setType(x.type())`), otherwise downstream Reshape
    nodes fail GE infershape because the actual tensor size is half the
    declared size.
    """

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
        """ONNX symbolic for SwiGLU — correctly halves the target dim."""
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
        # Build a TensorType that copies input dims but halves the target dim.
        # For our call sites dim is always -1 and the last dim is static
        # (fused MLP gate+up projection = 2*intermediate_size, always known).
        # torch>=2.8 removed torch.onnx.TensorType; use the jit type directly:
        # x.type() -> torch._C.TensorType, .sizes() -> [int|None|str],
        # .with_sizes([...]) rebuilds the type keeping dtype/device. Get the
        # rank from sizes(), NOT .dim() -- in torch 2.8/2.9 the symbolic input's
        # .dim() is unreliable on the quantized cat->SwiGlu chain, leaving the
        # output declared with the UNHALVED dim and breaking the downstream
        # Reshape [-1,4096] ("cannot be divided from [4096]").
        try:
            x_type = x.type()
            if not hasattr(x_type, "sizes"):
                return y
            sizes = list(x_type.sizes())
            if sizes is None:
                # Dynamic/unknown input type (e.g. prefill rotary chain): cannot
                # halve statically; leave untyped (downstream uses real infershape).
                return y
            x_rank = len(sizes)
            d = int(dim)
            if d < 0:
                d += x_rank
            new_sizes = []
            for i, s in enumerate(sizes):
                val = s if isinstance(s, int) else None
                if i == d and val is not None:
                    val = val // 2
                new_sizes.append(val)
            new_type = x_type.with_sizes(new_sizes)
            y.setType(new_type)
        except (RuntimeError, TypeError, ValueError, AttributeError):
            # Fallback: downstream depends on actual op infershape, avoid
            # declaring a 2x-wrong size that would guarantee Reshape failure.
            pass
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


class _AllReduceCustom(torch.autograd.Function):
    """Custom AllReduce(sum) op for ONNX export.

    Eager fallback is identity (shape-preserving) -- only used during trace; the
    real cross-rank sum happens at runtime where the plugin lowers the Custom
    (type=AllReduce, op=sum, group=hccl_world_group, rank_size, fusion) node to a
    GE HcomAllReduce. Requires the convert.cc group-injection fix + fusion attr set.

    Each AllReduce is emitted with fusion_i=0. This is REQUIRED for the converter's
    ConvertHcomFusionId (convert.cc ~3855) to inject the `group` attr into the
    MindIR — any non-zero fusion_id makes it early-return, leaving the node
    without `group`, so GE's hccl_graph_optimizer aborts at PreRun
    (`hcom_graph_optimizer get attr "group" failed`). Verified TP=2 is
    token-identical with fusion_i=0. (The earlier unique-fusion-id scheme was a
    workaround for a TP=4 precision bug; it breaks group injection on this
    CANN/mslite and is incompatible with the current TP=2 path.)
    """

    # Kept for backward compatibility with any caller referencing the counter;
    # fusion_i is fixed at 0 now (see note above).
    _ALLREDUCE_FUSION_ID = [0]

    @classmethod
    def reset_fusion_id(cls):
        """Reset the AllReduce fusion ID counter so every rank matches."""
        cls._ALLREDUCE_FUSION_ID[0] = 0

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
            fusion_i=0,
        )
        y.setType(x.type())
        return y


def allreduce_sum(x):
    """AllReduce-sum across the TP group (no-op when TP_SIZE == 1)."""
    if TP_SIZE <= 1:
        return x
    return _AllReduceCustom.apply(x)


# Chunk size for the lm_head AllReduce. The full vocab (151936 elements fp16)
# AllReduce corrupts at TP=4 on the 300I Duo (2×2 topology). Chunking into
# smaller pieces (~32K each) avoids the corruption — the layer AllReduces
# (4096 elements) work fine, so ~32K is safely under whatever threshold
# triggers the bug.
LM_HEAD_AR_CHUNK = 32768


def _allreduce_lm_head(lh_partial):
    """AllReduce the lm_head partial logits, chunked to avoid large-tensor corruption.

    At TP=4, a single AllReduce on the full vocab tensor (151936 fp16 elements)
    produces garbage (values beyond fp16 range — memory corruption, not a sum
    error). Chunking into ~32K-element AllReduces sidesteps the bug while keeping
    the model lightweight (same compute as the original sharded approach).
    """
    if TP_SIZE <= 1:
        return lh_partial.float()
    v = lh_partial.shape[-1]
    # Only chunk at TP>=4 (the large-tensor corruption is 4-rank-specific;
    # 2p single AllReduce works fine, so preserve it for efficiency).
    if TP_SIZE < 4 or v <= LM_HEAD_AR_CHUNK:
        return allreduce_sum(lh_partial).float()
    chunks = []
    for start in range(0, v, LM_HEAD_AR_CHUNK):
        end = min(start + LM_HEAD_AR_CHUNK, v)
        chunks.append(allreduce_sum(lh_partial[..., start:end]))
    return torch.cat(chunks, dim=-1).float()


# ---------------------------------------------------------------------------
# Linear projection dispatch (always 2D MatMul — see _proj_linear)
# ---------------------------------------------------------------------------


def _proj_linear(x, weight, bias=None):
    """Linear projection x @ weight.T + bias, emitted as a 2D MatMul (always-on).

    Reshape x from (..., in) to (M, in) and use `torch.matmul(x2d, weight.t())`
    (NOT F.linear, which on 2D lowers to ONNX `Gemm`/transB and a slower kernel).
    weight.t() folds to a pre-transposed constant. Output is reshaped back to
    (*x.shape[:-1], out). Bit-identical to F.linear.

    Note: after export, each MatMul node gets an allow_nz=true attribute (see
    _set_allow_nz_on_matmul), letting GE's IsAllowNzMatmul allow MatMul to use the
    FRACTAL_NZ format (under a full fp16 graph, a plain MatMul defaults to ND,
    causing inefficient vector-unit data rearrangement; see heavy_format_propagation).
    """
    in_shape = x.shape
    y = torch.matmul(x.reshape(-1, in_shape[-1]), weight.t())
    if bias is not None:
        y = y + bias
    return y.reshape(*in_shape[:-1], -1)


# ---------------------------------------------------------------------------
# A8W8 quantized projection (AMCT A2 eager path)
#
# AMCT quantizes the torch model and saves the weights (a2_quant_extract.py ->
# --quant-dir); this script emits the same Custom(AscendQuant)/
# Custom(AscendDequant) chain as AMCT's deploy onnx (rewrite_custom3.py):
#
#   Mul(x, scale_factor)                       smoothquant smooth weight [k]
#   Custom(AscendQuant): 1 input; attrs {scale=scale_d, offset=act_offset,
#       dst_type=2(CANN INT8), dtype=3(onnx INT8 out)}
#   MatMul(int8, int8) -> int32                (w_int8 [k,n], already transposed)
#   Add(int32 bias)                            quantized int32 bias [n]
#   Custom(AscendDequant): 2 inputs (x int32, deq_scale u64 [n]); NO dtype attr
#
# Column-parallel (q/k/v/gate/up, TP): slice w_int8[:,s:e], bias[s:e],
# deq_scale[s:e]. Row-parallel (o_proj, TP): slice w_int8[s:e,:], scale_factor[s:e].
# skip_layers (down_proj, lm_head) stay fp16 -> plain _proj_linear.
# ---------------------------------------------------------------------------
QUANT_PACK = None  # {module_path: {w_int8, scale_factor, scale_d, act_offset, bias, deq_scale}}


def _load_quant_pack(quant_dir):
    """Load AMCT-extracted quantized weights (manifest.json + npy files)."""
    global QUANT_PACK
    if not quant_dir:
        QUANT_PACK = None
        return
    with open(Path(quant_dir) / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    raw = {}
    for key, meta in manifest.items():
        name, field = key.split("::")
        arr = np.load(Path(quant_dir) / meta["path"])
        raw.setdefault(name, {})[field] = arr
    pack = {}
    for name, d in raw.items():
        t = {}
        for k, v in d.items():
            if k == "w_int8":
                t[k] = torch.from_numpy(v)  # int8 [k, n]
            elif k == "bias":
                t[k] = torch.from_numpy(v)  # int32 [n]
            elif k == "deq_scale":
                t[k] = torch.from_numpy(v)  # uint64 [n]
            elif k == "scale_factor":
                t[k] = torch.from_numpy(v).to(torch.float16).squeeze(0)  # [1,k] -> [k]; TP row-slice (sf[s:e]) then cuts the K axis
            else:
                t[k] = torch.from_numpy(v).float()
        pack[name] = t
    QUANT_PACK = pack
    print(f"[quant] loaded {len(pack)} quantized modules from {quant_dir}")


def _quant_for(name):
    """Return quant dict for module path ('model.layers.0.self_attn.q_proj') or None."""
    if not QUANT_PACK:
        return None
    return QUANT_PACK.get(name)


class _AscendQuantCustom(torch.autograd.Function):
    """Custom(AscendQuant): fp16 -> int8, attrs {scale=scale_d, offset=act_offset,
    dst_type=2, dtype=3}. torch forward is a simulation only (tracing); the GE
    runtime executes the real kernel."""

    @staticmethod
    def forward(ctx, x, scale: float, offset: float):
        del ctx
        xq = torch.clamp(torch.round(x.float() / scale) + offset, -128, 127)
        return xq.to(torch.int8)

    @staticmethod
    def symbolic(g, x, scale: float, offset: float):
        """Emit Custom(AscendQuant) with GE's multiplicative scale semantics."""
        # GE interprets scale multiplicatively (x*scale+offset) while the torch
        # fallback divides (scale_d = 1/act_scale); export scale_f=1/scale_d so
        # both semantics match.
        y = g.op("Custom", x,
                 type_s="AscendQuant",
                 input_names_s=["x"],
                 output_names_s=["y"],
                 output_num_i=1,
                 input_index_i=[0],
                 scale_f=float(1.0 / scale),
                 offset_f=float(offset),
                 dst_type_i=2,
                 dtype_i=3)
        return y.setType(x.type().with_dtype(torch.int8))


class _AscendDequantCustom(torch.autograd.Function):
    """Custom(AscendDequant): (int32, u64 deq_scale) -> fp16. NO dtype attr
    (converter parses dtype as output datatype -> UNDEFINED -> mindir crash)."""

    @staticmethod
    def forward(ctx, x, deq_scale):
        del ctx
        # Simulation only: y ~= x * deq_scale (fp32 domain, then fp16).
        return (x.float() * deq_scale.float()).to(torch.float16)

    @staticmethod
    def symbolic(g, x, deq_scale):
        y = g.op("Custom", x, deq_scale,
                 type_s="AscendDequant",
                 input_names_s=["x", "deq_scale"],
                 output_names_s=["y"],
                 output_num_i=1,
                 input_index_i=[0, 1])
        return y.setType(x.type().with_dtype(torch.float16))


class _QBMCV3Custom(torch.autograd.Function):
    """Custom(QuantBatchMatmulV3): fused int8@int8 matmul + bias + dequant -> fp16.

    Slots match the rewrite_qdq3_to_qbmv3.py graph rewrite: x1=int8 act,
    x2=int8 weight, scale=uint64 deq_scale, offset=float32 zeros (placeholder),
    bias=int32. The kernel reads the low 32 bits of scale as float32. forward is
    a shape-only simulation; GE runs the real kernel. split_matmul_by_chunk
    keeps each chunk M<=512 to avoid the QBMV3 tiling SIGSEGV at M>=1024.
    """

    @staticmethod
    def forward(ctx, x1, x2, scale_u64, offset_f32, bias_i32):
        """Trace-only shape simulation of the fused QBMV3 kernel (real values from GE)."""
        # Trace-only shape simulation (fp16 [M, N]); real values come from the
        # GE kernel. Avoid bit tricks (view/int64 AND) -- trace emits them as
        # aten::view ONNX nodes and fails. ctx/scale_u64/offset_f32 exist only
        # for the .apply signature; the GE kernel reads them at runtime.
        del ctx, scale_u64, offset_f32
        y = x1.float() @ x2.float()  # [M, N]
        y = y * 1.0  # deq approximated as 1.0; shape propagation only
        if bias_i32 is not None:
            y = y + bias_i32.float()
        return y.to(torch.float16)

    @staticmethod
    def symbolic(g, x1, x2, scale_u64, offset_f32, bias_i32):
        """Emit Custom(QuantBatchMatmulV3); output type is [M, N] (w's columns)."""
        y = g.op("Custom", x1, x2, scale_u64, offset_f32, bias_i32,
                 type_s="QuantBatchMatmulV3",
                 input_names_s=_as_list_str(["x1", "x2", "scale", "offset", "bias"]),
                 output_names_s=_as_list_str(["y"]),
                 dtype_i=10)
        # Output type must reflect w's column count [M, N], not x1's [M, K]:
        # declaring [M, K] (4096) while the kernel emits N=2048 doubles the
        # gate/up sizes and breaks the downstream Reshape [-1,4096]
        # ("cannot be divided from [4096]").
        try:
            x1_t = x1.type()
            x2_t = x2.type()
            s1 = list(x1_t.sizes()) if hasattr(x1_t, "sizes") else None
            s2 = list(x2_t.sizes()) if hasattr(x2_t, "sizes") else None
            if s1 is not None and s2 is not None and len(s1) >= 2 and len(s2) >= 2:
                # x1 [M, K], x2(w) [K, N] -> out [M, N]
                new_sizes = list(s1[:-1]) + [s2[-1]]
                y.setType(x1_t.with_sizes(new_sizes).with_dtype(torch.float16))
        except (RuntimeError, TypeError, ValueError, AttributeError):
            pass
        return y


def _quant_proj(x, qd, mode="col", slice_=None):
    """Quantized linear projection: smooth -> AscendQuant -> int8 QBMV3(fused
    matmul+bias+dequant -> fp16, chunk_size=512 split).

    qd: dict from _quant_for(); mode 'col' = column-parallel (slice output n),
    'row' = row-parallel (slice input k); slice_=(s,e) applied when TP_SIZE>1.
    """
    w = qd["w_int8"]  # [k, n] int8 (already transposed)
    sf = qd["scale_factor"]  # [k] smooth weight
    scale_d = float(qd["scale_d"][0]) if qd["scale_d"].dim() else float(qd["scale_d"])
    offset = float(qd["act_offset"][0]) if qd["act_offset"].dim() else float(qd["act_offset"])
    bias = qd.get("bias")
    deq = qd["deq_scale"]  # [n] u64
    if TP_SIZE > 1 and slice_ is not None:
        s, e = slice_
        if mode == "col":
            w = w[:, s:e]
            deq = deq[s:e]
            if bias is not None:
                bias = bias[s:e]
        else:
            w = w[s:e, :]
            sf = sf[s:e]
    # Move quantized weights to x's device (NPU).
    dev = x.device
    w = w.to(dev)
    sf = sf.to(dev)
    deq = deq.to(dev)
    if bias is not None:
        bias = bias.to(dev)
    in_shape = x.shape
    x = x * sf  # smoothquant (broadcast [k] over last dim)
    xq = _AscendQuantCustom.apply(x, scale_d, offset)
    # One fused QBMV3 (matmul + bias + dequant -> fp16), no chunking.
    offset_zeros = torch.zeros(deq.shape, dtype=torch.float32)
    y = _QBMCV3Custom.apply(xq.reshape(-1, in_shape[-1]), w.to(torch.int8),
                            deq, offset_zeros, bias)  # int8 x int8 -> fp16 (fused QBMV3)
    return y.reshape(*in_shape[:-1], -1)


def split_matmul_by_chunk(x: torch.Tensor, w: torch.Tensor, deq: torch.Tensor,
                          offset_f32: torch.Tensor, bias: torch.Tensor,
                          chunk_size: int = 512):
    """Chunk the activation along dim 0 into chunk_size-row pieces, emit one
    QBMV3 (fused matmul+bias+dequant -> fp16) per chunk, then concat(dim=0).

    x: int8 [M, K] (AscendQuant output), w: int8 [K, N], deq: uint64 [N],
    offset_f32: float32 [N] (all-zero placeholder), bias: int32 [N] or None.
    Only the M dim is chunked; w/deq/bias stay whole (TP slicing of
    w/deq/bias/sf already happened in _quant_proj).
    """
    parts = []
    rows = x.shape[0]
    for start in range(0, rows, chunk_size):
        end = start + chunk_size
        x_chunk = x[start:end, :]
        part = _QBMCV3Custom.apply(x_chunk, w, deq, offset_f32, bias)
        parts.append(part)
    out = torch.cat(parts, dim=0)
    return out


def _set_allow_nz_on_matmul(onnx_path):
    """Add the allow_nz=true attribute to all MatMul/MatMulV2/BatchMatMulV2 nodes in an ONNX graph.

    Corresponds to IsAllowNzMatmul in the GE source ops_kernel_common.cc:
        bool allow_nz = false;
        AttrUtils::GetBool(node->GetOpDesc(), "allow_nz", allow_nz);
        return allow_nz;
    Only MatMul nodes with allow_nz=true are permitted by heavy_format_propagation
    to use FRACTAL_NZ. In ONNX, int 1 is used to represent bool true (the GE-side
    GetBool is compatible with int values).

    IMPORTANT: only the model header is rewritten (load_external_data=False +
    onnx.save). Do NOT use onnx.save_model(save_as_external_data=True): onnx
    1.17 inflates fp16 external data ~3x (1.24GB -> 3.7GB) and converter_lite
    fails with a memcpy_s size mismatch. External files written by
    torch.onnx.export stay untouched (correct fp16 sizes).
    """
    from onnx import helper
    m = onnx.load(onnx_path, load_external_data=False)
    added = 0
    for node in m.graph.node:
        if node.op_type in ("MatMul", "MatMulV2", "BatchMatMulV2"):
            # Skip if an allow_nz attribute already exists
            if any(a.name == "allow_nz" for a in node.attribute):
                continue
            node.attribute.append(helper.make_attribute("allow_nz", 1))
            added += 1
    if added:
        # Header-only write: initializers keep data_location=EXTERNAL and the
        # original location/offset/length references; external files are NOT
        # touched, so the fp16 weights stay byte-exact.
        onnx.save(m, onnx_path)
    print(f"[allow_nz] {onnx_path}: set allow_nz=true on {added} MatMul nodes")


def _compute_qkv(attn_mod, hidden_states, layer_idx):
    """Compute fused QKV projection and reshape to (batch, seq, heads, head_dim).

    Applies q_norm / k_norm (per-head RMSNorm, TP-sliced) when present.
    Returns (query_states, key_states, value_states, input_shape).

    A8W8 quantized path: q/k/v each have their OWN act scale/offset (per-tensor
    calibration), so the fused projection is split into three independent
    quantized projections (smooth -> Quant -> int8 MatMul -> Dequant each), then
    re-cat into the same [q | k | v] layout. Skip (non-quantized) modules keep
    the original single fused MatMul.
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
    q_b, k_b, v_b = attn_mod.q_proj.bias, attn_mod.k_proj.bias, attn_mod.v_proj.bias

    q_out = q_per
    kv_out = kv_per
    pfx = "model.layers.%d.self_attn" % layer_idx
    qd_q, qd_k, qd_v = (_quant_for(pfx + ".q_proj"), _quant_for(pfx + ".k_proj"),
                        _quant_for(pfx + ".v_proj"))
    if qd_q and qd_k and qd_v:
        q = _quant_proj(hidden_states, qd_q, "col", (qs, qe))      # [..., q_out]
        k = _quant_proj(hidden_states, qd_k, "col", (ks, ke))      # [..., kv_out]
        v = _quant_proj(hidden_states, qd_v, "col", (ks, ke))      # [..., kv_out]
        qkv = torch.cat([q, k, v], dim=-1)
    else:
        w = torch.cat([q_w[qs:qe], k_w[ks:ke], v_w[ks:ke]], dim=0)
        if q_b is None:
            b = None
        else:
            b = torch.cat([q_b[qs:qe], k_b[ks:ke], v_b[ks:ke]], dim=0)
        qkv = _proj_linear(hidden_states, w, b)

    query = qkv[..., :q_out].view(hidden_shape)
    key = qkv[..., q_out:q_out + kv_out].view(hidden_shape)
    value = qkv[..., q_out + kv_out:].view(hidden_shape)
    if hasattr(attn_mod, "q_norm"):
        query = _rms_norm_layer(attn_mod.q_norm, query)
    if hasattr(attn_mod, "k_norm"):
        key = _rms_norm_layer(attn_mod.k_norm, key)
    return query, key, value, input_shape


def _run_prefill_attention(query, key, value, attention_mask, num_heads,
                           num_kv_heads, scaling, past_len=0):
    """Run prefill-path attention via PromptFlashAttention (BSND).

    query/key/value are (batch, seq, heads, head_dim) = BSND. For 1p/2p
    (num_kv_heads >= 4), uses PFA's native GQA path directly -- no manual
    Expand, saving 2 Expand ops/layer (72 total at 36 layers). For TP>=4
    (2 KV heads/rank), manual expand+reshape is still used because PFA's
    native GQA produces wrong partials at small per-rank KV-head counts
    (diagnosed: 4p prefill logits cosine 0.23 vs 1p with native GQA).
    KV returned to the cache is the ORIGINAL (un-repeated) per-rank shard,
    matching decode's past_key_cache layout.
    """
    q_len, k_len = query.shape[1], key.shape[1]
    flash_mask = _make_flash_attn_mask(
        attention_mask, q_len, k_len, past_len)
    k_pfa, v_pfa, n_kv_pfa = key, value, num_kv_heads
    if 0 < num_kv_heads < num_heads and TP_SIZE >= 4:
        rep = num_heads // num_kv_heads
        b, s, _, d = key.shape
        k_pfa = key.unsqueeze(3).expand(b, s, num_kv_heads, rep, d).reshape(b, s, num_heads, d)
        v_pfa = value.unsqueeze(3).expand(b, s, num_kv_heads, rep, d).reshape(b, s, num_heads, d)
        n_kv_pfa = num_heads
    attn_output = prompt_flash_attention(
        query, k_pfa, v_pfa, atten_mask=flash_mask,
        num_heads=num_heads, scale_value=float(scaling),
        input_layout="BSND", num_key_value_heads=n_kv_pfa,
        sparse_mode=0, inner_precise=1)
    # cache: original (un-repeated) kv, BNSD fp16 (matches decode past_key_cache)
    return attn_output, key.permute(0, 2, 1, 3), value.permute(0, 2, 1, 3)


def _run_suffix_attention(query, key, value, attention_mask, past_key, past_value,
                          num_heads, num_kv_heads, scaling):
    """Suffix attention using the same PromptFlashAttention path as prefix."""
    past_len = past_key.shape[2]
    # Prefix cache is BNSD while the shared PFA helper consumes BSND.
    key = torch.cat([past_key.permute(0, 2, 1, 3), key], dim=1)
    value = torch.cat([past_value.permute(0, 2, 1, 3), value], dim=1)
    return _run_prefill_attention(
        query, key, value, attention_mask, num_heads, num_kv_heads,
        scaling, past_len=past_len)


def _text_attn_forward_suffix(attn_mod, hidden_states, cos4, sin4,
                              attention_mask, past_key, past_value, layer_idx=0):
    """Run one suffix-prefill attention layer using reusable prefix KV."""
    num_heads = attn_mod.config.num_attention_heads // TP_SIZE
    num_kv_heads = attn_mod.config.num_key_value_heads // TP_SIZE
    query, key, value, input_shape = _compute_qkv(attn_mod, hidden_states, layer_idx)
    query = rotary_mul(query, cos4, sin4)
    key = rotary_mul(key, cos4, sin4)
    output, key, value = _run_suffix_attention(
        query, key, value, attention_mask, past_key, past_value,
        num_heads, num_kv_heads,
        getattr(attn_mod, "scaling", 1.0 / (attn_mod.head_dim ** 0.5)))
    output = output.reshape(*input_shape, -1)

    qd_o = _quant_for("model.layers.%d.self_attn.o_proj" % layer_idx)
    if qd_o:
        if TP_SIZE > 1:
            q_dim_local = num_heads * attn_mod.head_dim
            output = allreduce_sum(_quant_proj(
                output, qd_o, "row",
                (TP_RANK * q_dim_local, (TP_RANK + 1) * q_dim_local)))
        else:
            output = _quant_proj(output, qd_o, "row", None)
    elif TP_SIZE > 1:
        q_dim_local = num_heads * attn_mod.head_dim
        weight = attn_mod.o_proj.weight[
            :, TP_RANK * q_dim_local:(TP_RANK + 1) * q_dim_local]
        output = allreduce_sum(_proj_linear(output, weight, attn_mod.o_proj.bias))
    else:
        output = _proj_linear(output, attn_mod.o_proj.weight, attn_mod.o_proj.bias)
    return output, key, value


def _run_decode_attention(query, key, value, attention_mask, attn_mod, num_heads, num_kv_heads):
    """Run decode-path attention via IncreFlashAttention (BNSD).

    For 1p/2p (num_kv_heads >= 4), uses IncreFlash's native GQA path directly --
    no manual Expand, saving 2 Expand ops/layer (72 total at 36 layers). For
    TP>=4 (2 KV heads/rank), manual expand+reshape is still used because the
    kernel's GQA path is unreliable at small per-rank KV-head counts (verified:
    native GQA gives random garbage at TP=4). expand+reshape (not
    repeat_interleave) is used because repeat_interleave exports as a Split op
    that GE can't compile. Returns attn_output in BNSD layout.
    """
    scaling = getattr(attn_mod, "scaling", 1.0 / (attn_mod.head_dim ** 0.5))
    if 0 < num_kv_heads < num_heads and TP_SIZE >= 4:
        rep = num_heads // num_kv_heads
        b, _, length, d = key.shape
        key = key.unsqueeze(2).expand(b, num_kv_heads, rep, length, d).reshape(b, num_heads, length, d)
        value = value.unsqueeze(2).expand(b, num_kv_heads, rep, length, d).reshape(b, num_heads, length, d)
        num_kv_heads = num_heads  # now MHA: every q head has its own (repeated) kv head
    pad_mask = attention_mask[:, None, None, :] == 0
    return incre_flash_attention(
        query, key, value, pad_mask,
        num_heads=num_heads, scale_value=float(scaling),
        input_layout="BNSD", num_key_value_heads=num_kv_heads, inner_precise=1)


def _text_attn_forward(attn_mod, hidden_states, cos4, sin4, attention_mask,
                       cache_pos, past_key, past_value, layer_idx=0):
    """Dispatch attention computation to prefill or decode path.

    Prefill: PromptFlashAttention (BSND) with causal+padding mask.
    Decode: IncreFlashAttention (BNSD, native GQA) with KV cache scatter update.
    """
    num_heads = attn_mod.config.num_attention_heads
    num_kv_heads = attn_mod.config.num_key_value_heads

    query, key, value, input_shape = _compute_qkv(attn_mod, hidden_states, layer_idx)

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

    num_heads_local = num_heads // TP_SIZE
    num_kv_heads_local = num_kv_heads // TP_SIZE
    if past_key is None:
        attn_output, key, value = _run_prefill_attention(
            query, key, value, attention_mask, num_heads_local, num_kv_heads_local,
            getattr(attn_mod, "scaling", 1.0 / (attn_mod.head_dim ** 0.5)))
        out = attn_output.reshape(*input_shape, -1)
    else:
        # Decode: IncreFlash with native GQA for 1p/2p (num_kv_heads >= 4),
        # manual expand for TP>=4 (kernel GQA unreliable at 2 KV heads/rank).
        attn_output = _run_decode_attention(
            query, key, value, attention_mask, attn_mod, num_heads_local, num_kv_heads_local)
        out = attn_output.transpose(1, 2).reshape(*input_shape, -1)

    if DEBUG_TAP:
        _TAP_RAW_OUT.append(out)  # raw attention output (heads concatenated), before o_proj

    # o_proj: row-parallel under TP (input dim = local q-heads * head_dim), then AllReduce.
    qd_o = _quant_for("model.layers.%d.self_attn.o_proj" % layer_idx)
    if qd_o:
        # Quantized o_proj: row slice on w_int8 [k,n] dim0 + scale_factor [k].
        # TP row-parallel: each rank's QBMV output is a partial sum over its
        # input slice, must AllReduce before use (mirrors the non-quant path).
        if TP_SIZE > 1:
            q_dim_local = num_heads_local * attn_mod.head_dim
            out_proj = allreduce_sum(_quant_proj(out, qd_o, "row",
                                                 (TP_RANK * q_dim_local, (TP_RANK + 1) * q_dim_local)))
        else:
            out_proj = _quant_proj(out, qd_o, "row", None)
    elif TP_SIZE > 1:
        q_dim_local = num_heads_local * attn_mod.head_dim
        o_w = attn_mod.o_proj.weight[:, TP_RANK * q_dim_local:(TP_RANK + 1) * q_dim_local]
        out_proj = allreduce_sum(_proj_linear(out, o_w, attn_mod.o_proj.bias))
    else:
        out_proj = _proj_linear(out, attn_mod.o_proj.weight, attn_mod.o_proj.bias)
    return out_proj, key, value


# ---------------------------------------------------------------------------
# MLP helpers
# ---------------------------------------------------------------------------


def _mlp_gate_up_linear(mlp_mod, x, layer_idx):
    """Fused gate+up projection -> returns the merged [gate | up] tensor directly.

    The fused GEMM weight cat([gate_w, up_w]) makes the output [gate | up] in
    gate-then-up order, which is EXACTLY what SwiGlu consumes (it chunks the last
    dim in half internally). Returning the merged tensor AVOIDS a split-then-recat
    pair (StridedSliceD + ConcatD) per layer that is a pure identity no-op.
    TP: column-parallel -- each rank holds intermediate/TP rows of gate and up.

    A8W8 quantized path: gate/up have independent act scales -> two separate
    quantized projections, re-cat into [gate | up] (SwiGlu chunks internally,
    so the re-cat keeps the same layout as the fused path).
    """
    gate_w = mlp_mod.gate_proj.weight
    up_w = mlp_mod.up_proj.weight
    gate_b = mlp_mod.gate_proj.bias
    up_b = mlp_mod.up_proj.bias
    g_per = int(gate_w.shape[0]) // TP_SIZE
    gs, ge = TP_RANK * g_per, (TP_RANK + 1) * g_per
    pfx = "model.layers.%d.mlp" % layer_idx
    qd_g, qd_u = _quant_for(pfx + ".gate_proj"), _quant_for(pfx + ".up_proj")
    if qd_g and qd_u:
        gate = _quant_proj(x, qd_g, "col", (gs, ge))   # [..., intermediate]
        up = _quant_proj(x, qd_u, "col", (gs, ge))     # [..., intermediate]
        return torch.cat([gate, up], dim=-1)           # [gate | up], feeds SwiGlu
    w = torch.cat([gate_w[gs:ge], up_w[gs:ge]], dim=0)
    b = None if gate_b is None else torch.cat([gate_b[gs:ge], up_b[gs:ge]], dim=0)
    return _proj_linear(x, w, b)   # already [gate | up]; feed directly to SwiGlu


def _run_mlp(layer, hidden_states, layer_idx=0):
    """Run MLP forward pass with fused gate+up projection and SwiGLU activation."""
    mlp = layer.mlp
    if hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj") and hasattr(mlp, "down_proj"):
        gate_up = _mlp_gate_up_linear(mlp, hidden_states, layer_idx)   # [gate | up] fused
        act = swiglu(gate_up, dim=-1)   # SwiGlu chunks in half internally; no cat needed
        if TP_SIZE > 1:
            # down_proj row-parallel: input dim = intermediate/TP, then AllReduce.
            d_w = mlp.down_proj.weight
            in_per = int(d_w.shape[1]) // TP_SIZE
            d_w_local = d_w[:, TP_RANK * in_per:(TP_RANK + 1) * in_per]
            return allreduce_sum(_proj_linear(act, d_w_local, mlp.down_proj.bias))
        return _proj_linear(act, mlp.down_proj.weight, mlp.down_proj.bias)
    return mlp(hidden_states)


def _pad_kv_by_output_tokens(kv_tensor, extra_len=MAX_OUTPUT_TOKENS):
    """Pad KV cache tensor by a *fixed* number of extra KV slots on axis 2.

    No longer called from the prefill graph — all paths (1p/2p/4p) now pad
    KV on the host side instead, eliminating 72 ConcatD ops from the graph.
    Kept for reference / potential future use.

    Always append exactly `extra_len` (default MAX_OUTPUT_TOKENS=512) zero rows,
    so the KV output length = current seq + 512 regardless of bucket. Since
    `extra_len` is a pure python int constant baked into the ONNX concat-shape,
    GE correctly specializes the final KV length per bucket when it recompiles
    for a different input seq (ge.dynamicDims 6 buckets):

        bucket seq=512  -> KV output len =  512 + 512 = 1024
        bucket seq=3072 -> KV output len = 3072 + 512 = 3584
    """
    if extra_len <= 0:
        return kv_tensor
    zeros = kv_tensor.new_zeros(kv_tensor.shape[0], kv_tensor.shape[1],
                                int(extra_len), kv_tensor.shape[3])
    return torch.cat([kv_tensor, zeros], dim=2)


# ---------------------------------------------------------------------------
# Prefill / Decode wrapper modules
# ---------------------------------------------------------------------------


class Qwen3LlmPrefill(torch.nn.Module):
    """Qwen3-8B LLM Prefill wrapper — processes full prompt and outputs KV
    cache of length seq. The inference script pads KV to seq + 512 on the
    host side, eliminating 72 ConcatD ops from the graph.

    3 inputs (input_ids, attention_mask, position_ids) — KV output dim is seq;
    host-side padding to kv_len = seq + 512 is done by the inference script
    (1p: infer_qwen3_8b_mslite_1p._prefill; 2p: _reconstruct_kv_tp; 4p: _tp_hybrid_prefill).
    """

    def __init__(self, model, lm_head, pad_kv=True):
        """Initialize prefill wrapper; pad_kv controls decode-tail padding."""
        self.pad_kv = bool(pad_kv)
        super().__init__()
        self.model = model.model
        self.lm_head = lm_head

    def forward(self, input_ids, attention_mask, position_ids):
        """Run prefill: 3 inputs, 3 outputs (logits, K, V).

        KV output dim = seq (no graph-side padding). The inference script
        pads to kv_len = seq + 512 on the host side, eliminating 72 ConcatD
        ops from the graph.
        """
        inputs_embeds = self.model.embed_tokens(input_ids)
        cos, sin = self.model.rotary_emb(inputs_embeds, position_ids)
        cos4 = cos.unsqueeze(2) if cos.dim() == 3 else cos
        sin4 = sin.unsqueeze(2) if sin.dim() == 3 else sin

        present_k, present_v = [], []
        hidden_states = inputs_embeds
        for i, layer in enumerate(self.model.layers):
            residual = hidden_states
            hidden_states = _rms_norm_layer(layer.input_layernorm, hidden_states)
            attn_out, pk, pv = _text_attn_forward(
                layer.self_attn, hidden_states, cos4, sin4, attention_mask,
                None, None, None, i)
            hidden_states = residual + attn_out
            residual = hidden_states
            hidden_states = _rms_norm_layer(layer.post_attention_layernorm, hidden_states)
            hidden_states = residual + _run_mlp(layer, hidden_states, i)
            present_k.append(pk)
            present_v.append(pv)

        hidden_states = _rms_norm_layer(self.model.norm, hidden_states)
        # Sampling only needs logits[0, -1, :] (the real last token, considering right padding).
        # Use attention_mask.sum-1 to locate the real last token, index it directly to 1D [hidden],
        # then do matmul — compared with index_select keeping [1,1,hidden], this saves the final logits
        # copy time and the extra 1x1 dimension transfer (output logits is 1D [vocab], downstream reshape compatible).
        idx = attention_mask.sum(dim=1) - 1
        hidden_states = hidden_states[0, idx[0], :]
        if TP_SIZE > 1:
            h_per = hidden_states.shape[-1] // TP_SIZE
            hs, he = TP_RANK * h_per, (TP_RANK + 1) * h_per
            lh_w = self.lm_head.weight[:, hs:he]
            lh_partial = _proj_linear(hidden_states[hs:he], lh_w, self.lm_head.bias)
            logits = _allreduce_lm_head(lh_partial)
        else:
            logits = _proj_linear(hidden_states, self.lm_head.weight, self.lm_head.bias).float()
        # Pad the KV cache with 512 empty slots inside the graph: output kv_len = seq + 512,
        # matching the decode past_key_cache / past_value_cache inputs
        # (decode inputs kv_len = seq + MAX_OUTPUT_TOKENS), avoiding host-side padding.
        k_out = torch.stack(present_k, dim=0)  # [L, B, H, seq, D]
        v_out = torch.stack(present_v, dim=0)
        if self.pad_kv and MAX_OUTPUT_TOKENS > 0:
            k_pad = k_out.new_zeros(k_out.shape[0], k_out.shape[1], k_out.shape[2],
                                    MAX_OUTPUT_TOKENS, k_out.shape[4])
            v_pad = v_out.new_zeros(v_out.shape[0], v_out.shape[1], v_out.shape[2],
                                    MAX_OUTPUT_TOKENS, v_out.shape[4])
            k_out = torch.cat([k_out, k_pad], dim=3)
            v_out = torch.cat([v_out, v_pad], dim=3)
        return logits, k_out, v_out


class Qwen3LlmPrefix(torch.nn.Module):
    """Common-prefix graph: compute reusable per-layer K/V only."""

    def __init__(self, model):
        super().__init__()
        self.model = model.model

    def forward(self, input_ids, attention_mask, position_ids):
        """Run the common-prefix prefill and return stacked per-layer K/V only."""
        inputs_embeds = self.model.embed_tokens(input_ids)
        cos, sin = self.model.rotary_emb(inputs_embeds, position_ids)
        cos4 = cos.unsqueeze(2) if cos.dim() == 3 else cos
        sin4 = sin.unsqueeze(2) if sin.dim() == 3 else sin
        present_k, present_v = [], []
        hidden_states = inputs_embeds
        for i, layer in enumerate(self.model.layers):
            residual = hidden_states
            hidden_states = _rms_norm_layer(layer.input_layernorm, hidden_states)
            attn_out, pk, pv = _text_attn_forward(
                layer.self_attn, hidden_states, cos4, sin4, attention_mask,
                None, None, None, i)
            hidden_states = residual + attn_out
            residual = hidden_states
            hidden_states = _rms_norm_layer(layer.post_attention_layernorm, hidden_states)
            hidden_states = residual + _run_mlp(layer, hidden_states, i)
            present_k.append(pk)
            present_v.append(pv)
        # Do not run lm_head: a common prefix is cached, not sampled.
        return torch.stack(present_k, dim=0), torch.stack(present_v, dim=0)


class Qwen3LlmSuffix(torch.nn.Module):
    """Suffix graph: consume prefix KV and return logits plus decode-ready KV."""

    def __init__(self, model, lm_head):
        super().__init__()
        self.model = model.model
        self.lm_head = lm_head

    def forward(self, input_ids, attention_mask, position_ids,
                past_key_cache, past_value_cache):
        """Consume prefix K/V, run suffix layers, and return last-token logits + full K/V."""
        inputs_embeds = self.model.embed_tokens(input_ids)
        cos, sin = self.model.rotary_emb(inputs_embeds, position_ids)
        cos4 = cos.unsqueeze(2) if cos.dim() == 3 else cos
        sin4 = sin.unsqueeze(2) if sin.dim() == 3 else sin
        hidden_states = inputs_embeds
        present_k, present_v = [], []
        for i, layer in enumerate(self.model.layers):
            residual = hidden_states
            hidden_states = _rms_norm_layer(layer.input_layernorm, hidden_states)
            attn_out, pk, pv = _text_attn_forward_suffix(
                layer.self_attn, hidden_states, cos4, sin4, attention_mask,
                past_key_cache[i], past_value_cache[i], i)
            hidden_states = residual + attn_out
            residual = hidden_states
            hidden_states = _rms_norm_layer(layer.post_attention_layernorm, hidden_states)
            hidden_states = residual + _run_mlp(layer, hidden_states, i)
            present_k.append(pk)
            present_v.append(pv)
        hidden_states = _rms_norm_layer(self.model.norm, hidden_states)[:, -1:, :]
        if TP_SIZE > 1 and not REPLICATE_LM_HEAD:
            hidden_per_rank = hidden_states.shape[-1] // TP_SIZE
            start = TP_RANK * hidden_per_rank
            end = (TP_RANK + 1) * hidden_per_rank
            partial = _proj_linear(
                hidden_states[..., start:end], self.lm_head.weight[:, start:end],
                self.lm_head.bias)
            logits = _allreduce_lm_head(partial)
        else:
            logits = _proj_linear(
                hidden_states, self.lm_head.weight, self.lm_head.bias).float()
        # Keep the complete cache so the existing decode graph can continue
        # generation after the suffix produces the first token.
        key_cache = torch.stack(present_k, dim=0)
        value_cache = torch.stack(present_v, dim=0)
        if MAX_OUTPUT_TOKENS > 0:
            key_pad = key_cache.new_zeros(
                key_cache.shape[0], key_cache.shape[1], key_cache.shape[2],
                MAX_OUTPUT_TOKENS, key_cache.shape[4])
            value_pad = value_cache.new_zeros(
                value_cache.shape[0], value_cache.shape[1], value_cache.shape[2],
                MAX_OUTPUT_TOKENS, value_cache.shape[4])
            key_cache = torch.cat([key_cache, key_pad], dim=3)
            value_cache = torch.cat([value_cache, value_pad], dim=3)
        return logits, key_cache, value_cache


class Qwen3LlmDecode(torch.nn.Module):
    """Qwen3-8B LLM Decode wrapper — single-token generation with KV cache update."""

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

        # NOTE: unbind(0) exports as a single Split with N outputs that GE can
        # miscompile in deep TP graphs (silent precision corruption). Index each
        # layer separately instead -> N independent Slice ops, which compile
        # correctly.
        num_layers = len(self.model.layers)
        past_k_layers = [past_key_cache[i] for i in range(num_layers)]
        past_v_layers = [past_value_cache[i] for i in range(num_layers)]
        present_k, present_v = [], []
        hidden_states = inputs_embeds
        tap_attn_out = tap_post_attn = tap_post_mlp = None  # only set when DEBUG_TAP and i==0
        global _TAP_RAW_OUT
        if DEBUG_TAP:
            _TAP_RAW_OUT = []

        for i, layer in enumerate(self.model.layers):
            residual = hidden_states
            # Use Custom RmsNorm for ALL norms (fused Ascend C op, much faster than
            # native Pow/ReduceMean/Sqrt/Div on Vector). The TP=4 GE miscompile concern
            # does not apply to 1p/2p. For TP=4, DEBUG_TAP already works around it.
            hidden_states = _rms_norm_layer(layer.input_layernorm, hidden_states)
            attn_out, pk, pv = _text_attn_forward(
                layer.self_attn, hidden_states, cos4, sin4, attention_mask,
                position_ids, past_k_layers[i], past_v_layers[i], i)
            hidden_states = residual + attn_out
            if DEBUG_TAP and i == 0:
                tap_attn_out = attn_out  # layer-0 attention output (post o_proj AllReduce)
                tap_post_attn = hidden_states  # embed + attn_out_0 (before MLP)
            residual = hidden_states
            hidden_states = _rms_norm_layer(layer.post_attention_layernorm, hidden_states)
            hidden_states = residual + _run_mlp(layer, hidden_states, i)
            if DEBUG_TAP and i == 0:
                tap_post_mlp = hidden_states  # layer-0 full output
            present_k.append(pk)
            present_v.append(pv)

        hidden_states = _rms_norm_layer(self.model.norm, hidden_states)
        if TP_SIZE > 1:
            # lm_head row-parallel: slice the hidden input dim, then AllReduce full-vocab logits.
            h_per = hidden_states.shape[-1] // TP_SIZE
            hs, he = TP_RANK * h_per, (TP_RANK + 1) * h_per
            lh_w = self.lm_head.weight[:, hs:he]
            lh_partial = _proj_linear(hidden_states[..., hs:he], lh_w, self.lm_head.bias)
            logits = _allreduce_lm_head(lh_partial)
        else:
            logits = _proj_linear(hidden_states, self.lm_head.weight, self.lm_head.bias).float()
        if DEBUG_TAP:
            return logits, torch.stack(present_k, dim=0), torch.stack(present_v, dim=0), \
                tap_attn_out, tap_post_attn, tap_post_mlp, _TAP_RAW_OUT[0]
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
    prefill = Qwen3LlmPrefill(model, lm_head).to(device).eval()
    decode = Qwen3LlmDecode(model, lm_head).to(device).eval()
    return prefill, decode, lm_head


def _prepare_llm_prefill_wrapper(model, device: str, pad_kv=True):
    """Build only the prefill wrapper."""
    prefill = _prepare_llm_modules(model, device)[0]
    if not pad_kv:
        prefill.pad_kv = False
    return prefill


def _prepare_llm_decode_wrapper(model, device: str):
    """Build only the decode wrapper."""
    return _prepare_llm_modules(model, device)[1]


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
    return prefill_dir / "qwen3_8b_llm_prefill.onnx", decode_dir / "qwen3_8b_llm_decode.onnx"


def _create_prefill_dummy_inputs(device: str, dummy_seq_len: int):
    """Create random dummy inputs for prefill model export (3 inputs)."""
    seq = int(dummy_seq_len)
    ids = torch.randint(0, 1000, (1, seq), dtype=torch.int64, device=device)
    mask = torch.ones(1, seq, dtype=torch.int64, device=device)
    pos = torch.arange(seq, device=device, dtype=torch.int64).view(1, -1)
    return seq, ids, mask, pos


def _read_external_tensor(t, base_dir):
    """Manually read an external-data tensor, bypassing numpy_helper's checker path."""
    loc, offset, length = "", 0, 0
    for ed in t.external_data:
        if ed.key == "location":
            loc = ed.value
        elif ed.key == "offset":
            offset = int(ed.value)
        elif ed.key == "length":
            length = int(ed.value)
    with open(os.path.join(base_dir, loc), "rb") as f:
        f.seek(offset)
        raw = f.read(length if length > 0 else -1)
    dt = {onnx.TensorProto.INT64: np.int64, onnx.TensorProto.INT32: np.int32,
          onnx.TensorProto.FLOAT: np.float32, onnx.TensorProto.FLOAT16: np.float16,
          onnx.TensorProto.INT8: np.int8, onnx.TensorProto.UINT64: np.uint64}
    return np.frombuffer(raw, dtype=dt[t.data_type]).reshape(list(t.dims))


def _postproc_scale_uint64(m, base_dir):
    """Post-export fix: rewrite the QBMM scale int64 Constant -> uint64 initializer.

    torch.onnx.export has no uint64, so deq_scale is emitted as an int64
    Constant in external data; the GE kernel needs uint64
    ((16384<<32)|f32_bits). Read external data manually to bypass the checker.
    """
    g = m.graph
    prod = {n.output[0]: n for n in g.node if len(n.output) > 0}
    init_names = {i.name for i in g.initializer}
    n_fixed = 0
    for n in list(g.node):
        if n.op_type == "Custom" and any(a.name == "type" and a.s == b"QuantBatchMatmulV3" for a in n.attribute):
            if len(n.input) < 3:
                continue
            scale_name = n.input[2]
            const_node = prod.get(scale_name)
            if const_node is None or const_node.op_type != "Constant":
                continue
            t = const_node.attribute[0].t
            if t.data_type != onnx.TensorProto.INT64:
                continue
            if t.data_location == onnx.TensorProto.EXTERNAL:
                val = _read_external_tensor(t, base_dir).astype(np.uint64)
            else:
                val = onnx.numpy_helper.to_array(t).astype(np.uint64)
            init = onnx.helper.make_tensor(scale_name, onnx.TensorProto.UINT64,
                                           list(val.shape), val.tolist())
            g.node.remove(const_node)
            if scale_name not in init_names:
                g.initializer.append(init)
                init_names.add(scale_name)
            n_fixed += 1
    return n_fixed


def _inline_postproc(path):
    """Inline post-export processing: scale int64 -> uint64.

    Parse and write back with raw protobuf to bypass the onnx checker (fresh
    external-data references from torch.export fail onnx.load/save).
    """
    import time
    from google.protobuf.message import DecodeError

    m = onnx.ModelProto()
    for _ in range(10):
        try:
            with open(str(path), "rb") as f:
                m.ParseFromString(f.read())
            break
        except (OSError, DecodeError):
            time.sleep(1)
    else:
        raise RuntimeError(f"inline postproc: onnx parse failed after retries: {path}")
    n = _postproc_scale_uint64(m, os.path.dirname(os.path.abspath(str(path))))
    with open(str(path), "wb") as f:
        f.write(m.SerializeToString())
    print(f"  [postproc-inline] {n} QBMM scales -> uint64 @ {path}")


def _export_prefill_onnx(prefill, prefill_path: Path, dummy_inputs, use_dynamo: bool, static: bool = False):
    """Export prefill subgraph to ONNX (3 inputs).

    1p GE multi-bucket (static=False): input seq dims + KV cache dim are
    marked dynamic. KV padding appends a CONSTANT 512 extra slots, so
    KV len = seq + 512 is automatically correct for every bucket when GE
    recompiles via ge.dynamicDims.
    """
    print(f"Exporting LLM prefill to {prefill_path} (static={static})...")
    dynamic = None
    if not static:
        dynamic = {"input_ids": {1: "seq"},
                   "attention_mask": {1: "seq"},
                   "position_ids": {1: "seq"},
                   "present_key_cache": {3: "kv_len"},
                   "present_value_cache": {3: "kv_len"}}
    pf_out_names = ["logits", "present_key_cache", "present_value_cache"]
    with torch.no_grad():
        torch.onnx.export(
            prefill, dummy_inputs, str(prefill_path),
            input_names=["input_ids", "attention_mask", "position_ids"],
            output_names=pf_out_names,
            opset_version=18, do_constant_folding=True, dynamo=use_dynamo,
            dynamic_axes=dynamic)
    _inline_postproc(prefill_path)
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


def _export_decode_onnx(decode, decode_path: Path, dummy_inputs, use_dynamo: bool, static: bool = False,
                        dynamic_kv_only: bool = False):
    """Export decode subgraph to ONNX.

    static=True       -> no dynamic_axes (fixed batch=1, used by TP>=2 online convert).
    static=False + dynamic_kv_only=True -> only KV-cache/attention-mask seq dims dynamic
                        (used by 1p GE online convert for multi-bucket ge.dynamicDims;
                        batch dims stay fixed at 1).
    static=False + dynamic_kv_only=False -> all batch/seq dims dynamic (offline ACL).
    """
    print(f"Exporting LLM decode to {decode_path}...")
    dynamic = None
    if not static:
        if dynamic_kv_only:
            # 1p GE: batch dims fixed (1), only KV seq dims dynamic for multi-bucket.
            # attention_mask: [1, -1]; past/present_*_cache: [layers, 1, kv_heads, -1, head_dim]
            dynamic = {"attention_mask": {1: "kv_len"},
                       "past_key_cache": {3: "kv_len"}, "past_value_cache": {3: "kv_len"},
                       "present_key_cache": {3: "kv_len"}, "present_value_cache": {3: "kv_len"}}
        else:
            dynamic = {"input_ids": {0: "batch"}, "attention_mask": {0: "batch"},
                       "position_ids": {0: "batch"}, "logits": {0: "batch"},
                       "past_key_cache": {1: "batch"}, "past_value_cache": {1: "batch"},
                       "present_key_cache": {1: "batch"}, "present_value_cache": {1: "batch"}}
    out_names = ["logits", "present_key_cache", "present_value_cache"]
    if DEBUG_TAP:
        out_names += ["tap_attn_out", "tap_post_attn", "tap_post_mlp", "tap_raw_out"]
    with torch.no_grad():
        torch.onnx.export(
            decode, dummy_inputs, str(decode_path),
            input_names=["input_ids", "attention_mask", "position_ids",
                          "past_key_cache", "past_value_cache"],
            output_names=out_names,
            opset_version=18, do_constant_folding=True, dynamo=use_dynamo,
            dynamic_axes=dynamic)
    _inline_postproc(decode_path)
    print("LLM decode exported successfully.")


def export_llm_prefill_decode(model, output_dir, device="cpu", dummy_seq_len=8, use_dynamo=False):
    """Export Qwen3-8B as two ONNX subgraphs (prefill + decode)."""
    prefill, decode, _ = _prepare_llm_modules(model, device=device)
    kv_dtype = next(model.parameters()).dtype
    num_layers, num_kv_heads, head_dim = _get_kv_cache_config(model)
    prefill_path, decode_path = _prepare_output_paths(output_dir)

    _, ids, mask, pos = _create_prefill_dummy_inputs(device=device, dummy_seq_len=dummy_seq_len)
    _export_prefill_onnx(prefill, prefill_path, (ids, mask, pos), use_dynamo)

    decode_inputs = _create_decode_dummy_inputs(device, num_layers, num_kv_heads, head_dim, kv_dtype)
    _export_decode_onnx(decode, decode_path, decode_inputs, use_dynamo)


def _tp_rank_dir(output_dir, rank, tp_size, sub):
    """Per-rank ONNX output dir for TP>=2.

    Each rank exports into its own directory (rank{r}/{sub}/) because the ONNX
    external-data files are named after tensor names (onnx__MatMul_*, layer
    weights, ...) and would silently overwrite each other when rank0/rank1 are
    written into the same directory. tp_size==1 keeps the legacy flat layout
    ({output_dir}/{sub}/) for 1p compatibility.
    """
    if int(tp_size) > 1:
        return Path(output_dir) / f"rank{rank}" / sub
    return Path(output_dir) / sub


def export_tp_decode(model, output_dir, rank, tp_size, device="cpu", use_dynamo=False, static=True,
                     dynamic_kv_only=False):
    """Export the TP-sharded decode subgraph for one rank.

    static=True (TP>=2): no dynamic axes, for online (optimize=none) convert.
    static=False + dynamic_kv_only=True (1p GE): only KV seq dims dynamic, batch=1 fixed.
    """
    global TP_SIZE, TP_RANK
    TP_SIZE = int(tp_size)
    TP_RANK = int(rank)
    _AllReduceCustom.reset_fusion_id()  # reset so every rank matches
    print(f"Exporting decode rank={rank}/{tp_size} (static={static}, dynamic_kv_only={dynamic_kv_only})...")
    decode = _prepare_llm_decode_wrapper(model, device)
    kv_dtype = next(model.parameters()).dtype
    num_layers, num_kv_heads, head_dim = _get_kv_cache_config(model)
    decode_dir = _tp_rank_dir(output_dir, rank, tp_size, "decode")
    decode_dir.mkdir(parents=True, exist_ok=True)
    decode_path = decode_dir / f"qwen3_8b_llm_decode_rank{rank}.onnx"
    decode_inputs = _create_decode_dummy_inputs(str(device), num_layers, num_kv_heads, head_dim, kv_dtype)
    _export_decode_onnx(decode, decode_path, decode_inputs, use_dynamo, static=static,
                        dynamic_kv_only=dynamic_kv_only)
    TP_SIZE, TP_RANK = 1, 0


def export_tp_prefill(model, output_dir, rank, tp_size, device="cpu", dummy_seq_len=64, use_dynamo=False, static=True):
    """Export the TP-sharded prefill subgraph for one rank.

    static=True (TP>=2): no dynamic axes, for online GE convert.
    static=False (1p GE multi-bucket, 6 buckets): seq dims + KV len dim dynamic, KV padding
        append 512 zeros (constant). KV len = seq + 512 auto per bucket.
    """
    global TP_SIZE, TP_RANK
    TP_SIZE = int(tp_size)
    TP_RANK = int(rank)
    _AllReduceCustom.reset_fusion_id()
    print(f"Exporting prefill rank={rank}/{tp_size} (static={static})...")
    prefill = _prepare_llm_prefill_wrapper(model, device, pad_kv=True)
    prefill_dir = _tp_rank_dir(output_dir, rank, tp_size, "prefill")
    prefill_dir.mkdir(parents=True, exist_ok=True)
    prefill_path = prefill_dir / f"qwen3_8b_llm_prefill_rank{rank}.onnx"
    _, ids, mask, pos = _create_prefill_dummy_inputs(device=device, dummy_seq_len=dummy_seq_len)
    _export_prefill_onnx(prefill, prefill_path, (ids, mask, pos), use_dynamo, static=static)
    TP_SIZE, TP_RANK = 1, 0


def export_common_prefix_suffix(model, output_dir, rank, tp_size, device="cpu",
                                prefix_len=COMMON_PREFIX_LEN,
                                suffix_len=COMMON_SUFFIX_LEN, use_dynamo=False):
    """Export common-prefix KV producer and suffix consumer for one TP rank."""
    global TP_SIZE, TP_RANK
    TP_SIZE, TP_RANK = int(tp_size), int(rank)
    _AllReduceCustom.reset_fusion_id()
    model.eval().to(device)
    prefix = Qwen3LlmPrefix(model).to(device).eval()
    suffix = Qwen3LlmSuffix(model, model.lm_head).to(device).eval()
    num_layers, num_kv_heads, head_dim = _get_kv_cache_config(model)
    local_kv_heads = num_kv_heads // TP_SIZE
    kv_dtype = next(model.parameters()).dtype

    prefix_dir = _tp_rank_dir(output_dir, rank, tp_size, "prefix")
    suffix_dir = _tp_rank_dir(output_dir, rank, tp_size, "suffix")
    prefix_dir.mkdir(parents=True, exist_ok=True)
    suffix_dir.mkdir(parents=True, exist_ok=True)
    prefix_path = prefix_dir / f"qwen3_8b_prefix_rank{rank}.onnx"
    suffix_path = suffix_dir / f"qwen3_8b_suffix_rank{rank}.onnx"

    _, prefix_ids, prefix_mask, prefix_pos = _create_prefill_dummy_inputs(
        device, prefix_len)
    prefix_dynamic = {
        "input_ids": {0: "batch", 1: "prefix_len"},
        "attention_mask": {0: "batch", 1: "prefix_len"},
        "position_ids": {0: "batch", 1: "prefix_len"},
        "present_key_cache": {1: "batch", 3: "prefix_len"},
        "present_value_cache": {1: "batch", 3: "prefix_len"},
    }
    with torch.no_grad():
        torch.onnx.export(
            prefix, (prefix_ids, prefix_mask, prefix_pos), str(prefix_path),
            input_names=["input_ids", "attention_mask", "position_ids"],
            output_names=["present_key_cache", "present_value_cache"],
            opset_version=18, do_constant_folding=True, dynamo=use_dynamo,
            dynamic_axes=prefix_dynamic)
    _inline_postproc(prefix_path)

    suffix_ids = torch.randint(
        0, 1000, (1, suffix_len), dtype=torch.int64, device=device)
    # Suffix receives the complete prefix+suffix mask. This preserves prefix
    # padding and lets callers left-pad suffix buckets without changing logits.
    suffix_mask = torch.ones(
        1, prefix_len + suffix_len, dtype=torch.int64, device=device)
    suffix_pos = torch.arange(
        prefix_len, prefix_len + suffix_len, dtype=torch.int64,
        device=device).view(1, -1)
    past_k = torch.zeros(
        num_layers, 1, local_kv_heads, prefix_len, head_dim,
        dtype=kv_dtype, device=device)
    past_v = torch.zeros_like(past_k)
    suffix_dynamic = {
        "input_ids": {0: "batch", 1: "suffix_len"},
        "attention_mask": {0: "batch", 1: "total_len"},
        "position_ids": {0: "batch", 1: "suffix_len"},
        "past_key_cache": {1: "batch", 3: "prefix_len"},
        "past_value_cache": {1: "batch", 3: "prefix_len"},
        "logits": {0: "batch"},
        "present_key_cache": {1: "batch", 3: "kv_len"},
        "present_value_cache": {1: "batch", 3: "kv_len"},
    }
    with torch.no_grad():
        torch.onnx.export(
            suffix, (suffix_ids, suffix_mask, suffix_pos, past_k, past_v),
            str(suffix_path),
            input_names=["input_ids", "attention_mask", "position_ids",
                         "past_key_cache", "past_value_cache"],
            output_names=["logits", "present_key_cache", "present_value_cache"],
            opset_version=18,
            do_constant_folding=True, dynamo=use_dynamo,
            dynamic_axes=suffix_dynamic)
    _inline_postproc(suffix_path)
    print(f"Common-prefix graphs exported: {prefix_path}, {suffix_path}")
    TP_SIZE, TP_RANK = 1, 0


def _save_reference_data(model, output_dir, device, dtype_str):
    """Save torch reference inputs/outputs for MindIR accuracy comparison.

    Uses seq=512 (first ge.dynamicDims bucket) for prefill and kv_len=1024
    (first decode bucket) for decode. Prefill KV output feeds decode's
    past_key_cache/past_value_cache, creating an end-to-end reference.
    Only called for tp_size=1 (TP>1 AllReduce eager fallback is identity).
    """
    ref_dir = Path(output_dir) / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)

    prefill, decode, _ = _prepare_llm_modules(model, device)
    num_layers, num_kv_heads, head_dim = _get_kv_cache_config(model)

    seq = 512  # first ge.dynamicDims prefill bucket
    kv_len = seq + MAX_OUTPUT_TOKENS  # 1024, first decode bucket

    # --- Prefill reference ---
    torch.manual_seed(42)  # reproducible random inputs
    ids = torch.randint(0, 1000, (1, seq), dtype=torch.int64, device=device)
    mask = torch.ones(1, seq, dtype=torch.int64, device=device)
    pos = torch.arange(seq, device=device, dtype=torch.int64).view(1, -1)

    with torch.no_grad():
        pf_logits, pf_k, pf_v = prefill(ids, mask, pos)

    np.save(ref_dir / "prefill_input_0_input_ids.npy", ids.cpu().numpy())
    np.save(ref_dir / "prefill_input_1_attention_mask.npy", mask.cpu().numpy())
    np.save(ref_dir / "prefill_input_2_position_ids.npy", pos.cpu().numpy())
    np.save(ref_dir / "prefill_output_0_logits.npy", pf_logits.cpu().numpy())
    np.save(ref_dir / "prefill_output_1_present_key_cache.npy", pf_k.cpu().numpy())
    np.save(ref_dir / "prefill_output_2_present_value_cache.npy", pf_v.cpu().numpy())

    # --- Decode reference ---
    # Prefill model already pads KV to kv_len = seq + MAX_OUTPUT_TOKENS inside
    # the graph (see Qwen3LlmPrefill.forward lines ~974-985), so pf_k/pf_v are
    # already (L, 1, H, kv_len, D). No host-side padding needed.
    pf_k_padded = pf_k
    pf_v_padded = pf_v

    # First decode step: position=seq, mask has seq+1 valid tokens
    dec_ids = torch.tensor([[int(ids[0, -1].item())]], dtype=torch.int64, device=device)
    dec_mask = torch.zeros(1, kv_len, dtype=torch.int64, device=device)
    dec_mask[0, :seq + 1] = 1
    dec_pos = torch.tensor([[seq]], dtype=torch.int64, device=device)

    with torch.no_grad():
        dec_logits, dec_k, dec_v = decode(dec_ids, dec_mask, dec_pos, pf_k_padded, pf_v_padded)

    np.save(ref_dir / "decode_input_0_input_ids.npy", dec_ids.cpu().numpy())
    np.save(ref_dir / "decode_input_1_attention_mask.npy", dec_mask.cpu().numpy())
    np.save(ref_dir / "decode_input_2_position_ids.npy", dec_pos.cpu().numpy())
    np.save(ref_dir / "decode_input_3_past_key_cache.npy", pf_k_padded.cpu().numpy())
    np.save(ref_dir / "decode_input_4_past_value_cache.npy", pf_v_padded.cpu().numpy())
    np.save(ref_dir / "decode_output_0_logits.npy", dec_logits.cpu().numpy())
    np.save(ref_dir / "decode_output_1_present_key_cache.npy", dec_k.cpu().numpy())
    np.save(ref_dir / "decode_output_2_present_value_cache.npy", dec_v.cpu().numpy())

    # Save metadata
    meta = {
        "prefill": {
            "seq": seq,
            "inputs": ["input_ids", "attention_mask", "position_ids"],
            "outputs": ["logits", "present_key_cache", "present_value_cache"],
            "shapes": {
                "input_ids": [1, seq],
                "attention_mask": [1, seq],
                "position_ids": [1, seq],
                "logits": list(pf_logits.shape),
                "present_key_cache": list(pf_k.shape),
                "present_value_cache": list(pf_v.shape),
            },
        },
        "decode": {
            "kv_len": kv_len,
            "inputs": ["input_ids", "attention_mask", "position_ids",
                       "past_key_cache", "past_value_cache"],
            "outputs": ["logits", "present_key_cache", "present_value_cache"],
            "shapes": {
                "input_ids": [1, 1],
                "attention_mask": [1, kv_len],
                "position_ids": [1, 1],
                "past_key_cache": [num_layers, 1, num_kv_heads, kv_len, head_dim],
                "past_value_cache": [num_layers, 1, num_kv_heads, kv_len, head_dim],
                "logits": list(dec_logits.shape),
                "present_key_cache": list(dec_k.shape),
                "present_value_cache": list(dec_v.shape),
            },
        },
        "dtype": dtype_str,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
    }
    with open(ref_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Reference data saved to {ref_dir}/ (prefill seq={seq}, decode kv_len={kv_len})")


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
    parser.add_argument("--quant-dir", type=str, default=None,
                        help="Directory of AMCT-extracted A8W8 quantized weights "
                             "(manifest.json + npy from a2_quant_extract.py). When set, "
                             "q/k/v/o_proj/gate/up projections are emitted as quantized "
                             "Custom(AscendQuant)->int8 MatMul->Custom(AscendDequant) "
                             "chains; down_proj/lm_head (AMCT skip_layers) stay fp16.")
    parser.add_argument("--use-dynamo", action="store_true",
                        help="Use torch dynamo exporter path")
    parser.add_argument("--tp-size", type=int, default=1,
                        help="Tensor-parallel size; >1 exports a TP-sharded prefill+decode "
                             "per rank. Custom(AllReduce) is inserted after o_proj/"
                             "down_proj/lm_head; convert with --optimize=none, run with provider=ge.")
    parser.add_argument("--num-layers", type=int, default=0,
                        help="If >0, slice the model to N layers (debugging; 0 = all 36).")
    parser.add_argument("--kv-cache-len", type=int, default=0,
                        help="Override KV_CACHE_LEN (default 256). Set >=1024 to enable prefill "
                             "seq > 128 for long-prompt perf sweep (perf-only; decode KV cache grows "
                             "proportionally). 0 = keep default 256.")
    parser.add_argument("--tp-dynamic", action="store_true",
                        help="TP>=2 prefill: export with dynamic batch+seq axes (one ONNX serving "
                             "multiple seq buckets via runtime online-GE ge.dynamicDims). Default off "
                             "(static prefill — the current TP flow). Decode is always static. "
                             "Requires runtime config with ge.inputShape (U-names) + ge.dynamicDims "
                             "+ ge.dynamicNodeType=1 on the online-GE (provider=ge) path.")
    parser.add_argument("--common-prefix", action="store_true",
                        help="Export prefix/suffix/decode graphs with reusable prefix KV.")
    parser.add_argument("--prefix-len", type=int, default=COMMON_PREFIX_LEN,
                        help="Dummy common-prefix length (310P aligned default: 768).")
    parser.add_argument("--suffix-len", type=int, default=COMMON_SUFFIX_LEN,
                        help="Dummy suffix length; real length remains dynamic.")
    return parser.parse_args()


def main():
    """Load Qwen3-8B model and export to ONNX prefill + decode subgraphs."""
    args = _parse_export_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.kv_cache_len and args.kv_cache_len > 0:
        global KV_CACHE_LEN
        KV_CACHE_LEN = int(args.kv_cache_len)
        print(f"[kv-cache-len] override KV_CACHE_LEN = {KV_CACHE_LEN}")

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    if args.quant_dir:
        _load_quant_pack(args.quant_dir)

    print(f"\nLoading model {args.model_id} for export (dtype={args.dtype})...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=False,
        attn_implementation="eager").to(device)

    if args.num_layers and args.num_layers > 0:
        model.model.layers = model.model.layers[:args.num_layers]  # type: ignore[attr-defined]
        model.config.num_hidden_layers = args.num_layers  # type: ignore[attr-defined]
        print(f"Sliced model to {args.num_layers} layers (debug)")

    if args.common_prefix:
        if args.prefix_len <= 0 or args.suffix_len <= 0:
            raise ValueError("--prefix-len and --suffix-len must be positive")
        global REPLICATE_LM_HEAD
        REPLICATE_LM_HEAD = args.tp_size >= 4
        for rank in range(args.tp_size):
            export_common_prefix_suffix(
                model, output_dir, rank, args.tp_size, str(device),
                args.prefix_len, args.suffix_len, args.use_dynamo)
            export_tp_decode(
                model, output_dir, rank, args.tp_size, str(device),
                args.use_dynamo, static=False,
                dynamic_kv_only=True)
    elif args.tp_size > 1:
        # TP=4 (36 layers, 4 ranks / 2 cards) can hit a GE graph-optimization
        # miscompile that dampens decode logits. Exporting the decode with
        # layer-0 intermediate output taps (DEBUG_TAP) forces GE to materialize
        # those intermediates, which blocks the bad optimization and restores
        # correct precision. Taps are extra outputs of existing intermediates ->
        # negligible overhead; the infer allocates the extra output buffers and
        # ignores them.
        global DEBUG_TAP
        DEBUG_TAP = args.tp_size >= 4
        tp_prefill_static = not args.tp_dynamic
        if not tp_prefill_static:
            print("[tp-dynamic] TP prefill exported with dynamic seq axis AND decode "
                  "exported with dynamic KV-len axis (dynamic_kv_only). Serve the 6 seq/KV "
                  "buckets at runtime via online-GE ge.dynamicDims + ge.dynamicNodeType=1 "
                  "(see configs/tp2/ge_{prefill,decode}.cfg).")
        for rank in range(args.tp_size):
            export_tp_prefill(model, output_dir, rank, args.tp_size, str(device),
                              args.dummy_seq_len, args.use_dynamo, static=tp_prefill_static)
            # tp-dynamic: decode KV-len dim must be dynamic too so the same MindIR
            # serves every decode KV bucket (1024..3584); otherwise decode is locked
            # to the dummy KV len and ge.dynamicDims cannot re-specialize it.
            export_tp_decode(model, output_dir, rank, args.tp_size, str(device), args.use_dynamo,
                             static=tp_prefill_static,
                             dynamic_kv_only=not tp_prefill_static)
    else:
        # tp_size=1: single-shard (rank0). Multi-bucket GE flow:
        #   * prefill: static=False -> seq dim + KV len dim dynamic for
        #     ge.dynamicDims 6 buckets. KV padding = 512 constant zeros, so per
        #     bucket KV len = seq + 512 automatically (GE re-specializes).
        #   * decode:  static=False + dynamic_kv_only=True -> KV cache /
        #     attention_mask dims dynamic for 6-bucket ge.dynamicDims, batch fixed.
        export_tp_prefill(model, output_dir, 0, 1, str(device), 64, args.use_dynamo,
                          static=False)
        export_tp_decode(model, output_dir, 0, 1, str(device), args.use_dynamo,
                         static=False, dynamic_kv_only=True)

    # Save reference data for MindIR accuracy comparison (1p only;
    # TP>1 AllReduce eager fallback is identity, so reference would be wrong)
    if args.tp_size == 1 and not args.common_prefix:
        _save_reference_data(model, args.output_dir, str(device), args.dtype)

    # Add allow_nz=true to MatMul/MatMulV2/BatchMatMulV2 nodes in all exported ONNX files,
    # allowing GE to use FRACTAL_NZ for fp16 MatMul (otherwise the whole graph falls to ND,
    # causing inefficient vector-unit data rearrangement).
    export_subgraphs = ("prefix", "suffix", "decode") if args.common_prefix else ("prefill", "decode")
    for sub in export_subgraphs:
        for r in range(args.tp_size):
            filename = (f"qwen3_8b_{sub}_rank{r}.onnx"
                        if args.common_prefix and sub in ("prefix", "suffix")
                        else f"qwen3_8b_llm_{sub}_rank{r}.onnx")
            onnx_path = _tp_rank_dir(output_dir, r, args.tp_size, sub) / filename
            if onnx_path.exists():
                _set_allow_nz_on_matmul(str(onnx_path))

    print("Clearing memory after export...")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"\nExport finished. Files saved in {args.output_dir}")


if __name__ == "__main__":
    main()
