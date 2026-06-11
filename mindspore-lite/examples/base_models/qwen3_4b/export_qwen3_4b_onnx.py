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
Export Qwen3-4B model to ONNX format.
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
    """Convert each element of *items* to a string and return the list.

    Args:
        items: An iterable of values to stringify.

    Returns:
        A list of string representations of the input items.
    """
    return [str(x) for x in items]


def _rotate_half(x):
    """Rotate the second half of the last dimension to the front with a sign flip.

    Splits the last dimension in half, swaps the two halves and negates the
    former second half, as required by rotary position embeddings.

    Args:
        x: Input tensor whose last dimension is even.

    Returns:
        Tensor with the same shape as *x* but with the halves swapped.
    """
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat([-x2, x1], dim=-1)


class _RotaryMulCustom(torch.autograd.Function):
    """Custom RotaryMul op for ONNX export."""

    @staticmethod
    def forward(ctx, x, cos4, sin4):
        """Forward for RotaryMul (eager fallback).

        Args:
            ctx: Autograd context (unused).
            x: Input tensor.
            cos4: Cosine component of the rotary embedding.
            sin4: Sine component of the rotary embedding.

        Returns:
            Tensor with rotary embedding applied element-wise.
        """
        del ctx
        return (x * cos4) + (_rotate_half(x) * sin4)

    @staticmethod
    def symbolic(g, x, cos4, sin4):
        """ONNX symbolic for RotaryMul.

        Args:
            g: ONNX graph.
            x: Input tensor.
            cos4: Cosine component.
            sin4: Sine component.

        Returns:
            ONNX node representing the custom RotaryMul operation.
        """
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
    """Apply rotary multiplication via the custom RotaryMul operator.

    Args:
        x: Input tensor.
        cos4: Cosine component of the rotary embedding.
        sin4: Sine component of the rotary embedding.

    Returns:
        Tensor with rotary multiplication applied.
    """
    return _RotaryMulCustom.apply(x, cos4, sin4)


class _ApplyRotaryPosEmbCustom(torch.autograd.Function):
    """Custom ApplyRotaryPosEmb op for ONNX export."""

    @staticmethod
    def forward(ctx, query, key, cos, sin, layout: int, rotary_mode: str):
        """Forward for rotary position embedding (eager fallback).

        Args:
            ctx: Autograd context (unused).
            query: Query tensor of shape ``(B, S, num_heads, head_dim)`` or ``(B, num_heads, S, head_dim)``.
            key: Key tensor with the same shape convention as *query*.
            cos: Cosine part of the rotary embedding.
            sin: Sine part of the rotary embedding.
            layout: Layout indicator — ``1`` for ``BNSD``, otherwise ``BSND``.
            rotary_mode: Rotary mode string (unused in eager fallback).

        Returns:
            Tuple of (rotated_query, rotated_key).
        """
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
        """ONNX symbolic for rotary position embedding.

        Args:
            g: ONNX graph.
            query: Query tensor.
            key: Key tensor.
            cos: Cosine component.
            sin: Sine component.
            layout: Layout indicator — ``1`` for ``BNSD``, otherwise ``BSND``.
            rotary_mode: Rotary mode string passed as an ONNX attribute.

        Returns:
            Tuple of ONNX nodes (rotated_query, rotated_key).
        """
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
    """Apply rotary position embedding via the custom operator.

    Args:
        query: Query tensor.
        key: Key tensor.
        cos: Cosine component.
        sin: Sine component.
        layout: Layout indicator (default 3).
        rotary_mode: Rotary mode string (default ``"half"``).

    Returns:
        Tuple of (rotated_query, rotated_key).
    """
    return _ApplyRotaryPosEmbCustom.apply(query, key, cos, sin, int(layout), str(rotary_mode))


class _RmsNormCustom(torch.autograd.Function):
    """Custom RMSNorm op for ONNX export."""

    @staticmethod
    def forward(ctx, x, gamma, epsilon: float):
        """Forward for RMSNorm (eager fallback).

        Args:
            ctx: Autograd context (unused).
            x: Input tensor.
            gamma: Scale (weight) tensor.
            epsilon: Small constant for numerical stability.

        Returns:
            Tuple of (normalized_output, reciprocal_std).
        """
        del ctx
        x_fp32 = x.to(torch.float32)
        var = (x_fp32 * x_fp32).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + float(epsilon))
        y = (x_fp32 * rstd).to(x.dtype) * gamma
        return y, rstd

    @staticmethod
    def symbolic(g, x, gamma, epsilon: float):
        """ONNX symbolic for RMSNorm.

        Args:
            g: ONNX graph.
            x: Input tensor.
            gamma: Scale (weight) tensor.
            epsilon: Small constant for numerical stability.

        Returns:
            Tuple of ONNX nodes (normalized_output, reciprocal_std).
        """
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
    """Apply RMS normalization via the custom RmsNorm operator.

    Args:
        x: Input tensor.
        gamma: Scale (weight) tensor.
        epsilon: Small constant for numerical stability (default 1e-6).

    Returns:
        Tuple of (normalized_output, reciprocal_std).
    """
    return _RmsNormCustom.apply(x, gamma, float(epsilon))


def _make_flash_attn_mask(attention_mask, q_len, k_len, past_len):
    """Build a boolean flash-attention mask combining causal and padding constraints.

    Args:
        attention_mask: Integer mask of shape ``(B, k_len)`` where ``1`` = valid, ``0`` = padding.
        q_len: Query sequence length.
        k_len: Key sequence length.
        past_len: Number of past (cached) key/value positions.

    Returns:
        Boolean mask of shape ``(B, 1, q_len, k_len)`` — ``True`` means *masked out*.
    """
    ar_q = torch.arange(q_len, device=attention_mask.device)
    ar_k = torch.arange(k_len, device=attention_mask.device)
    causal = ar_k[None, :] > (past_len + ar_q[:, None])
    causal = causal[None, None, :, :].expand(attention_mask.shape[0], 1, q_len, k_len)
    padding = attention_mask[:, None, None, :].to(torch.bool).logical_not()
    return (causal | padding).to(torch.bool)


def _expand_gqa_kv(k, v, num_heads, num_kv_heads):
    """Expand key/value tensors for grouped-query attention by repeating heads.

    Args:
        k: Key tensor of shape ``(B, num_kv_heads, S, head_dim)``.
        v: Value tensor of shape ``(B, num_kv_heads, S, head_dim)``.
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads.

    Returns:
        Tuple (k_expanded, v_expanded) with heads repeated to match *num_heads*.
    """
    if 0 < num_kv_heads < num_heads:
        rep = num_heads // num_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    return k, v


def _build_flash_symbolic(g, op_type, tensors, input_idx, extra_attrs):
    """Build an ONNX Custom op node for flash attention variants.

    Args:
        g: ONNX graph builder.
        op_type: Custom op type string (e.g. ``"IncreFlashAttention"``).
        tensors: List of input tensors (query, key, value[, atten_mask]).
        input_idx: List of input indices matching tensors.
        extra_attrs: Dict of additional op attributes passed as kwargs.

    Returns:
        ONNX node with type set from the first tensor.
    """
    y = g.op(
        "Custom", *tensors,
        type_s=op_type,
        input_names_s=_as_list_str(["query", "key", "value", "atten_mask"]),
        optional_input_names_s=_as_list_str(["atten_mask"]),
        output_names_s=_as_list_str(["attention_out"]),
        output_num_i=1,
        input_index_i=input_idx,
        **extra_attrs,
    )
    y.setType(tensors[0].type())
    return y


def _eager_flash_attn(query, key, value, atten_mask, num_heads, scale_value,
                      input_layout, num_key_value_heads, has_sparse=False,
                      sparse_mode=0):
    """Eager fallback for flash attention (prefill and decode).

    Performs layout conversion, GQA expansion, scaled dot-product attention,
    mask application, and softmax. Returns the attention output tensor.
    """
    layout = str(input_layout).upper()
    q, k, v = query, key, value
    if layout in ("BSND", "SBND"):
        q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
    if 0 < num_key_value_heads < num_heads:
        rep = num_heads // num_key_value_heads
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    attn = torch.matmul(q, k.transpose(2, 3)) * float(scale_value)
    if atten_mask is not None:
        m = atten_mask.to(torch.bool)
        if m.dim() == 4 and m.shape[1] == 1:
            m = m.expand(attn.shape[0], attn.shape[1], m.shape[2], m.shape[3])
        attn = attn.masked_fill(m, torch.finfo(attn.dtype).min)
    elif has_sparse and int(sparse_mode) in (2, 3):
        attn = _apply_causal_mask(attn)
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(attn, v)
    if layout in ("BSND", "SBND"):
        out = out.permute(0, 2, 1, 3)
    return out


def _apply_causal_mask(attn):
    """Apply causal mask to attention scores for sparse modes 2/3."""
    q_len, k_len = attn.shape[-2], attn.shape[-1]
    ar_q, ar_k = torch.arange(q_len, device=attn.device), torch.arange(k_len, device=attn.device)
    causal = ar_k[None, :] > ar_q[:, None]
    causal = causal[None, None, :, :].expand(attn.shape[0], attn.shape[1], q_len, k_len)
    return attn.masked_fill(causal, torch.finfo(attn.dtype).min)


class _IncreFlashAttentionCustom(torch.autograd.Function):
    """Custom IncreFlashAttention op for ONNX export."""

    @staticmethod
    def forward(ctx, query, key, value, atten_mask, num_heads, scale_value,
                input_layout, num_key_value_heads, block_size, inner_precise):
        """Eager fallback: scaled dot-product attention with optional GQA."""
        del ctx, block_size, inner_precise
        return _eager_flash_attn(
            query, key, value, atten_mask, num_heads, scale_value,
            input_layout, num_key_value_heads,
        )

    @staticmethod
    def symbolic(g, query, key, value, atten_mask, num_heads, scale_value,
                 input_layout, num_key_value_heads, block_size, inner_precise):
        """ONNX symbolic for IncreFlashAttention custom op."""
        tensors = [query, key, value]
        idx = [0, 1, 2]
        if atten_mask is not None:
            tensors.append(atten_mask)
            idx.append(3)
        return _build_flash_symbolic(
            g, "IncreFlashAttention", tensors, idx,
            {"num_heads_i": int(num_heads), "scale_value_f": float(scale_value),
             "input_layout_s": str(input_layout),
             "num_key_value_heads_i": int(num_key_value_heads),
             "block_size_i": int(block_size), "inner_precise_i": int(inner_precise)},
        )


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
    """Functional wrapper for incremental flash attention.

    Args:
        query: Query tensor.
        key: Key tensor.
        value: Value tensor.
        atten_mask: Attention mask or ``None``.
        num_heads: Number of query attention heads.
        scale_value: Softmax scaling factor.
        input_layout: Layout string.
        num_key_value_heads: Number of key/value heads.
        block_size: Block size hint (default 0).
        inner_precise: Precision hint (default 1).

    Returns:
        Attention output tensor.
    """
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
        """Forward for prompt flash attention (eager fallback).

        Args:
            ctx: Autograd context (unused).
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            atten_mask: Attention mask or ``None``.
            num_heads: Number of query attention heads.
            scale_value: Softmax scaling factor.
            input_layout: Layout string.
            num_key_value_heads: Number of key/value heads.
            sparse_mode: Sparse attention mode indicator.
            inner_precise: Precision hint (unused in eager).
            pre_tokens: Pre-token count (unused in eager).
            next_tokens: Next-token count (unused in eager).

        Returns:
            Attention output tensor.
        """
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
        """ONNX symbolic for prompt flash attention.

        Args:
            g: ONNX graph.
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            atten_mask: Attention mask or ``None``.
            num_heads: Number of query attention heads.
            scale_value: Softmax scaling factor.
            input_layout: Layout string.
            num_key_value_heads: Number of key/value heads.
            sparse_mode: Sparse attention mode indicator.
            inner_precise: Precision hint.
            pre_tokens: Pre-token count.
            next_tokens: Next-token count.

        Returns:
            ONNX node representing the custom PromptFlashAttention operation.
        """
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
    """Functional wrapper for prompt flash attention.

    Args:
        query: Query tensor.
        key: Key tensor.
        value: Value tensor.
        atten_mask: Attention mask or ``None``.
        num_heads: Number of query attention heads.
        scale_value: Softmax scaling factor.
        input_layout: Layout string.
        num_key_value_heads: Number of key/value heads.
        sparse_mode: Sparse mode indicator (default 0).
        inner_precise: Precision hint (default 1).
        pre_tokens: Pre-token count (default 214748647).
        next_tokens: Next-token count (default 0).

    Returns:
        Attention output tensor.
    """
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
        """Forward for SwiGLU (eager fallback).

        Args:
            ctx: Autograd context (unused).
            x: Input tensor whose *dim*-th dimension is even.
            dim: Dimension along which to split and apply the gated activation.

        Returns:
            Tensor with SwiGLU activation applied.
        """
        del ctx
        d = int(dim)
        if d < 0:
            d = x.dim() + d
        a, b = torch.chunk(x, 2, dim=d)
        return F.silu(a) * b

    @staticmethod
    def symbolic(g, x, dim: int):
        """ONNX symbolic for SwiGLU.

        Args:
            g: ONNX graph.
            x: Input tensor.
            dim: Split dimension.

        Returns:
            ONNX node representing the custom SwiGLU operation.
        """
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
    """Apply SwiGLU activation via the custom operator.

    Args:
        x: Input tensor whose *dim*-th dimension is even.
        dim: Dimension along which to split (default -1, i.e. the last dimension).

    Returns:
        Tensor with SwiGLU activation applied.
    """
    return _SwiGluCustom.apply(x, int(dim))


class _ScatterCustom(torch.autograd.Function):
    """Custom scatter op for ONNX export."""

    @staticmethod
    def forward(ctx, var, indices, updates, reduce: str, axis: int):
        """Forward for scatter (eager fallback).

        Args:
            ctx: Autograd context (unused).
            var: 4-D source tensor of shape ``(B, H, S, D)``.
            indices: Position indices of shape ``(B,)`` or ``(B, 1)``.
            updates: Update values of shape ``(B, H, D)`` or ``(B, H, 1, D)``.
            reduce: Reduction mode — only ``"update"`` is supported.
            axis: Scatter axis (only ``-2`` / ``2`` is supported).

        Returns:
            A new tensor with the updates scattered into *var* at *indices*.

        Raises:
            RuntimeError: If *reduce* is not ``"update"`` or the axis/shape is unsupported.
        """
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
        """ONNX symbolic for scatter.

        Args:
            g: ONNX graph.
            var: Source tensor.
            indices: Position indices.
            updates: Update values.
            reduce: Reduction mode string.
            axis: Scatter axis.

        Returns:
            ONNX node representing the custom Scatter operation.
        """
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
    """Scatter updates into a tensor at the given indices via the custom operator.

    Args:
        var: Source tensor of shape ``(B, H, S, D)``.
        indices: Position indices of shape ``(B,)`` or ``(B, 1)``.
        updates: Values to scatter.
        reduce: Reduction mode (default ``"update"``).
        axis: Axis along which to scatter (default -2).

    Returns:
        Tensor with updates scattered in.
    """
    return _ScatterCustom.apply(var, indices, updates, str(reduce), int(axis))


def _make_additive_causal_mask(attention_mask, q_len, k_len, past_len, dtype):
    """Build an additive causal mask combining causal masking and padding.

    Args:
        attention_mask: Integer mask of shape ``(B, k_len)`` (``1`` = valid).
        q_len: Query sequence length.
        k_len: Key sequence length.
        past_len: Number of past cached positions.
        dtype: Floating dtype used to determine the mask fill value.

    Returns:
        Float mask of shape ``(B, 1, q_len, k_len)`` with large negative values
        where attention should be suppressed.
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


def _compute_qkv_proj(attn_mod, hidden_states):
    """Compute fused QKV projection and split into query, key, value.

    Concatenates the weights and (optionally) biases of the q/k/v projection
    layers into a single linear operation, then slices the result.

    Args:
        attn_mod: The attention module providing ``q_proj``, ``k_proj``, ``v_proj``.
        hidden_states: Input tensor of shape ``(B, S, hidden_size)``.

    Returns:
        Tuple of (query_linear, key_linear, value_linear) tensors after the
        fused linear projection but before reshaping.
    """
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
    return q_lin, k_lin, v_lin


def _apply_rotary_and_cache(
    query_states, key_states, value_states, cos4, sin4,
    cache_pos, past_key, past_value,
):
    """Apply rotary embeddings and optionally update the KV cache.

    When *past_key* / *past_value* are provided (decode mode), the new key/value
    are scattered into the cache tensors at the position indicated by
    *cache_pos*.

    Args:
        query_states: Query tensor (already reshaped to multi-head form).
        key_states: Key tensor (already reshaped to multi-head form).
        value_states: Value tensor (already reshaped to multi-head form).
        cos4: Cosine rotary component.
        sin4: Sine rotary component.
        cache_pos: Position indices for cache update (decode mode), or ``None``.
        past_key: Past key cache tensor, or ``None`` for prefill.
        past_value: Past value cache tensor, or ``None`` for prefill.

    Returns:
        Tuple of (query_states, key_states, value_states) after rotary
        embedding and (optionally) cache scatter.
    """
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


def _prefill_attention(query_states, key_states, value_states, attention_mask, scaling, num_heads, num_kv_heads):
    """Compute attention scores during the prefill phase (no KV cache).

    Transposes tensors to ``BNSD`` layout, optionally expands GQA heads,
    computes scaled dot-product attention with a flash mask, and returns
    the output in ``BSND`` layout along with the transposed key/value.

    Args:
        query_states: Query tensor in ``BSND`` layout.
        key_states: Key tensor in ``BSND`` layout.
        value_states: Value tensor in ``BSND`` layout.
        attention_mask: Integer attention mask of shape ``(B, S)``.
        scaling: Softmax scale factor (typically ``1 / sqrt(head_dim)``).
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads.

    Returns:
        Tuple of (attn_output, key_states_bnsd, value_states_bnsd).
        *attn_output* is in ``BSND`` layout, while key/value are in ``BNSD``.
    """
    q = query_states.permute(0, 2, 1, 3)
    k = key_states.permute(0, 2, 1, 3)
    v = value_states.permute(0, 2, 1, 3)
    k, v = _expand_gqa_kv(k, v, num_heads, num_kv_heads)

    attn = torch.matmul(q, k.transpose(2, 3)) * float(scaling)
    q_len = attn.shape[-2]
    k_len = attn.shape[-1]
    flash_mask = _make_flash_attn_mask(attention_mask, q_len, k_len, 0)
    if flash_mask.dim() == 4 and flash_mask.shape[1] == 1:
        flash_mask = flash_mask.expand(
            attn.shape[0], attn.shape[1], flash_mask.shape[2], flash_mask.shape[3]
        )
    attn = attn.masked_fill(flash_mask.to(torch.bool), torch.finfo(attn.dtype).min)
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_output = torch.matmul(attn, v).permute(0, 2, 1, 3)
    return attn_output, key_states.transpose(1, 2), value_states.transpose(1, 2)


def _decode_attention(query_states, key_states, value_states, attention_mask, scaling, num_heads, num_kv_heads):
    """Compute attention scores during the decode phase using KV cache.

    Uses the custom ``IncreFlashAttention`` operator with a padding mask.

    Args:
        query_states: Query tensor in ``BNSD`` layout.
        key_states: Key tensor (updated cache) in ``BNSD`` layout.
        value_states: Value tensor (updated cache) in ``BNSD`` layout.
        attention_mask: Integer attention mask of shape ``(B, S)``.
        scaling: Softmax scale factor.
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads.

    Returns:
        Attention output tensor in ``BNSD`` layout.
    """
    pad_mask = attention_mask[:, None, None, :].to(torch.bool).logical_not()
    return incre_flash_attention(
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


def _reshape_and_norm_qkv(attn_mod, q_lin, k_lin, v_lin, hidden_shape, is_decode):
    """Reshape QKV projections and apply optional per-head RMS normalization.

    Converts the flat QKV linear outputs into multi-head form, applies
    ``q_norm`` / ``k_norm`` if present, and transposes to ``BNSD`` layout
    when in decode mode.

    Args:
        attn_mod: The attention module (may have ``q_norm`` / ``k_norm``).
        q_lin: Query projection output (flat).
        k_lin: Key projection output (flat).
        v_lin: Value projection output (flat).
        hidden_shape: Target shape ``(B, S, num_heads, head_dim)`` for view.
        is_decode: Whether this is a decode step (triggers BNSD transpose).

    Returns:
        Tuple of (query_states, key_states, value_states) in the appropriate
        layout for the current phase.
    """
    query_states = q_lin.view(hidden_shape)
    key_states = k_lin.view(hidden_shape)
    if hasattr(attn_mod, "q_norm"):
        query_states = _rms_norm_layer(attn_mod.q_norm, query_states)
    if hasattr(attn_mod, "k_norm"):
        key_states = _rms_norm_layer(attn_mod.k_norm, key_states)

    value_states = v_lin.view(hidden_shape)
    if is_decode:
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

    return query_states, key_states, value_states


def _text_attn_forward(
    attn_mod, hidden_states, cos4, sin4, attention_mask, cache_pos, past_key, past_value
):
    """Full attention forward covering both prefill and decode phases.

    Args:
        attn_mod: The attention module.
        hidden_states: Input hidden states ``(B, S, hidden_size)``.
        cos4: Cosine rotary component.
        sin4: Sine rotary component.
        attention_mask: Integer attention mask.
        cache_pos: Position indices for cache update, or ``None``.
        past_key: Past key cache, or ``None`` for prefill.
        past_value: Past value cache, or ``None`` for prefill.

    Returns:
        Tuple of (attn_output, key_states, value_states).
    """
    input_shape = hidden_states.shape[:-1]
    head_dim = attn_mod.head_dim
    num_heads = attn_mod.config.num_attention_heads
    num_kv_heads = attn_mod.config.num_key_value_heads
    hidden_shape = (*input_shape, -1, head_dim)
    is_decode = past_key is not None

    q_lin, k_lin, v_lin = _compute_qkv_proj(attn_mod, hidden_states)
    query_states, key_states, value_states = _reshape_and_norm_qkv(
        attn_mod, q_lin, k_lin, v_lin, hidden_shape, is_decode,
    )

    query_states, key_states, value_states = _apply_rotary_and_cache(
        query_states, key_states, value_states, cos4, sin4,
        cache_pos, past_key, past_value,
    )

    scaling = getattr(attn_mod, "scaling", 1.0 / (head_dim ** 0.5))
    if not is_decode:
        attn_output, key_states, value_states = _prefill_attention(
            query_states, key_states, value_states,
            attention_mask, scaling, num_heads, num_kv_heads,
        )
        attn_output = attn_output.reshape(*input_shape, -1)
    else:
        attn_output = _decode_attention(
            query_states, key_states, value_states,
            attention_mask, scaling, num_heads, num_kv_heads,
        )
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1)

    attn_output = attn_mod.o_proj(attn_output)
    return attn_output, key_states, value_states


def _rms_norm_layer(norm_mod, x):
    """Apply RMS normalization using the custom RmsNorm operator.

    Args:
        norm_mod: A normalization module that provides ``weight`` and
            optionally ``variance_epsilon``.
        x: Input tensor.

    Returns:
        Normalized output tensor (same dtype as *x*).
    """
    gamma = norm_mod.weight
    eps = getattr(norm_mod, "variance_epsilon", 1e-6)
    y, _ = rms_norm(x, gamma, eps)
    return y


def _mlp_gate_up_linear(mlp_mod, x):
    """Merge gate_proj and up_proj into a single linear and split outputs.

    Args:
        mlp_mod: MLP module with ``gate_proj`` and ``up_proj`` sub-modules.
        x: Input tensor.

    Returns:
        Tuple of (gate_output, up_output) tensors.
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


def _run_mlp(mlp_mod, hidden_states):
    """Run the MLP block, using fused gate/up when available.

    Checks whether the MLP has the standard ``gate_proj`` / ``up_proj`` /
    ``down_proj`` structure; if so, uses the fused path with SwiGLU.
    Otherwise falls back to calling the MLP module directly.

    Args:
        mlp_mod: MLP module.
        hidden_states: Input tensor.

    Returns:
        MLP output tensor.
    """
    has_fused = (
        hasattr(mlp_mod, "gate_proj")
        and hasattr(mlp_mod, "up_proj")
        and hasattr(mlp_mod, "down_proj")
    )
    if has_fused:
        gate, up = _mlp_gate_up_linear(mlp_mod, hidden_states)
        x = torch.cat([gate, up], dim=-1)
        return mlp_mod.down_proj(swiglu(x, dim=-1))
    return mlp_mod(hidden_states)


def _prefill_layer_forward(layer, hidden_states, cos4, sin4, attention_mask):
    """Run a single transformer layer during the prefill phase.

    Applies input layernorm, self-attention, post-attention layernorm,
    and the MLP (residual connections included).  Key/value caches are
    padded to ``KV_CACHE_LEN``.

    Args:
        layer: A single transformer layer module.
        hidden_states: Input hidden states.
        cos4: Cosine rotary component.
        sin4: Sine rotary component.
        attention_mask: Integer attention mask.

    Returns:
        Tuple of (hidden_states, padded_key_cache, padded_value_cache).
    """
    residual = hidden_states
    hidden_states = _rms_norm_layer(layer.input_layernorm, hidden_states)
    attn_out, pk, pv = _text_attn_forward(
        layer.self_attn, hidden_states, cos4, sin4, attention_mask, None, None, None,
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
    hidden_states = residual + _run_mlp(layer.mlp, hidden_states)
    return hidden_states, pk, pv


class Qwen3LlmPrefill(torch.nn.Module):
    """Qwen3-4B LLM Prefill wrapper.

    Runs the full prefill pass (all layers) and returns logits together
    with initialised key/value caches padded to ``KV_CACHE_LEN``.
    """

    def __init__(self, model, lm_head):
        """Initialise the prefill wrapper.

        Args:
            model: The HuggingFace causal LM model.
            lm_head: The language-model head (linear projection to vocab).
        """
        super().__init__()
        self.model = model.model
        self.lm_head = lm_head

    def forward(self, input_ids, attention_mask, position_ids):
        """Run prefill forward for Qwen3-4B.

        Args:
            input_ids: Token IDs of shape ``(B, S)``.
            attention_mask: Binary mask of shape ``(B, S)``.
            position_ids: Position indices of shape ``(B, S)``.

        Returns:
            Tuple of (logits, present_key_cache, present_value_cache).
        """
        inputs_embeds = self.model.embed_tokens(input_ids)

        cos, sin = self.model.rotary_emb(inputs_embeds, position_ids)
        cos4 = cos.unsqueeze(2) if cos.dim() == 3 else cos
        sin4 = sin.unsqueeze(2) if sin.dim() == 3 else sin
        hidden_states = inputs_embeds
        present_k = []
        present_v = []

        for layer in self.model.layers:
            hidden_states, pk, pv = _prefill_layer_forward(
                layer, hidden_states, cos4, sin4, attention_mask,
            )
            present_k.append(pk)
            present_v.append(pv)

        hidden_states = _rms_norm_layer(self.model.norm, hidden_states)
        logits = self.lm_head(hidden_states)
        present_k = torch.stack(present_k, dim=0)
        present_v = torch.stack(present_v, dim=0)
        return logits, present_k, present_v


def _decode_layer_forward(layer, hidden_states, cos4, sin4, attention_mask, position_ids, pk_in, pv_in):
    """Run a single transformer layer during the decode phase.

    Applies input layernorm, self-attention (with KV cache scatter),
    post-attention layernorm, and the MLP with residual connections.

    Args:
        layer: A single transformer layer module.
        hidden_states: Input hidden states.
        cos4: Cosine rotary component.
        sin4: Sine rotary component.
        attention_mask: Integer attention mask.
        position_ids: Position indices for cache update.
        pk_in: Input key cache tensor for this layer.
        pv_in: Input value cache tensor for this layer.

    Returns:
        Tuple of (hidden_states, updated_key_cache, updated_value_cache).
    """
    residual = hidden_states
    hidden_states = layer.input_layernorm(hidden_states)
    attn_out, pk, pv = _text_attn_forward(
        layer.self_attn, hidden_states, cos4, sin4,
        attention_mask, position_ids, pk_in, pv_in,
    )
    hidden_states = residual + attn_out
    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = residual + _run_mlp(layer.mlp, hidden_states)
    return hidden_states, pk, pv


class Qwen3LlmDecode(torch.nn.Module):
    """Qwen3-4B LLM Decode wrapper.

    Runs a single decode step (all layers) and returns logits together
    with updated key/value caches.
    """

    def __init__(self, model, lm_head):
        """Initialise the decode wrapper.

        Args:
            model: The HuggingFace causal LM model.
            lm_head: The language-model head (linear projection to vocab).
        """
        super().__init__()
        self.model = model.model
        self.lm_head = lm_head

    def forward(self, input_ids, attention_mask, position_ids, past_key_cache, past_value_cache):
        """Run decode forward for Qwen3-4B.

        Args:
            input_ids: Token IDs of shape ``(B, 1)``.
            attention_mask: Binary mask of shape ``(B, past_len)``.
            position_ids: Position indices of shape ``(B, 1)``.
            past_key_cache: Stacked key cache of shape ``(num_layers, B, H, S, D)``.
            past_value_cache: Stacked value cache of shape ``(num_layers, B, H, S, D)``.

        Returns:
            Tuple of (logits, present_key_cache, present_value_cache).
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
            hidden_states, pk, pv = _decode_layer_forward(
                layer, hidden_states, cos4, sin4,
                attention_mask, position_ids,
                past_k_layers[i], past_v_layers[i],
            )
            present_k.append(pk)
            present_v.append(pv)

        hidden_states = _rms_norm_layer(self.model.norm, hidden_states)
        logits = self.lm_head(hidden_states)
        present_k = torch.stack(present_k, dim=0)
        present_v = torch.stack(present_v, dim=0)
        return logits, present_k, present_v


def _prepare_llm_modules(model, device: str):
    """Prepare LLM modules for Qwen3-4B inference.

    Moves the model and language-model head to the target device and wraps
    them in :class:`Qwen3LlmPrefill` and :class:`Qwen3LlmDecode`.

    Args:
        model: The HuggingFace causal LM model.
        device: Target device string (e.g. ``"cpu"``, ``"cuda"``).

    Returns:
        Tuple of (prefill_wrapper, decode_wrapper, lm_head).
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
    """Get KV cache configuration for Qwen3-4B inference.

    Args:
        model: The HuggingFace causal LM model.

    Returns:
        Tuple of (num_layers, num_kv_heads, head_dim).
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
    """Prepare output paths for Qwen3-4B inference.

    Creates ``prefill`` and ``decode`` sub-directories under *output_dir*.

    Args:
        output_dir: Root output directory path.

    Returns:
        Tuple of (prefill_onnx_path, decode_onnx_path).
    """
    prefill_dir = Path(output_dir) / "prefill"
    decode_dir = Path(output_dir) / "decode"
    prefill_dir.mkdir(parents=True, exist_ok=True)
    decode_dir.mkdir(parents=True, exist_ok=True)
    prefill_path = prefill_dir / "qwen3_4b_llm_prefill.onnx"
    decode_path = decode_dir / "qwen3_4b_llm_decode.onnx"
    return prefill_path, decode_path


def _create_prefill_dummy_inputs(device: str, dummy_seq_len: int):
    """Create dummy inputs for Qwen3-4B prefill export.

    Args:
        device: Device string for tensor placement.
        dummy_seq_len: Sequence length for the dummy inputs.

    Returns:
        Tuple of (dummy_seq_len, dummy_input_ids, dummy_attention_mask,
        dummy_position_ids).
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
    """Export LLM prefill to ONNX format.

    Args:
        prefill: The prefill wrapper module.
        prefill_path: Destination path for the ONNX file.
        dummy_inputs: Tuple of (input_ids, attention_mask, position_ids) dummy tensors.
        use_dynamo: Whether to use the TorchDynamo exporter path.
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
    """Create dummy inputs for Qwen3-4B decode export.

    Args:
        device: Device string for tensor placement.
        dummy_seq: Sequence length (unused, kept for API compatibility).
        num_layers: Number of transformer layers.
        num_kv_heads: Number of key/value attention heads.
        head_dim: Dimension of each attention head.
        kv_dtype: Data type for the KV cache tensors.

    Returns:
        Tuple of (dummy_input_ids, dummy_attention_mask,
        dummy_position_ids, dummy_key_cache, dummy_value_cache).
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
    """Export LLM decode to ONNX format.

    Args:
        decode: The decode wrapper module.
        decode_path: Destination path for the ONNX file.
        dummy_inputs: Tuple of dummy tensors for the decode model.
        use_dynamo: Whether to use the TorchDynamo exporter path.
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
    """Export Qwen3-4B model to ONNX format.

    Orchestrates the full export pipeline: prepares modules, creates dummy
    inputs, and exports both prefill and decode graphs.

    Args:
        model: The HuggingFace causal LM model.
        output_dir: Root directory for ONNX output files.
        device: Device string for export (default ``"cpu"``).
        dummy_seq_len: Dummy sequence length for export shapes (default 8).
        use_dynamo: Whether to use the TorchDynamo exporter path (default False).
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


def _parse_export_args():
    """Parse command-line arguments for the ONNX export script.

    Returns:
        An ``argparse.Namespace`` with attributes ``model_id``,
        ``output_dir``, ``device``, ``dummy_seq_len``, ``dtype``,
        and ``use_dynamo``.
    """
    parser = argparse.ArgumentParser(description="Export Qwen3-4B to ONNX")
    parser.add_argument(
        "--model-id",
        type=str,
        default="./Qwen3-4B",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./qwen3_4b_onnx", help="Output directory"
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
    return parser.parse_args()


def _load_model_for_export(model_id: str, dtype: str, device: str):
    """Load a HuggingFace causal LM model for ONNX export.

    Args:
        model_id: HuggingFace model ID or local path.
        dtype: String dtype specifier (``"fp16"``, ``"bf16"``, or ``"fp32"``).
        device: Device string for model placement.

    Returns:
        Tuple of (model, torch_device).
    """
    print(f"\nLoading model {model_id} for export (dtype={dtype})...")
    if dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    torch_device = torch.device(device)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=False,
        attn_implementation="eager",
    ).to(torch_device)
    return model, torch_device


def main():
    """Main function to export Qwen3-4B model to ONNX format.

    Parses command-line arguments, loads the model, runs the export
    pipeline, and cleans up GPU memory afterwards.
    """
    args = _parse_export_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, device = _load_model_for_export(args.model_id, args.dtype, args.device)

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
