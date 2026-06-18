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
Export GLM-OCR (zai-org/GLM-OCR) to ONNX.

GLM-OCR is a 0.9B multimodal OCR model (GLM4v encoder-decoder): a CogViT-style
vision encoder + a GLM text decoder with multimodal RoPE (mRoPE). The model is
exported into three ONNX sub-graphs:

  1. Vision encoder  : pixel_values (flattened patches) -> image embeddings
  2. LLM prefill     : full prompt -> logits + KV cache (PromptFlashAttention)
  3. LLM decode       : single token + KV cache -> logits + updated KV cache
                        (IncreFlashAttention + Scatter cache update)
"""

import argparse
import gc
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    import torch._dynamo

    torch._dynamo.disable()
except Exception:
    pass

try:
    from transformers import GlmOcrForConditionalGeneration
except ImportError:
    print("Error: transformers package not found or version too low.")
    print("Please install transformers>=5.0 (supports glm_ocr natively).")
    sys.exit(1)


_USE_CUSTOM_OP = True
_USE_CUSTOM_ATTN = True
_CUSTOM_ATTN_LAYOUT = "BNSD"


# ---------------------------------------------------------------------------
# Custom CANN operators (exported as ONNX ``Custom`` nodes for Ascend).
# ---------------------------------------------------------------------------


def _as_list_str(items):
    """Convert items to a list of string representations for ONNX attr."""
    return [str(x) for x in items]


class _CustomRmsNorm(torch.autograd.Function):
    """Custom RMSNorm operator for ONNX export."""

    @staticmethod
    def forward(ctx, x, gamma, epsilon_f):
        """Compute RMSNorm (fp32 accumulator) returning y and rstd."""
        del ctx
        x_fp32 = x.to(torch.float32)
        var = (x_fp32 * x_fp32).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + float(epsilon_f))
        y = (x_fp32 * rstd).to(dtype=x.dtype) * gamma.to(dtype=x.dtype)
        return y, rstd

    @staticmethod
    def symbolic(g, x, gamma, epsilon_f):
        """Export a Custom node for RMSNorm."""
        y, rstd = g.op(
            "Custom", x, gamma,
            outputs=2,
            type_s="RmsNorm",
            epsilon_f=float(epsilon_f),
            input_names_s=_as_list_str(["x", "gamma"]),
            optional_input_names_s=_as_list_str([]),
            output_names_s=_as_list_str(["y", "rstd"]),
            output_num_i=2,
            input_index_i=[0, 1],
        )
        y.setType(x.type())
        return y, rstd


class _CustomSwiGlu(torch.autograd.Function):
    """Custom SwiGLU operator for ONNX export (silu(a) * b)."""

    @staticmethod
    def forward(ctx, x, dim_i):
        """Split ``x`` along ``dim_i`` and return silu(first) * second."""
        del ctx
        a, b = torch.chunk(x, 2, dim=int(dim_i))
        return F.silu(a) * b

    @staticmethod
    def symbolic(g, x, dim_i):
        """Export a Custom node for SwiGlu."""
        y = g.op(
            "Custom", x,
            type_s="SwiGlu",
            dim_i=int(dim_i),
            input_names_s=_as_list_str(["x"]),
            output_names_s=_as_list_str(["y"]),
        )
        y.setType(x.type())
        return y


class _CustomPromptFlashAttention(torch.autograd.Function):
    """Custom prompt flash attention operator (prefill) for ONNX export."""

    @staticmethod
    def forward(ctx, query, key, value, atten_mask, num_heads_i,
                num_kv_heads_i, scale_value_f, input_layout_s):
        """Fallback full attention used during tracing and shape inference."""
        del ctx, input_layout_s
        key, value = _expand_kv(key, value, int(num_heads_i), int(num_kv_heads_i))
        scores = torch.matmul(query, key.transpose(-2, -1)) * float(scale_value_f)
        if atten_mask is not None:
            scores = scores.masked_fill(atten_mask, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        return torch.matmul(probs, value)

    @staticmethod
    def symbolic(g, query, key, value, atten_mask, num_heads_i,
                 num_kv_heads_i, scale_value_f, input_layout_s):
        """Export a Custom node for prompt flash attention."""
        y = g.op(
            "Custom", query, key, value, atten_mask,
            type_s="PromptFlashAttention",
            num_heads_i=int(num_heads_i),
            num_key_value_heads_i=int(num_kv_heads_i),
            scale_value_f=float(scale_value_f),
            input_layout_s=str(input_layout_s),
            pre_tokens_i=2147483647,
            next_tokens_i=0,
            sparse_mode_i=0,
            inner_precise_i=1,
            input_names_s=_as_list_str(["query", "key", "value", "atten_mask"]),
            optional_input_names_s=_as_list_str(["atten_mask"]),
            output_names_s=_as_list_str(["attention_out"]),
            output_num_i=1,
            input_index_i=[0, 1, 2, 3],
        )
        y.setType(query.type())
        return y


class _CustomIncreFlashAttention(torch.autograd.Function):
    """Custom incremental flash attention operator (decode) for ONNX export."""

    @staticmethod
    def forward(ctx, query, key, value, atten_mask, num_heads_i,
                num_kv_heads_i, scale_value_f, input_layout_s):
        """Fallback single-step attention used during tracing."""
        del ctx, input_layout_s
        key, value = _expand_kv(key, value, int(num_heads_i), int(num_kv_heads_i))
        scores = torch.matmul(query, key.transpose(-2, -1)) * float(scale_value_f)
        if atten_mask is not None:
            scores = scores.masked_fill(atten_mask, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        return torch.matmul(probs, value)

    @staticmethod
    def symbolic(g, query, key, value, atten_mask, num_heads_i,
                 num_kv_heads_i, scale_value_f, input_layout_s):
        """Export a Custom node for incremental flash attention."""
        y = g.op(
            "Custom", query, key, value, atten_mask,
            type_s="IncreFlashAttention",
            num_heads_i=int(num_heads_i),
            num_key_value_heads_i=int(num_kv_heads_i),
            scale_value_f=float(scale_value_f),
            input_layout_s=str(input_layout_s),
            block_size_i=0,
            inner_precise_i=1,
            input_names_s=_as_list_str(["query", "key", "value", "atten_mask"]),
            optional_input_names_s=_as_list_str(["atten_mask"]),
            output_names_s=_as_list_str(["attention_out"]),
            output_num_i=1,
            input_index_i=[0, 1, 2, 3],
        )
        y.setType(query.type())
        return y


class _CustomScatter(torch.autograd.Function):
    """Custom scatter-update operator for fixed-length KV cache update."""

    @staticmethod
    def forward(ctx, var, indices, updates, axis_i):
        """Scatter ``updates`` into ``var`` at ``indices`` along ``axis_i``."""
        del ctx
        bsz = var.shape[0]
        ax = int(axis_i)
        idx = indices.view(bsz, 1, 1, 1).to(dtype=torch.int64)
        idx = idx.expand(bsz, var.shape[1], 1, var.shape[3])
        return var.scatter(ax, idx, updates.to(dtype=var.dtype))

    @staticmethod
    def symbolic(g, var, indices, updates, axis_i):
        """Export a Custom node for scatter cache update."""
        y = g.op(
            "Custom", var, indices, updates,
            type_s="Scatter",
            reduce_s="update",
            axis_i=int(axis_i),
            input_names_s=_as_list_str(["var", "indices", "updates"]),
            optional_input_names_s=_as_list_str([]),
            output_names_s=_as_list_str(["var"]),
        )
        y.setType(var.type())
        return y


# ---------------------------------------------------------------------------
# Helpers shared by attention / mlp / rotary.
# ---------------------------------------------------------------------------


def _expand_kv(key, value, num_heads, num_kv_heads):
    """Expand GQA key/value to num_heads via repeat_interleave on dim=1."""
    if num_kv_heads == num_heads:
        return key, value
    if num_heads % num_kv_heads != 0:
        raise RuntimeError(f"num_heads({num_heads}) not divisible by num_kv_heads({num_kv_heads})")
    rep = num_heads // num_kv_heads
    key = key.repeat_interleave(rep, dim=1)
    value = value.repeat_interleave(rep, dim=1)
    return key, value


def _rms_norm(norm_mod, x):
    """Apply RMSNorm via the Custom op (or eager fallback)."""
    if not _USE_CUSTOM_OP:
        return norm_mod(x)
    eps = getattr(norm_mod, "variance_epsilon", getattr(norm_mod, "eps", 1e-5))
    y, _ = _CustomRmsNorm.apply(x, norm_mod.weight, float(eps))
    return y


def _apply_interleaved_rotary(x, cos, sin):
    """Apply GLM interleaved mRoPE to one tensor, export/converter-friendly.

    GLM's ``rotate_half_llm`` uses ``x[..., 0::2]`` / ``x[..., 1::2]`` which the
    legacy ONNX exporter mis-ranks; and a reshape over leading (symbolic) dims
    breaks the converter's shape inference. We instead gather even/odd channels
    via ``index_select`` (ONNX Gather) and recombine with ``stack`` + ``flatten``
    (reshape only of the concrete last two dims). ``cos``/``sin`` are already
    mrope-merged of shape ``(bs, seq, D)``.
    """
    d_full = x.shape[-1]
    d_half = d_full // 2
    cos = cos.unsqueeze(1)[..., :d_half]              # (bs,1,seq,d_half)
    sin = sin.unsqueeze(1)[..., :d_half]
    even_idx = torch.arange(0, d_full, 2, dtype=torch.long)
    odd_idx = torch.arange(1, d_full, 2, dtype=torch.long)
    x_even = x.index_select(-1, even_idx)             # (...,d_half)
    x_odd = x.index_select(-1, odd_idx)
    out_even = x_even * cos - x_odd * sin
    out_odd = x_odd * cos + x_even * sin
    return torch.stack((out_even, out_odd), dim=-1).flatten(-2)


def _rotary(q, k, cos, sin):
    """Apply GLM interleaved mRoPE to q and k."""
    q = _apply_interleaved_rotary(q, cos, sin)
    k = _apply_interleaved_rotary(k, cos, sin)
    return q, k


class GlmMrope(torch.nn.Module):
    """GLM multimodal RoPE with a *persistent* inv_freq buffer.

    The HF ``GlmOcrTextRotaryEmbedding`` registers ``inv_freq`` as a
    non-persistent buffer, which the legacy ONNX exporter drops, breaking the
    rotary embedding silently. We reimplement the mRoPE forward (incl. the
    ``apply_mrope`` section selection) with a persistent buffer so it exports.
    """

    def __init__(self, head_dim, rope_theta, mrope_section):
        """Precompute inv_freq and store the mrope section split."""
        super().__init__()
        dim = int(head_dim)
        inv_freq = 1.0 / (float(rope_theta) ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=True)
        self.section = [int(x) for x in mrope_section]

    def forward(self, position_ids):
        """Return (cos, sin) of shape (bs, seq, head_dim) for positions [3, bs, seq]."""
        freqs = position_ids.unsqueeze(-1).float() * self.inv_freq  # (3,bs,seq,dim/2)
        chunks = torch.split(freqs, self.section, dim=-1)            # 3 chunks
        merged = torch.cat([chunks[i][i] for i in range(len(self.section))], dim=-1)
        emb = torch.cat([merged, merged], dim=-1)                    # (bs,seq,dim)
        # NOTE: keep float32 (do NOT cast to position_ids.dtype which is int).
        return emb.cos(), emb.sin()


def _mlp_forward(layer, x):
    """Run fused GLM MLP (gate_up_proj -> SwiGlu -> down_proj)."""
    mlp = layer.mlp
    if not _USE_CUSTOM_OP:
        return mlp(x)
    gate_up = mlp.gate_up_proj(x)
    act = _CustomSwiGlu.apply(gate_up, -1)
    return mlp.down_proj(act)


def _fused_qkv(attn_mod, hidden_states):
    """Fuse q/k/v projections into a single linear, return (q, k, v)."""
    wq = attn_mod.q_proj.weight
    wk = attn_mod.k_proj.weight
    wv = attn_mod.v_proj.weight
    weight = torch.cat([wq, wk, wv], dim=0)
    bq = getattr(attn_mod.q_proj, "bias", None)
    bk = getattr(attn_mod.k_proj, "bias", None)
    bv = getattr(attn_mod.v_proj, "bias", None)
    bias = None
    if bq is not None or bk is not None or bv is not None:
        bq = torch.zeros((wq.shape[0],), dtype=wq.dtype) if bq is None else bq
        bk = torch.zeros((wk.shape[0],), dtype=wk.dtype) if bk is None else bk
        bv = torch.zeros((wv.shape[0],), dtype=wv.dtype) if bv is None else bv
        bias = torch.cat([bq, bk, bv], dim=0)
    qkv = F.linear(hidden_states, weight, bias)
    q, k, v = torch.split(qkv, [wq.shape[0], wk.shape[0], wv.shape[0]], dim=-1)
    return q, k, v


def _make_prefill_causal_mask(attention_mask, q_len, kv_len):
    """Boolean causal + padding mask (True == masked out) for prefill attention."""
    ar_q = torch.arange(q_len, device=attention_mask.device)
    ar_k = torch.arange(kv_len, device=attention_mask.device)
    causal = ar_k[None, :] > ar_q[:, None]
    pad = attention_mask.to(torch.bool).logical_not()
    full = causal[None, None, :, :] | pad[:, None, None, :]
    return full


def _make_decode_mask(attention_mask, cache_pos, max_seq_len, dtype):
    """Boolean mask (True == masked out) for single-step decode attention."""
    del dtype
    bsz = attention_mask.shape[0]
    pos = cache_pos.view(bsz, 1).to(dtype=torch.int64)
    ar_k = torch.arange(max_seq_len, device=attention_mask.device).view(1, -1)
    causal = ar_k > pos
    pad = attention_mask.to(torch.bool).logical_not()
    full = causal[:, None, None, :] | pad[:, None, None, :]
    return full


# ---------------------------------------------------------------------------
# Vision encoder wrapper.
# ---------------------------------------------------------------------------


class VisionWrapper(torch.nn.Module):
    """Wrap GLM-OCR vision encoder to export a single image at a fixed grid."""

    def __init__(self, visual, grid_thw):
        """Cache the fixed grid_thw so the ONNX graph only takes pixel_values."""
        super().__init__()
        self.visual = visual
        # NOTE: grid_thw must be a plain constant attribute (not a registered buffer);
        # as a buffer it changes how get_vision_position_ids traces and breaks a cat op.
        self.grid_thw = grid_thw.to(torch.int64).detach()

    def forward(self, pixel_values):
        """Run vision encoder and return pooled image embeddings."""
        outputs = self.visual(pixel_values, grid_thw=self.grid_thw)
        if hasattr(outputs, "pooler_output"):
            return outputs.pooler_output
        return outputs[1]


# ---------------------------------------------------------------------------
# LLM prefill / decode wrappers (Gemma-style post-norm decoder layers).
# ---------------------------------------------------------------------------


def _layer_forward_prefill(layer, hidden_states, cos, sin, attn_mask):
    """Run one GLM decoder layer in prefill mode, returning hidden + k/v cache."""
    residual = hidden_states
    hidden_states = _rms_norm(layer.input_layernorm, hidden_states)
    attn_out, pk, pv = _attn_prefill(layer.self_attn, hidden_states, cos, sin, attn_mask)
    hidden_states = _rms_norm(layer.post_self_attn_layernorm, attn_out)
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = _rms_norm(layer.post_attention_layernorm, hidden_states)
    hidden_states = _mlp_forward(layer, hidden_states)
    hidden_states = _rms_norm(layer.post_mlp_layernorm, hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states, pk, pv


def _standard_attention(q, k, v, bool_mask, num_heads, num_kv_heads, scale):
    """Standard matmul attention (traces to plain ONNX ops, no Custom node)."""
    k, v = _expand_kv(k, v, num_heads, num_kv_heads)
    scores = torch.matmul(q, k.transpose(-2, -1)) * float(scale)
    scores = scores.masked_fill(bool_mask, torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(probs, v)


def _standard_scatter(key_cache, value_cache, cache_pos, k, v):
    """Standard scatter-update of the fixed KV cache (traces to ONNX Scatter)."""
    bsz = key_cache.shape[0]
    idx = cache_pos.view(bsz, 1, 1, 1).to(dtype=torch.int64)
    idx = idx.expand(bsz, key_cache.shape[1], 1, key_cache.shape[3])
    return key_cache.scatter(2, idx, k), value_cache.scatter(2, idx, v)


def _attn_prefill(attn_mod, hidden_states, cos, sin, bool_mask):
    """Compute prefill attention (custom PromptFlashAttention or standard)."""
    bsz, seq_len, _ = hidden_states.shape
    head_dim = attn_mod.head_dim
    num_heads = attn_mod.num_heads
    num_kv_heads = attn_mod.num_key_value_heads
    q, k, v = _fused_qkv(attn_mod, hidden_states)
    q = q.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    v = v.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    q, k = _rotary(q, k, cos, sin)
    if _USE_CUSTOM_ATTN:
        attn_out = _CustomPromptFlashAttention.apply(
            q, k, v, bool_mask, int(num_heads), int(num_kv_heads),
            float(attn_mod.scaling), _CUSTOM_ATTN_LAYOUT)
    else:
        attn_out = _standard_attention(q, k, v, bool_mask, num_heads, num_kv_heads, attn_mod.scaling)
    attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1).contiguous()
    attn_out = attn_mod.o_proj(attn_out)
    return attn_out, k, v


def _layer_forward_decode(layer, hidden_states, cos, sin, bool_mask, cache_pos,
                          key_cache, value_cache):
    """Run one GLM decoder layer in decode mode with fixed KV cache + scatter."""
    residual = hidden_states
    hidden_states = _rms_norm(layer.input_layernorm, hidden_states)
    attn_out, pk, pv = _attn_decode(
        layer.self_attn, hidden_states, cos, sin, bool_mask, cache_pos, key_cache, value_cache)
    hidden_states = _rms_norm(layer.post_self_attn_layernorm, attn_out)
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = _rms_norm(layer.post_attention_layernorm, hidden_states)
    hidden_states = _mlp_forward(layer, hidden_states)
    hidden_states = _rms_norm(layer.post_mlp_layernorm, hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states, pk, pv


def _attn_decode(attn_mod, hidden_states, cos, sin, bool_mask, cache_pos,
                 key_cache, value_cache):
    """Compute decode-step attention with IncreFlashAttention + scatter cache."""
    bsz, seq_len, _ = hidden_states.shape
    head_dim = attn_mod.head_dim
    num_heads = attn_mod.num_heads
    num_kv_heads = attn_mod.num_key_value_heads
    q, k, v = _fused_qkv(attn_mod, hidden_states)
    q = q.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    v = v.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    q, k = _rotary(q, k, cos, sin)
    cache_dtype = key_cache.dtype
    k = k.to(dtype=cache_dtype)
    v = v.to(dtype=cache_dtype)
    if _USE_CUSTOM_OP:
        key_cache = _CustomScatter.apply(key_cache, cache_pos.to(torch.int64), k, 2)
        value_cache = _CustomScatter.apply(value_cache, cache_pos.to(torch.int64), v, 2)
    else:
        key_cache, value_cache = _standard_scatter(key_cache, value_cache, cache_pos, k, v)
    if _USE_CUSTOM_ATTN:
        attn_out = _CustomIncreFlashAttention.apply(
            q, key_cache, value_cache, bool_mask, int(num_heads), int(num_kv_heads),
            float(attn_mod.scaling), _CUSTOM_ATTN_LAYOUT)
    else:
        attn_out = _standard_attention(q, key_cache, value_cache, bool_mask, num_heads, num_kv_heads, attn_mod.scaling)
    attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1).contiguous()
    attn_out = attn_mod.o_proj(attn_out)
    return attn_out, key_cache, value_cache


class GlmOcrLlmPrefill(torch.nn.Module):
    """GLM-OCR LLM prefill wrapper: full prompt -> logits + KV cache."""

    def __init__(self, text_model, lm_head, image_token_id):
        """Initialize prefill wrapper with text model, lm_head and image token id."""
        super().__init__()
        self.text_model = text_model
        self.lm_head = lm_head
        self.image_token_id = int(image_token_id)
        cfg = text_model.config
        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        rope_params = cfg.rope_parameters
        self.rotary = GlmMrope(head_dim, rope_params["rope_theta"],
                               rope_params.get("mrope_section", [16, 24, 24]))

    def forward(self, input_ids, attention_mask, position_ids, image_embeds):
        """Run prefill: embed -> scatter image embeds -> layers -> logits + kv."""
        inputs_embeds = self.text_model.embed_tokens(input_ids)
        image_mask = input_ids == self.image_token_id
        image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(
            image_mask, image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype))
        cos, sin = self.rotary(position_ids)
        _, q_len = input_ids.shape
        bool_mask = _make_prefill_causal_mask(attention_mask, q_len, q_len)
        hidden_states = inputs_embeds
        present = []
        for layer in self.text_model.layers:
            hidden_states, pk, pv = _layer_forward_prefill(layer, hidden_states, cos, sin, bool_mask)
            present.append(pk)
            present.append(pv)
        hidden_states = _rms_norm(self.text_model.norm, hidden_states)
        logits = self.lm_head(hidden_states)
        present_kv = torch.stack(present, dim=0)
        return logits, present_kv


class GlmOcrLlmDecode(torch.nn.Module):
    """GLM-OCR LLM decode wrapper: single token + KV cache -> logits + kv."""

    def __init__(self, text_model, lm_head, max_seq_len):
        """Initialize decode wrapper with text model, lm_head and fixed seq len."""
        super().__init__()
        self.text_model = text_model
        self.lm_head = lm_head
        self.max_seq_len = int(max_seq_len)
        cfg = text_model.config
        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        rope_params = cfg.rope_parameters
        self.rotary = GlmMrope(head_dim, rope_params["rope_theta"],
                               rope_params.get("mrope_section", [16, 24, 24]))

    def forward(self, input_ids, attention_mask, position_ids, past_key_values, cache_pos):
        """Run decode: embed -> layers (scatter cache) -> logits + kv."""
        inputs_embeds = self.text_model.embed_tokens(input_ids)
        cos, sin = self.rotary(position_ids)
        bsz = input_ids.shape[0]
        cache_pos = cache_pos.view(bsz).to(dtype=torch.int64)
        bool_mask = _make_decode_mask(attention_mask, cache_pos, self.max_seq_len, inputs_embeds.dtype)
        hidden_states = inputs_embeds
        present = []
        for i, layer in enumerate(self.text_model.layers):
            pk_cache = past_key_values[2 * i]
            pv_cache = past_key_values[2 * i + 1]
            hidden_states, pk_cache, pv_cache = _layer_forward_decode(
                layer, hidden_states, cos, sin, bool_mask, cache_pos, pk_cache, pv_cache)
            present.append(pk_cache)
            present.append(pv_cache)
        hidden_states = _rms_norm(self.text_model.norm, hidden_states)
        logits = self.lm_head(hidden_states)
        present_kv = torch.stack(present, dim=0)
        return logits, present_kv


# ---------------------------------------------------------------------------
# Model metadata + module preparation.
# ---------------------------------------------------------------------------


def _get_text_meta(model):
    """Return (text_model, lm_head, num_layers, num_kv_heads, head_dim, image_token_id)."""
    text_model = model.model.language_model
    lm_head = model.lm_head
    cfg = text_model.config
    num_layers = cfg.num_hidden_layers
    num_kv_heads = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    image_token_id = model.config.image_token_id
    return text_model, lm_head, num_layers, num_kv_heads, head_dim, image_token_id


def _load_model(model_id, dtype):
    """Load GLM-OCR on CPU with eager attention at the requested dtype."""
    return GlmOcrForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, low_cpu_mem_usage=True,
        attn_implementation="eager")


def _clear_cache():
    """Release torch caches and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Export routines.
# ---------------------------------------------------------------------------


def export_vision(model, output_dir, device, vision_image_size):
    """Export the GLM-OCR vision encoder to ONNX at a fixed image grid."""
    patch_size = model.config.vision_config.patch_size
    in_channels = getattr(model.config.vision_config, "in_channels", 3)
    temporal_patch_size = model.config.vision_config.temporal_patch_size
    if vision_image_size % patch_size != 0:
        raise ValueError(f"vision_image_size({vision_image_size}) must be divisible by patch_size({patch_size})")
    grid = vision_image_size // patch_size
    grid_thw = torch.tensor([[1, grid, grid]], dtype=torch.int64, device=device)
    num_patches = int(grid_thw[0, 0].item() * grid_thw[0, 1].item() * grid_thw[0, 2].item())
    patch_dim = in_channels * temporal_patch_size * patch_size * patch_size
    dummy_pixels = torch.randn(num_patches, patch_dim, device=device, dtype=torch.float32)

    visual = model.model.visual
    visual.eval().to(device)
    wrapper = VisionWrapper(visual, grid_thw).to(device).eval()

    vision_path = Path(output_dir) / "glm_ocr_vision.onnx"
    print(f"Exporting Vision encoder to {vision_path} (grid={grid}x{grid}, patches={num_patches})...")
    from torch.onnx import utils as onnx_utils
    with torch.no_grad():
        onnx_utils.export(
            wrapper, (dummy_pixels,), str(vision_path),
            input_names=["pixel_values"],
            output_names=["image_embeds"],
            opset_version=14, do_constant_folding=True)
    print("Vision encoder exported successfully.")
    return vision_path


def export_llm(model, output_dir, device, kv_cache_len, dummy_seq, dummy_num_img, dtype):
    """Export GLM-OCR LLM prefill and decode sub-graphs to ONNX."""
    text_model, lm_head, num_layers, num_kv_heads, head_dim, image_token_id = _get_text_meta(model)
    text_model.eval().to(device)
    lm_head.eval().to(device)
    prefill = GlmOcrLlmPrefill(text_model, lm_head, image_token_id).to(device).eval()
    decode = GlmOcrLlmDecode(text_model, lm_head, kv_cache_len).to(device).eval()

    prefill_path = Path(output_dir) / "glm_ocr_llm_prefill.onnx"
    decode_path = Path(output_dir) / "glm_ocr_llm_decode.onnx"

    # Prefill dummy inputs.
    dummy_ids = torch.randint(0, 1000, (1, dummy_seq), dtype=torch.int64, device=device)
    dummy_mask = torch.ones(1, dummy_seq, dtype=torch.int64, device=device)
    dummy_pos = torch.arange(dummy_seq, device=device, dtype=torch.int64).view(1, 1, -1).expand(3, 1, dummy_seq)
    dummy_img = torch.randn(dummy_num_img, text_model.config.hidden_size, device=device, dtype=dtype)

    print(f"Exporting LLM prefill to {prefill_path}...")
    with torch.no_grad():
        torch.onnx.export(
            prefill, (dummy_ids, dummy_mask, dummy_pos, dummy_img), str(prefill_path),
            input_names=["input_ids", "attention_mask", "position_ids", "image_embeds"],
            output_names=["logits", "present_key_values"],
            opset_version=18, do_constant_folding=True, dynamo=False,
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq_len"},
                "attention_mask": {0: "batch", 1: "seq_len"},
                "position_ids": {1: "batch", 2: "seq_len"},
                "image_embeds": {0: "num_image_tokens"},
                "logits": {0: "batch", 1: "seq_len"},
                "present_key_values": {1: "batch", 3: "seq_len"},
            })
    print("LLM prefill exported successfully.")

    # Decode dummy inputs.
    step_ids = torch.randint(0, 1000, (1, 1), dtype=torch.int64, device=device)
    step_mask = torch.ones(1, kv_cache_len, dtype=torch.int64, device=device)
    cache_pos_val = int(min(8, kv_cache_len - 1))
    cache_pos = torch.tensor([cache_pos_val], dtype=torch.int64, device=device)
    step_pos = cache_pos.view(1, 1).expand(3, 1, 1)
    dummy_past = torch.zeros(2 * num_layers, 1, num_kv_heads, kv_cache_len, head_dim,
                             dtype=dtype, device=device)

    print(f"Exporting LLM decode to {decode_path}...")
    with torch.no_grad():
        torch.onnx.export(
            decode, (step_ids, step_mask, step_pos, dummy_past, cache_pos), str(decode_path),
            input_names=["input_ids", "attention_mask", "position_ids", "past_key_values", "cache_pos"],
            output_names=["logits", "present_key_values"],
            opset_version=18, do_constant_folding=True, dynamo=False,
            dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"}})
    print("LLM decode exported successfully.")


def _export_vision_step(model_id, output_dir, device, vision_image_size):
    """Load model, export vision encoder, then release it."""
    print("\nStep 1/2: Exporting Vision encoder...")
    print(f"Loading {model_id} (fp32) for vision export...")
    model = _load_model(model_id, torch.float32)
    try:
        del model.model.language_model
        del model.lm_head
    except (AttributeError, TypeError):
        pass
    _clear_cache()
    export_vision(model, output_dir, device, vision_image_size)
    del model
    _clear_cache()


def _export_llm_step(model_id, output_dir, device, kv_cache_len, dtype_str):
    """Load model, export LLM prefill+decode, then release it."""
    print("\nStep 2/2: Exporting LLM (prefill + decode)...")
    dtype_map = {"fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[dtype_str]
    print(f"Loading {model_id} ({dtype_str}) for LLM export...")
    model = _load_model(model_id, dtype)
    try:
        del model.model.visual
    except (AttributeError, TypeError):
        pass
    _clear_cache()
    export_llm(model, output_dir, device, int(kv_cache_len), dummy_seq=8, dummy_num_img=16, dtype=dtype)
    del model
    _clear_cache()


def _parse_args():
    """Parse command-line arguments for the export script."""
    parser = argparse.ArgumentParser(description="Export GLM-OCR to ONNX")
    parser.add_argument("--model-id", type=str, default="./GLM-OCR",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--output-dir", type=str, default="./glm_ocr_onnx",
                        help="Output directory for ONNX files")
    parser.add_argument("--device", type=str, default="cpu", help="Export device (cpu)")
    parser.add_argument("--vision-image-size", type=int, default=896,
                        help="Fixed image size (must be divisible by patch_size=14).")
    parser.add_argument("--kv-cache-len", type=int, default=2048,
                        help="Fixed KV cache length for decode export.")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"],
                        help="LLM export dtype")
    parser.add_argument("--no-custom-op", action="store_true",
                        help="Disable exporting fused CANN operators as Custom nodes.")
    parser.add_argument("--no-custom-attn", action="store_true",
                        help="Use standard matmul attention instead of flash-attention custom ops.")
    return parser.parse_args()


def main():
    """Parse args and export vision + LLM prefill/decode ONNX sub-graphs."""
    args = _parse_args()
    global _USE_CUSTOM_OP, _USE_CUSTOM_ATTN
    _USE_CUSTOM_OP = not bool(args.no_custom_op)
    _USE_CUSTOM_ATTN = not bool(args.no_custom_attn)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _export_vision_step(args.model_id, output_dir, args.device, args.vision_image_size)
    _export_llm_step(args.model_id, output_dir, args.device, args.kv_cache_len, args.dtype)

    print(f"\nExport finished. Files saved in {args.output_dir}")


if __name__ == "__main__":
    main()
