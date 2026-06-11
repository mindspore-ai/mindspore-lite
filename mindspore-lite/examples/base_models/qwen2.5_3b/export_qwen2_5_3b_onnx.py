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
Export Qwen2.5-3B model to ONNX format (prefill + decode).
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

def _as_list_str(items):
    """Convert items to a list of strings for ONNX custom op attributes."""
    return [str(x) for x in items]


# ---------------------------------------------------------------------------
# Custom ONNX operators: RotaryMul
# ---------------------------------------------------------------------------


class _RotaryMulCustom(torch.autograd.Function):
    """Custom RotaryMul op for ONNX export with eager fallback."""

    @staticmethod
    def forward(ctx, x, cos4, sin4):
        """Eager forward: compute rotary embedding via half-rotation."""
        del ctx
        d = x.shape[-1]
        x1 = x[..., : d // 2]
        x2 = x[..., d // 2 :]
        return torch.cat([-x2, x1], dim=-1) * sin4 + x * cos4

    @staticmethod
    def symbolic(g, x, cos4, sin4):
        """ONNX symbolic: emit Custom op node of type RotaryMul."""
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
    """Apply custom rotary multiplication for position embedding."""
    return _RotaryMulCustom.apply(x, cos4, sin4)


# ---------------------------------------------------------------------------
# Custom ONNX operators: RmsNorm
# ---------------------------------------------------------------------------


class _RmsNormCustom(torch.autograd.Function):
    """Custom RMSNorm op for ONNX export with eager fallback."""

    @staticmethod
    def forward(ctx, x, gamma, epsilon: float):
        """Eager forward: compute root mean square normalization."""
        del ctx
        x_fp32 = x.to(torch.float32)
        var = (x_fp32 * x_fp32).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + float(epsilon))
        y = (x_fp32 * rstd).to(x.dtype) * gamma
        return y, rstd

    @staticmethod
    def symbolic(g, x, gamma, epsilon: float):
        """ONNX symbolic: emit Custom op node of type RmsNorm."""
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
    """Apply custom RMS normalization."""
    return _RmsNormCustom.apply(x, gamma, float(epsilon))


# ---------------------------------------------------------------------------
# Custom ONNX operators: IncreFlashAttention
# ---------------------------------------------------------------------------


def _make_flash_attn_mask(attention_mask, q_len, k_len, past_len):
    """Build a combined causal + padding boolean mask for flash attention."""
    ar_q = torch.arange(q_len, device=attention_mask.device)
    ar_k = torch.arange(k_len, device=attention_mask.device)
    causal = ar_k[None, :] > (past_len + ar_q[:, None])
    causal = causal[None, None, :, :].expand(attention_mask.shape[0], 1, q_len, k_len)
    padding = attention_mask[:, None, None, :].to(torch.bool).logical_not()
    return (causal | padding).to(torch.bool)


def _expand_gqa_kv(k, v, num_heads, num_kv_heads):
    """Expand key/value tensors for grouped-query attention (GQA)."""
    if 0 < num_kv_heads < num_heads:
        rep = num_heads // num_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    return k, v


def _apply_attn_mask(attn, atten_mask):
    """Apply attention mask to attention scores, expanding dims if needed."""
    if atten_mask is not None:
        m = atten_mask.to(torch.bool)
        if m.dim() == 4 and m.shape[1] == 1:
            m = m.expand(attn.shape[0], attn.shape[1], m.shape[2], m.shape[3])
        attn = attn.masked_fill(m, torch.finfo(attn.dtype).min)
    return attn


def _maybe_permute_bnsd(tensor, layout):
    """Permute tensor from BSND/SBND to BNSD layout if needed."""
    if str(layout).upper() in ("BSND", "SBND"):
        return tensor.permute(0, 2, 1, 3)
    return tensor


def _maybe_permute_back(tensor, layout):
    """Permute tensor from BNSD back to BSND/SBND layout if needed."""
    if str(layout).upper() in ("BSND", "SBND"):
        return tensor.permute(0, 2, 1, 3)
    return tensor


class _IncreFlashAttentionCustom(torch.autograd.Function):
    """Custom IncreFlashAttention op for ONNX export with eager fallback."""

    @staticmethod
    def forward(ctx, query, key, value, atten_mask,
                num_heads, scale_value, input_layout,
                num_key_value_heads, block_size, inner_precise):
        """Eager forward: compute incremental flash attention step."""
        del ctx, block_size, inner_precise
        q = _maybe_permute_bnsd(query, input_layout)
        k = _maybe_permute_bnsd(key, input_layout)
        v = _maybe_permute_bnsd(value, input_layout)
        k, v = _expand_gqa_kv(k, v, num_heads, num_key_value_heads)
        attn = torch.matmul(q, k.transpose(2, 3)) * float(scale_value)
        attn = _apply_attn_mask(attn, atten_mask)
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v)
        return _maybe_permute_back(out, input_layout)

    @staticmethod
    def symbolic(g, query, key, value, atten_mask,
                 num_heads, scale_value, input_layout,
                 num_key_value_heads, block_size, inner_precise):
        """ONNX symbolic: emit Custom op node of type IncreFlashAttention."""
        has_mask = atten_mask is not None
        inputs = [query, key, value, atten_mask] if has_mask else [query, key, value]
        input_idx = [0, 1, 2, 3] if has_mask else [0, 1, 2]
        y = g.op(
            "Custom", *inputs,
            type_s="IncreFlashAttention",
            input_names_s=_as_list_str(["query", "key", "value", "atten_mask"]),
            optional_input_names_s=_as_list_str(["atten_mask"]),
            output_names_s=_as_list_str(["attention_out"]),
            output_num_i=1,
            input_index_i=input_idx,
            num_heads_i=int(num_heads),
            scale_value_f=float(scale_value),
            input_layout_s=str(input_layout),
            num_key_value_heads_i=int(num_key_value_heads),
            block_size_i=int(block_size),
            inner_precise_i=int(inner_precise),
        )
        y.setType(query.type())
        return y


def incre_flash_attention(query, key, value, atten_mask,
                          num_heads, scale_value, input_layout,
                          num_key_value_heads,
                          block_size=0, inner_precise=1):
    """Apply custom incremental flash attention for decode step."""
    return _IncreFlashAttentionCustom.apply(
        query, key, value, atten_mask,
        int(num_heads), float(scale_value), str(input_layout),
        int(num_key_value_heads), int(block_size), int(inner_precise),
    )


# ---------------------------------------------------------------------------
# Custom ONNX operators: SwiGLU
# ---------------------------------------------------------------------------


class _SwiGluCustom(torch.autograd.Function):
    """Custom SwiGLU op for ONNX export with eager fallback."""

    @staticmethod
    def forward(ctx, x, dim: int):
        """Eager forward: compute SwiGLU activation (silu(gate) * up)."""
        del ctx
        d = int(dim)
        if d < 0:
            d = x.dim() + d
        a, b = torch.chunk(x, 2, dim=d)
        return F.silu(a) * b

    @staticmethod
    def symbolic(g, x, dim: int):
        """ONNX symbolic: emit Custom op node of type SwiGlu."""
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
    """Apply custom SwiGLU activation function."""
    return _SwiGluCustom.apply(x, int(dim))


# ---------------------------------------------------------------------------
# Custom ONNX operators: Scatter
# ---------------------------------------------------------------------------


class _ScatterCustom(torch.autograd.Function):
    """Custom Scatter op for ONNX export with eager fallback."""

    @staticmethod
    def forward(ctx, var, indices, updates, reduce: str, axis: int):
        """Eager forward: scatter updates into a 4D tensor along axis=2."""
        del ctx
        if str(reduce) != "update":
            raise RuntimeError("Only reduce='update' is supported.")
        ax = var.dim() + int(axis) if int(axis) < 0 else int(axis)
        if var.dim() != 4 or ax != 2:
            raise RuntimeError("Only 4D var with axis=-2/2 is supported.")
        bsz, num_heads = var.shape[0], var.shape[1]
        pos = indices.squeeze(-1).to(torch.long).view(bsz) if indices.dim() == 2 else indices.to(torch.long).view(bsz)
        upd = updates[:, :, 0, :] if updates.dim() == 4 and updates.shape[2] == 1 else updates
        out = var.clone()
        b = torch.arange(bsz, device=out.device).view(bsz, 1).expand(bsz, num_heads)
        h = torch.arange(num_heads, device=out.device).view(1, num_heads).expand(bsz, num_heads)
        out[b, h, pos.view(bsz, 1).expand(bsz, num_heads), :] = upd
        return out

    @staticmethod
    def symbolic(g, var, indices, updates, reduce: str, axis: int):
        """ONNX symbolic: emit Custom op node of type Scatter."""
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


def scatter(var, indices, updates, reduce: str = "update", axis: int = -2):
    """Apply custom scatter operation for KV cache update."""
    return _ScatterCustom.apply(var, indices, updates, str(reduce), int(axis))


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _rms_norm_layer(norm_mod, x):
    """Apply RMS normalization using a module's weight and epsilon."""
    gamma = norm_mod.weight
    eps = getattr(norm_mod, "variance_epsilon", 1e-6)
    y, _ = rms_norm(x, gamma, eps)
    return y


def _fuse_qkv_linear(attn_mod, hidden_states):
    """Fuse Q/K/V projections into a single linear and split outputs.

    Returns (query_states, key_states, value_states) reshaped to
    (batch, seq, num_heads, head_dim) for attention computation.
    """
    q_w, k_w, v_w = attn_mod.q_proj.weight, attn_mod.k_proj.weight, attn_mod.v_proj.weight
    q_b, k_b, v_b = attn_mod.q_proj.bias, attn_mod.k_proj.bias, attn_mod.v_proj.bias
    w = torch.cat([q_w, k_w, v_w], dim=0)
    b = torch.cat([q_b, k_b, v_b], dim=0) if q_b is not None else None
    qkv = F.linear(hidden_states, w, b)
    q_f, kv_f = int(q_w.shape[0]), int(k_w.shape[0])
    head_dim = attn_mod.head_dim
    hidden_shape = (*hidden_states.shape[:-1], -1, head_dim)
    return (qkv[..., :q_f].view(hidden_shape),
            qkv[..., q_f:q_f + kv_f].view(hidden_shape),
            qkv[..., q_f + kv_f:].view(hidden_shape))


def _compute_prefill_attn(query_states, key_states, value_states,
                          attention_mask, num_heads, num_kv_heads, scaling):
    """Compute prefill attention with manual matmul + causal/padding mask.

    Returns (attn_output, key_states, value_states) in BSHD layout.
    """
    q, k, v = query_states.permute(0, 2, 1, 3), key_states.permute(0, 2, 1, 3), value_states.permute(0, 2, 1, 3)
    k_cache, v_cache = k, v  # save unexpanded KV for cache output (BNSD format)
    k, v = _expand_gqa_kv(k, v, num_heads, num_kv_heads)
    attn = torch.matmul(q, k.transpose(2, 3)) * float(scaling)
    flash_mask = _make_flash_attn_mask(attention_mask, attn.shape[-2], attn.shape[-1], 0)
    if flash_mask.dim() == 4 and flash_mask.shape[1] == 1:
        flash_mask = flash_mask.expand(attn.shape[0], attn.shape[1], flash_mask.shape[2], flash_mask.shape[3])
    attn = attn.masked_fill(flash_mask.to(torch.bool), torch.finfo(attn.dtype).min)
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_output = torch.matmul(attn, v).permute(0, 2, 1, 3)
    return attn_output, k_cache, v_cache


def _text_attn_forward(attn_mod, hidden_states, cos4, sin4,
                       attention_mask, cache_pos, past_key, past_value):
    """Forward pass for Qwen2.5 attention layer.

    Handles both prefill (past_key=None) and decode (past_key provided).
    Qwen2.5 does NOT have q_norm/k_norm (unlike Qwen3).
    """
    input_shape = hidden_states.shape[:-1]
    num_heads = attn_mod.config.num_attention_heads
    num_kv_heads = attn_mod.config.num_key_value_heads
    scaling = getattr(attn_mod, "scaling", 1.0 / (attn_mod.head_dim ** 0.5))

    query_states, key_states, value_states = _fuse_qkv_linear(attn_mod, hidden_states)

    if past_key is not None:
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

    query_states = rotary_mul(query_states, cos4, sin4)
    key_states = rotary_mul(key_states, cos4, sin4)

    if past_key is not None:
        pos = cache_pos[:, -1] if cache_pos.dim() == 2 else cache_pos
        key_states = scatter(past_key, pos, key_states, reduce="update", axis=-2)
        value_states = scatter(past_value, pos, value_states, reduce="update", axis=-2)

    if past_key is None:
        attn_output, key_states, value_states = _compute_prefill_attn(
            query_states, key_states, value_states, attention_mask, num_heads, num_kv_heads, scaling)
        attn_output = attn_output.reshape(*input_shape, -1)
    else:
        pad_mask = attention_mask[:, None, None, :].to(torch.bool).logical_not()
        attn_output = incre_flash_attention(
            query_states, key_states, value_states, pad_mask,
            num_heads=num_heads, scale_value=float(scaling),
            input_layout="BNSD", num_key_value_heads=num_kv_heads, inner_precise=1)
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1)

    return attn_mod.o_proj(attn_output), key_states, value_states


def _mlp_gate_up_linear(mlp_mod, x):
    """Fuse gate_proj and up_proj into a single linear and split outputs.

    Returns (gate, up) tensors for subsequent SwiGLU activation.
    """
    gate_w, up_w = mlp_mod.gate_proj.weight, mlp_mod.up_proj.weight
    gate_b, up_b = mlp_mod.gate_proj.bias, mlp_mod.up_proj.bias
    w = torch.cat([gate_w, up_w], dim=0)
    b = torch.cat([gate_b, up_b], dim=0) if gate_b is not None else None
    y = F.linear(x, w, b)
    gate_f = int(gate_w.shape[0])
    return y[..., :gate_f], y[..., gate_f:]


def _run_mlp(layer, hidden_states):
    """Run MLP forward with fused gate+up projection and SwiGLU."""
    mlp = layer.mlp
    if hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj") and hasattr(mlp, "down_proj"):
        gate, up = _mlp_gate_up_linear(mlp, hidden_states)
        return mlp.down_proj(swiglu(torch.cat([gate, up], dim=-1), dim=-1))
    return mlp(hidden_states)


def _run_transformer_layer(layer, hidden_states, cos4, sin4,
                           attention_mask, cache_pos, past_key, past_value):
    """Run a single transformer layer: attention + MLP with residual."""
    residual = hidden_states
    hidden_states = _rms_norm_layer(layer.input_layernorm, hidden_states)
    attn_out, pk, pv = _text_attn_forward(
        layer.self_attn, hidden_states, cos4, sin4,
        attention_mask, cache_pos, past_key, past_value)
    hidden_states = residual + attn_out
    residual = hidden_states
    hidden_states = _rms_norm_layer(layer.post_attention_layernorm, hidden_states)
    return residual + _run_mlp(layer, hidden_states), pk, pv


# ---------------------------------------------------------------------------
# Prefill / Decode wrappers
# ---------------------------------------------------------------------------


class Qwen2LlmPrefill(torch.nn.Module):
    """Qwen2.5-3B LLM Prefill wrapper for ONNX export.

    Processes the full input prompt and outputs logits plus zero-padded KV cache.
    """

    def __init__(self, model, lm_head):
        super().__init__()
        self.model = model.model
        self.lm_head = lm_head

    def forward(self, input_ids, attention_mask, position_ids):
        """Run prefill: embed tokens, iterate layers, return logits and KV cache."""
        inputs_embeds = self.model.embed_tokens(input_ids)
        cos, sin = self.model.rotary_emb(inputs_embeds, position_ids)
        cos4 = cos.unsqueeze(2) if cos.dim() == 3 else cos
        sin4 = sin.unsqueeze(2) if sin.dim() == 3 else sin
        hidden_states = inputs_embeds
        present_k, present_v = [], []

        for layer in self.model.layers:
            hidden_states, pk, pv = _run_transformer_layer(
                layer, hidden_states, cos4, sin4, attention_mask, None, None, None)
            pk = torch.cat(
                [pk, pk.new_zeros(pk.shape[0], pk.shape[1], KV_CACHE_LEN, pk.shape[3])],
                dim=2)[:, :, :KV_CACHE_LEN, :]
            pv = torch.cat(
                [pv, pv.new_zeros(pv.shape[0], pv.shape[1], KV_CACHE_LEN, pv.shape[3])],
                dim=2)[:, :, :KV_CACHE_LEN, :]
            present_k.append(pk)
            present_v.append(pv)

        hidden_states = _rms_norm_layer(self.model.norm, hidden_states)
        return self.lm_head(hidden_states), torch.stack(present_k, dim=0), torch.stack(present_v, dim=0)


class Qwen2LlmDecode(torch.nn.Module):
    """Qwen2.5-3B LLM Decode wrapper for ONNX export.

    Processes a single token with past KV cache for auto-regressive generation.
    """

    def __init__(self, model, lm_head):
        super().__init__()
        self.model = model.model
        self.lm_head = lm_head

    def forward(self, input_ids, attention_mask, position_ids,
                past_key_cache, past_value_cache):
        """Run decode: embed single token, iterate layers with KV cache, return logits."""
        inputs_embeds = self.model.embed_tokens(input_ids)
        cos, sin = self.model.rotary_emb(inputs_embeds, position_ids)
        cos4 = cos.unsqueeze(1) if cos.dim() == 3 else cos
        sin4 = sin.unsqueeze(1) if sin.dim() == 3 else sin
        hidden_states = inputs_embeds
        present_k, present_v = [], []
        past_k_layers = past_key_cache.unbind(0)
        past_v_layers = past_value_cache.unbind(0)

        for i, layer in enumerate(self.model.layers):
            hidden_states, pk, pv = _run_transformer_layer(
                layer, hidden_states, cos4, sin4,
                attention_mask, position_ids, past_k_layers[i], past_v_layers[i])
            present_k.append(pk)
            present_v.append(pv)

        hidden_states = _rms_norm_layer(self.model.norm, hidden_states)
        return self.lm_head(hidden_states), torch.stack(present_k, dim=0), torch.stack(present_v, dim=0)


# ---------------------------------------------------------------------------
# Export logic
# ---------------------------------------------------------------------------


def _prepare_llm_modules(model, device: str):
    """Wrap model into prefill/decode modules and move to device."""
    lm_head = model.lm_head
    model.eval()
    lm_head.eval()
    model.to(device)
    lm_head.to(device)
    prefill = Qwen2LlmPrefill(model, lm_head).to(device).eval()
    decode = Qwen2LlmDecode(model, lm_head).to(device).eval()
    return prefill, decode, lm_head


def _get_kv_cache_config(model):
    """Extract (num_layers, num_kv_heads, head_dim) from model config."""
    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    head_dim = getattr(model.config, "head_dim",
                       model.config.hidden_size // model.config.num_attention_heads)
    return num_layers, num_kv_heads, head_dim


def _prepare_output_paths(output_dir):
    """Create prefill/decode output directories and return ONNX file paths."""
    prefill_dir = Path(output_dir) / "prefill"
    decode_dir = Path(output_dir) / "decode"
    prefill_dir.mkdir(parents=True, exist_ok=True)
    decode_dir.mkdir(parents=True, exist_ok=True)
    return prefill_dir / "qwen2_5_3b_llm_prefill.onnx", decode_dir / "qwen2_5_3b_llm_decode.onnx"


def _create_prefill_dummy_inputs(device: str, dummy_seq_len: int):
    """Create random dummy inputs for prefill ONNX export."""
    seq = int(dummy_seq_len)
    input_ids = torch.randint(0, 1000, (1, seq), dtype=torch.int64, device=device)
    attention_mask = torch.ones(1, seq, dtype=torch.int64, device=device)
    position_ids = torch.arange(seq, device=device, dtype=torch.int64).view(1, -1)
    return input_ids, attention_mask, position_ids


def _export_prefill_onnx(prefill, prefill_path: Path, dummy_inputs, use_dynamo: bool):
    """Export the prefill model to ONNX with dynamic sequence length."""
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
            },
        )
    print("LLM prefill exported successfully.")


def _create_decode_dummy_inputs(device, num_layers, num_kv_heads, head_dim, kv_dtype):
    """Create zero-filled dummy inputs for decode ONNX export."""
    dummy_input_ids = torch.randint(0, 1000, (1, 1), dtype=torch.int64, device=device)
    dummy_mask = torch.ones(1, KV_CACHE_LEN, dtype=torch.int64, device=device)
    dummy_pos = torch.tensor([[KV_CACHE_LEN - 1]], dtype=torch.int64, device=device)
    shape = (num_layers, 1, num_kv_heads, KV_CACHE_LEN, head_dim)
    dummy_k = torch.zeros(shape, dtype=kv_dtype, device=device)
    dummy_v = torch.zeros(shape, dtype=kv_dtype, device=device)
    return dummy_input_ids, dummy_mask, dummy_pos, dummy_k, dummy_v


def _export_decode_onnx(decode, decode_path: Path, dummy_inputs, use_dynamo: bool):
    """Export the decode model to ONNX with fixed KV cache shape."""
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
            },
        )
    print("LLM decode exported successfully.")


def export_llm_prefill_decode(model, output_dir, device="cpu",
                               dummy_seq_len=8, use_dynamo=False):
    """Export Qwen2.5-3B model as prefill + decode ONNX subgraphs."""
    prefill, decode, _ = _prepare_llm_modules(model, device=device)
    kv_dtype = next(model.parameters()).dtype
    num_layers, num_kv_heads, head_dim = _get_kv_cache_config(model)
    prefill_path, decode_path = _prepare_output_paths(output_dir)

    prefill_dummies = _create_prefill_dummy_inputs(device=device, dummy_seq_len=dummy_seq_len)
    _export_prefill_onnx(prefill, prefill_path, prefill_dummies, use_dynamo)

    decode_dummies = _create_decode_dummy_inputs(device, num_layers, num_kv_heads, head_dim, kv_dtype)
    _export_decode_onnx(decode, decode_path, decode_dummies, use_dynamo)


def _parse_export_args():
    """Parse command-line arguments for ONNX export."""
    parser = argparse.ArgumentParser(description="Export Qwen2.5-3B to ONNX")
    parser.add_argument("--model-id", type=str, default="./Qwen2.5-3B-Instruct",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--output-dir", type=str, default="./qwen2_5_3b_onnx",
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


def _resolve_dtype(dtype_str: str):
    """Convert string dtype name to torch dtype."""
    if dtype_str == "fp16":
        return torch.float16
    if dtype_str == "bf16":
        return torch.bfloat16
    return torch.float32


def main():
    """Main entry: load model, export prefill + decode ONNX, cleanup."""
    args = _parse_export_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = _resolve_dtype(args.dtype)
    device = torch.device(args.device)
    print(f"\nLoading model {args.model_id} for export (dtype={args.dtype})...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch_dtype,
        low_cpu_mem_usage=False, attn_implementation="eager").to(device)

    export_llm_prefill_decode(model, output_dir, str(device), args.dummy_seq_len, args.use_dynamo)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"\nExport finished. Files saved in {args.output_dir}")


if __name__ == "__main__":
    main()
