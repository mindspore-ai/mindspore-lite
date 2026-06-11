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
Export Qwen2-7B-Instruct model to ONNX format.
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
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
except Exception:
    apply_rotary_pos_emb = None


def _as_list_str(items):
    """Convert a list of items to a list of string representations."""
    return [str(x) for x in items]


def _rotate_half(x):
    """Rotate half the hidden dims of the input tensor for rotary embeddings."""
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _build_custom_op(g, op_type, inputs, input_names, optional_names, output_names,
                     output_num, input_index, **attrs):
    """Build a custom ONNX operator node with the given attributes."""
    y = g.op(
        "Custom",
        *inputs,
        type_s=op_type,
        input_names_s=_as_list_str(input_names),
        optional_input_names_s=_as_list_str(optional_names),
        output_names_s=_as_list_str(output_names),
        output_num_i=output_num,
        input_index_i=input_index,
        **attrs,
    )
    if isinstance(y, tuple):
        for item in y:
            item.setType(inputs[0].type())
    else:
        y.setType(inputs[0].type())
    return y


class _RotaryMulCustom(torch.autograd.Function):
    """Custom RotaryMul op for ONNX export."""

    @staticmethod
    def forward(ctx, x, cos4, sin4):
        """Eager fallback: compute rotary multiplication."""
        del ctx
        return (x * cos4) + (_rotate_half(x) * sin4)

    @staticmethod
    def symbolic(g, x, cos4, sin4):
        """ONNX symbolic for RotaryMul."""
        y = _build_custom_op(
            g, "RotaryMul", [x, cos4, sin4],
            ["x", "r1", "r2"], [], ["y"], 1, [0, 1, 2],
        )
        return y


def rotary_mul(x, cos4, sin4):
    """Apply rotary position embedding multiplication to input tensor."""
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
        q, k = _build_custom_op(
            g, "ApplyRotaryPosEmb", [query, key, cos4, sin4],
            ["query", "key", "cos", "sin"], [], ["query", "key"], 2,
            [0, 1, 2, 3], layout_i=int(layout), rotary_mode_s=str(rotary_mode),
            outputs=2,
        )
        k.setType(key.type())
        return q, k


def apply_rotary_pos_emb_custom(query, key, cos, sin, layout: int = 3, rotary_mode: str = "half"):
    """Apply rotary position embedding with custom ONNX operator support."""
    return _ApplyRotaryPosEmbCustom.apply(query, key, cos, sin, int(layout), str(rotary_mode))


class _RmsNormCustom(torch.autograd.Function):
    """Custom RMSNorm op for ONNX export."""

    @staticmethod
    def forward(ctx, x, gamma, epsilon: float):
        """Eager fallback: compute RMS normalization."""
        del ctx
        x_fp32 = x.to(torch.float32)
        var = (x_fp32 * x_fp32).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + float(epsilon))
        y = (x_fp32 * rstd).to(x.dtype) * gamma
        return y, rstd

    @staticmethod
    def symbolic(g, x, gamma, epsilon: float):
        """ONNX symbolic for RMSNorm."""
        y, rstd = _build_custom_op(
            g, "RmsNorm", [x, gamma],
            ["x", "gamma"], [], ["y", "rstd"], 2, [0, 1],
            epsilon_f=float(epsilon), outputs=2,
        )
        return y, rstd


def rms_norm(x, gamma, epsilon: float = 1e-6):
    """Apply RMS normalization with custom ONNX operator support."""
    return _RmsNormCustom.apply(x, gamma, float(epsilon))


def _make_flash_attn_mask(attention_mask, q_len, k_len, past_len):
    """Create boolean flash attention mask combining causal and padding masks."""
    ar_q = torch.arange(q_len, device=attention_mask.device)
    ar_k = torch.arange(k_len, device=attention_mask.device)
    causal = ar_k[None, :] > (past_len + ar_q[:, None])
    causal = causal[None, None, :, :].expand(attention_mask.shape[0], 1, q_len, k_len)
    padding = attention_mask[:, None, None, :].to(torch.bool).logical_not()
    return (causal | padding).to(torch.bool)


def _eager_attention(q, k, v, scale_value, num_kv_heads, num_heads, atten_mask):
    """Compute scaled dot-product attention in eager mode."""
    if 0 < num_kv_heads < num_heads:
        rep = num_heads // num_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    attn = torch.matmul(q, k.transpose(2, 3)) * float(scale_value)
    if atten_mask is not None:
        m = atten_mask.to(torch.bool)
        if m.dim() == 4 and m.shape[1] == 1:
            m = m.expand(attn.shape[0], attn.shape[1], m.shape[2], m.shape[3])
        attn = attn.masked_fill(m, torch.finfo(attn.dtype).min)
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(attn, v)


def _apply_layout_permute(tensors, input_layout, inverse=False):
    """Permute tensors between BSND and BNSD layouts."""
    q, k, v = tensors
    layout = str(input_layout).upper()
    if layout in ("BSND", "SBND"):
        if inverse:
            return tuple(t.permute(0, 2, 1, 3) for t in (q, k, v))
        return tuple(t.permute(0, 2, 1, 3) for t in (q, k, v))
    return q, k, v


class _IncreFlashAttentionCustom(torch.autograd.Function):
    """Custom IncreFlashAttention op for ONNX export."""

    @staticmethod
    def forward(ctx, query, key, value, atten_mask, num_heads, scale_value,
                input_layout, num_key_value_heads, block_size, inner_precise):
        """Eager fallback: compute incremental flash attention for decode."""
        del ctx, block_size, inner_precise
        q, k, v = _apply_layout_permute((query, key, value), input_layout)
        out = _eager_attention(q, k, v, scale_value, num_key_value_heads, num_heads, atten_mask)
        layout = str(input_layout).upper()
        if layout in ("BSND", "SBND"):
            out = out.permute(0, 2, 1, 3)
        return out

    @staticmethod
    def symbolic(g, query, key, value, atten_mask, num_heads, scale_value,
                 input_layout, num_key_value_heads, block_size, inner_precise):
        """ONNX symbolic for incremental flash attention."""
        base_inputs = [query, key, value]
        base_index = [0, 1, 2]
        if atten_mask is not None:
            base_inputs.append(atten_mask)
            base_index.append(3)
        y = _build_custom_op(
            g, "IncreFlashAttention", base_inputs,
            ["query", "key", "value", "atten_mask"],
            ["atten_mask"], ["attention_out"], 1, base_index,
            num_heads_i=int(num_heads), scale_value_f=float(scale_value),
            input_layout_s=str(input_layout),
            num_key_value_heads_i=int(num_key_value_heads),
            block_size_i=int(block_size), inner_precise_i=int(inner_precise),
        )
        return y


def incre_flash_attention(query, key, value, atten_mask, num_heads: int,
                          scale_value: float, input_layout: str,
                          num_key_value_heads: int, block_size: int = 0,
                          inner_precise: int = 1):
    """Functional wrapper for incremental flash attention."""
    return _IncreFlashAttentionCustom.apply(
        query, key, value, atten_mask, int(num_heads), float(scale_value),
        str(input_layout), int(num_key_value_heads), int(block_size), int(inner_precise),
    )


class _PromptFlashAttentionCustom(torch.autograd.Function):
    """Custom prompt flash attention op for ONNX export."""

    @staticmethod
    def forward(ctx, query, key, value, atten_mask, num_heads, scale_value,
                input_layout, num_key_value_heads, sparse_mode,
                inner_precise, pre_tokens, next_tokens):
        """Eager fallback: compute prompt flash attention for prefill."""
        del ctx, inner_precise, pre_tokens, next_tokens
        q, k, v = _apply_layout_permute((query, key, value), input_layout)
        out = _eager_attention(q, k, v, scale_value, num_key_value_heads, num_heads, atten_mask)
        if atten_mask is None and int(sparse_mode) in (2, 3):
            out = _apply_causal_mask(out)
        layout = str(input_layout).upper()
        if layout in ("BSND", "SBND"):
            out = out.permute(0, 2, 1, 3)
        return out

    @staticmethod
    def symbolic(g, query, key, value, atten_mask, num_heads, scale_value,
                 input_layout, num_key_value_heads, sparse_mode,
                 inner_precise, pre_tokens, next_tokens):
        """ONNX symbolic for prompt flash attention."""
        base_inputs = [query, key, value]
        base_index = [0, 1, 2]
        if atten_mask is not None:
            base_inputs.append(atten_mask)
            base_index.append(3)
        y = _build_custom_op(
            g, "PromptFlashAttention", base_inputs,
            ["query", "key", "value", "atten_mask"],
            ["atten_mask"], ["attention_out"], 1, base_index,
            num_heads_i=int(num_heads), scale_value_f=float(scale_value),
            pre_tokens_i=int(pre_tokens), next_tokens_i=int(next_tokens),
            input_layout_s=str(input_layout),
            num_key_value_heads_i=int(num_key_value_heads),
            sparse_mode_i=int(sparse_mode), inner_precise_i=int(inner_precise),
        )
        return y


def _apply_causal_mask(attn):
    """Apply causal mask to attention scores when sparse_mode indicates causal."""
    q_len = attn.shape[-2]
    k_len = attn.shape[-1]
    ar_q = torch.arange(q_len, device=attn.device)
    ar_k = torch.arange(k_len, device=attn.device)
    causal = ar_k[None, :] > ar_q[:, None]
    causal = causal[None, None, :, :].expand(attn.shape[0], attn.shape[1], q_len, k_len)
    return attn.masked_fill(causal, torch.finfo(attn.dtype).min)


def prompt_flash_attention(query, key, value, atten_mask, num_heads: int,
                            scale_value: float, input_layout: str,
                            num_key_value_heads: int, sparse_mode: int = 0,
                            inner_precise: int = 1, pre_tokens: int = 214748647,
                            next_tokens: int = 0):
    """Functional wrapper for prompt flash attention."""
    return _PromptFlashAttentionCustom.apply(
        query, key, value, atten_mask, int(num_heads), float(scale_value),
        str(input_layout), int(num_key_value_heads), int(sparse_mode),
        int(inner_precise), int(pre_tokens), int(next_tokens),
    )


class _SwiGluCustom(torch.autograd.Function):
    """Custom SwiGLU op for ONNX export."""

    @staticmethod
    def forward(ctx, x, dim: int):
        """Eager fallback: compute SwiGLU activation (silu(gate) * up)."""
        del ctx
        d = int(dim)
        if d < 0:
            d = x.dim() + d
        a, b = torch.chunk(x, 2, dim=d)
        return F.silu(a) * b

    @staticmethod
    def symbolic(g, x, dim: int):
        """ONNX symbolic for SwiGLU."""
        y = _build_custom_op(
            g, "SwiGlu", [x], ["x"], [], ["y"], 1, [0], dim_i=int(dim),
        )
        return y


def swiglu(x, dim: int = -1):
    """Apply SwiGLU activation (silu(gate) * up) with custom ONNX operator support."""
    return _SwiGluCustom.apply(x, int(dim))


class _ScatterCustom(torch.autograd.Function):
    """Custom scatter op for ONNX export."""

    @staticmethod
    def forward(ctx, var, indices, updates, reduce: str, axis: int):
        """Eager fallback: scatter update along a specified axis."""
        del ctx
        if str(reduce) != "update":
            raise RuntimeError("Only reduce='update' is supported.")
        ax = int(axis)
        if ax < 0:
            ax = var.dim() + ax
        if var.dim() != 4 or ax != 2:
            raise RuntimeError("Only 4D var with axis=-2/2 is supported.")
        bsz, num_heads, _, _ = var.shape
        pos = _flatten_indices(indices, bsz)
        upd = _flatten_updates(updates)
        out = _scatter_update(var, pos, upd, bsz, num_heads)
        return out

    @staticmethod
    def symbolic(g, var, indices, updates, reduce: str, axis: int):
        """ONNX symbolic for scatter."""
        y = _build_custom_op(
            g, "Scatter", [var, indices, updates],
            ["var", "indices", "updates"], [], ["var"], 1, [0, 1, 2],
            reduce_s=str(reduce), axis_i=int(axis),
        )
        return y


def _flatten_indices(indices, bsz):
    """Flatten index tensor to 1D shape (bsz,) for scatter operation."""
    pos = indices
    if pos.dim() == 2 and pos.shape[-1] == 1:
        pos = pos.squeeze(-1)
    return pos.to(torch.long).view(bsz)


def _flatten_updates(updates):
    """Flatten update tensor by removing the singleton sequence dimension."""
    if updates.dim() == 4 and updates.shape[2] == 1:
        return updates[:, :, 0, :]
    return updates


def _scatter_update(var, pos, upd, bsz, num_heads):
    """Perform the actual scatter update at the specified positions."""
    out = var.clone()
    b = torch.arange(bsz, device=out.device).view(bsz, 1).expand(bsz, num_heads)
    h = torch.arange(num_heads, device=out.device).view(1, num_heads).expand(bsz, num_heads)
    s = pos.view(bsz, 1).expand(bsz, num_heads)
    out[b, h, s, :] = upd
    return out


def scatter(var, indices, updates, reduce: str = "update", axis: int = -2):
    """Scatter update tensor values at specified indices along an axis."""
    return _ScatterCustom.apply(var, indices, updates, str(reduce), int(axis))


def _make_additive_causal_mask(attention_mask, q_len, k_len, past_len, dtype):
    """Create additive causal mask combining causal and padding components."""
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
    """Compute fused QKV projection and reshape to (batch, seq, heads, head_dim)."""
    head_dim = attn_mod.head_dim
    hidden_shape = (*hidden_states.shape[:-1], -1, head_dim)
    q_w = attn_mod.q_proj.weight
    k_w = attn_mod.k_proj.weight
    v_w = attn_mod.v_proj.weight
    w = torch.cat([q_w, k_w, v_w], dim=0)
    b = _cat_biases(attn_mod)
    q_out = int(q_w.shape[0])
    kv_out = int(k_w.shape[0])
    qkv = F.linear(hidden_states, w, b)
    q = qkv[..., :q_out].view(hidden_shape)
    k = qkv[..., q_out:q_out + kv_out].view(hidden_shape)
    v = qkv[..., q_out + kv_out:].view(hidden_shape)
    return q, k, v


def _cat_biases(attn_mod):
    """Concatenate Q/K/V bias into a single bias vector, or return None."""
    q_b = attn_mod.q_proj.bias
    k_b = attn_mod.k_proj.bias
    v_b = attn_mod.v_proj.bias
    if q_b is None:
        return None
    return torch.cat([q_b, k_b, v_b], dim=0)


def _apply_rotary_and_cache(cos4, sin4, query, key, value, past_key,
                            past_value, cache_pos):
    """Apply rotary embedding and update KV cache via scatter operation."""
    query = rotary_mul(query, cos4, sin4)
    key = rotary_mul(key, cos4, sin4)
    if past_key is None:
        return query, key, value
    pos = cache_pos
    if pos is None:
        raise RuntimeError("cache_pos is required when past_key_values is provided.")
    if pos.dim() == 2:
        pos = pos[:, -1]
    key = scatter(past_key, pos, key, reduce="update", axis=-2)
    value = scatter(past_value, pos, value, reduce="update", axis=-2)
    return query, key, value


def _prefill_attention(query, key, value, attention_mask, num_heads,
                       num_kv_heads, scaling):
    """Compute attention for the prefill phase using standard scaled dot-product."""
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    k_cache, v_cache = k, v
    if 0 < num_kv_heads < num_heads:
        rep = num_heads // num_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    q_len, k_len = q.shape[2], k_cache.shape[2]
    attn = torch.matmul(q, k.transpose(2, 3)) * float(scaling)
    flash_mask = _make_flash_attn_mask(attention_mask, q_len, k_len, 0)
    if flash_mask.dim() == 4 and flash_mask.shape[1] == 1:
        flash_mask = flash_mask.expand(attn.shape[0], attn.shape[1], flash_mask.shape[2], flash_mask.shape[3])
    attn = attn.masked_fill(flash_mask.to(torch.bool), torch.finfo(attn.dtype).min)
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_output = torch.matmul(attn, v).permute(0, 2, 1, 3)
    return attn_output, k_cache, v_cache


def _decode_attention(query, key, value, attention_mask, num_heads,
                      num_kv_heads, scaling):
    """Compute attention for the decode phase using incremental flash attention."""
    pad_mask = attention_mask[:, None, None, :].to(torch.bool).logical_not()
    attn_output = incre_flash_attention(
        query, key, value, pad_mask,
        num_heads=num_heads, scale_value=float(scaling),
        input_layout="BNSD", num_key_value_heads=num_kv_heads, inner_precise=1,
    )
    return attn_output.transpose(1, 2)


def _text_attn_forward(attn_mod, hidden_states, cos4, sin4, attention_mask,
                        cache_pos, past_key, past_value):
    """Full attention forward supporting both prefill and decode modes."""
    num_heads = attn_mod.config.num_attention_heads
    num_kv_heads = attn_mod.config.num_key_value_heads
    input_shape = hidden_states.shape[:-1]

    query, key, value = _compute_qkv_proj(attn_mod, hidden_states)
    if past_key is not None:
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

    query, key, value = _apply_rotary_and_cache(
        cos4, sin4, query, key, value, past_key, past_value, cache_pos)

    scaling = getattr(attn_mod, "scaling", 1.0 / (attn_mod.head_dim ** 0.5))
    if past_key is None:
        attn_output, key, value = _prefill_attention(
            query, key, value, attention_mask, num_heads, num_kv_heads, scaling)
        attn_output = attn_output.reshape(*input_shape, -1)
    else:
        attn_output = _decode_attention(
            query, key, value, attention_mask, num_heads, num_kv_heads, scaling)
        attn_output = attn_output.reshape(*input_shape, -1)
    return attn_mod.o_proj(attn_output), key, value


def _rms_norm_layer(norm_mod, x):
    """Apply RMS normalization using the custom ONNX-friendly operator."""
    gamma = norm_mod.weight
    eps = getattr(norm_mod, "variance_epsilon", 1e-6)
    y, _ = rms_norm(x, gamma, eps)
    return y


def _mlp_gate_up_linear(mlp_mod, x):
    """Merge gate_proj and up_proj into a single linear and split outputs."""
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


def _mlp_forward(mlp_mod, hidden_states):
    """Compute MLP forward with fused gate+up projection and SwiGLU activation."""
    if hasattr(mlp_mod, "gate_proj") and hasattr(mlp_mod, "up_proj") and hasattr(mlp_mod, "down_proj"):
        gate, up = _mlp_gate_up_linear(mlp_mod, hidden_states)
        x = torch.cat([gate, up], dim=-1)
        return mlp_mod.down_proj(swiglu(x, dim=-1))
    return mlp_mod(hidden_states)


def _run_transformer_layer(layer, hidden_states, cos4, sin4, attention_mask,
                            cache_pos, past_key, past_value):
    """Execute a single transformer layer: attention + MLP with residual connections."""
    residual = hidden_states
    hidden_states = _rms_norm_layer(layer.input_layernorm, hidden_states)
    attn_out, pk, pv = _text_attn_forward(
        layer.self_attn, hidden_states, cos4, sin4,
        attention_mask, cache_pos, past_key, past_value,
    )
    hidden_states = residual + attn_out
    residual = hidden_states
    hidden_states = _rms_norm_layer(layer.post_attention_layernorm, hidden_states)
    hidden_states = residual + _mlp_forward(layer.mlp, hidden_states)
    return hidden_states, pk, pv


class Qwen2LlmPrefill(torch.nn.Module):
    """Qwen2-7B LLM Prefill wrapper for processing the full input prompt."""

    def __init__(self, model, lm_head):
        """Initialize prefill wrapper with the base model and lm_head."""
        super().__init__()
        self.model = model.model
        self.lm_head = lm_head

    def forward(self, input_ids, attention_mask, position_ids):
        """Run prefill: embed tokens, process all layers, output logits and KV cache."""
        inputs_embeds = self.model.embed_tokens(input_ids)
        cos, sin = self.model.rotary_emb(inputs_embeds, position_ids)
        cos4 = cos.unsqueeze(2) if cos.dim() == 3 else cos
        sin4 = sin.unsqueeze(2) if sin.dim() == 3 else sin
        hidden_states = inputs_embeds
        present_k, present_v = [], []

        for layer in self.model.layers:
            hidden_states, pk, pv = _run_transformer_layer(
                layer, hidden_states, cos4, sin4, attention_mask, None, None, None)
            pk = _pad_kv_cache(pk)
            pv = _pad_kv_cache(pv)
            present_k.append(pk)
            present_v.append(pv)

        hidden_states = _rms_norm_layer(self.model.norm, hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, torch.stack(present_k, dim=0), torch.stack(present_v, dim=0)


def _pad_kv_cache(cache_tensor):
    """Pad KV cache tensor to KV_CACHE_LEN along the sequence dimension."""
    pad_len = KV_CACHE_LEN - cache_tensor.shape[2]
    padding = cache_tensor.new_zeros(
        cache_tensor.shape[0], cache_tensor.shape[1], pad_len, cache_tensor.shape[3])
    return torch.cat([cache_tensor, padding], dim=2)[:, :, :KV_CACHE_LEN, :]


class Qwen2LlmDecode(torch.nn.Module):
    """Qwen2-7B LLM Decode wrapper for single-token autoregressive generation."""

    def __init__(self, model, lm_head):
        """Initialize decode wrapper with the base model and lm_head."""
        super().__init__()
        self.model = model.model
        self.lm_head = lm_head

    def forward(self, input_ids, attention_mask, position_ids, past_key_cache,
                past_value_cache):
        """Run decode: embed token, process all layers with cached KV, output logits."""
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
        logits = self.lm_head(hidden_states)
        return logits, torch.stack(present_k, dim=0), torch.stack(present_v, dim=0)


def _prepare_llm_modules(model, device: str):
    """Set model to eval mode, move to device, and create prefill/decode wrappers."""
    lm_head = model.lm_head
    model.eval()
    lm_head.eval()
    model.to(device)
    lm_head.to(device)
    prefill = Qwen2LlmPrefill(model, lm_head).to(device).eval()
    decode = Qwen2LlmDecode(model, lm_head).to(device).eval()
    return prefill, decode, lm_head


def _get_kv_cache_config(model):
    """Extract KV cache shape config: num_layers, num_kv_heads, head_dim."""
    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    head_dim = getattr(
        model.config, "head_dim",
        model.config.hidden_size // model.config.num_attention_heads,
    )
    return num_layers, num_kv_heads, head_dim


def _prepare_output_paths(output_dir):
    """Create prefill/ and decode/ subdirectories and return ONNX output paths."""
    prefill_dir = Path(output_dir) / "prefill"
    decode_dir = Path(output_dir) / "decode"
    prefill_dir.mkdir(parents=True, exist_ok=True)
    decode_dir.mkdir(parents=True, exist_ok=True)
    return prefill_dir / "qwen2_7b_llm_prefill.onnx", decode_dir / "qwen2_7b_llm_decode.onnx"


def _create_prefill_dummy_inputs(device: str, dummy_seq_len: int):
    """Create random dummy inputs for prefill ONNX export."""
    dummy_seq = int(dummy_seq_len)
    dummy_input_ids = torch.randint(0, 1000, (1, dummy_seq), dtype=torch.int64, device=device)
    dummy_attention_mask = torch.ones(1, dummy_seq, dtype=torch.int64, device=device)
    dummy_position_ids = torch.arange(dummy_seq, device=device, dtype=torch.int64).view(1, -1)
    return dummy_seq, dummy_input_ids, dummy_attention_mask, dummy_position_ids


def _export_prefill_onnx(prefill, prefill_path: Path, dummy_inputs, use_dynamo: bool):
    """Export LLM prefill model to ONNX with dynamic sequence length axes."""
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
    """Create dummy inputs for decode ONNX export (single token + KV cache)."""
    past_len = int(KV_CACHE_LEN)
    dummy_input_ids = torch.randint(0, 1000, (1, 1), dtype=torch.int64, device=device)
    dummy_attention_mask = torch.ones(1, past_len, dtype=torch.int64, device=device)
    dummy_position_ids = torch.tensor([[past_len - 1]], dtype=torch.int64, device=device)
    dummy_k = torch.zeros(num_layers, 1, num_kv_heads, past_len, head_dim, dtype=kv_dtype, device=device)
    dummy_v = torch.zeros(num_layers, 1, num_kv_heads, past_len, head_dim, dtype=kv_dtype, device=device)
    return dummy_input_ids, dummy_attention_mask, dummy_position_ids, dummy_k, dummy_v


def _export_decode_onnx(decode, decode_path: Path, dummy_inputs, use_dynamo: bool):
    """Export LLM decode model to ONNX with fixed input shapes."""
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


def export_llm_prefill_decode(model, output_dir, device="cpu", dummy_seq_len=8, use_dynamo=False):
    """Export Qwen2-7B model to ONNX as separate prefill and decode subgraphs."""
    prefill, decode, _ = _prepare_llm_modules(model, device=device)
    kv_dtype = next(model.parameters()).dtype
    num_layers, num_kv_heads, head_dim = _get_kv_cache_config(model)
    prefill_path, decode_path = _prepare_output_paths(output_dir)

    _, dummy_input_ids, dummy_attention_mask, dummy_position_ids = (
        _create_prefill_dummy_inputs(device=device, dummy_seq_len=dummy_seq_len))
    _export_prefill_onnx(prefill, prefill_path,
                          (dummy_input_ids, dummy_attention_mask, dummy_position_ids), use_dynamo)

    decode_inputs = _create_decode_dummy_inputs(device, num_layers, num_kv_heads, head_dim, kv_dtype)
    _export_decode_onnx(decode, decode_path, decode_inputs, use_dynamo)


def _resolve_dtype(dtype_str: str):
    """Resolve string dtype name to torch dtype."""
    mapping = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return mapping[dtype_str]


def _load_model(model_id: str, dtype_str: str, device: str):
    """Load the HuggingFace model with specified dtype and device."""
    torch_dtype = _resolve_dtype(dtype_str)
    device_obj = torch.device(device)
    return AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch_dtype,
        low_cpu_mem_usage=False, attn_implementation="eager",
    ).to(device_obj)


def main():
    """Main entry point: parse args, load model, export ONNX, and cleanup."""
    parser = argparse.ArgumentParser(description="Export Qwen2-7B to ONNX")
    parser.add_argument("--model-id", type=str, default="./Qwen2-7B-Instruct/Qwen2-7B-Instruct",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--output-dir", type=str, default="./qwen2_7b_onnx", help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device for export (cpu or cuda)")
    parser.add_argument("--dummy-seq-len", type=int, default=8, help="Dummy sequence length for export")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"],
                        help="Export dtype")
    parser.add_argument("--use-dynamo", action="store_true", help="Use torch dynamo exporter path")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading model {args.model_id} for export (dtype={args.dtype})...")
    model = _load_model(args.model_id, args.dtype, args.device)

    export_llm_prefill_decode(model, output_dir, args.device, args.dummy_seq_len, args.use_dynamo)

    print("Clearing memory after export...")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"\nExport finished. Files saved in {args.output_dir}")


if __name__ == "__main__":
    main()
