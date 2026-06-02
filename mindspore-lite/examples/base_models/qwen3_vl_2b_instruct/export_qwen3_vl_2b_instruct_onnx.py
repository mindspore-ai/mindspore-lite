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
Export Qwen3-VL-2B-Instruct to ONNX.
"""

import sys
import argparse
import gc
from pathlib import Path
import torch

try:
    import torch._dynamo

    torch._dynamo.disable()
except:
    pass

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    print("Error: transformers package not found or version too low.")
    print(
        "Please install the latest version: pip install git+https://github.com/huggingface/transformers"
    )
    sys.exit(1)


try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb
except Exception:
    apply_rotary_pos_emb = None


_USE_CUSTOM_OP = True
_CUSTOM_ATTN_LAYOUT = "BNSD"


class _CustomRmsNorm(torch.autograd.Function):
    """Custom RMSNorm operator for ONNX export."""

    @staticmethod
    def forward(ctx, x, gamma, epsilon_f: float):
        """Compute RMSNorm with custom ONNX symbolic."""
        del ctx
        x_float = x.float()
        var = x_float.pow(2).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + float(epsilon_f))
        y = (x_float * rstd).to(dtype=x.dtype) * gamma.to(dtype=x.dtype)
        return y, rstd

    @staticmethod
    def symbolic(g, x, gamma, epsilon_f: float):
        return g.op(
            "Custom",
            x,
            gamma,
            outputs=2,
            type_s="RmsNorm",
            epsilon_f=float(epsilon_f),
            input_names_s=["x", "gamma"],
            output_names_s=["y", "rstd"],
        )


class _CustomSwiGlu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim_i: int):
        del ctx
        a, b = torch.chunk(x, 2, dim=int(dim_i))
        return torch.nn.functional.silu(a) * b

    @staticmethod
    def symbolic(g, x, dim_i: int):
        return g.op(
            "Custom",
            x,
            type_s="SwiGlu",
            dim_i=int(dim_i),
            input_names_s=["x"],
            output_names_s=["y"],
        )


def _rope_to_b1sd(t):
    if t is None:
        return None
    if t.ndim == 2:
        return t.unsqueeze(0).unsqueeze(0)
    if t.ndim == 3:
        return t.unsqueeze(1)
    return t


def _rotate_half(x):
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


class _CustomRotaryMul(torch.autograd.Function):
    """Custom rotary multiplication operator for ONNX export."""

    @staticmethod
    def forward(ctx, x, r1, r2):
        del ctx
        r1 = r1.to(dtype=x.dtype)
        r2 = r2.to(dtype=x.dtype)
        return x * r1 + _rotate_half(x) * r2

    @staticmethod
    def symbolic(g, x, r1, r2):
        return g.op(
            "Custom",
            x,
            r1,
            r2,
            type_s="RotaryMul",
            input_names_s=["x", "r1", "r2"],
            output_names_s=["y"],
        )


class _CustomPromptFlashAttention(torch.autograd.Function):
    """Custom prompt flash attention operator for ONNX export."""

    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        atten_mask,
        num_heads_i: int,
        num_key_value_heads_i: int,
        scale_value_f: float,
        input_layout_s: str,
    ):
        """Fallback forward used during tracing and shape inference."""
        del ctx, input_layout_s
        if key.dtype != query.dtype:
            key = key.to(dtype=query.dtype)
        if value.dtype != query.dtype:
            value = value.to(dtype=query.dtype)
        if num_key_value_heads_i != num_heads_i:
            if num_heads_i % num_key_value_heads_i != 0:
                raise RuntimeError(
                    f"num_heads({num_heads_i}) not divisible by num_kv_heads({num_key_value_heads_i})"
                )
            repeat = num_heads_i // num_key_value_heads_i
            b, kv, s, d = key.shape
            key = key[:, :, None, :, :].expand(b, kv, repeat, s, d).reshape(
                b, kv * repeat, s, d
            )
            b, kv, s, d = value.shape
            value = value[:, :, None, :, :].expand(b, kv, repeat, s, d).reshape(
                b, kv * repeat, s, d
            )
        scale = float(scale_value_f)
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        if atten_mask is not None:
            scores = scores.masked_fill(atten_mask, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        return torch.matmul(probs, value)

    @staticmethod
    def symbolic(
        g,
        query,
        key,
        value,
        atten_mask,
        num_heads_i: int,
        num_key_value_heads_i: int,
        scale_value_f: float,
        input_layout_s: str,
    ):
        """Export a Custom node for prompt flash attention."""
        return g.op(
            "Custom",
            query,
            key,
            value,
            atten_mask,
            type_s="PromptFlashAttention",
            num_heads_i=int(num_heads_i),
            num_key_value_heads_i=int(num_key_value_heads_i),
            scale_value_f=float(scale_value_f),
            input_layout_s=str(input_layout_s),
            input_names_s=["query", "key", "value", "atten_mask"],
            output_names_s=["attention_out"],
        )


class _CustomIncreFlashAttention(torch.autograd.Function):
    """Custom incremental flash attention operator for ONNX export (decode step)."""

    @staticmethod
    def forward(
        unused_ctx,
        query,
        key,
        value,
        atten_mask,
        num_heads_i: int,
        num_key_value_heads_i: int,
        scale_value_f: float,
        unused_input_layout_s: str,
    ):
        """Fallback forward used during tracing and shape inference."""
        del unused_ctx, unused_input_layout_s
        if key.dtype != query.dtype:
            key = key.to(dtype=query.dtype)
        if value.dtype != query.dtype:
            value = value.to(dtype=query.dtype)
        if num_key_value_heads_i != num_heads_i:
            if num_heads_i % num_key_value_heads_i != 0:
                raise RuntimeError(
                    f"num_heads({num_heads_i}) not divisible by num_kv_heads({num_key_value_heads_i})"
                )
            repeat = num_heads_i // num_key_value_heads_i
            b, kv, s, d = key.shape
            key = key[:, :, None, :, :].expand(b, kv, repeat, s, d).reshape(
                b, kv * repeat, s, d
            )
            b, kv, s, d = value.shape
            value = value[:, :, None, :, :].expand(b, kv, repeat, s, d).reshape(
                b, kv * repeat, s, d
            )
        scale = float(scale_value_f)
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        if atten_mask is not None:
            scores = scores.masked_fill(atten_mask, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        return torch.matmul(probs, value)

    @staticmethod
    def symbolic(
        g,
        query,
        key,
        value,
        atten_mask,
        num_heads_i: int,
        num_key_value_heads_i: int,
        scale_value_f: float,
        input_layout_s: str,
    ):
        """Export a Custom node for incremental flash attention."""
        return g.op(
            "Custom",
            query,
            key,
            value,
            atten_mask,
            type_s="IncreFlashAttention",
            num_heads_i=int(num_heads_i),
            num_key_value_heads_i=int(num_key_value_heads_i),
            scale_value_f=float(scale_value_f),
            input_layout_s=str(input_layout_s),
            input_names_s=["query", "key", "value", "atten_mask"],
            output_names_s=["attention_out"],
        )


class _CustomScatter(torch.autograd.Function):
    """Custom scatter update operator for KV cache update."""

    @staticmethod
    def forward(unused_ctx, var, indices, updates, axis_i: int):
        """Update a single cache position along the given scatter axis."""
        del unused_ctx
        bsz = var.shape[0]
        axis_i = int(axis_i)
        idx = indices.view(bsz, 1, 1, 1).to(dtype=torch.int64)
        idx = idx.expand(bsz, var.shape[1], 1, var.shape[3])
        return var.scatter(axis_i, idx, updates.to(dtype=var.dtype))

    @staticmethod
    def symbolic(g, var, indices, updates, axis_i: int):
        return g.op(
            "Custom",
            var,
            indices,
            updates,
            type_s="Scatter",
            reduce_s="update",
            axis_i=int(axis_i),
            input_names_s=["var", "indices", "updates"],
            output_names_s=["var"],
        )


def _rms_norm(norm_mod, x):
    if not _USE_CUSTOM_OP:
        return norm_mod(x)
    epsilon = getattr(norm_mod, "variance_epsilon", getattr(norm_mod, "eps", 1e-6))
    y, _ = _CustomRmsNorm.apply(x, norm_mod.weight, float(epsilon))
    return y


def _mlp_forward(layer, x):
    """Run fused MLP forward with optional SwiGLU activation."""
    mlp = layer.mlp
    if not _USE_CUSTOM_OP:
        return mlp(x)
    gate_proj = getattr(mlp, "gate_proj", None)
    up_proj = getattr(mlp, "up_proj", None)
    down_proj = getattr(mlp, "down_proj", None)
    if gate_proj is None or up_proj is None or down_proj is None:
        return mlp(x)
    fused = _fused_gate_up(gate_proj, up_proj, x)
    if fused is None:
        gate = gate_proj(x)
        up = up_proj(x)
    else:
        gate, up = fused
    fused_in = torch.cat([gate, up], dim=-1)
    act = _CustomSwiGlu.apply(fused_in, -1)
    return down_proj(act)


def _fused_gate_up(gate_proj, up_proj, x):
    """Fuse gate and up projections into a single linear operation."""
    w_gate = getattr(gate_proj, "weight", None)
    w_up = getattr(up_proj, "weight", None)
    if w_gate is None or w_up is None:
        return None
    if w_gate.ndim != 2 or w_up.ndim != 2:
        return None
    if w_gate.shape[1] != w_up.shape[1]:
        return None
    weight = torch.cat([w_gate, w_up], dim=0)
    b_gate = getattr(gate_proj, "bias", None)
    b_up = getattr(up_proj, "bias", None)
    if b_gate is None and b_up is None:
        bias = None
    else:
        if b_gate is None:
            b_gate = torch.zeros((w_gate.shape[0],), dtype=w_gate.dtype, device=w_gate.device)
        if b_up is None:
            b_up = torch.zeros((w_up.shape[0],), dtype=w_up.dtype, device=w_up.device)
        bias = torch.cat([b_gate, b_up], dim=0)
    out = torch.nn.functional.linear(x, weight, bias)
    gate, up = torch.split(out, [w_gate.shape[0], w_up.shape[0]], dim=-1)
    return gate, up


def _fused_qkv(attn_mod, hidden_states):
    """Fuse Q/K/V projections into a single linear operation."""
    q_proj = getattr(attn_mod, "q_proj", None)
    k_proj = getattr(attn_mod, "k_proj", None)
    v_proj = getattr(attn_mod, "v_proj", None)
    if q_proj is None or k_proj is None or v_proj is None:
        return None
    wq = getattr(q_proj, "weight", None)
    wk = getattr(k_proj, "weight", None)
    wv = getattr(v_proj, "weight", None)
    if wq is None or wk is None or wv is None:
        return None
    if wq.ndim != 2 or wk.ndim != 2 or wv.ndim != 2:
        return None
    if wq.shape[1] != wk.shape[1] or wq.shape[1] != wv.shape[1]:
        return None
    weight = torch.cat([wq, wk, wv], dim=0)
    bq = getattr(q_proj, "bias", None)
    bk = getattr(k_proj, "bias", None)
    bv = getattr(v_proj, "bias", None)
    if bq is None and bk is None and bv is None:
        bias = None
    else:
        if bq is None:
            bq = torch.zeros((wq.shape[0],), dtype=wq.dtype, device=wq.device)
        if bk is None:
            bk = torch.zeros((wk.shape[0],), dtype=wk.dtype, device=wk.device)
        if bv is None:
            bv = torch.zeros((wv.shape[0],), dtype=wv.dtype, device=wv.device)
        bias = torch.cat([bq, bk, bv], dim=0)
    qkv = torch.nn.functional.linear(hidden_states, weight, bias)
    q, k, v = torch.split(qkv, [wq.shape[0], wk.shape[0], wv.shape[0]], dim=-1)
    return q, k, v


class VisionTowerWrapper(torch.nn.Module):
    """
    Wrapper for Qwen3-VL Vision Tower to cache position embeddings.
    """

    def __init__(self, vision_tower, dummy_grid_thw):
        """
        Initialize Qwen3-VL Vision Tower wrapper.
        """
        super().__init__()
        self.vision_tower = vision_tower
        self.dummy_grid_thw = dummy_grid_thw
        with torch.no_grad():
            cached_pos_embeds = vision_tower.fast_pos_embed_interpolate(dummy_grid_thw)
            cached_rot_pos_emb = vision_tower.rot_pos_emb(dummy_grid_thw)
        vision_tower.fast_pos_embed_interpolate = lambda x: cached_pos_embeds
        vision_tower.rot_pos_emb = lambda x: cached_rot_pos_emb

    def forward(self, pixel_values):
        """
        Forward pass for Qwen3-VL Vision Tower wrapper.
        """
        outputs = self.vision_tower(
            pixel_values, grid_thw=self.dummy_grid_thw, return_dict=True
        )
        image_embeds = outputs.pooler_output
        deepstack = outputs.deepstack_features
        if isinstance(deepstack, (list, tuple)):
            if len(deepstack) == 0:
                deepstack = image_embeds.new_zeros(
                    (0, image_embeds.shape[0], image_embeds.shape[1])
                )
            else:
                deepstack = torch.stack(deepstack, dim=0)
        return image_embeds, deepstack


def export_vision_tower(model, output_path, device="cpu", vision_image_size=128):
    """
    Export Qwen3-VL Vision Tower to ONNX.
    """
    print(f"Exporting Vision Tower to {output_path}...")

    # Qwen3-VL uses a visual encoder (usually model.model.visual)
    vision_tower = model.model.visual
    vision_tower.eval()
    vision_tower.to(device)

    patch_size = model.config.vision_config.patch_size
    grid_h = int(vision_image_size) // int(patch_size)
    grid_w = int(vision_image_size) // int(patch_size)
    dummy_grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.int64).to(device)
    dummy_seq_len = int(
        dummy_grid_thw[0, 0].item()
        * dummy_grid_thw[0, 1].item()
        * dummy_grid_thw[0, 2].item()
    )
    dummy_pixel_values = torch.randn(
        dummy_seq_len, 1536, device=device, dtype=torch.float16
    )

    wrapper = VisionTowerWrapper(vision_tower, dummy_grid_thw)

    try:
        print("Exporting Vision Tower with legacy exporter...")

        # Instead of patching tolist, we use the legacy exporter's internal function
        # which bypasses the new capture logic and handles the vision tower's
        # complex return types and symbolic shapes more gracefully.
        from torch.onnx import utils as onnx_utils

        vision_tower.eval()
        wrapper.eval()

        with torch.no_grad():
            onnx_utils.export(
                wrapper,
                (dummy_pixel_values,),
                output_path,
                input_names=["pixel_values"],
                output_names=["image_embeds", "deepstack_embeds"],
                opset_version=14,
                do_constant_folding=True,
            )
        print("Vision Tower exported successfully.")
    except (RuntimeError, ValueError, TypeError, OSError) as e:
        print(f"Failed to export Vision Tower with trace, trying direct export: {e}")
        try:
            print(f"Exporting Vision Tower to {output_path}...")
            vision_tower.eval()
            wrapper.eval()
            with torch.no_grad():
                torch.onnx.export(
                    wrapper,
                    (dummy_pixel_values,),
                    output_path,
                    input_names=["pixel_values"],
                    output_names=["image_embeds", "deepstack_embeds"],
                    opset_version=18,
                    do_constant_folding=True,
                )
            print("Vision Tower exported successfully.")
        except (RuntimeError, ValueError, TypeError, OSError) as e2:
            print(f"Failed to export Vision Tower: {e2}")
            import traceback

            traceback.print_exc()


class LLMWrapper(torch.nn.Module):
    """
    Wrapper for Qwen3-VL-2B-Instruct LLM to cache position embeddings.
    """

    def __init__(self, llm):
        """
        Initialize Qwen3-VL-2B-Instruct LLM wrapper.
        """
        super().__init__()
        self.llm = llm

    def forward(self, input_ids, attention_mask, position_ids):
        """
        Forward pass for Qwen3-VL-2B-Instruct LLM decode.
        """
        # Explicitly pass arguments to the LLM's forward method
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=False,
        )
        # return_dict=False returns a tuple
        if isinstance(outputs, (list, tuple)):
            return outputs[0]
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        return outputs


def _make_additive_causal_mask(attention_mask, q_len, k_len, past_len, dtype):
    """
    Create additive causal mask for Qwen3-VL-2B-Instruct text model.
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


def _text_position_ids(position_ids, batch, seq_len, device):
    """
    Process position ids for Qwen3-VL-2B-Instruct text model.
    """
    if position_ids is None:
        base = torch.arange(seq_len, device=device).view(1, -1).expand(batch, -1)
        position_ids = base
    if position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)
    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        mm_position_ids = position_ids[1:]
        return text_position_ids, mm_position_ids
    return position_ids, None


def _text_attn_forward(
    attn_mod, hidden_states, cos, sin, attention_mask, past_key, past_value
):
    """
    Forward pass for Qwen3-VL-2B-Instruct text attention.
    """
    input_shape = hidden_states.shape[:-1]
    head_dim = attn_mod.head_dim
    hidden_shape = (*input_shape, -1, head_dim)
    qkv = _fused_qkv(attn_mod, hidden_states)
    if qkv is None:
        q = attn_mod.q_proj(hidden_states)
        k = attn_mod.k_proj(hidden_states)
        v = attn_mod.v_proj(hidden_states)
    else:
        q, k, v = qkv
    query_states = _rms_norm(attn_mod.q_norm, q.view(hidden_shape)).transpose(1, 2)
    key_states = _rms_norm(attn_mod.k_norm, k.view(hidden_shape)).transpose(1, 2)
    value_states = v.view(hidden_shape).transpose(1, 2)
    if _USE_CUSTOM_OP:
        query_states = _CustomRotaryMul.apply(query_states, cos, sin)
        key_states = _CustomRotaryMul.apply(key_states, cos, sin)
    else:
        if apply_rotary_pos_emb is None:
            raise RuntimeError("apply_rotary_pos_emb not available")
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    if past_key is not None:
        key_states = torch.cat([past_key, key_states], dim=2)
        value_states = torch.cat([past_value, value_states], dim=2)
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3_vl.modeling_qwen3_vl import eager_attention_forward

    interface = getattr(attn_mod.config, "_attn_implementation", "eager")
    attention_fn = ALL_ATTENTION_FUNCTIONS.get_interface(
        interface, eager_attention_forward
    )
    if _USE_CUSTOM_OP:
        atten_mask = attention_mask < 0
        num_heads = getattr(attn_mod, "num_heads", None)
        if num_heads is None:
            num_heads = getattr(attn_mod, "num_attention_heads", None)
        if num_heads is None:
            num_heads = getattr(attn_mod.config, "num_attention_heads", None)
        if num_heads is None:
            num_heads = getattr(attn_mod.config, "num_heads", None)
        num_kv_heads = getattr(attn_mod, "num_key_value_heads", None)
        if num_kv_heads is None:
            num_kv_heads = getattr(attn_mod.config, "num_key_value_heads", None)
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if past_key is None:
            attn_output, _ = attention_fn(
                attn_mod,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0,
                scaling=attn_mod.scaling,
                is_causal=True,
            )
        else:
            if int(num_kv_heads) != int(num_heads):
                repeat = int(num_heads) // int(num_kv_heads)
                b, kv, s, d = key_states.shape
                key_states = key_states[:, :, None, :, :].expand(
                    b, kv, repeat, s, d
                ).reshape(b, kv * repeat, s, d)
                b, kv, s, d = value_states.shape
                value_states = value_states[:, :, None, :, :].expand(
                    b, kv, repeat, s, d
                ).reshape(b, kv * repeat, s, d)
            attn_output = _CustomIncreFlashAttention.apply(
                query_states,
                key_states,
                value_states,
                atten_mask,
                int(num_heads),
                int(num_heads),
                float(attn_mod.scaling),
                _CUSTOM_ATTN_LAYOUT,
            )
    else:
        attn_output, _ = attention_fn(
            attn_mod,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=attn_mod.scaling,
            is_causal=True,
        )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attn_mod.o_proj(attn_output)
    return attn_output, key_states, value_states


def _make_decode_additive_mask_fixed(attention_mask, cache_pos, max_seq_len, dtype):
    """Create fixed-size causal mask for single-step decode."""
    mask_value = torch.finfo(dtype).min
    bsz = attention_mask.shape[0]
    cache_pos = cache_pos.view(bsz, 1).to(dtype=torch.int64)
    ar_k = torch.arange(max_seq_len, device=attention_mask.device).view(1, -1)
    causal = ar_k > cache_pos
    causal = causal.to(dtype) * mask_value
    causal = causal[:, None, None, :].expand(bsz, 1, 1, max_seq_len)
    padding = (1.0 - attention_mask.to(dtype)) * mask_value
    padding = padding[:, None, None, :]
    return causal + padding


def _scatter_cache_update(cache, update, cache_pos):
    update = update.to(cache.dtype)
    if _USE_CUSTOM_OP:
        return _CustomScatter.apply(cache, cache_pos.to(dtype=torch.int64), update, 2)
    bsz = cache.shape[0]
    idx = cache_pos.view(bsz, 1, 1, 1).to(dtype=torch.int64)
    idx = idx.expand(bsz, cache.shape[1], 1, cache.shape[3])
    return cache.scatter(2, idx, update)


def _text_attn_forward_decode_fixed(
    attn_mod, hidden_states, cos, sin, attention_mask, cache_pos, key_cache, value_cache
):
    """Decode-step attention with fixed-length KV cache using scatter update."""
    input_shape = hidden_states.shape[:-1]
    head_dim = attn_mod.head_dim
    hidden_shape = (*input_shape, -1, head_dim)
    qkv = _fused_qkv(attn_mod, hidden_states)
    if qkv is None:
        q = attn_mod.q_proj(hidden_states)
        k = attn_mod.k_proj(hidden_states)
        v = attn_mod.v_proj(hidden_states)
    else:
        q, k, v = qkv
    query_states = _rms_norm(attn_mod.q_norm, q.view(hidden_shape)).transpose(1, 2)
    key_states = _rms_norm(attn_mod.k_norm, k.view(hidden_shape)).transpose(1, 2)
    value_states = v.view(hidden_shape).transpose(1, 2)
    if _USE_CUSTOM_OP:
        query_states = _CustomRotaryMul.apply(query_states, cos, sin)
        key_states = _CustomRotaryMul.apply(key_states, cos, sin)
    else:
        if apply_rotary_pos_emb is None:
            raise RuntimeError("apply_rotary_pos_emb not available")
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    cache_dtype = key_cache.dtype
    if query_states.dtype != cache_dtype:
        query_states = query_states.to(dtype=cache_dtype)
    if key_states.dtype != cache_dtype:
        key_states = key_states.to(dtype=cache_dtype)
    if value_states.dtype != cache_dtype:
        value_states = value_states.to(dtype=cache_dtype)

    key_cache = _scatter_cache_update(key_cache, key_states, cache_pos)
    value_cache = _scatter_cache_update(value_cache, value_states, cache_pos)

    key_for_attn = key_cache
    value_for_attn = value_cache

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3_vl.modeling_qwen3_vl import eager_attention_forward

    interface = getattr(attn_mod.config, "_attn_implementation", "eager")
    attention_fn = ALL_ATTENTION_FUNCTIONS.get_interface(interface, eager_attention_forward)

    if _USE_CUSTOM_OP:
        num_heads = getattr(attn_mod, "num_heads", None)
        if num_heads is None:
            num_heads = getattr(attn_mod, "num_attention_heads", None)
        if num_heads is None:
            num_heads = getattr(attn_mod.config, "num_attention_heads", None)
        if num_heads is None:
            num_heads = getattr(attn_mod.config, "num_heads", None)
        num_kv_heads = getattr(attn_mod, "num_key_value_heads", None)
        if num_kv_heads is None:
            num_kv_heads = getattr(attn_mod.config, "num_key_value_heads", None)
        if num_kv_heads is None:
            num_kv_heads = num_heads

        if int(num_kv_heads) != int(num_heads):
            if int(num_heads) % int(num_kv_heads) != 0:
                raise RuntimeError(
                    f"num_heads({int(num_heads)}) not divisible by num_kv_heads({int(num_kv_heads)})"
                )
            repeat = int(num_heads) // int(num_kv_heads)
            b, kv, s, d = key_for_attn.shape
            key_for_attn = key_for_attn[:, :, None, :, :].expand(
                b, kv, repeat, s, d
            ).reshape(b, kv * repeat, s, d)
            b, kv, s, d = value_for_attn.shape
            value_for_attn = value_for_attn[:, :, None, :, :].expand(
                b, kv, repeat, s, d
            ).reshape(b, kv * repeat, s, d)
        atten_mask = attention_mask < 0
        attn_output = _CustomIncreFlashAttention.apply(
            query_states,
            key_for_attn,
            value_for_attn,
            atten_mask,
            int(num_heads),
            int(num_heads),
            float(attn_mod.scaling),
            _CUSTOM_ATTN_LAYOUT,
        )
    else:
        attn_output, _ = attention_fn(
            attn_mod,
            query_states,
            key_cache,
            value_cache,
            attention_mask,
            dropout=0.0,
            scaling=attn_mod.scaling,
            is_causal=True,
        )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attn_mod.o_proj(attn_output)
    return attn_output, key_cache, value_cache


class Qwen3VLLlmPrefill(torch.nn.Module):
    """
    Qwen3-VL-2B-Instruct LLM prefill model.
    """

    def __init__(self, text_model, lm_head, image_token_id: int, num_deepstack: int):
        """
        Initialize Qwen3-VL-2B-Instruct LLM prefill model.
        """
        super().__init__()
        self.text_model = text_model
        self.lm_head = lm_head
        self.num_hidden_layers = text_model.config.num_hidden_layers
        self.image_token_id = int(image_token_id)
        self.num_deepstack = int(num_deepstack)

    def forward(
        self, input_ids, attention_mask, position_ids, image_embeds, deepstack_embeds
    ):
        """
        Forward pass for Qwen3-VL-2B-Instruct LLM prefill.
        """
        inputs_embeds = self.text_model.embed_tokens(input_ids)
        image_mask = input_ids == self.image_token_id
        image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(
            image_mask,
            image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
        )
        deepstack_dense = []
        for i in range(self.num_deepstack):
            dense = inputs_embeds.new_zeros(inputs_embeds.shape)
            dense = dense.masked_scatter(
                image_mask,
                deepstack_embeds[i].to(
                    device=inputs_embeds.device, dtype=inputs_embeds.dtype
                ),
            )
            deepstack_dense.append(dense)
        bsz, q_len = input_ids.shape
        text_pos, mm_pos = _text_position_ids(
            position_ids, bsz, q_len, inputs_embeds.device
        )
        if mm_pos is None:
            mm_pos = text_pos[None, ...].expand(3, bsz, q_len)
        cos, sin = self.text_model.rotary_emb(inputs_embeds, mm_pos)
        if _USE_CUSTOM_OP:
            cos = _rope_to_b1sd(cos)
            sin = _rope_to_b1sd(sin)
        k_len = q_len
        attn_mask = _make_additive_causal_mask(
            attention_mask, q_len, k_len, 0, inputs_embeds.dtype
        )
        hidden_states = inputs_embeds
        present = []
        for layer_idx, layer in enumerate(self.text_model.layers):
            residual = hidden_states
            hidden_states = _rms_norm(layer.input_layernorm, hidden_states)
            attn_out, pk, pv = _text_attn_forward(
                layer.self_attn,
                hidden_states,
                cos,
                sin,
                attn_mask,
                None,
                None,
            )
            hidden_states = residual + attn_out
            residual = hidden_states
            hidden_states = _rms_norm(layer.post_attention_layernorm, hidden_states)
            hidden_states = residual + _mlp_forward(layer, hidden_states)
            if layer_idx < self.num_deepstack:
                hidden_states = hidden_states + deepstack_dense[layer_idx]
            present.append(pk)
            present.append(pv)
        hidden_states = _rms_norm(self.text_model.norm, hidden_states)
        logits = self.lm_head(hidden_states)
        present_kv = torch.stack(present, dim=0)
        return logits, present_kv


class Qwen3VLLlmDecode(torch.nn.Module):
    """
    Qwen3-VL-2B-Instruct LLM decode model.
    """

    def __init__(self, text_model, lm_head, max_seq_len: int = 512):
        """
        Qwen3-VL-2B-Instruct LLM decode model.
        """
        super().__init__()
        self.text_model = text_model
        self.lm_head = lm_head
        self.num_hidden_layers = text_model.config.num_hidden_layers
        self.max_seq_len = int(max_seq_len)

    def forward(self, input_ids, attention_mask, position_ids, past_key_values, cache_pos):
        """
        Forward pass for Qwen3-VL-2B-Instruct LLM decode.
        """
        inputs_embeds = self.text_model.embed_tokens(input_ids)
        bsz, q_len = input_ids.shape
        text_pos, mm_pos = _text_position_ids(
            position_ids, bsz, q_len, inputs_embeds.device
        )
        if mm_pos is None:
            mm_pos = text_pos[None, ...].expand(3, bsz, q_len)
        cos, sin = self.text_model.rotary_emb(inputs_embeds, mm_pos)
        if _USE_CUSTOM_OP:
            cos = _rope_to_b1sd(cos)
            sin = _rope_to_b1sd(sin)
        cache_pos = cache_pos.view(bsz).to(dtype=torch.int64)
        attn_mask = _make_decode_additive_mask_fixed(
            attention_mask, cache_pos, self.max_seq_len, inputs_embeds.dtype
        )
        hidden_states = inputs_embeds
        present = []
        for i, layer in enumerate(self.text_model.layers):
            pk_cache = past_key_values[2 * i]
            pv_cache = past_key_values[2 * i + 1]
            residual = hidden_states
            hidden_states = _rms_norm(layer.input_layernorm, hidden_states)
            attn_out, pk_cache, pv_cache = _text_attn_forward_decode_fixed(
                layer.self_attn,
                hidden_states,
                cos,
                sin,
                attn_mask,
                cache_pos,
                pk_cache,
                pv_cache,
            )
            hidden_states = residual + attn_out
            residual = hidden_states
            hidden_states = _rms_norm(layer.post_attention_layernorm, hidden_states)
            hidden_states = residual + _mlp_forward(layer, hidden_states)
            present.append(pk_cache)
            present.append(pv_cache)
        hidden_states = _rms_norm(self.text_model.norm, hidden_states)
        logits = self.lm_head(hidden_states)
        present_kv = torch.stack(present, dim=0)
        return logits, present_kv


def _get_llm_export_meta(model):
    """
    Get metadata for Qwen3-VL-2B-Instruct LLM export.
    """
    text_model = model.model.language_model
    lm_head = model.lm_head
    num_layers = text_model.config.num_hidden_layers
    num_kv_heads = text_model.config.num_key_value_heads
    head_dim = getattr(
        text_model.config,
        "head_dim",
        text_model.config.hidden_size // text_model.config.num_attention_heads,
    )
    image_token_id = model.config.image_token_id
    num_deepstack = len(
        getattr(model.config.vision_config, "deepstack_visual_indexes", [])
    )
    return (
        text_model,
        lm_head,
        num_layers,
        num_kv_heads,
        head_dim,
        image_token_id,
        num_deepstack,
    )


def _prepare_llm_modules(model, device):
    """
    Prepare Qwen3-VL-2B-Instruct LLM modules for export.
    """
    text_model, lm_head, *_ = _get_llm_export_meta(model)
    text_model.eval()
    lm_head.eval()
    text_model.to(device)
    lm_head.to(device)
    return text_model, lm_head


def _build_llm_wrappers(
    text_model, lm_head, image_token_id, num_deepstack, device, max_seq_len: int
):
    """
    Build Qwen3-VL-2B-Instruct LLM wrappers for export.
    """
    prefill = (
        Qwen3VLLlmPrefill(
            text_model,
            lm_head,
            image_token_id=image_token_id,
            num_deepstack=num_deepstack,
        )
        .to(device)
        .eval()
    )
    decode = (
        Qwen3VLLlmDecode(text_model, lm_head, max_seq_len=int(max_seq_len))
        .to(device)
        .eval()
    )
    return prefill, decode


def _make_prefill_dummy_inputs(
    text_model, num_deepstack, device, dummy_seq=8, dummy_num_img_tokens=16
):
    """
    Create dummy inputs for Qwen3-VL-2B-Instruct LLM prefill.
    """
    dummy_input_ids = torch.randint(
        0, 1000, (1, dummy_seq), dtype=torch.int64, device=device
    )
    dummy_attention_mask = torch.ones(1, dummy_seq, dtype=torch.int64, device=device)
    base_pos = torch.arange(dummy_seq, device=device, dtype=torch.int64).view(1, -1)
    dummy_position_ids = base_pos.unsqueeze(0).expand(4, 1, dummy_seq)
    dummy_image_embeds = torch.randn(
        dummy_num_img_tokens,
        text_model.config.hidden_size,
        device=device,
        dtype=torch.float16,
    )
    dummy_deepstack = torch.randn(
        num_deepstack,
        dummy_num_img_tokens,
        text_model.config.hidden_size,
        device=device,
        dtype=torch.float16,
    )
    return (
        dummy_input_ids,
        dummy_attention_mask,
        dummy_position_ids,
        dummy_image_embeds,
        dummy_deepstack,
    )


def _make_decode_dummy_inputs(
    num_layers, num_kv_heads, head_dim, device, max_seq_len=512, dummy_step=1
):
    """
    Create dummy inputs for Qwen3-VL-2B-Instruct LLM decode.
    """
    dummy_input_ids_step = torch.randint(
        0, 1000, (1, dummy_step), dtype=torch.int64, device=device
    )
    dummy_attention_mask_step = torch.ones(
        1, max_seq_len, dtype=torch.int64, device=device
    )
    cache_pos_val = int(min(8, int(max_seq_len) - 1))
    cache_pos = torch.tensor([cache_pos_val], dtype=torch.int64, device=device)
    step_pos = cache_pos.view(1, 1)
    dummy_position_ids_step = step_pos.unsqueeze(0).expand(4, 1, dummy_step)
    dummy_past = torch.zeros(
        2 * num_layers,
        1,
        num_kv_heads,
        max_seq_len,
        head_dim,
        dtype=torch.float16,
        device=device,
    )
    return (
        dummy_input_ids_step,
        dummy_attention_mask_step,
        dummy_position_ids_step,
        dummy_past,
        cache_pos,
    )


def _export_onnx(
    prefill_or_decode, onnx_path, args, input_names, output_names, dynamic_axes
):
    """
    Export Qwen3-VL-2B-Instruct LLM prefill or decode model to ONNX.
    """
    print(f"Exporting {args} to {onnx_path}...")
    with torch.no_grad():
        torch.onnx.export(
            prefill_or_decode,
            args,
            str(onnx_path),
            input_names=input_names,
            output_names=output_names,
            dynamo=False,
            opset_version=18,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
        )


def export_llm_prefill_decode(model, output_dir, device="cpu", kv_cache_len: int = 512):
    """
    Export Qwen3-VL-2B-Instruct LLM prefill and decode models to ONNX.
    """
    (
        text_model,
        lm_head,
        num_layers,
        num_kv_heads,
        head_dim,
        image_token_id,
        num_deepstack,
    ) = _get_llm_export_meta(model)
    _prepare_llm_modules(model, device)
    prefill, decode = _build_llm_wrappers(
        text_model,
        lm_head,
        image_token_id,
        num_deepstack,
        device,
        max_seq_len=int(kv_cache_len),
    )

    prefill_path = Path(output_dir) / "qwen3_vl_llm_prefill.onnx"
    decode_path = Path(output_dir) / "qwen3_vl_llm_decode.onnx"

    dummy_inputs = _make_prefill_dummy_inputs(text_model, num_deepstack, device)

    prefill_input_names = [
        "input_ids",
        "attention_mask",
        "position_ids",
        "image_embeds",
        "deepstack_embeds",
    ]
    prefill_output_names = ["logits", "present_key_values"]
    prefill_dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq_len"},
        "attention_mask": {0: "batch", 1: "seq_len"},
        "position_ids": {1: "batch", 2: "seq_len"},
        "logits": {0: "batch", 1: "seq_len"},
        "present_key_values": {1: "batch", 3: "seq_len"},
        "image_embeds": {0: "num_image_tokens"},
        "deepstack_embeds": {1: "num_image_tokens"},
    }

    _export_onnx(
        prefill,
        prefill_path,
        dummy_inputs,
        prefill_input_names,
        prefill_output_names,
        prefill_dynamic_axes,
    )
    print("LLM prefill exported successfully.")

    (
        dummy_input_ids_step,
        dummy_attention_mask_step,
        dummy_position_ids_step,
        dummy_past,
        dummy_cache_pos,
    ) = _make_decode_dummy_inputs(
        num_layers,
        num_kv_heads,
        head_dim,
        device,
        max_seq_len=int(kv_cache_len),
    )

    decode_input_names = [
        "input_ids",
        "attention_mask",
        "position_ids",
        "past_key_values",
        "cache_pos",
    ]
    decode_output_names = ["logits", "present_key_values"]

    decode_dynamic_axes = {}

    _export_onnx(
        decode,
        decode_path,
        (
            dummy_input_ids_step,
            dummy_attention_mask_step,
            dummy_position_ids_step,
            dummy_past,
            dummy_cache_pos,
        ),
        decode_input_names,
        decode_output_names,
        decode_dynamic_axes,
    )
    print("LLM decode exported successfully.")


def _clear_torch_cache():
    """
    Clear PyTorch cache to free memory.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _export_vision_step(model_id, output_dir, device, vision_image_size):
    """
    Export Qwen3-VL-2B-Instruct vision tower model to ONNX.
    """
    print("\nStep 1/2: Exporting Vision Tower...")
    print(f"Loading model {model_id} in FP16 for Vision export...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    try:
        del model.model.language_model
        del model.lm_head
    except (AttributeError, TypeError):
        pass
    _clear_torch_cache()
    vision_path = Path(output_dir) / "qwen3_vl_vision.onnx"
    export_vision_tower(model, str(vision_path), device, vision_image_size)
    del model
    _clear_torch_cache()
    return vision_path


def _export_llm_step(model_id, output_dir, device, kv_cache_len: int):
    """
    Export Qwen3-VL-2B-Instruct LLM prefill and decode models to ONNX.
    """
    print("\nStep 2/2: Exporting LLM...")
    print(f"Loading model {model_id} in FP16 for LLM export...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    try:
        del model.model.visual
    except (AttributeError, TypeError):
        pass
    _clear_torch_cache()
    export_llm_prefill_decode(model, output_dir, device, kv_cache_len=int(kv_cache_len))
    del model
    _clear_torch_cache()


def main():
    """
    Main function for Qwen3-VL-2B-Instruct export to ONNX.
    """
    parser = argparse.ArgumentParser(description="Export Qwen3-VL-2B-Instruct to ONNX")
    parser.add_argument(
        "--model-id",
        type=str,
        default="./Qwen3-VL-2B-Instruct",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./qwen3_vl_2b_instruct_onnx", help="Output directory"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device for export (cpu or cuda)"
    )
    parser.add_argument(
        "--vision-image-size",
        type=int,
        default=128,
        help="Vision export image size. Must be divisible by vision_config.patch_size.",
    )
    parser.add_argument(
        "--kv-cache-len",
        type=int,
        default=512,
        help="KV cache length used for fixed-shape decode export.",
    )
    parser.add_argument(
        "--no-custom-op",
        action="store_true",
        help="Disable exporting fused CANN operators as Custom nodes.",
    )

    args = parser.parse_args()

    global _USE_CUSTOM_OP
    _USE_CUSTOM_OP = not bool(args.no_custom_op)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _export_vision_step(args.model_id, output_dir, args.device, args.vision_image_size)
    _export_llm_step(args.model_id, output_dir, args.device, args.kv_cache_len)

    print(f"\nExport finished. Files saved in {args.output_dir}")


if __name__ == "__main__":
    main()
