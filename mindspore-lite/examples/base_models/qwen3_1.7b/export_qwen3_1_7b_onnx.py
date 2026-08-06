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
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

KV_CACHE_LEN = 512
TORCH_PTQ_INT8 = True
TORCH_PTQ_CALIB_JSONL = ""
TORCH_PTQ_MAX_SAMPLES = 32
TORCH_PTQ_MAX_DECODE_STEPS = 32
SMOOTH_ALPHA = 0.5
WEIGHT_CLIP_RATIO = 0.0

try:
    import torch._dynamo

    getattr(torch, "_dynamo").disable()
except (ImportError, AttributeError, TypeError):
    pass

try:
    from torch.fx import wrap as _fx_wrap

    _fx_wrap("rotary_mul")
    _fx_wrap("apply_rotary_pos_emb_custom")
    _fx_wrap("rms_norm")
    _fx_wrap("incre_flash_attention")
    _fx_wrap("swiglu")
    _fx_wrap("scatter")
except (ImportError, AttributeError):
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
        """Apply rotary multiplication (FP32 simulation)."""
        del ctx
        return (x * cos4) + (_rotate_half(x) * sin4)

    @staticmethod
    def symbolic(g, x, cos4, sin4):
        """Build ONNX Custom RotaryMul node."""
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
        """Apply rotary position embedding (FP32 simulation)."""
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
        """Build ONNX Custom ApplyRotaryPosEmb node."""
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
    """Custom RmsNorm op for ONNX export."""

    @staticmethod
    def forward(ctx, x, gamma, epsilon: float):
        """Apply RMS normalization (FP32 simulation)."""
        del ctx
        x_fp32 = x.to(torch.float32)
        var = (x_fp32 * x_fp32).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + float(epsilon))
        y = (x_fp32 * rstd).to(x.dtype) * gamma
        return y, rstd

    @staticmethod
    def symbolic(g, x, gamma, epsilon: float):
        """Build ONNX Custom RmsNorm node."""
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
        """Incremental flash attention fallback to manual matmul."""
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
        """Build ONNX Custom IncreFlashAttention node."""
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
    """Run incremental flash attention custom op."""
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
    """Custom PromptFlashAttention op for ONNX export."""

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
        """Prompt flash attention fallback to manual matmul."""
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
        """Build ONNX Custom PromptFlashAttention node."""
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
    """Run prompt flash attention custom op."""
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
        """Apply SwiGLU activation (FP32 simulation)."""
        del ctx
        d = int(dim)
        if d < 0:
            d = x.dim() + d
        a, b = torch.chunk(x, 2, dim=d)
        return F.silu(a) * b

    @staticmethod
    def symbolic(g, x, dim: int):
        """Build ONNX Custom SwiGLU node."""
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
    """Custom Scatter op for ONNX export."""

    @staticmethod
    def forward(ctx, var, indices, updates, reduce: str, axis: int):
        """Apply scatter update on KV cache (FP32 simulation)."""
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
        """Build ONNX Custom Scatter node."""
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


def _onnx_cast_to_i_from_dtype(dtype: torch.dtype) -> int:
    if dtype == torch.float16:
        return 10
    if dtype == torch.float32:
        return 1
    if dtype == torch.bfloat16:
        return 16
    raise RuntimeError(f"Unsupported dtype for ONNX Cast: {dtype}")


def _compute_smooth_quant_scale(max_act_per_ch, max_w_per_col, alpha=0.5, eps=1e-8):
    """Compute SmoothQuant smoothing factor per input channel.
    s[channel] = max_act[channel]^alpha / max_w[channel]^(1-alpha)

    Args:
        max_act_per_ch: per-channel activation max [in_features]
        max_w_per_col:  per-column weight max [in_features]
        alpha: smoothing strength (0.5 = equal smoothing)
    Returns:
        s: smoothing factor [in_features]
    """
    max_act = max_act_per_ch.clamp_min(eps)
    max_w = max_w_per_col.clamp_min(eps)
    s = (max_act ** alpha) / (max_w ** (1.0 - alpha))
    # Clamp to avoid extreme values
    s = s.clamp(1e-4, 1e4)
    return s.detach()


def _quantize_weight_symmetric_int8(
    weight_fp: torch.Tensor,
    eps: float = 1e-8,
    per_channel: bool = True,
    clip_ratio: float = 0.0,
):
    """Quantize weight to int8, optionally clipping weight outliers.

    Args:
        weight_fp: FP32 weight tensor
        per_channel: per-channel quantization
        clip_ratio: if > 0, clip top clip_ratio extreme values before quant.
                    0.01 = clip top 1% of outliers.
    """
    weight = weight_fp.detach()
    if clip_ratio > 0 and weight.numel() > 100:
        # Clip extreme weight outliers to improve quantization quality.
        # A few outlier weights can dominate the scale factor.
        flat = weight.abs().view(-1)
        k = max(1, int(flat.numel() * clip_ratio))
        threshold, _ = flat.topk(k)
        threshold = threshold[-1].clamp_min(eps)
        weight = weight.clamp(-threshold, threshold)

    if per_channel:
        # Per-channel: one scale per output channel (dim=0).
        # weight shape: [out_features, in_features]
        maxabs = weight.abs().max(dim=1, keepdim=True).values.clamp_min(float(eps))
        w_scale = (maxabs / 127.0).to(dtype=torch.float32)  # [out_features, 1]
        w_q = torch.clamp(torch.round(weight / w_scale), -127, 127).to(torch.int8)
        return w_q, w_scale.view(-1)  # [out_features]
    # Per-tensor (original behavior)
    maxabs = weight.abs().max().clamp_min(float(eps))
    w_scale = (maxabs / 127.0).to(dtype=torch.float32)
    w_q = torch.clamp(torch.round(weight / w_scale), -127, 127).to(torch.int8)
    return w_q, w_scale


class _QuantLinearSymInt8(torch.autograd.Function):
    """Quantized linear layer: FP32 forward, int8 symbolic for ONNX."""

    @staticmethod
    def forward(
        ctx, x, weight_fp, bias_fp, x_scale_f: float, w_q, correction,
        w_scale_mean: float, smooth_scale, out_to_i: int,
    ):
        """Forward pass: run plain FP32 linear (symbolic builds int8 graph)."""
        del ctx, x_scale_f, w_q, correction, w_scale_mean, smooth_scale, out_to_i
        # Forward uses FP32 weight for correct tracing; the ONNX graph
        # (via symbolic) uses int8 weight + per-channel dequant + smoothquant.
        return F.linear(x, weight_fp, bias_fp)

    @staticmethod
    def symbolic(
        g, x, weight_fp, bias_fp, x_scale_f: float, w_q, correction,
        w_scale_mean: float, smooth_scale, out_to_i: int,
    ):
        """Build the ONNX int8 quantized linear graph."""
        del weight_fp
        import struct

        x_scale_f = float(x_scale_f)
        w_scale_mean = float(w_scale_mean)
        per_channel = correction is not None

        # SmoothQuant: divide input by s before quantization
        if smooth_scale is not None:
            # smooth_scale is a pre-computed constant tensor [in_features]
            x = g.op("Div", x, smooth_scale)

        # AscendQuant: quantize activation to int8.
        ascend_scale = 1.0 / max(x_scale_f, 1e-8)
        x_i8 = g.op(
            "Custom", x,
            type_s="AscendQuant",
            input_names_s=_as_list_str(["x"]),
            optional_input_names_s=_as_list_str([]),
            output_names_s=_as_list_str(["y"]),
            output_num_i=1,
            input_index_i=[0],
            src_t_i=1,  # kNumberTypeFloat32
            dst_t_i=3,  # kNumberTypeInt8
            scale_f=float(ascend_scale),
            offset_f=0.0,
        )

        # Pack x_scale * w_scale_mean into uint64.
        combined_scale = x_scale_f * w_scale_mean
        scale_bits = struct.unpack("<I", struct.pack("<f", combined_scale))[0]
        packed_scale = scale_bits
        scale_tensor = torch.tensor([packed_scale], dtype=torch.int64)
        scale_const = g.op("Constant", value_t=scale_tensor)

        # QuantBatchMatmul: int8 x int8 with combined (mean) scale.
        y = g.op(
            "Custom", x_i8, w_q, scale_const,
            type_s="QuantBatchMatmul",
            input_names_s=_as_list_str(
                ["x1", "x2", "scale", "offset", "bias", "pertoken_scale"]),
            optional_input_names_s=_as_list_str(
                ["offset", "bias", "pertoken_scale"]),
            output_names_s=_as_list_str(["y"]),
            output_num_i=1,
            input_index_i=[0, 1, 2],
            transpose_x1_s="false",
            transpose_x2_s="true",
            dtype_i=1,  # Force float32 intermediate for per-channel Mul
        )

        # Per-channel correction: Mul by (w_scale / w_scale_mean)
        if per_channel:
            y = g.op("Mul", y, correction)

        # Bias
        y = g.op("Add", y, bias_fp)

        # Cast to target dtype
        y = g.op("Cast", y, to_i=int(out_to_i))
        y.setType(x.type())
        return y


def quant_linear_symmetric_int8(x, weight_fp, bias_fp, x_scale, w_q, w_scale, smooth_scale=None):
    """Run symmetric int8 quantized linear (FP32 forward, int8 ONNX graph)."""
    # w_scale can be a 1D per-channel tensor [out_features] or a scalar
    # smooth_scale: optional 1D tensor [in_features] for SmoothQuant
    out_to_i = int(_onnx_cast_to_i_from_dtype(x.dtype))
    x_scale_f = float(x_scale)

    if smooth_scale is not None:
        # Cast smooth_scale to match the model's export dtype so that the
        # ONNX Constant node inserted for the Div operand has the same dtype
        # as the input (matching --dtype specified at export time).
        smooth_scale = smooth_scale.to(x.dtype)

    if isinstance(w_scale, torch.Tensor) and w_scale.dim() == 1:
        # Per-channel: pre-compute correction factor and mean scale
        w_scale_np = w_scale.detach().cpu().numpy()
        w_scale_mean = float(w_scale_np.mean())
        correction = (w_scale_np / w_scale_mean).astype(np.float32)
        correction = torch.from_numpy(correction)
    else:
        # Per-tensor fallback
        if isinstance(w_scale, torch.Tensor):
            w_scale_mean = float(w_scale.cpu().item())
        else:
            w_scale_mean = float(w_scale)
        correction = None

    return _QuantLinearSymInt8.apply(
        x, weight_fp, bias_fp, x_scale_f, w_q, correction, w_scale_mean, smooth_scale, out_to_i,
    )



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

    if TORCH_PTQ_INT8 and hasattr(attn_mod, "ptq_qkv_act_obs"):
        attn_mod.ptq_qkv_act_obs(hidden_states.detach())
        # Track per-channel activation max for SmoothQuant
        flat_h = hidden_states.detach().reshape(-1, hidden_states.shape[-1])
        per_ch = flat_h.abs().max(dim=0).values
        if not hasattr(attn_mod, "ptq_qkv_act_per_ch_max") or attn_mod.ptq_qkv_act_per_ch_max is None:
            attn_mod.ptq_qkv_act_per_ch_max = per_ch
        else:
            attn_mod.ptq_qkv_act_per_ch_max = torch.maximum(attn_mod.ptq_qkv_act_per_ch_max, per_ch)

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
    if TORCH_PTQ_INT8 and hasattr(attn_mod, "ptq_qkv_w_q"):
        if b is None:
            b = hidden_states.new_zeros((w.shape[0],))
        qkv_smooth = getattr(attn_mod, "ptq_qkv_smooth_scale", None)
        qkv = quant_linear_symmetric_int8(
            hidden_states,
            w,
            b,
            attn_mod.ptq_qkv_x_scale,
            attn_mod.ptq_qkv_w_q,
            attn_mod.ptq_qkv_w_scale,
            smooth_scale=qkv_smooth,
        )
    else:
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

    query_states = rotary_mul(query_states, cos4, sin4)
    key_states = rotary_mul(key_states, cos4, sin4)

    if past_key is not None:
        pos = cache_pos[:, -1]
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
    if past_key is not None and TORCH_PTQ_INT8 and hasattr(attn_mod.o_proj, "ptq_w_q"):
        b = attn_mod.o_proj.bias
        if b is None:
            b = attn_output.new_zeros((attn_mod.o_proj.weight.shape[0],))
        o_smooth = getattr(attn_mod.o_proj, "ptq_smooth_scale", None)
        attn_output = quant_linear_symmetric_int8(
            attn_output,
            attn_mod.o_proj.weight,
            b,
            attn_mod.o_proj.ptq_x_scale,
            attn_mod.o_proj.ptq_w_q,
            attn_mod.o_proj.ptq_w_scale,
            smooth_scale=o_smooth,
        )
    else:
        attn_output = attn_mod.o_proj(attn_output)
    return attn_output, key_states, value_states


def _rms_norm_layer(norm_mod, x):
    gamma = norm_mod.weight
    eps = getattr(norm_mod, "variance_epsilon", 1e-6)
    y, _ = rms_norm(x, gamma, eps)
    return y


def _mlp_gate_up_linear(mlp_mod, x):
    """Run merged MLP gate/up projection, optionally with int8 quantization."""
    gate_w = mlp_mod.gate_proj.weight
    up_w = mlp_mod.up_proj.weight
    gate_b = mlp_mod.gate_proj.bias
    up_b = mlp_mod.up_proj.bias
    w = torch.cat([gate_w, up_w], dim=0)
    if gate_b is None:
        b = None
    else:
        b = torch.cat([gate_b, up_b], dim=0)
    if TORCH_PTQ_INT8 and hasattr(mlp_mod, "ptq_gate_up_act_obs"):
        mlp_mod.ptq_gate_up_act_obs(x.detach())
        # Track per-channel activation max for SmoothQuant
        flat_x_gate = x.detach().reshape(-1, x.shape[-1])
        per_ch_g = flat_x_gate.abs().max(dim=0).values
        if not hasattr(mlp_mod, "ptq_gate_up_act_per_ch_max") or mlp_mod.ptq_gate_up_act_per_ch_max is None:
            mlp_mod.ptq_gate_up_act_per_ch_max = per_ch_g
        else:
            mlp_mod.ptq_gate_up_act_per_ch_max = torch.maximum(mlp_mod.ptq_gate_up_act_per_ch_max, per_ch_g)
    if TORCH_PTQ_INT8 and hasattr(mlp_mod, "ptq_gate_up_w_q"):
        if b is None:
            b = x.new_zeros((w.shape[0],))
        gu_smooth = getattr(mlp_mod, "ptq_gate_up_smooth_scale", None)
        y = quant_linear_symmetric_int8(
            x,
            w,
            b,
            mlp_mod.ptq_gate_up_x_scale,
            mlp_mod.ptq_gate_up_w_q,
            mlp_mod.ptq_gate_up_w_scale,
            smooth_scale=gu_smooth,
        )
    else:
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
                sw = swiglu(x, dim=-1)
                if TORCH_PTQ_INT8 and hasattr(mlp.down_proj, "ptq_w_q"):
                    b = mlp.down_proj.bias
                    if b is None:
                        b = sw.new_zeros((mlp.down_proj.weight.shape[0],))
                    down_smooth = getattr(mlp.down_proj, "ptq_smooth_scale", None)
                    mlp_out = quant_linear_symmetric_int8(
                        sw,
                        mlp.down_proj.weight,
                        b,
                        mlp.down_proj.ptq_x_scale,
                        mlp.down_proj.ptq_w_q,
                        mlp.down_proj.ptq_w_scale,
                        smooth_scale=down_smooth,
                    )
                else:
                    mlp_out = mlp.down_proj(sw)
                hidden_states = residual + mlp_out
            else:
                hidden_states = residual + mlp(hidden_states)
            present_k.append(pk)
            present_v.append(pv)

        hidden_states = _rms_norm_layer(self.model.norm, hidden_states)
        if TORCH_PTQ_INT8 and hasattr(self.lm_head, "ptq_w_q"):
            b = getattr(self.lm_head, "bias", None)
            if b is None:
                b = hidden_states.new_zeros((self.lm_head.weight.shape[0],))
            logits = quant_linear_symmetric_int8(
                hidden_states,
                self.lm_head.weight,
                b,
                self.lm_head.ptq_x_scale,
                self.lm_head.ptq_w_q,
                self.lm_head.ptq_w_scale,
                smooth_scale=getattr(self.lm_head, "ptq_smooth_scale", None),
            )
        else:
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
        cos4 = cos.unsqueeze(1)
        sin4 = sin.unsqueeze(1)
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
                sw = swiglu(x, dim=-1)
                if TORCH_PTQ_INT8 and hasattr(mlp.down_proj, "ptq_w_q"):
                    b = mlp.down_proj.bias
                    if b is None:
                        b = sw.new_zeros((mlp.down_proj.weight.shape[0],))
                    down_smooth = getattr(mlp.down_proj, "ptq_smooth_scale", None)
                    mlp_out = quant_linear_symmetric_int8(
                        sw,
                        mlp.down_proj.weight,
                        b,
                        mlp.down_proj.ptq_x_scale,
                        mlp.down_proj.ptq_w_q,
                        mlp.down_proj.ptq_w_scale,
                        smooth_scale=down_smooth,
                    )
                else:
                    mlp_out = mlp.down_proj(sw)
                hidden_states = residual + mlp_out
            else:
                hidden_states = residual + mlp(hidden_states)
            present_k.append(pk)
            present_v.append(pv)

        hidden_states = _rms_norm_layer(self.model.norm, hidden_states)
        if TORCH_PTQ_INT8 and hasattr(self.lm_head, "ptq_w_q"):
            b = getattr(self.lm_head, "bias", None)
            if b is None:
                b = hidden_states.new_zeros((self.lm_head.weight.shape[0],))
            logits = quant_linear_symmetric_int8(
                hidden_states,
                self.lm_head.weight,
                b,
                self.lm_head.ptq_x_scale,
                self.lm_head.ptq_w_q,
                self.lm_head.ptq_w_scale,
                smooth_scale=getattr(self.lm_head, "ptq_smooth_scale", None),
            )
        else:
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
    decode_name = "qwen3_1_7b_llm_decode.onnx"
    if TORCH_PTQ_INT8:
        decode_name = "qwen3_1_7b_llm_decode_ptq_int8.onnx"
    decode_path = decode_dir / decode_name
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


def _load_calib_records_jsonl(path: str, max_samples: int):
    """Load PTQ calibration records from a JSONL file."""
    records = []
    if not path:
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if len(records) >= int(max_samples):
                break
    return records


def _make_synthetic_calib_records(num_samples: int, seq_len: int):
    """Generate synthetic calibration records when no JSONL file is given."""
    records = []
    seq_len = int(seq_len)
    for _ in range(int(num_samples)):
        input_ids = torch.randint(0, 1000, (1, seq_len), dtype=torch.int64).tolist()
        attention_mask = torch.ones(1, seq_len, dtype=torch.int64).tolist()
        position_ids = torch.arange(seq_len, dtype=torch.int64).view(1, -1).tolist()
        gen_len = max(4, min(64, seq_len))
        generated_ids = torch.randint(0, 1000, (gen_len,), dtype=torch.int64).tolist()
        records.append(
            {
                "prefill_input_ids": input_ids,
                "prefill_attention_mask": attention_mask,
                "prefill_position_ids": position_ids,
                "generated_ids": generated_ids,
            }
        )
    return records


def _torch_ptq_static_int8_quantize_decode(
    prefill,
    decode,
    device: torch.device,
    calib_records,
    decode_example_inputs,
    max_decode_steps: int,
):
    """Calibrate and quantize decode Linear layers to symmetric int8 with SmoothQuant."""
    from torch.ao.quantization.observer import MinMaxObserver
    _ = decode_example_inputs

    decode.eval()
    prefill.eval()

    hooks = []
    linear_modules = []
    for m in decode.modules():
        if isinstance(m, torch.nn.Linear):
            linear_modules.append(m)
            obs = MinMaxObserver(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric).to(device)
            m.ptq_act_obs = obs

            def _pre_hook(mod, inputs):
                x = inputs[0]
                mod.ptq_act_obs(x.detach())
                # Track per-channel activation max for SmoothQuant
                flat_x = x.detach().reshape(-1, x.shape[-1])
                per_ch = flat_x.abs().max(dim=0).values
                if not hasattr(mod, "ptq_act_per_ch_max") or mod.ptq_act_per_ch_max is None:
                    mod.ptq_act_per_ch_max = per_ch
                else:
                    mod.ptq_act_per_ch_max = torch.maximum(mod.ptq_act_per_ch_max, per_ch)

            hooks.append(m.register_forward_pre_hook(_pre_hook))

    # QKV merged linear is not a module; it is created by cat in _text_attn_forward.
    # We attach one observer per attention module to collect hidden_states range.
    for layer in decode.model.layers:
        attn = layer.self_attn
        obs = MinMaxObserver(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric).to(device)
        attn.ptq_qkv_act_obs = obs

    # MLP gate/up merged linear is not a module; attach observer on mlp module.
    for layer in decode.model.layers:
        mlp = layer.mlp
        if hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj"):
            obs = MinMaxObserver(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric).to(device)
            mlp.ptq_gate_up_act_obs = obs

    calib_start = time.monotonic()
    num_records = len(calib_records)
    with torch.no_grad():
        for rec_idx, rec in enumerate(calib_records, start=1):
            prefill_input_ids = rec.get("prefill_input_ids", None)
            prefill_attention_mask = rec.get("prefill_attention_mask", None)
            prefill_position_ids = rec.get("prefill_position_ids", None)
            gen_ids = rec.get("generated_ids", None)
            if prefill_input_ids is None or prefill_attention_mask is None or prefill_position_ids is None:
                print(f"[PTQ] sample {rec_idx}/{num_records}: skip (missing prefill fields)")
                continue
            print(
                f"[PTQ] sample {rec_idx}/{num_records}: prefill start (seq_len={len(prefill_input_ids[0])})",
                flush=True,
            )

            input_ids = torch.tensor(prefill_input_ids, dtype=torch.int64, device=device)
            attention_mask = torch.tensor(
                prefill_attention_mask, dtype=torch.int64, device=device
            )
            position_ids = torch.tensor(
                prefill_position_ids, dtype=torch.int64, device=device
            )
            _, past_k, past_v = prefill(input_ids, attention_mask, position_ids)
            print(f"[PTQ] sample {rec_idx}/{num_records}: prefill done", flush=True)

            actual_len = int(attention_mask[0].sum().item())
            cur_attention_mask = torch.zeros(
                (1, int(KV_CACHE_LEN)), dtype=torch.int64, device=device
            )
            if actual_len > 0:
                cur_attention_mask[0, :actual_len] = 1
            valid_len = int(actual_len)

            if not gen_ids:
                print(f"[PTQ] sample {rec_idx}/{num_records}: skip (no generated ids)")
                continue
            gen_ids = [int(x) for x in gen_ids]
            steps = min(int(max_decode_steps), max(len(gen_ids) - 1, 1))
            for t in range(steps):
                if valid_len >= int(KV_CACHE_LEN):
                    break
                token_id = gen_ids[t]
                next_input_ids = torch.tensor([[token_id]], dtype=torch.int64, device=device)
                cur_attention_mask[0, valid_len] = 1
                next_position_ids = torch.tensor(
                    [[valid_len]], dtype=torch.int64, device=device
                )
                _, past_k, past_v = decode(
                    next_input_ids, cur_attention_mask, next_position_ids, past_k, past_v
                )
                valid_len += 1
                if t % 4 == 0 or t == steps - 1:
                    elapsed = time.monotonic() - calib_start
                    print(
                        f"[PTQ] sample {rec_idx}/{num_records} decode step {t + 1}/{steps} "
                        f"(elapsed {elapsed:.1f}s)",
                        flush=True,
                    )
            elapsed = time.monotonic() - calib_start
            print(
                f"[PTQ] sample {rec_idx}/{num_records} done (total elapsed {elapsed:.1f}s)",
                flush=True,
            )

    for h in hooks:
        h.remove()

    for m in linear_modules:
        obs = getattr(m, "ptq_act_obs", None)
        if obs is None or obs.max_val is None or obs.min_val is None:
            maxabs = torch.tensor(1.0, device=device, dtype=torch.float32)
            per_ch_max = None
        else:
            maxabs = torch.maximum(obs.max_val.abs(), obs.min_val.abs()).to(torch.float32)
            maxabs = maxabs.clamp_min(1e-8)
            per_ch_max = getattr(m, "ptq_act_per_ch_max", None)
        if int(m.weight.t().shape[-1]) > 0x7FFFFFFF:
            if hasattr(m, "ptq_act_obs"):
                delattr(m, "ptq_act_obs")
            continue

        w = m.weight.detach()
        if per_ch_max is not None and w.dim() >= 2:
            max_w_per_col = w.abs().max(dim=0).values
            smooth_scale = _compute_smooth_quant_scale(per_ch_max, max_w_per_col, alpha=SMOOTH_ALPHA)
            smooth_scale = smooth_scale.to(device=device, dtype=torch.float32)
            w_smoothed = w * smooth_scale.unsqueeze(0)
            max_act_smoothed = (per_ch_max / smooth_scale).max().clamp_min(1e-8)
            x_scale_smoothed = float((max_act_smoothed / 127.0).cpu().item())
            w_q, w_scale = _quantize_weight_symmetric_int8(w_smoothed, per_channel=True, clip_ratio=WEIGHT_CLIP_RATIO)
            m.ptq_smooth_scale = smooth_scale.detach().cpu()
            m.ptq_x_scale = x_scale_smoothed
        else:
            x_scale = float((maxabs / 127.0).to(dtype=torch.float32).cpu().item())
            w_q, w_scale = _quantize_weight_symmetric_int8(w, per_channel=True, clip_ratio=WEIGHT_CLIP_RATIO)
            m.ptq_x_scale = x_scale
        m.ptq_w_q = w_q
        m.ptq_w_scale = w_scale
        for attr in ["ptq_act_obs", "ptq_act_per_ch_max"]:
            if hasattr(m, attr):
                delattr(m, attr)

    # Quantize merged QKV weights per layer with SmoothQuant.
    for layer in decode.model.layers:
        attn = layer.self_attn
        obs = getattr(attn, "ptq_qkv_act_obs", None)
        if obs is None or obs.max_val is None or obs.min_val is None:
            maxabs = torch.tensor(1.0, device=device, dtype=torch.float32)
            per_ch_max = None
        else:
            maxabs = torch.maximum(obs.max_val.abs(), obs.min_val.abs()).to(torch.float32).clamp_min(1e-8)
            per_ch_max = getattr(attn, "ptq_qkv_act_per_ch_max", None)
        q_w = attn.q_proj.weight
        k_w = attn.k_proj.weight
        v_w = attn.v_proj.weight
        w = torch.cat([q_w, k_w, v_w], dim=0)
        if int(w.t().shape[-1]) > 0x7FFFFFFF:
            continue

        if per_ch_max is not None:
            max_w_per_col = w.abs().max(dim=0).values
            smooth_scale = _compute_smooth_quant_scale(per_ch_max, max_w_per_col, alpha=SMOOTH_ALPHA)
            smooth_scale = smooth_scale.to(device=device, dtype=torch.float32)
            w_smoothed = w * smooth_scale.unsqueeze(0)
            max_act_smoothed = (per_ch_max / smooth_scale).max().clamp_min(1e-8)
            x_scale_smoothed = float((max_act_smoothed / 127.0).cpu().item())
            w_q, w_scale = _quantize_weight_symmetric_int8(w_smoothed, per_channel=True, clip_ratio=WEIGHT_CLIP_RATIO)
            attn.ptq_qkv_smooth_scale = smooth_scale.detach().cpu()
            attn.ptq_qkv_x_scale = x_scale_smoothed
        else:
            x_scale = float((maxabs / 127.0).to(dtype=torch.float32).cpu().item())
            w_q, w_scale = _quantize_weight_symmetric_int8(w, per_channel=True, clip_ratio=WEIGHT_CLIP_RATIO)
            attn.ptq_qkv_x_scale = x_scale
        attn.ptq_qkv_w_q = w_q
        attn.ptq_qkv_w_scale = w_scale
        for attr in ["ptq_qkv_act_obs", "ptq_qkv_act_per_ch_max"]:
            if hasattr(attn, attr):
                delattr(attn, attr)

    # Quantize merged MLP gate/up weights per layer with SmoothQuant.
    for layer in decode.model.layers:
        mlp = layer.mlp
        if not (hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj")):
            continue
        obs = getattr(mlp, "ptq_gate_up_act_obs", None)
        if obs is None or obs.max_val is None or obs.min_val is None:
            maxabs = torch.tensor(1.0, device=device, dtype=torch.float32)
            per_ch_max = None
        else:
            maxabs = torch.maximum(obs.max_val.abs(), obs.min_val.abs()).to(torch.float32).clamp_min(1e-8)
            per_ch_max = getattr(mlp, "ptq_gate_up_act_per_ch_max", None)
        w = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], dim=0)
        if int(w.t().shape[-1]) > 0x7FFFFFFF:
            for attr in ["ptq_gate_up_act_obs", "ptq_gate_up_act_per_ch_max"]:
                if hasattr(mlp, attr):
                    delattr(mlp, attr)
            continue

        if per_ch_max is not None:
            max_w_per_col = w.abs().max(dim=0).values
            smooth_scale = _compute_smooth_quant_scale(per_ch_max, max_w_per_col, alpha=SMOOTH_ALPHA)
            smooth_scale = smooth_scale.to(device=device, dtype=torch.float32)
            w_smoothed = w * smooth_scale.unsqueeze(0)
            max_act_smoothed = (per_ch_max / smooth_scale).max().clamp_min(1e-8)
            x_scale_smoothed = float((max_act_smoothed / 127.0).cpu().item())
            w_q, w_scale = _quantize_weight_symmetric_int8(w_smoothed, per_channel=True, clip_ratio=WEIGHT_CLIP_RATIO)
            mlp.ptq_gate_up_smooth_scale = smooth_scale.detach().cpu()
            mlp.ptq_gate_up_x_scale = x_scale_smoothed
        else:
            x_scale = float((maxabs / 127.0).to(dtype=torch.float32).cpu().item())
            w_q, w_scale = _quantize_weight_symmetric_int8(w, per_channel=True, clip_ratio=WEIGHT_CLIP_RATIO)
            mlp.ptq_gate_up_x_scale = x_scale
        mlp.ptq_gate_up_w_q = w_q
        mlp.ptq_gate_up_w_scale = w_scale
        for attr in ["ptq_gate_up_act_obs", "ptq_gate_up_act_per_ch_max"]:
            if hasattr(mlp, attr):
                delattr(mlp, attr)

    print(
        f"[PTQ] calibration finished: {num_records} samples, "
        f"total {time.monotonic() - calib_start:.1f}s",
        flush=True,
    )
    return decode


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
    if TORCH_PTQ_INT8:
        calib_records = _load_calib_records_jsonl(
            TORCH_PTQ_CALIB_JSONL, TORCH_PTQ_MAX_SAMPLES
        )
        if not calib_records:
            calib_records = _make_synthetic_calib_records(
                num_samples=min(4, int(TORCH_PTQ_MAX_SAMPLES)),
                seq_len=max(8, int(dummy_seq_len)),
            )
        decode = _torch_ptq_static_int8_quantize_decode(
            prefill=prefill,
            decode=decode,
            device=torch.device(device),
            calib_records=calib_records,
            decode_example_inputs=decode_dummy_inputs,
            max_decode_steps=TORCH_PTQ_MAX_DECODE_STEPS,
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
        "--kv-cache-len",
        type=int,
        default=512,
        help="Fixed KV cache sequence length for export",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        choices=["fp16", "bf16", "fp32"],
        help="Export dtype",
    )
    parser.add_argument(
        "--use-dynamo", action="store_true", help="Use torch dynamo exporter path"
    )
    parser.add_argument(
        "--disable-torch-ptq-int8",
        action="store_true",
        dest="disable_torch_ptq_int8",
        default=False,
        help="Disable PTQ static int8 quantization (enabled by default).",
    )
    parser.add_argument(
        "--torch-ptq-calib-jsonl",
        type=str,
        dest="torchptq_calib_jsonl",
        default="calib.jsonl",
        help="Calibration JSONL file exported by infer_qwen3_1_7b_mslite.py --dump-calib.",
    )
    parser.add_argument(
        "--torch-ptq-max-samples",
        type=int,
        dest="torchptq_max_samples",
        default=32,
        help="Max calibration samples to load from JSONL.",
    )
    parser.add_argument(
        "--torch-ptq-max-decode-steps",
        type=int,
        dest="torchptq_max_decode_steps",
        default=32,
        help="Max decode steps per sample during calibration.",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.65,
        help="SmoothQuant alpha (0.0-1.0). Lower = more activation smoothing.",
    )
    parser.add_argument(
        "--weight-clip-ratio",
        type=float,
        default=0.0,
        help="Clip weight outliers before quantization (0.01 = top 1%%).",
    )
    args = parser.parse_args()
    global KV_CACHE_LEN
    KV_CACHE_LEN = int(args.kv_cache_len)
    global TORCH_PTQ_INT8, TORCH_PTQ_CALIB_JSONL, TORCH_PTQ_MAX_SAMPLES, TORCH_PTQ_MAX_DECODE_STEPS
    global SMOOTH_ALPHA, WEIGHT_CLIP_RATIO
    TORCH_PTQ_INT8 = not bool(args.disable_torch_ptq_int8)
    TORCH_PTQ_CALIB_JSONL = str(args.torchptq_calib_jsonl or "")
    TORCH_PTQ_MAX_SAMPLES = int(args.torchptq_max_samples)
    TORCH_PTQ_MAX_DECODE_STEPS = int(args.torchptq_max_decode_steps)
    SMOOTH_ALPHA = float(args.smooth_alpha)
    WEIGHT_CLIP_RATIO = float(args.weight_clip_ratio)
    print(f"  SmoothQuant alpha={SMOOTH_ALPHA:.2f}, weight_clip={WEIGHT_CLIP_RATIO:.4f}")

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
