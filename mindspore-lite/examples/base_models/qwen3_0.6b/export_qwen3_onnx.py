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
Export Qwen3-0.6B model to ONNX format.
"""

import sys
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
try:
    import torch._dynamo

    torch._dynamo.disable()
except:
    pass

try:
    from transformers import AutoModelForCausalLM
except ImportError:
    print("Error: transformers package not found or version too low.")
    print(
        "Please install the latest version: pip install git+https://github.com/huggingface/transformers"
    )
    sys.exit(1)

try:
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
except Exception:
    try:
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    except Exception:
        apply_rotary_pos_emb = None


# PTQ int8 globals (set by CLI in main()). Default ON to match qwen3_1.7b
# convention; --disable-torch-ptq-int8 turns it off.
TORCH_PTQ_INT8 = False
TORCH_PTQ_CALIB_JSONL = ""
TORCH_PTQ_MAX_SAMPLES = 32
TORCH_PTQ_MAX_DECODE_STEPS = 32
SMOOTH_ALPHA = 0.5
WEIGHT_CLIP_RATIO = 0.0
# KV cache length used during PTQ calibration decode replay. Doesn't need to
# match a deployment bucket — calibration just runs forward passes to collect
# activation stats; the resulting quant params are shape-independent.
PTQ_CALIB_KV_LEN = 512


# ---------------------------------------------------------------------------
# CANN Custom Op implementations for ONNX export (used when --fusion-opt on)
# ---------------------------------------------------------------------------
# These torch.autograd.Function subclasses trace a pure-PyTorch reference
# implementation in forward() (so the exporter sees the right numerics during
# tracing) but emit a Custom ONNX node in symbolic() (so MindSpore Lite maps
# the subgraph to a single CANN fused op at conversion time).


class _CannRmsNorm(torch.autograd.Function):
    """RmsNorm -> Custom(RmsNorm)."""

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
        """Emit ONNX Custom node mapping to the CANN RmsNorm op."""
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
        y, _ = _CannRmsNorm.apply(x, self.weight, self.epsilon)
        return y


class _CannAddRmsNorm(torch.autograd.Function):
    """Add + RmsNorm -> Custom(AddRmsNorm)."""

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
        """Emit ONNX Custom node mapping to the fused CANN Add+RmsNorm op."""
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
    """rotate_half + cos/sin multiply -> Custom(RotaryMul)."""

    @staticmethod
    def forward(ctx, x, r1, r2):
        del ctx
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        rotated = torch.cat([-x2, x1], dim=-1)
        y = x * r1 + rotated * r2
        return y

    @staticmethod
    def symbolic(g, x, r1, r2):
        """Emit ONNX Custom node mapping to the CANN RotaryMul op (rotate_half + cos/sin mul)."""
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
    """QK^T + softmax + V -> Custom(PromptFlashAttention)."""

    @staticmethod
    def forward(ctx, query, key, value, atten_mask, num_heads, num_key_value_heads, scale_value):
        """PyTorch reference impl (traced for ONNX export numerics).

        Returns [B, S, N, D] (transposed) to match CANN PFA op output layout.
        """
        del ctx
        if num_key_value_heads < num_heads:
            key = key.repeat_interleave(num_heads // num_key_value_heads, dim=1)
            value = value.repeat_interleave(num_heads // num_key_value_heads, dim=1)
        scale = float(scale_value)
        attn = torch.matmul(query, key.transpose(2, 3)) * scale
        if atten_mask is not None:
            attn = attn + atten_mask
        attn = torch.nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_output = torch.matmul(attn, value)
        return attn_output

    @staticmethod
    def symbolic(g, query, key, value, atten_mask, num_heads, num_key_value_heads, scale_value):
        """Emit ONNX Custom node mapping to the CANN PromptFlashAttention op."""
        y = g.op(
            "Custom",
            query,
            key,
            value,
            atten_mask,
            type_s="PromptFlashAttention",
            input_names_s=["query", "key", "value", "atten_mask"],
            optional_input_names_s=["atten_mask"],
            output_names_s=["attention_out"],
            output_num_i=1,
            input_index_i=[0, 1, 2, 3],
            num_heads_i=int(num_heads),
            num_key_value_heads_i=int(num_key_value_heads),
            scale_value_f=float(scale_value),
            input_layout_s="BNSD",
            inner_precise_i=0,
        )
        y.setType(query.type())
        return y


class _CannSwiGlu(torch.autograd.Function):
    """SiLU(gate) * up -> Custom(SwiGlu)."""

    @staticmethod
    def forward(ctx, x, dim):
        del ctx
        d = int(dim)
        split = x.shape[d] // 2
        a, b = torch.split(x, [split, split], dim=d)
        return torch.nn.functional.silu(a) * b

    @staticmethod
    def symbolic(g, x, dim):
        """Emit ONNX Custom node mapping to the CANN SwiGlu op (SiLU(gate) * up)."""
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
    """MatMul -> Custom(MatMulV2) — used to lower BMM to MM for Linear layers."""

    @staticmethod
    def forward(ctx, x1, x2):
        del ctx
        return torch.matmul(x1, x2)

    @staticmethod
    def symbolic(g, x1, x2):
        """Emit ONNX Custom node mapping to the CANN MatMulV2 op."""
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
    """Apply a Linear via 2D MatMulV2 to avoid BatchMatMul lowering."""
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
    out = torch.matmul(x, linear_mod.weight.t())
    if linear_mod.bias is not None:
        out = out + linear_mod.bias
    return out


# ---------------------------------------------------------------------------
# PTQ int8 quantization scaffolding (AscendQuant + QuantBatchMatmul Custom ops)
# ---------------------------------------------------------------------------
# Strategy: forward stays FP32 (so PyTorch tracing numerics are correct);
# symbolic emits Ascend Quant/MatMul Custom nodes for the ONNX graph. After
# converter_lite runs with plugin_custom_ops=All, these become int8 ops in
# the MindIR. Weight is per-channel symmetric int8; activation is per-tensor
# symmetric int8. Optional SmoothQuant shifts quantization difficulty from
# activation to weight to mitigate outliers.


def _onnx_cast_to_i_from_dtype(dtype: torch.dtype) -> int:
    if dtype == torch.float16:
        return 10
    if dtype == torch.float32:
        return 1
    if dtype == torch.bfloat16:
        return 16
    raise RuntimeError(f"Unsupported dtype for ONNX Cast: {dtype}")


def _compute_smooth_quant_scale(max_act_per_ch, max_w_per_col, alpha=0.5, eps=1e-8):
    """s[ch] = max_act[ch]^alpha / max_w[ch]^(1-alpha)."""
    max_act = max_act_per_ch.clamp_min(eps)
    max_w = max_w_per_col.clamp_min(eps)
    s = (max_act ** alpha) / (max_w ** (1.0 - alpha))
    s = s.clamp(1e-4, 1e4)
    return s.detach()


def _quantize_weight_symmetric_int8(weight_fp, eps=1e-8, per_channel=True, clip_ratio=0.0):
    """Per-channel symmetric int8 weight quantization."""
    weight = weight_fp.detach()
    if clip_ratio > 0 and weight.numel() > 100:
        flat = weight.abs().view(-1)
        k = max(1, int(flat.numel() * clip_ratio))
        threshold, _ = flat.topk(k)
        threshold = threshold[-1].clamp_min(eps)
        weight = weight.clamp(-threshold, threshold)
    if per_channel:
        maxabs = weight.abs().max(dim=1, keepdim=True).values.clamp_min(float(eps))
        w_scale = (maxabs / 127.0).to(dtype=torch.float32)
        w_q = torch.clamp(torch.round(weight / w_scale), -127, 127).to(torch.int8)
        return w_q, w_scale.view(-1)
    maxabs = weight.abs().max().clamp_min(float(eps))
    w_scale = (maxabs / 127.0).to(dtype=torch.float32)
    w_q = torch.clamp(torch.round(weight / w_scale), -127, 127).to(torch.int8)
    return w_q, w_scale


class _QuantLinearSymInt8(torch.autograd.Function):
    """Quantized linear (int8): forward FP32 for tracing, symbolic for ONNX."""

    @staticmethod
    # pylint: disable=unused-argument
    def forward(ctx, x, weight_fp, bias_fp, x_scale_f, w_q, correction,
                w_scale_mean, smooth_scale, out_to_i):
        return torch.nn.functional.linear(x, weight_fp, bias_fp)

    @staticmethod
    # pylint: disable=unused-argument
    def symbolic(g, x, weight_fp, bias_fp, x_scale_f, w_q, correction,
                 w_scale_mean, smooth_scale, out_to_i):
        """Emit ONNX Custom nodes mapping to the Ascend int8 Quant + MatMul ops."""
        import struct

        x_scale_f = float(x_scale_f)
        w_scale_mean = float(w_scale_mean)
        per_channel = correction is not None

        if smooth_scale is not None:
            x = g.op("Div", x, smooth_scale)

        ascend_scale = 1.0 / max(x_scale_f, 1e-8)
        x_i8 = g.op(
            "Custom", x,
            type_s="AscendQuant",
            input_names_s=["x"],
            optional_input_names_s=[],
            output_names_s=["y"],
            output_num_i=1,
            input_index_i=[0],
            src_t_i=1,  # float32
            dst_t_i=3,  # int8
            scale_f=float(ascend_scale),
            offset_f=0.0,
        )

        combined_scale = x_scale_f * w_scale_mean
        scale_bits = struct.unpack("<I", struct.pack("<f", combined_scale))[0]
        scale_tensor = torch.tensor([scale_bits], dtype=torch.int64)
        scale_const = g.op("Constant", value_t=scale_tensor)

        y = g.op(
            "Custom", x_i8, w_q, scale_const,
            type_s="QuantBatchMatmul",
            input_names_s=["x1", "x2", "scale", "offset", "bias", "pertoken_scale"],
            optional_input_names_s=["offset", "bias", "pertoken_scale"],
            output_names_s=["y"],
            output_num_i=1,
            input_index_i=[0, 1, 2],
            transpose_x1_s="false",
            transpose_x2_s="true",
            dtype_i=1,  # float32 intermediate for per-channel Mul
        )

        if per_channel:
            y = g.op("Mul", y, correction)

        y = g.op("Add", y, bias_fp)
        y = g.op("Cast", y, to_i=int(out_to_i))
        y.setType(x.type())
        return y


def quant_linear_symmetric_int8(x, weight_fp, bias_fp, x_scale, w_q, w_scale, smooth_scale=None):
    """Quantized linear wrapper: pre-compute correction & mean, dispatch to apply()."""
    out_to_i = int(_onnx_cast_to_i_from_dtype(x.dtype))
    x_scale_f = float(x_scale)

    if smooth_scale is not None:
        smooth_scale = smooth_scale.to(x.dtype)

    if isinstance(w_scale, torch.Tensor) and w_scale.dim() == 1:
        w_scale_np = w_scale.detach().cpu().numpy()
        w_scale_mean = float(w_scale_np.mean())
        correction = (w_scale_np / w_scale_mean).astype(np.float32)
        correction = torch.from_numpy(correction)
    else:
        if isinstance(w_scale, torch.Tensor):
            w_scale_mean = float(w_scale.cpu().item())
        else:
            w_scale_mean = float(w_scale)
        correction = None

    return _QuantLinearSymInt8.apply(
        x, weight_fp, bias_fp, x_scale_f, w_q, correction,
        w_scale_mean, smooth_scale, out_to_i,
    )


def _linear_ptq_aware(linear_mod, x):
    """Linear dispatch: quant_linear_symmetric_int8 if PTQ params attached, else F.linear."""
    if TORCH_PTQ_INT8 and hasattr(linear_mod, "_ptq_w_q"):
        b = linear_mod.bias
        if b is None:
            b = x.new_zeros((linear_mod.weight.shape[0],))
        smooth = getattr(linear_mod, "_ptq_smooth_scale", None)
        return quant_linear_symmetric_int8(
            x, linear_mod.weight, b,
            linear_mod._ptq_x_scale,
            linear_mod._ptq_w_q,
            linear_mod._ptq_w_scale,
            smooth_scale=smooth,
        )
    return linear_mod(x)


def _mlp_ptq_aware_forward(mlp_mod, x):
    """MLP forward with PTQ-aware gate/up/down projections."""
    if TORCH_PTQ_INT8 and hasattr(mlp_mod.gate_proj, "_ptq_w_q"):
        gate = _linear_ptq_aware(mlp_mod.gate_proj, x)
        up = _linear_ptq_aware(mlp_mod.up_proj, x)
        gate_up = mlp_mod.act_fn(gate) * up
        return _linear_ptq_aware(mlp_mod.down_proj, gate_up)
    return mlp_mod(x)


# ---------------------------------------------------------------------------
# PTQ calibration & quantization-parameter computation
# ---------------------------------------------------------------------------


def _load_calib_records_jsonl(path, max_samples):
    """Load up to ``max_samples`` calibration records from a JSONL file."""
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


def _make_synthetic_calib_records(num_samples, seq_len):
    """Synthetic records when no JSONL provided — keeps export runnable end-to-end."""
    records = []
    seq_len = int(seq_len)
    for _ in range(int(num_samples)):
        input_ids = torch.randint(0, 1000, (1, seq_len), dtype=torch.int64).tolist()
        attention_mask = torch.ones(1, seq_len, dtype=torch.int64).tolist()
        position_ids = torch.arange(seq_len, dtype=torch.int64).view(1, -1).tolist()
        gen_len = max(4, min(64, seq_len))
        generated_ids = torch.randint(0, 1000, (gen_len,), dtype=torch.int64).tolist()
        records.append({
            "prefill_input_ids": input_ids,
            "prefill_attention_mask": attention_mask,
            "prefill_position_ids": position_ids,
            "generated_ids": generated_ids,
        })
    return records


def _setup_linear_hooks(decode, device):
    """Attach MinMax observers + per-channel max trackers to every Linear."""
    from torch.ao.quantization.observer import MinMaxObserver

    hooks = []
    linear_modules = []
    for m in decode.modules():
        if isinstance(m, torch.nn.Linear):
            linear_modules.append(m)
            obs = MinMaxObserver(
                dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
            ).to(device)
            m._ptq_act_obs = obs

            def _make_pre_hook():
                def _pre_hook(mod, inputs):
                    x = inputs[0]
                    mod._ptq_act_obs(x.detach())
                    flat_x = x.detach().reshape(-1, x.shape[-1])
                    per_ch = flat_x.abs().max(dim=0).values
                    if not hasattr(mod, "_ptq_act_per_ch_max") or mod._ptq_act_per_ch_max is None:
                        mod._ptq_act_per_ch_max = per_ch
                    else:
                        mod._ptq_act_per_ch_max = torch.maximum(mod._ptq_act_per_ch_max, per_ch)
                return _pre_hook

            hooks.append(m.register_forward_pre_hook(_make_pre_hook()))
    return hooks, linear_modules


def _run_calibration_loop(prefill, decode, device, calib_records, max_decode_steps):
    """Replay prefill + decode on calibration records to collect activation stats."""
    with torch.no_grad():
        for rec in calib_records:
            prefill_input_ids = rec.get("prefill_input_ids")
            prefill_attention_mask = rec.get("prefill_attention_mask")
            prefill_position_ids = rec.get("prefill_position_ids")
            gen_ids = rec.get("generated_ids")
            if prefill_input_ids is None or prefill_attention_mask is None or prefill_position_ids is None:
                continue

            input_ids = torch.tensor(prefill_input_ids, dtype=torch.int64, device=device)
            attention_mask = torch.tensor(prefill_attention_mask, dtype=torch.int64, device=device)
            position_ids = torch.tensor(prefill_position_ids, dtype=torch.int64, device=device)
            _, past_kv = prefill(input_ids, attention_mask, position_ids)
            # past_kv shape: (2*num_layers, 1, num_kv_heads, prefill_seq, head_dim)

            actual_len = int(attention_mask[0].sum().item())
            valid_len = actual_len

            if not gen_ids:
                continue
            gen_ids = [int(x) for x in gen_ids]
            steps = min(int(max_decode_steps), max(len(gen_ids) - 1, 1))
            for _ in range(steps):
                if valid_len >= int(PTQ_CALIB_KV_LEN):
                    break
                # Build decode inputs with KV cache padded/used as-is.
                # past_kv already has the right shape from prefill / previous decode.
                cur_attn_len = past_kv.shape[3] + 1
                next_input_ids = torch.tensor([[gen_ids[0]]], dtype=torch.int64, device=device)
                next_attention_mask = torch.ones((1, cur_attn_len), dtype=torch.int64, device=device)
                next_position_ids = torch.tensor([[valid_len]], dtype=torch.int64, device=device)
                _, past_kv = decode(next_input_ids, next_attention_mask, next_position_ids, past_kv)
                valid_len += 1


def _quantize_one_weight(m, maxabs, per_ch_max, device):
    """Compute per-channel int8 quant params (optionally SmoothQuant-scaled) and attach to ``m``."""
    w = m.weight.detach()
    if per_ch_max is not None and w.dim() >= 2:
        max_w_per_col = w.abs().max(dim=0).values
        smooth_scale = _compute_smooth_quant_scale(per_ch_max, max_w_per_col, alpha=SMOOTH_ALPHA)
        smooth_scale = smooth_scale.to(device=device, dtype=torch.float32)
        w_smoothed = w * smooth_scale.unsqueeze(0)
        max_act_smoothed = (per_ch_max / smooth_scale).max().clamp_min(1e-8)
        x_scale_smoothed = float((max_act_smoothed / 127.0).cpu().item())
        w_q, w_scale = _quantize_weight_symmetric_int8(
            w_smoothed, per_channel=True, clip_ratio=WEIGHT_CLIP_RATIO
        )
        m._ptq_smooth_scale = smooth_scale.detach().cpu()
        m._ptq_x_scale = x_scale_smoothed
    else:
        x_scale = float((maxabs / 127.0).to(dtype=torch.float32).cpu().item())
        w_q, w_scale = _quantize_weight_symmetric_int8(
            w, per_channel=True, clip_ratio=WEIGHT_CLIP_RATIO
        )
        m._ptq_x_scale = x_scale
    m._ptq_w_q = w_q
    m._ptq_w_scale = w_scale


def _quantize_linear_modules(linear_modules, device):
    """Quantize every observed Linear module and clean up its activation observers."""
    for m in linear_modules:
        obs = getattr(m, "_ptq_act_obs", None)
        if obs is None or obs.max_val is None or obs.min_val is None:
            maxabs = torch.tensor(1.0, device=device, dtype=torch.float32)
            per_ch_max = None
        else:
            maxabs = torch.maximum(obs.max_val.abs(), obs.min_val.abs()).to(torch.float32).clamp_min(1e-8)
            per_ch_max = getattr(m, "_ptq_act_per_ch_max", None)
        _quantize_one_weight(m, maxabs, per_ch_max, device)
        for attr in ["_ptq_act_obs", "_ptq_act_per_ch_max"]:
            if hasattr(m, attr):
                delattr(m, attr)


def _torch_ptq_static_int8_quantize_decode(prefill, decode, device, calib_records,
                                            max_decode_steps):
    """Run calibration + compute quant params, attach to decode's Linear modules."""
    decode.eval()
    prefill.eval()

    hooks, linear_modules = _setup_linear_hooks(decode, device)
    _run_calibration_loop(
        prefill, decode, device, calib_records, max_decode_steps,
    )
    for h in hooks:
        h.remove()
    _quantize_linear_modules(linear_modules, device)
    return decode


def _qwen3_rotary_emb_matmul2d(rotary_emb, x, position_ids):
    """Qwen3 rotary cos/sin with matmul-friendly tensor shapes (matches jina_v3)."""
    rope_type = getattr(rotary_emb, "rope_type", "") or getattr(
        rotary_emb, "config", None
    ) and getattr(rotary_emb.config, "rope_type", "")
    if rope_type == "dynamic":
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
            rotary_emb.register_buffer(
                "inv_freq", rotary_emb.original_inv_freq, persistent=False
            )
            rotary_emb.max_seq_len_cached = rotary_emb.original_max_seq_len

    inv_freq = rotary_emb.inv_freq.to(device=x.device, dtype=torch.float32)
    position_ids_f = position_ids.to(dtype=torch.float32)
    bsz, seq_len = position_ids_f.shape
    freqs = (position_ids_f.reshape(-1, 1) @ inv_freq.reshape(1, -1)).reshape(
        bsz, seq_len, -1
    )
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * rotary_emb.attention_scaling
    sin = emb.sin() * rotary_emb.attention_scaling
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def _make_bool_causal_mask(attention_mask, q_len, k_len, past_len):
    """Boolean causal mask (True=masked) for CANN PromptFlashAttention."""
    device = attention_mask.device
    ar_q = torch.arange(q_len, device=device)
    ar_k = torch.arange(k_len, device=device)
    causal = ar_k[None, :] > (past_len + ar_q[:, None])
    padding = ~attention_mask.to(torch.bool)
    mask = causal[None, None, :, :].expand(attention_mask.shape[0], 1, q_len, k_len) | padding[:, None, None, :]
    return mask


def _make_additive_causal_mask(attention_mask, q_len, k_len, past_len, dtype):
    """
    Make additive causal mask for ONNX inference.
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
    attn_mod, hidden_states, position_embeddings, attention_mask, past_key, past_value
):
    """
    Forward pass for text attention.
    """
    input_shape = hidden_states.shape[:-1]
    head_dim = attn_mod.head_dim
    num_heads = attn_mod.config.num_attention_heads
    num_kv_heads = attn_mod.config.num_key_value_heads
    hidden_shape = (*input_shape, -1, head_dim)

    query_states = _linear_ptq_aware(attn_mod.q_proj, hidden_states).view(hidden_shape)
    key_states = _linear_ptq_aware(attn_mod.k_proj, hidden_states).view(hidden_shape)
    if hasattr(attn_mod, "q_norm"):
        query_states = attn_mod.q_norm(query_states)
    if hasattr(attn_mod, "k_norm"):
        key_states = attn_mod.k_norm(key_states)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = _linear_ptq_aware(attn_mod.v_proj, hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    if apply_rotary_pos_emb is not None:
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

    if past_key is not None:
        key_states = torch.cat([past_key, key_states], dim=2)
        value_states = torch.cat([past_value, value_states], dim=2)

    key_states_for_attn = key_states
    value_states_for_attn = value_states

    if num_kv_heads < num_heads:
        key_states_for_attn = key_states.repeat_interleave(
            num_heads // num_kv_heads, dim=1
        )
        value_states_for_attn = value_states.repeat_interleave(
            num_heads // num_kv_heads, dim=1
        )

    scaling = getattr(attn_mod, "scaling", 1.0 / (head_dim**0.5))
    attn_weights = (
        torch.matmul(query_states, key_states_for_attn.transpose(2, 3)) * scaling
    )
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = torch.nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states_for_attn)
    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1)
    attn_output = _linear_ptq_aware(attn_mod.o_proj, attn_output)
    return attn_output, key_states, value_states


# ---------------------------------------------------------------------------
# CANN-fused attention / mlp / norm helpers
# ---------------------------------------------------------------------------


def _cann_attn_forward(
    attn_mod,
    hidden_states,
    position_embeddings,
    bool_mask,
    past_key=None,
    past_value=None,
    enable_rotarymul=True,
    enable_pfa=True,
    enable_bmm2mm=False,
):
    """Attention forward with per-op fusion switches.

    enable_rotarymul: route RoPE through Custom(RotaryMul); else use transformers
                      apply_rotary_pos_emb (explicit PyTorch ops).
    enable_pfa:       route QK^T+softmax+V through Custom(PromptFlashAttention);
                      else use explicit matmul + softmax + matmul (with GQA repeat).
    """
    input_shape = hidden_states.shape[:-1]
    head_dim = attn_mod.head_dim
    num_heads = attn_mod.config.num_attention_heads
    num_kv_heads = attn_mod.config.num_key_value_heads
    hidden_shape = (*input_shape, -1, head_dim)

    query_states = _linear(attn_mod.q_proj, hidden_states, enable_bmm2mm).view(
        hidden_shape
    )
    key_states = _linear(attn_mod.k_proj, hidden_states, enable_bmm2mm).view(
        hidden_shape
    )
    if hasattr(attn_mod, "q_norm"):
        query_states = attn_mod.q_norm(query_states)
    if hasattr(attn_mod, "k_norm"):
        key_states = attn_mod.k_norm(key_states)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = (
        _linear(attn_mod.v_proj, hidden_states, enable_bmm2mm)
        .view(hidden_shape)
        .transpose(1, 2)
    )

    cos, sin = position_embeddings
    if enable_rotarymul:
        cos, sin = _expand_rotary_cos_sin(cos, sin, 4)
        query_states, key_states = _cann_apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
    else:
        if apply_rotary_pos_emb is not None:
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

    if past_key is not None:
        key_states = torch.cat([past_key, key_states], dim=2)
        value_states = torch.cat([past_value, value_states], dim=2)

    scaling = getattr(attn_mod, "scaling", 1.0 / (head_dim**0.5))

    if enable_pfa:
        attn_output = _CannPromptFlashAttention.apply(
            query_states,
            key_states,
            value_states,
            bool_mask,
            int(num_heads),
            int(num_kv_heads),
            float(scaling),
        )
    else:
        key_states_for_attn = key_states
        value_states_for_attn = value_states
        if num_kv_heads < num_heads:
            rep = num_heads // num_kv_heads
            key_states_for_attn = key_states.repeat_interleave(rep, dim=1)
            value_states_for_attn = value_states.repeat_interleave(rep, dim=1)
        attn_weights = torch.matmul(
            query_states, key_states_for_attn.transpose(2, 3)
        ) * scaling
        if bool_mask is not None:
            mask_value = torch.finfo(query_states.dtype).min
            additive = torch.where(
                bool_mask, torch.full((), mask_value, dtype=query_states.dtype),
                torch.zeros((), dtype=query_states.dtype),
            )
            attn_weights = attn_weights + additive
        attn_weights = torch.nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states_for_attn)

    # Both paths return [B,N,S,D], need transpose to [B,S,N,D]
    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = _linear(attn_mod.o_proj, attn_output, enable_bmm2mm)
    return attn_output, key_states, value_states


def _cann_mlp_forward(
    mlp_mod, hidden_states, enable_swiglu=True, enable_bmm2mm=False
):
    """MLP forward with per-op fusion switches.

    enable_swiglu=True : cat([gate, up]) + Custom(SwiGlu).
    enable_swiglu=False: silu(gate) * up directly — no cat, avoids the
                         FRACTAL_NZ→ND TransData that cat forces (defusion).
    """
    if enable_swiglu:
        gate_up = torch.cat(
            [
                _linear(mlp_mod.gate_proj, hidden_states, enable_bmm2mm),
                _linear(mlp_mod.up_proj, hidden_states, enable_bmm2mm),
            ],
            dim=-1,
        )
        gate_up = _CannSwiGlu.apply(gate_up, -1)
    else:
        gate = _linear(mlp_mod.gate_proj, hidden_states, enable_bmm2mm)
        up = _linear(mlp_mod.up_proj, hidden_states, enable_bmm2mm)
        gate_up = torch.nn.functional.silu(gate) * up
    return _linear(mlp_mod.down_proj, gate_up, enable_bmm2mm)


def _get_rmsnorm_epsilon(norm_mod):
    for attr in ("variance_epsilon", "eps", "epsilon"):
        val = getattr(norm_mod, attr, None)
        if val is not None:
            return float(val)
    return 1e-6


def _replace_rmsnorm_with_cann(model):
    """Swap Qwen3 RMSNorm modules for CANN-exportable wrappers (in-place)."""
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
    except Exception:
        return

    for layer in model.model.layers:
        for name in ("input_layernorm", "post_attention_layernorm"):
            mod = getattr(layer, name, None)
            if isinstance(mod, Qwen3RMSNorm) and hasattr(mod, "weight"):
                setattr(
                    layer, name, CannRmsNorm(mod.weight, _get_rmsnorm_epsilon(mod))
                )
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            for name in ("q_norm", "k_norm"):
                mod = getattr(attn, name, None)
                if isinstance(mod, Qwen3RMSNorm) and hasattr(mod, "weight"):
                    setattr(
                        attn, name, CannRmsNorm(mod.weight, _get_rmsnorm_epsilon(mod))
                    )

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


class Qwen3LlmPrefill(torch.nn.Module):
    """
    Qwen3-0.6B LLM Preffill model for ONNX inference.
    """

    def __init__(self, model, lm_head):
        """
        Initialize Qwen3LlmPrefill.
        """
        super().__init__()
        self.model = model.model
        self.lm_head = lm_head
        self.num_hidden_layers = model.config.num_hidden_layers

    def forward(self, input_ids, attention_mask, position_ids):
        """
        Forward pass for Qwen3LlmPrefill.
        """
        inputs_embeds = self.model.embed_tokens(input_ids)
        q_len = input_ids.shape[1]

        position_embeddings = self.model.rotary_emb(inputs_embeds, position_ids)
        k_len = q_len
        attn_mask = _make_additive_causal_mask(
            attention_mask, q_len, k_len, 0, inputs_embeds.dtype
        )

        hidden_states = inputs_embeds
        present = []

        for layer in self.model.layers:
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            attn_out, pk, pv = _text_attn_forward(
                layer.self_attn,
                hidden_states,
                position_embeddings,
                attn_mask,
                None,
                None,
            )
            hidden_states = residual + attn_out
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = residual + _mlp_ptq_aware_forward(layer.mlp, hidden_states)
            present.append(pk)
            present.append(pv)

        hidden_states = self.model.norm(hidden_states)
        logits = _linear_ptq_aware(self.lm_head, hidden_states)
        present_kv = torch.stack(present, dim=0)
        return logits, present_kv


class Qwen3LlmDecode(torch.nn.Module):
    """
    Qwen3-0.6B LLM Decode model for ONNX inference.
    """

    def __init__(self, model, lm_head):
        """
        Initialize Qwen3LlmDecode.
        """
        super().__init__()
        self.model = model.model
        self.lm_head = lm_head
        self.num_hidden_layers = model.config.num_hidden_layers

    def forward(self, input_ids, attention_mask, position_ids, past_key_values):
        """
        Forward pass for Qwen3LlmDecode.
        """
        inputs_embeds = self.model.embed_tokens(input_ids)
        q_len = input_ids.shape[1]

        position_embeddings = self.model.rotary_emb(inputs_embeds, position_ids)
        past_key_0 = past_key_values[0]
        past_len = past_key_0.shape[2]
        k_len = past_len + q_len
        attn_mask = _make_additive_causal_mask(
            attention_mask, q_len, k_len, past_len, inputs_embeds.dtype
        )

        hidden_states = inputs_embeds
        present = []

        for i, layer in enumerate(self.model.layers):
            pk_in = past_key_values[2 * i]
            pv_in = past_key_values[2 * i + 1]
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            attn_out, pk, pv = _text_attn_forward(
                layer.self_attn,
                hidden_states,
                position_embeddings,
                attn_mask,
                pk_in,
                pv_in,
            )
            hidden_states = residual + attn_out
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = residual + _mlp_ptq_aware_forward(layer.mlp, hidden_states)
            present.append(pk)
            present.append(pv)

        hidden_states = self.model.norm(hidden_states)
        logits = _linear_ptq_aware(self.lm_head, hidden_states)
        present_kv = torch.stack(present, dim=0)
        return logits, present_kv


# ---------------------------------------------------------------------------
# Fused wrappers (route RotaryMul / PFA / SwiGlu / RmsNorm / AddRmsNorm / MatMulV2 to CANN Custom ops)
# ---------------------------------------------------------------------------


class Qwen3LlmPrefillFused(torch.nn.Module):
    """Prefill wrapper with CANN fused Custom ops (per-op switches)."""

    def __init__(self, model, lm_head, flags):
        super().__init__()
        self.embed_tokens = model.model.embed_tokens
        self.layers = model.model.layers
        self.norm = model.model.norm
        self.rotary_emb = model.model.rotary_emb
        self.lm_head = lm_head
        self.num_layers = len(self.layers)
        self.flags = flags

    def forward(self, input_ids, attention_mask, position_ids):
        """Run prefill forward pass: embeddings + transformer layers + norm + lm_head."""
        q_len = input_ids.shape[1]
        inputs_embeds = self.embed_tokens(input_ids)
        position_embeddings = _qwen3_rotary_emb_matmul2d(
            self.rotary_emb, inputs_embeds, position_ids
        )

        bool_mask = _make_bool_causal_mask(attention_mask, q_len, q_len, 0)

        hidden_states = inputs_embeds
        residual = hidden_states
        hidden_states = self.layers[0].input_layernorm(hidden_states)

        present = []
        for i, layer in enumerate(self.layers):
            attn_out, pk, pv = _cann_attn_forward(
                layer.self_attn,
                hidden_states,
                position_embeddings,
                bool_mask,
                None,
                None,
                self.flags["enable_rotarymul"],
                self.flags["enable_pfa"],
                self.flags["enable_bmm2mm"],
            )
            present.append(pk)
            present.append(pv)

            hidden_states, residual = _cann_add_rms_norm(
                residual,
                attn_out,
                layer.post_attention_layernorm,
                self.flags["enable_add_rmsnorm"],
            )

            mlp_out = _cann_mlp_forward(
                layer.mlp,
                hidden_states,
                self.flags["enable_swiglu"],
                self.flags["enable_bmm2mm"],
            )
            if i < self.num_layers - 1:
                hidden_states, residual = _cann_add_rms_norm(
                    residual,
                    mlp_out,
                    self.layers[i + 1].input_layernorm,
                    self.flags["enable_add_rmsnorm"],
                )
            else:
                hidden_states, _ = _cann_add_rms_norm(
                    residual, mlp_out, self.norm, self.flags["enable_add_rmsnorm"]
                )

        logits = self.lm_head(hidden_states)
        present_kv = torch.stack(present, dim=0)
        return logits, present_kv


class Qwen3LlmDecodeFused(torch.nn.Module):
    """Decode wrapper with CANN fused Custom ops (per-op switches)."""

    def __init__(self, model, lm_head, flags):
        super().__init__()
        self.embed_tokens = model.model.embed_tokens
        self.layers = model.model.layers
        self.norm = model.model.norm
        self.rotary_emb = model.model.rotary_emb
        self.lm_head = lm_head
        self.num_layers = len(self.layers)
        self.flags = flags

    def forward(self, input_ids, attention_mask, position_ids, past_key_values):
        """Run single-step decode forward: embeddings + transformer layers + norm + lm_head."""
        q_len = input_ids.shape[1]
        inputs_embeds = self.embed_tokens(input_ids)
        position_embeddings = _qwen3_rotary_emb_matmul2d(
            self.rotary_emb, inputs_embeds, position_ids
        )

        past_key_0 = past_key_values[0]
        past_len = past_key_0.shape[2]
        k_len = past_len + q_len
        bool_mask = _make_bool_causal_mask(attention_mask, q_len, k_len, past_len)

        hidden_states = inputs_embeds
        residual = hidden_states
        hidden_states = self.layers[0].input_layernorm(hidden_states)

        present = []
        for i, layer in enumerate(self.layers):
            pk_in = past_key_values[2 * i]
            pv_in = past_key_values[2 * i + 1]
            attn_out, pk, pv = _cann_attn_forward(
                layer.self_attn,
                hidden_states,
                position_embeddings,
                bool_mask,
                pk_in,
                pv_in,
                self.flags["enable_rotarymul"],
                False,  # decode 不使用 PFA（GQA 与 CANN PFA 不兼容）
                self.flags["enable_bmm2mm"],
            )
            present.append(pk)
            present.append(pv)

            hidden_states, residual = _cann_add_rms_norm(
                residual,
                attn_out,
                layer.post_attention_layernorm,
                self.flags["enable_add_rmsnorm"],
            )

            mlp_out = _cann_mlp_forward(
                layer.mlp,
                hidden_states,
                self.flags["enable_swiglu"],
                self.flags["enable_bmm2mm"],
            )
            if i < self.num_layers - 1:
                hidden_states, residual = _cann_add_rms_norm(
                    residual,
                    mlp_out,
                    self.layers[i + 1].input_layernorm,
                    self.flags["enable_add_rmsnorm"],
                )
            else:
                hidden_states, _ = _cann_add_rms_norm(
                    residual, mlp_out, self.norm, self.flags["enable_add_rmsnorm"]
                )

        logits = self.lm_head(hidden_states)
        present_kv = torch.stack(present, dim=0)
        return logits, present_kv


def export_llm_prefill_decode(model, output_dir, device="cpu", flags=None):
    """Export Qwen3-0.6B LLM prefill and decode models to ONNX.

    flags: dict of per-op fusion switches. Defaults to all-off (non-fused
    baseline). When any flag is on, the Fused wrapper is used; otherwise the
    plain Qwen3Llm{Prefill,Decode} classes are used.
    """
    default_flags = {
        "enable_rmsnorm_replace": False,
        "enable_add_rmsnorm": False,
        "enable_rotarymul": True,
        "enable_pfa": True,
        "enable_swiglu": False,
        "enable_bmm2mm": False,
    }
    if flags:
        default_flags.update(flags)
    flags = default_flags

    lm_head = model.lm_head
    model.eval()
    lm_head.eval()
    model.to(device)
    lm_head.to(device)

    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    head_dim = getattr(
        model.config,
        "head_dim",
        model.config.hidden_size // model.config.num_attention_heads,
    )

    any_fusion_on = any(flags.values())
    if any_fusion_on:
        if flags["enable_rmsnorm_replace"]:
            _replace_rmsnorm_with_cann(model)
        prefill = Qwen3LlmPrefillFused(model, lm_head, flags).to(device).eval()
        decode = Qwen3LlmDecodeFused(model, lm_head, flags).to(device).eval()
        suffix = ""
        enabled = [k for k, v in flags.items() if v]
        print(f"Fusion opt ON: {', '.join(enabled)}")
    else:
        prefill = Qwen3LlmPrefill(model, lm_head).to(device).eval()
        decode = Qwen3LlmDecode(model, lm_head).to(device).eval()
        suffix = ""

    prefill_path = Path(output_dir) / f"qwen3_llm_prefill{suffix}.onnx"
    decode_path = Path(output_dir) / f"qwen3_llm_decode{suffix}.onnx"
    if TORCH_PTQ_INT8 and not any_fusion_on:
        # PTQ only supported on the non-fused path (fused path mixes Custom
        # fusion ops with quant Custom ops, which converter_lite rejects).
        decode_path = Path(output_dir) / f"qwen3_llm_decode{suffix}_ptq_int8.onnx"

    dummy_seq = 8
    dummy_input_ids = torch.randint(
        0, 1000, (1, dummy_seq), dtype=torch.int64, device=device
    )
    dummy_attention_mask = torch.ones(1, dummy_seq, dtype=torch.int64, device=device)
    dummy_position_ids = torch.arange(dummy_seq, device=device, dtype=torch.int64).view(
        1, -1
    )

    prefill_input_names = ["input_ids", "attention_mask", "position_ids"]
    prefill_output_names = ["logits", "present_key_values"]
    prefill_dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq_len"},
        "attention_mask": {0: "batch", 1: "seq_len"},
        "position_ids": {0: "batch", 1: "seq_len"},
        "logits": {0: "batch", 1: "seq_len"},
        "present_key_values": {1: "batch", 3: "seq_len"},
    }

    print(f"Exporting LLM prefill to {prefill_path}...")
    with torch.no_grad():
        torch.onnx.export(
            prefill,
            (dummy_input_ids, dummy_attention_mask, dummy_position_ids),
            str(prefill_path),
            input_names=prefill_input_names,
            output_names=prefill_output_names,
            opset_version=18,
            do_constant_folding=True,
            dynamic_axes=prefill_dynamic_axes,
        )
    print("LLM prefill exported successfully.")

    # Run PTQ calibration + attach quant params to decode's Linear modules.
    # Must happen AFTER prefill export (so prefill stays FP32) and BEFORE
    # decode export (so decode's ONNX contains AscendQuant/QuantBatchMatmul).
    if TORCH_PTQ_INT8 and not any_fusion_on:
        print(f"\nRunning PTQ int8 calibration (calib={TORCH_PTQ_CALIB_JSONL or '<synthetic>'}, "
              f"max_samples={TORCH_PTQ_MAX_SAMPLES}, max_decode_steps={TORCH_PTQ_MAX_DECODE_STEPS}, "
              f"smooth_alpha={SMOOTH_ALPHA}, weight_clip_ratio={WEIGHT_CLIP_RATIO})...")
        calib_records = _load_calib_records_jsonl(TORCH_PTQ_CALIB_JSONL, TORCH_PTQ_MAX_SAMPLES)
        if not calib_records:
            print("Warning: no calibration JSONL provided; using synthetic records.")
            calib_records = _make_synthetic_calib_records(
                num_samples=min(4, int(TORCH_PTQ_MAX_SAMPLES)),
                seq_len=max(8, dummy_seq),
            )
        decode = _torch_ptq_static_int8_quantize_decode(
            prefill=prefill,
            decode=decode,
            device=torch.device(device),
            calib_records=calib_records,
            max_decode_steps=TORCH_PTQ_MAX_DECODE_STEPS,
        )
        print(f"PTQ int8 quant params attached. Decode will export to {decode_path.name}")

    dummy_step = 1
    dummy_past_len = dummy_seq
    dummy_input_ids_step = torch.randint(
        0, 1000, (1, dummy_step), dtype=torch.int64, device=device
    )
    dummy_attention_mask_step = torch.ones(
        1, dummy_past_len + dummy_step, dtype=torch.int64, device=device
    )
    dummy_position_ids_step = torch.tensor(
        [[dummy_past_len]], dtype=torch.int64, device=device
    )
    dummy_past = torch.zeros(
        2 * num_layers,
        1,
        num_kv_heads,
        dummy_past_len,
        head_dim,
        dtype=torch.float16,
        device=device,
    )

    decode_input_names = [
        "input_ids",
        "attention_mask",
        "position_ids",
        "past_key_values",
    ]
    decode_output_names = ["logits", "present_key_values"]

    decode_dynamic_axes = {
        "input_ids": {0: "batch", 1: "step"},
        "attention_mask": {0: "batch", 1: "total_seq_len"},
        "position_ids": {0: "batch", 1: "step"},
        "logits": {0: "batch", 1: "step"},
        "past_key_values": {1: "batch", 3: "past_seq_len"},
        "present_key_values": {1: "batch", 3: "total_seq_len"},
    }

    print(f"Exporting LLM decode to {decode_path}...")
    with torch.no_grad():
        torch.onnx.export(
            decode,
            (
                dummy_input_ids_step,
                dummy_attention_mask_step,
                dummy_position_ids_step,
                dummy_past,
            ),
            str(decode_path),
            input_names=decode_input_names,
            output_names=decode_output_names,
            opset_version=18,
            do_constant_folding=True,
            dynamic_axes=decode_dynamic_axes,
        )
    print("LLM decode exported successfully.")


def main():
    """
    Main function to export Qwen3-0.6B LLM prefill and decode models to ONNX format.
    """
    parser = argparse.ArgumentParser(description="Export Qwen3-0.6B to ONNX")
    parser.add_argument(
        "--model-id", type=str, default="Qwen/Qwen3-0.6B", help="HuggingFace model ID"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./qwen3_onnx", help="Output directory"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device for export (cpu or cuda)"
    )
    parser.add_argument(
        "--enable-rmsnorm-replace",
        action="store_true",
        help="Replace Qwen3RMSNorm modules with CANN CannRmsNorm (Custom RmsNorm op).",
    )
    parser.add_argument(
        "--enable-add-rmsnorm",
        action="store_true",
        help="Fuse residual+RmsNorm into CANN AddRmsNorm inside fused wrapper.",
    )
    parser.add_argument(
        "--enable-rotarymul",
        action="store_true",
        default=True,
        help="Route RoPE through Custom(RotaryMul) inside fused wrapper (default: ON).",
    )
    parser.add_argument(
        "--enable-pfa",
        action="store_true",
        default=True,
        help="Route QK^T+softmax+V through Custom(PromptFlashAttention) (default: ON).",
    )
    parser.add_argument(
        "--enable-all-fusion",
        action="store_true",
        help="Convenience: turn on all per-op fusion switches.",
    )
    parser.add_argument(
        "--disable-fusion",
        action="store_true",
        help="Disable all fusion (export non-fused baseline).",
    )
    parser.add_argument(
        "--enable-swiglu",
        action="store_true",
        help="Route MLP SiLU(gate)*up through Custom(SwiGlu) (uses cat([gate,up])).",
    )
    parser.add_argument(
        "--enable-bmm2mm",
        action="store_true",
        help="Lower BatchMatMul to MatMulV2 (CANN Custom) for Linear layers.",
    )
    parser.add_argument(
        "--torch-ptq-int8",
        action="store_true",
        help="Enable PTQ int8 quantization for decode (AscendQuant + QuantBatchMatmul).",
    )
    parser.add_argument(
        "--disable-torch-ptq-int8",
        action="store_true",
        help="(Default) Disable PTQ int8 quantization; export plain FP32/fp16 decode.",
    )
    parser.add_argument(
        "--torch-ptq-calib-jsonl",
        type=str,
        default="",
        help="Calibration data JSONL file path (one record per line).",
    )
    parser.add_argument(
        "--torch-ptq-max-samples", type=int, default=32,
        help="Max number of calibration samples to use.",
    )
    parser.add_argument(
        "--torch-ptq-max-decode-steps", type=int, default=32,
        help="Max decode steps per calibration sample.",
    )
    parser.add_argument(
        "--smooth-alpha", type=float, default=0.5,
        help="SmoothQuant alpha (0=no activation smoothing, 1=full).",
    )
    parser.add_argument(
        "--weight-clip-ratio", type=float, default=0.0,
        help="Clip top fraction of weight outliers before quantization (0=off).",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.disable_fusion:
        flags = {
            "enable_rmsnorm_replace": False,
            "enable_add_rmsnorm": False,
            "enable_rotarymul": False,
            "enable_pfa": False,
            "enable_swiglu": False,
            "enable_bmm2mm": False,
        }
    else:
        flags = {
            "enable_rmsnorm_replace": args.enable_rmsnorm_replace or args.enable_all_fusion,
            "enable_add_rmsnorm": args.enable_add_rmsnorm or args.enable_all_fusion,
            "enable_rotarymul": args.enable_rotarymul or args.enable_all_fusion,
            "enable_pfa": args.enable_pfa or args.enable_all_fusion,
            "enable_swiglu": args.enable_swiglu or args.enable_all_fusion,
            "enable_bmm2mm": args.enable_bmm2mm or args.enable_all_fusion,
        }

    # Wire PTQ CLI args into module-level globals consumed by export_llm_prefill_decode.
    global TORCH_PTQ_INT8, TORCH_PTQ_CALIB_JSONL, TORCH_PTQ_MAX_SAMPLES
    global TORCH_PTQ_MAX_DECODE_STEPS, SMOOTH_ALPHA, WEIGHT_CLIP_RATIO
    if args.disable_torch_ptq_int8 and args.torch_ptq_int8:
        print("Error: --torch-ptq-int8 and --disable-torch-ptq-int8 are mutually exclusive.")
        sys.exit(1)
    TORCH_PTQ_INT8 = bool(args.torch_ptq_int8) and not bool(args.disable_torch_ptq_int8)
    TORCH_PTQ_CALIB_JSONL = args.torch_ptq_calib_jsonl
    TORCH_PTQ_MAX_SAMPLES = int(args.torch_ptq_max_samples)
    TORCH_PTQ_MAX_DECODE_STEPS = int(args.torch_ptq_max_decode_steps)
    SMOOTH_ALPHA = float(args.smooth_alpha)
    WEIGHT_CLIP_RATIO = float(args.weight_clip_ratio)

    print(f"\nLoading model {args.model_id} in FP16 for export...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map=args.device,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )

    export_llm_prefill_decode(model, output_dir, args.device, flags=flags)

    print("Clearing memory after export...")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nExport finished. Files saved in {args.output_dir}")


if __name__ == "__main__":
    main()
