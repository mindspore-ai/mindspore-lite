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
"""Offline SmoothQuant math; all returned weights keep their unpadded shapes."""

import torch

DEFAULT_ALPHA = 0.6


def compute_smooth_scale(weight, act_absmax, alpha):
    """SmoothQuant per-input-channel scale.

    ``scale[cin] = act_absmax[cin]^alpha / weight_absmax[cin]^(1-alpha)``,
    normalized to max 1 so the scaled weight cannot overflow.

    Args:
        weight: FP weight ``[cout, cin]``.
        act_absmax: fp32 ``[cin]`` activation absmax over all calibration
            samples.
        alpha: balance between activation and weight absorption (0.5 is the
            geometric mean; the validated Boogu-Image default is 0.6).

    Returns:
        fp32 ``[cin]`` smooth scale.
    """
    weight_absmax = weight.abs().amax(dim=0).clamp(min=1e-4).to(torch.float32)
    act = act_absmax.to(torch.float32).clamp(min=1e-4)
    scale = act.pow(alpha) / weight_absmax.pow(1.0 - alpha)
    scale = scale / scale.max().clamp(min=1e-4)
    return torch.clamp(scale, 1e-4, torch.finfo(torch.float32).max)


def quantize_weight_per_channel_int8(weight):
    """Symmetric per-output-channel MinMax int8 quantization.

    Args:
        weight: FP weight ``[cout, cin]`` (already smooth-scaled).

    Returns:
        ``(w_int8 [cout, cin] int8, scale fp32 [cout])``.
    """
    q_max, q_min = 127, -128
    w_absmax = weight.abs().amax(dim=1, keepdim=True)
    scale = (w_absmax / q_max).clamp(min=1e-9)
    w_int8 = (weight / scale).round().clamp(q_min, q_max).to(torch.int8)
    return w_int8, scale.reshape(-1).to(torch.float32)


def quantize_linear(linear, act_absmax, alpha):
    """Produce canonical [input, output] int8 tensors without runtime padding."""
    weight = linear.weight.detach()
    smooth = compute_smooth_scale(weight, act_absmax.to(weight.device), alpha)
    scaled = weight * smooth.unsqueeze(0).to(weight.dtype)
    quantized, scale = quantize_weight_per_channel_int8(scaled)
    tensors = {
        "qweight": quantized.t().contiguous().cpu(),
        "weight_scale": scale.cpu(),
        "inv_smooth_scale": smooth.reciprocal().to(torch.bfloat16).cpu(),
    }
    if linear.bias is not None:
        tensors["bias"] = linear.bias.detach().to(torch.bfloat16).cpu()
    return tensors


def quantize_fused_ffn(ffn, act_absmax, alpha):
    """Fold one shared scale into gate/up and concatenate without padding."""
    weight1 = ffn.linear_1.weight.detach()
    weight3 = ffn.linear_3.weight.detach()
    act = act_absmax.to(weight1.device)
    scale1 = compute_smooth_scale(weight1, act, alpha)
    scale3 = compute_smooth_scale(weight3, act, alpha)
    shared = (scale1 * scale3).sqrt()
    quant1, scale1 = quantize_weight_per_channel_int8(
        weight1 * shared.unsqueeze(0).to(weight1.dtype))
    quant3, scale3 = quantize_weight_per_channel_int8(
        weight3 * shared.unsqueeze(0).to(weight3.dtype))
    return {
        "qw13": torch.cat((quant1.t(), quant3.t()), dim=-1).contiguous().cpu(),
        "w13_scale": torch.cat((scale1, scale3)).cpu(),
        "inv_smooth13": shared.reciprocal().to(torch.bfloat16).cpu(),
    }
