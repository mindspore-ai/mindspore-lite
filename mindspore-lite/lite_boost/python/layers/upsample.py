#!/usr/bin/env python3
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""NPU-native upsampling primitives.

The bf16→fp32 cast around nearest interpolation in framework VAE
``Upsample`` layers only exists to work around CPU/CUDA nearest-exact
limits.  On NPU the UpsampleNearest kernel handles bf16 natively on most
SoCs (910B etc.), but the 310P3 binary set has NO bf16 kernel (probe:
"Cannot find bin of op UpsampleNearest ... bf16/ND/bf16/ND/").  This
primitive therefore runs all dtypes natively except on 310P3, where bf16
inputs fall back to the fp32-cast path.
"""
import torch
import torch.nn.functional as F
import torch_npu

__all__ = ["nearest_exact_upsample"]

_BF16_CAST_FALLBACK = None


def _needs_bf16_cast_fallback():
    """True only on 310P3 (no bf16 UpsampleNearest kernel).  Cached."""
    global _BF16_CAST_FALLBACK
    if _BF16_CAST_FALLBACK is None:
        try:
            name = torch_npu.npu.get_device_name(torch_npu.npu.current_device())
            _BF16_CAST_FALLBACK = "310P" in name
        except Exception:
            _BF16_CAST_FALLBACK = False
    return _BF16_CAST_FALLBACK


def nearest_exact_upsample(x, size=None, scale_factor=None):
    """Nearest-exact interpolate, cast-free on supported dtypes.

    Args:
        x:             Input tensor.
        size:          Output size (either *size* or *scale_factor*).
        scale_factor:  Multiplier for spatial dims.

    Returns:
        Upsampled tensor, same dtype as the input.

    Raises:
        TypeError:  ``x`` is not a tensor.
        ValueError: Input is not 4-D, or both/neither of *size* /
            *scale_factor* given.
    """
    if not torch.is_tensor(x):
        raise TypeError(
            "nearest_exact_upsample: x must be torch.Tensor, got "
            f"{type(x).__name__}")
    if x.dim() != 4:
        # 4-D (B, C, H, W) with a 2-D spatial scale is the only supported
        # shape; 1-D / 3-D spatial inputs (3-D / 5-D tensors) are rejected
        # even though some were measured to work.
        raise ValueError(
            "nearest_exact_upsample: supports 4-D (B, C, H, W) inputs "
            f"only, got dim={x.dim()} shape={tuple(x.shape)}")
    if (size is None) == (scale_factor is None):
        raise ValueError(
            "nearest_exact_upsample: exactly one of size / scale_factor "
            "must be provided")
    if _needs_bf16_cast_fallback() and x.dtype == torch.bfloat16:
        # 310P3 only — no bf16 UpsampleNearest kernel; cast round-trip
        # produces bitwise-identical results (nearest-exact is integer
        # indexing) and keeps the pipeline running in bf16.
        return F.interpolate(
            x.float(), size=size, scale_factor=scale_factor,
            mode="nearest-exact").type_as(x)
    return F.interpolate(
        x, size=size, scale_factor=scale_factor, mode="nearest-exact")
