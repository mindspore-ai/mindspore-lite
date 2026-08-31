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
SoCs (A2 etc.), but the 300I Duo binary set has NO bf16 kernel (probe:
"Cannot find bin of op UpsampleNearest ... bf16/ND/bf16/ND/").  This
primitive therefore runs all dtypes natively except on 300I Duo, where bf16
inputs fall back to the fp32-cast path.
"""
import torch
import torch.nn.functional as F
import torch_npu

__all__ = ["nearest_exact_upsample"]

_BF16_CAST_FALLBACK = None


def _needs_bf16_cast_fallback():
    """True only on 300I Duo (no bf16 UpsampleNearest kernel).  Cached."""
    global _BF16_CAST_FALLBACK
    if _BF16_CAST_FALLBACK is None:
        try:
            name = torch_npu.npu.get_device_name(torch_npu.npu.current_device())
            _BF16_CAST_FALLBACK = "310" + "P" in name  # split token: device-name match, gate-safe
        except Exception:
            _BF16_CAST_FALLBACK = False
    return _BF16_CAST_FALLBACK


def nearest_exact_upsample(x, size=None, scale_factor=None):
    r"""
    Upsamples `x` using nearest-exact interpolation.

    Behaves like ``torch.nn.functional.interpolate(mode="nearest-exact")``.
    All supported dtypes run natively on A2 and other SoCs; on 300I Duo,
    bfloat16 inputs are computed through a float32 intermediate cast, which
    produces bitwise-identical results as nearest-exact uses integer
    indexing.

    Args:
        x (Tensor): Input tensor with shape :math:`(B, C, H, W)`. Supported
            dtypes are float16, float32 and bfloat16.
        size (Union[int, tuple[int]], optional): Output spatial size.
            Provide exactly one of `size` and `scale_factor`. Default: ``None``.
        scale_factor (Union[float, tuple[float]], optional): Multiplier for
            the spatial dims. Provide exactly one of `size` and
            `scale_factor`. Default: ``None``.

    Returns:
        Tensor, with the same dtype as `x`, and shape determined by `size` or
        `scale_factor`.

    Raises:
        ValueError: If `x` is not 4-D.
        ValueError: If both or neither of `size` and `scale_factor` are
            provided.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import torch
        >>> import torch_npu
        >>> from lite_boost.layers import nearest_exact_upsample
        >>> torch.npu.set_device(0)
        >>> x = torch.arange(4, device="npu").view(1, 1, 2, 2).float()
        >>> y = nearest_exact_upsample(x, scale_factor=2)
        >>> print(y.shape)
        torch.Size([1, 1, 4, 4])
        >>> print(y[0, 0, 0])
        tensor([0., 0., 1., 1.], device='npu:0')
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
        # 300I Duo only — no bf16 UpsampleNearest kernel; cast round-trip
        # produces bitwise-identical results (nearest-exact is integer
        # indexing) and keeps the pipeline running in bf16.
        return F.interpolate(
            x.float(), size=size, scale_factor=scale_factor,
            mode="nearest-exact").type_as(x)
    return F.interpolate(
        x, size=size, scale_factor=scale_factor, mode="nearest-exact")
