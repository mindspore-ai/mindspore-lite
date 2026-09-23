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
NPU-safe kernel replacements for the ops Boogu-Image uses.

Model-agnostic implementations: NPU-safe SDPA (4-D bool mask expansion),
fused ``torch_npu.npu_swiglu``, and an NPU-safe rotary embedding. The
model-specific wiring (which boogu module attributes to patch with these
kernels) lives in ``boost.py``.

Once the kernels stabilize they can be promoted to ``lite_boost/layers/``
for reuse by other model adapters.
"""

import torch
import torch.nn.functional as F
import torch_npu

# Bounded cache: each entry pins its source freqs tensor (so its data_ptr
# cannot be recycled by the allocator for different content) and the derived
# cos/sin tables. A handful of entries cover all transformer blocks of a
# step; the cache is dropped wholesale when the cap is exceeded.
_COS_SIN_CACHE_MAX = 16
_cos_sin_cache = {}


def npu_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
             scale=None, enable_gqa=False):
    """SDPA wrapper that expands broadcast ``[B, 1, 1, S]`` bool masks to ``[B, 1, S, S]`` on NPU.

    The fused NPU SDPA kernel rejects them.
    """
    if (attn_mask is not None and query.device.type == "npu"
            and attn_mask.dim() == 4 and attn_mask.shape[1] == 1
            and attn_mask.shape[2] == 1 and attn_mask.dtype == torch.bool):
        attn_mask = attn_mask.expand(-1, -1, query.shape[-2], -1)
    return F.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
        is_causal=is_causal, scale=scale, enable_gqa=enable_gqa,
    )


def patch_sdpa_mask():
    """Install the NPU-safe SDPA wrapper on ``F.scaled_dot_product_attention`` globally (idempotent)."""
    if getattr(F, "_lb_npu_sdpa_patched", False):
        return
    orig_sdpa = F.scaled_dot_product_attention

    def npu_sdpa_patched(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                         scale=None, enable_gqa=False):
        if (attn_mask is not None and query.device.type == "npu"
                and attn_mask.dim() == 4 and attn_mask.shape[1] == 1
                and attn_mask.shape[2] == 1 and attn_mask.dtype == torch.bool):
            attn_mask = attn_mask.expand(-1, -1, query.shape[-2], -1)
        return orig_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                         is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)

    F.scaled_dot_product_attention = npu_sdpa_patched
    F._lb_npu_sdpa_patched = True


def npu_swiglu(x, y):
    """Fused SwiGLU via ``torch_npu.npu_swiglu`` on NPU; SiLU fallback elsewhere."""
    if x.device.type == "npu":
        return torch_npu.npu_swiglu(torch.cat([x, y], dim=-1), dim=-1)
    return F.silu(x.float(), inplace=False).to(x.dtype) * y


def patch_swiglu_attr(module):
    """Replace ``module.swiglu`` with the fused NPU variant (idempotent)."""
    if getattr(module, "_lb_npu_swiglu_patched", False) or not hasattr(module, "swiglu"):
        return
    module.swiglu = npu_swiglu
    module._lb_npu_swiglu_patched = True


def cached_cos_sin(freqs_cis):
    """Cached float32 ``(cos, sin)`` tables for a complex ``freqs_cis`` tensor.

    Each complex entry ``e^{i*theta}`` maps to interleaved-doubled
    ``(cos(theta), sin(theta))`` tables in the layout the fused
    ``npu_rotary_mul`` fallback consumes. Keeping the source tensor in the
    entry pins its ``data_ptr`` (the cache key), so the allocator cannot
    reuse the same address for different freqs content while the entry lives.
    """
    key = (freqs_cis.data_ptr(), tuple(freqs_cis.shape),
           str(freqs_cis.dtype), str(freqs_cis.device))
    cached = _cos_sin_cache.get(key)
    if cached is not None:
        return cached[1], cached[2]
    if len(_cos_sin_cache) >= _COS_SIN_CACHE_MAX:
        _cos_sin_cache.clear()
    lead = freqs_cis.shape[:-1]
    d2 = freqs_cis.shape[-1]
    cos = torch.stack([freqs_cis.real, freqs_cis.real], dim=-1).reshape(*lead, d2 * 2)
    sin = torch.stack([freqs_cis.imag, freqs_cis.imag], dim=-1).reshape(*lead, d2 * 2)
    cos = cos.unsqueeze(-2).contiguous()    # [B, S, 1, D]
    sin = sin.unsqueeze(-2).contiguous()
    _cos_sin_cache[key] = (freqs_cis, cos, sin)
    return cos, sin


_complex64_ok = {}


def _complex64_supported(device):
    """Probe (once per device type) whether the device executes complex64 mul.

    Complex64 multiply is the preferred RoPE path on NPU: as fast as the
    fused fp32 op and with higher precision. Devices without complex support
    fall back to the fused kernel.
    """
    dev_type = device.type
    if dev_type not in _complex64_ok:
        try:
            real = torch.randn(4, dtype=torch.float32, device=device)
            c = torch.view_as_complex(real.reshape(2, 2))
            out = torch.view_as_real(c * c).flatten(1)
            out.sum().item()
            _complex64_ok[dev_type] = True
        except RuntimeError:
            _complex64_ok[dev_type] = False
    return _complex64_ok[dev_type]


def _rotary_real_freqs(x, freqs_cis, use_real_unbind_dim):
    """Rotary embedding for diffusers-style real cos/sin tables (``use_real=True``)."""
    cos, sin = freqs_cis
    cos = cos[None, None].to(x.device)
    sin = sin[None, None].to(x.device)
    if use_real_unbind_dim == -1:
        d2 = x.shape[-1] // 2
        x_real, x_imag = x.reshape(*x.shape[:-1], d2, 2).unbind(dim=-1)
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    elif use_real_unbind_dim == -2:
        d2 = x.shape[-2] // 2
        x_real, x_imag = x.reshape(*x.shape[:-1], 2, d2).unbind(dim=-2)
        x_rotated = torch.cat([-x_imag, x_real], dim=-1)
    else:
        raise ValueError(f"Invalid use_real_unbind_dim: {use_real_unbind_dim}")
    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


def _rotary_complex_freqs(x, freqs_cis):
    """Rotary embedding for complex ``freqs_cis``.

    Prefers complex64 multiply (higher precision, same NPU speed as the
    fused op); falls back to the fused fp32 ``npu_rotary_mul`` on cached
    cos/sin tables when the device lacks complex support.
    """
    if not _complex64_supported(x.device):
        cos, sin = cached_cos_sin(freqs_cis)    # [B, S, 1, D] interleave-doubled
        # Pass [B, S, N, D] directly (the op broadcasts cos/sin over N).
        return torch_npu.npu_rotary_mul(x, cos.to(x.device), sin.to(x.device),
                                        rotary_mode='interleave')
    f = freqs_cis.unsqueeze(2)  # [B, S, 1, D//2] broadcast over heads
    d2 = x.shape[-1] // 2
    x_rot = torch.view_as_complex(x.float().reshape(*x.shape[:-1], d2, 2))
    return torch.view_as_real(x_rot * f).flatten(3).type_as(x)


def _rotary_half_freqs(x, freqs_cis):
    """Rotary embedding for precomputed real half-width ``[cos | sin]`` tables."""
    cos, sin = freqs_cis.chunk(2, dim=-1)
    cos = cos.unsqueeze(2).to(x.device)  # [B, S, 1, D//2] broadcast over heads
    sin = sin.unsqueeze(2).to(x.device)
    d2 = x.shape[-1] // 2
    x_r = x.float().reshape(*x.shape[:-1], d2, 2)
    x0 = x_r[..., 0]
    x1 = x_r[..., 1]
    out = torch.stack([x0 * cos - x1 * sin, x1 * cos + x0 * sin], dim=-1).flatten(3)
    return out.to(x.dtype)


def npu_apply_rotary_emb(x, freqs_cis, use_real=False, use_real_unbind_dim=-1):
    """NPU-safe rotary embedding with the diffusers ``apply_rotary_emb`` signature.

    Dispatches on the freqs layout: real tables (``use_real=True``), complex
    tables (complex64 mul with fused fp32 fallback), or precomputed real
    half-width ``[cos | sin]`` concat. Works on any sequence layout, so it
    serves TP (full-sequence freqs) and SP (per-rank freqs slices) alike.
    """
    if use_real:
        return _rotary_real_freqs(x, freqs_cis, use_real_unbind_dim)
    if freqs_cis.is_complex():
        return _rotary_complex_freqs(x, freqs_cis)
    return _rotary_half_freqs(x, freqs_cis)
