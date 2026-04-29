#!/usr/bin/env python3
# modified from
# https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/attention.py
# Copyright 2024-2025 The Alibaba Wan Team Authors
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
"""
reference from wan2.1
Flash attention with NPU fallback.

Supports FA3, FA2, and NPU (npu_prompt_flash_attention) backends,
auto-selected based on availability.
"""
import warnings

import torch

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    import torch_npu
    NPU_FUSION_ATTENTION_AVAILABLE = hasattr(torch_npu, 'npu_prompt_flash_attention')
except (ModuleNotFoundError, AttributeError):
    NPU_FUSION_ATTENTION_AVAILABLE = False

__all__ = ['flash_attention']


def _half(x, half_dtypes, dtype):
    return x if x.dtype in half_dtypes else x.to(dtype)


def _preprocess_qkv(q, k, v, q_lens, k_lens, b, lq, lk, half_dtypes, dtype):
    """Preprocess Q, K, V tensors for flash attention."""
    if q_lens is None:
        q = _half(q.flatten(0, 1), half_dtypes, dtype)
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = _half(torch.cat([u[:v] for u, v in zip(q, q_lens)]), half_dtypes, dtype)

    if k_lens is None:
        k = _half(k.flatten(0, 1), half_dtypes, dtype)
        v = _half(v.flatten(0, 1), half_dtypes, dtype)
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = _half(torch.cat([u[:v] for u, v in zip(k, k_lens)]), half_dtypes, dtype)
        v = _half(torch.cat([u[:v] for u, v in zip(v, k_lens)]), half_dtypes, dtype)

    return q, k, v, q_lens, k_lens


def _flash_attn_npu(q, k, v, q_lens, k_lens, b, lq, lk, softmax_scale):
    """NPU flash attention backend."""
    head_dim = q.size(-1)
    num_heads = q.size(1)

    if q_lens is not None and k_lens is not None:
        len_q = q_lens[0].item()
        len_k = k_lens[0].item()
        num_seqs = len(q_lens)
        q_4d = q.reshape(num_seqs, len_q, num_heads, head_dim)
        k_4d = k.reshape(num_seqs, len_k, num_heads, head_dim)
        v_4d = v.reshape(num_seqs, len_k, num_heads, head_dim)
    elif q_lens is not None:
        len_q = q_lens[0].item()
        num_seqs = q.size(0) // len_q
        q_4d = q.reshape(num_seqs, len_q, num_heads, head_dim)
        k_4d = k.reshape(num_seqs, -1, num_heads, head_dim)
        v_4d = v.reshape(num_seqs, -1, num_heads, head_dim)
    else:
        q_4d = q.reshape(b, lq, num_heads, head_dim)
        k_4d = k.reshape(b, lk, num_heads, head_dim)
        v_4d = v.reshape(b, lk, num_heads, head_dim)

    scale = softmax_scale if softmax_scale is not None else (1.0 / (head_dim ** 0.5))
    q_bnsd = q_4d.transpose(1, 2).contiguous().to(torch.float16)
    k_bnsd = k_4d.transpose(1, 2).contiguous().to(torch.float16)
    v_bnsd = v_4d.transpose(1, 2).contiguous().to(torch.float16)

    x = torch_npu.npu_prompt_flash_attention(
        q_bnsd, k_bnsd, v_bnsd,
        num_heads=num_heads,
        input_layout="BNSD",
        scale_value=scale,
        pre_tokens=2147483647,
        next_tokens=2147483647,
    )
    return x.to(torch.float32).transpose(1, 2).contiguous()


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.float32,
    version=None,
):
    """Flash attention with NPU fallback (FA3 → FA2 → NPU)."""
    half_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    assert dtype in half_dtypes
    assert q.device.type == 'npu' and q.size(-1) <= 256

    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    q, k, v, q_lens, k_lens = _preprocess_qkv(
        q, k, v, q_lens, k_lens, b, lq, lk, half_dtypes, dtype)

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    elif FLASH_ATTN_2_AVAILABLE:
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))
    elif NPU_FUSION_ATTENTION_AVAILABLE:
        x = _flash_attn_npu(q, k, v, q_lens, k_lens, b, lq, lk, softmax_scale)
    else:
        raise RuntimeError(
            "No attention implementation available. "
            "Please install flash_attn or use NPU."
        )

    return x.type(out_dtype)
