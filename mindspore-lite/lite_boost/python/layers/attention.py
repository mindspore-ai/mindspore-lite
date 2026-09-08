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
#
# Third-party code attribution:
#
#   File structure derived from Wan2.1 (Apache 2.0):
#     https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/attention.py
#     Copyright 2024-2025 The Alibaba Wan Team Authors.
#
#   The varlen flash-attention API call pattern (cu_seqlens construction +
#   flash_attn_varlen_func invocation) derived from flash-attention:
#     https://github.com/Dao-AILab/flash-attention
#     Copyright (c) 2022, Intelligent Systems Lab Org.
#     Copyright (c) 2020, Alexey.
#     Copyright (c) 2019, Intel ISL (Intel Intelligent Systems Lab).
#
#   See the LICENSE files in those projects for full notices.
"""
Flash attention with NPU fallback.

Supports FA3, FA2, and NPU (npu_prompt_flash_attention) backends,
auto-selected according to availability.
"""

__all__ = ['flash_attention']

import warnings

import torch

try:
    import flash_attn_interface
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False
else:
    FLASH_ATTN_3_AVAILABLE = True

try:
    import flash_attn
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False
else:
    FLASH_ATTN_2_AVAILABLE = True

try:
    import torch_npu
except (ModuleNotFoundError, AttributeError):
    NPU_FUSION_ATTENTION_AVAILABLE = False
else:
    NPU_FUSION_ATTENTION_AVAILABLE = hasattr(torch_npu, 'npu_prompt_flash_attention')


def _preprocess_qkv(q, k, v, q_lens, k_lens, b, lq, lk, dtype):
    """Flatten Q/K/V to packed (B*S, N, D), cast them to the target
    `dtype`, and normalize the length arrays.

    Varlen inputs keep their padded full-length tokens (no prefix
    truncation here): the NPU backend applies the per-sequence true
    lengths through ``actual_seq_lengths``, and the GPU backends through
    the cu_seqlens of flash_attn_varlen_func.
    """
    if q_lens is None:
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    elif not isinstance(q_lens, torch.Tensor):
        q_lens = torch.tensor(q_lens, dtype=torch.int32, device=q.device)

    if k_lens is None:
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    elif not isinstance(k_lens, torch.Tensor):
        k_lens = torch.tensor(k_lens, dtype=torch.int32, device=k.device)

    q = q.flatten(0, 1).to(dtype)
    k = k.flatten(0, 1).to(dtype)
    v = v.flatten(0, 1).to(dtype)

    return q, k, v, q_lens, k_lens


def _flash_attn_npu(q, k, v, q_lens, k_lens, b, lq, lk, softmax_scale):
    """NPU flash attention backend.

    q/k/v are the flattened full-length (B*S, N, D) tensors.  The padded
    (B, S, N, D) layout is restored here and the per-sequence true
    lengths are forwarded as ``actual_seq_lengths`` /
    ``actual_seq_lengths_kv``: npu_prompt_flash_attention varlen mode
    masks each sequence to its own prefix, so UNEQUAL lengths across the
    batch are supported (verified on 300I Duo).  The output keeps the
    padded (B, S, N, D) layout of the input `q`: rows beyond a
    sequence's true length (``q_lens[i]``) are kernel output for masked
    positions with undefined values and must be sliced away by the
    caller.
    """
    head_dim = q.size(-1)
    num_heads = q.size(1)

    q_4d = q.reshape(b, lq, num_heads, head_dim)
    k_4d = k.reshape(b, lk, num_heads, head_dim)
    v_4d = v.reshape(b, lk, num_heads, head_dim)

    scale = softmax_scale if softmax_scale is not None else (1.0 / (head_dim ** 0.5))
    compute_dtype = q_4d.dtype if q_4d.dtype in (torch.float16, torch.bfloat16) else torch.float16
    q_bnsd = q_4d.transpose(1, 2).contiguous().to(compute_dtype)
    k_bnsd = k_4d.transpose(1, 2).contiguous().to(compute_dtype)
    v_bnsd = v_4d.transpose(1, 2).contiguous().to(compute_dtype)

    x = torch_npu.npu_prompt_flash_attention(
        q_bnsd, k_bnsd, v_bnsd,
        num_heads=num_heads,
        input_layout="BNSD",
        scale_value=scale,
        pre_tokens=2147483647,
        next_tokens=2147483647,
        actual_seq_lengths=q_lens.cpu().tolist(),
        actual_seq_lengths_kv=k_lens.cpu().tolist(),
    )
    return x.transpose(1, 2).contiguous()  # padded (B, S, N, D)


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
    dtype=None,
    version=None,
):
    r"""
    Computes flash attention with automatic backend fallback.

    Selects FA3, FA2, or the NPU backend (``npu_prompt_flash_attention``)
    according to availability.  Supports varlen sequences through
    `q_lens`/`k_lens`; when both are ``None`` every sequence is treated
    as full length.

    Notation: `B` is the batch size, `N` the number of heads, `D` the
    head dim (``D <= 256`` on the NPU), `S` the sequence length of the
    input `q`/`k`/`v` tensors, `lq` the query sequence length
    (``q.size(1)``, the same value as `S`), and `lk` the key sequence
    length (``k.size(1)``).  The output always keeps the padded
    (B, S, N, D) layout of the input `q` (same shape as `q`) on every
    backend.  In varlen mode `q_lens`/`k_lens` carry the per-sequence
    true lengths, which may be shorter than the full length and, on the
    NPU backend, may differ from each other across the batch; only the
    first `q_lens[i]` rows of sequence `i` are valid then -- the rows
    beyond are masked kernel output with undefined values.  When both
    are ``None`` every sequence is treated as full length (`lq`/`lk`).

    Supported only on A2; 300I Duo is not supported.  The
    `dropout_p`, `causal`, `window_size`, `deterministic` and
    `version` arguments are GPU-only features of the flash_attn
    (FA2/FA3) backends; the NPU backend ignores them and always
    computes global attention with no dropout.

    Varlen support differs by backend: the GPU flash_attn (FA2/FA3)
    backends require every per-sequence length to equal the full
    length (``q_lens[i] == lq`` and ``k_lens[i] == lk``); the NPU
    backend additionally accepts shorter lengths, equal or unequal
    across the batch.

    Args:
        q (Tensor): Query tensor with shape :math:`(B, S, N, D)`, on an
            NPU device with head dim :math:`D \le 256`.  Supported dtypes
            are float16, float32 and bfloat16.
        k (Tensor): Key tensor with shape :math:`(B, S, N, D)`, same dtype
            as `q`.
        v (Tensor): Value tensor with shape :math:`(B, S, N, D)`, same
            dtype as `q`.
        q_lens (Union[list[int], Tensor[int32]], optional): Per-sequence
            query lengths (length `B`) in varlen mode; may be shorter
            than the full length. Default: ``None``.
        k_lens (Union[list[int], Tensor[int32]], optional): Per-sequence
            key lengths (length `B`) in varlen mode; may be shorter than
            the full key length. Default: ``None``.
        dropout_p (float, optional): Dropout probability; effective on the
            GPU flash_attn FA2 backend only. Default: ``0.``.
        softmax_scale (float, optional): Attention scaling factor; ``None``
            means :math:`1/\sqrt{D}`. Default: ``None``.
        q_scale (float, optional): Pre-scale applied to `q` as
            ``q = q * q_scale``. Default: ``None``.
        causal (bool, optional): Causal mask; effective on the GPU
            flash_attn FA2 backend only (the NPU backend always computes
            global attention, so ``True`` is not honored there).
            Default: ``False``.
        window_size (Tuple[int, int], optional): Sliding-window
            restriction; effective on the GPU flash_attn FA2 backend
            only. Default: ``(-1, -1)``.
        deterministic (bool, optional): Deterministic mode; effective on
            the GPU flash_attn FA3/FA2 backends only. Default: ``False``.
        dtype (torch.dtype, optional): Compute and output dtype, one of
            float16, bfloat16 or float32: `q`/`k`/`v` are cast to it before
            computing and the output is returned in it.  ``None`` keeps the
            input dtype end to end (float16 in, float16 out); float32
            inputs then still compute on the fp16 NPU kernel and come back
            as float32. Default: ``None``.
        version (int, optional): ``3`` forces FA3, falling back to FA2
            with a warning when FA3 is unavailable; ``None`` selects
            automatically. Effective on the GPU flash_attn FA3/FA2
            backends only. Default: ``None``.

    Returns:
        Tensor with the dtype requested via `dtype` (``None`` keeps the
        input `q` dtype) and the same padded shape as the input `q`
        (:math:`(B, S, N, D)`) on every backend.  In varlen mode only
        the first `q_lens[i]` rows of sequence `i` are valid; the rows
        beyond are masked kernel output with undefined values and should
        be sliced away (``out[i, :q_lens[i]]``).

    Raises:
        ValueError: If `dtype` is neither ``None`` nor in {float16,
            bfloat16, float32}, if `q`/`k`/`v` do not share a dtype in
            {float16, bfloat16, float32}, if `q` is not on an NPU device,
            if the head dim exceeds 256, if any per-sequence length
            exceeds the full length, or if `q_lens`/`k_lens` do not have
            one element per batch item (length ``B``).
        NotImplementedError: On the GPU flash_attn (FA2/FA3) backends,
            if varlen mode is used with lengths that do not equal the
            full length.
        RuntimeError: If no backend is available (neither ``flash_attn``
            nor NPU fusion attention).

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import torch
        >>> import torch_npu
        >>> from lite_boost.layers import flash_attention
        >>> torch.npu.set_device(0)
        >>> q = torch.randn(1, 16, 8, 32, device="npu")
        >>> k = torch.randn(1, 16, 8, 32, device="npu")
        >>> v = torch.randn(1, 16, 8, 32, device="npu")
        >>> out = flash_attention(q, k, v)
        >>> print(out.shape)
        torch.Size([1, 16, 8, 32])
        >>> print(out.dtype)
        torch.float32
    """
    half_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    if dtype is not None and dtype not in half_dtypes:
        raise ValueError(
            f"dtype must be one of {half_dtypes} or None, but got {dtype}.")
    if q.dtype not in half_dtypes or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError(
            f"q/k/v must share a dtype in {half_dtypes}, but got "
            f"q={q.dtype}, k={k.dtype}, v={v.dtype}.")
    if q.device.type != 'npu' or q.size(-1) > 256:
        raise ValueError(
            f"q must be on 'npu' device with head_dim <= 256, "
            f"but got device={q.device.type}, head_dim={q.size(-1)}.")

    b, lq, lk = q.size(0), q.size(1), k.size(1)
    # dtype=None keeps the input dtype for compute and output; an explicit
    # dtype casts every input below and defines the output dtype.  The NPU
    # kernel computes float16/bfloat16 only: explicit float32 (or float32
    # inputs with dtype=None) still compute on the fp16 kernel and are
    # returned as float32.
    dtype = q.dtype if dtype is None else dtype
    out_dtype = dtype

    q, k, v, q_lens, k_lens = _preprocess_qkv(
        q, k, v, q_lens, k_lens, b, lq, lk, dtype)

    if q_lens.numel() != b or k_lens.numel() != b:
        raise ValueError(
            "q_lens/k_lens must have one length per batch element "
            f"(batch size {b}), but got q_lens={q_lens.numel()} and "
            f"k_lens={k_lens.numel()} elements.")

    if (q_lens > lq).any().item() or (k_lens > lk).any().item():
        raise ValueError(
            "q_lens/k_lens must not exceed the full lengths "
            f"(q.size(1)={lq}, k.size(1)={lk}).")

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    use_fa3 = (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE
    if use_fa3 or FLASH_ATTN_2_AVAILABLE:
        # GPU flash_attn varlen limitation: flash_attn_varlen_func
        # returns sum(q_lens) real tokens that are restored below via
        # unflatten(0, (b, lq)), so every per-sequence length must
        # equal the full length.  Shorter lengths -- equal (e.g.
        # q_lens=[12, 12] with S=16) or unequal -- would make
        # sum(q_lens) != b*lq and crash inside unflatten; the NPU
        # backend supports them through actual_seq_lengths instead.
        if (q_lens != lq).any().item() or (k_lens != lk).any().item():
            raise NotImplementedError(
                "flash_attn varlen requires full-length sequences "
                "(q_lens == q.size(1) and k_lens == k.size(1)); "
                "shorter or unequal lengths are supported on the NPU "
                "backend only.")

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
