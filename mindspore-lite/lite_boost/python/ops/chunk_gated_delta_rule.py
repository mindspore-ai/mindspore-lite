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
"""
ChunkGatedDeltaRule operator Python binding (prefill / chunk-level).

Wraps ``torch.ops.lite_boost.chunk_gated_delta_rule`` (-> the ascend_a2
AscendC op ``aclnnChunkGatedDeltaRule``). Converts the user-friendly BNSD layout to
the TND layout the CANN op expects, casts q/k/v/beta/state to the low dtype (bf16 OR
fp16 — the A2 op accepts both via DataTypeList; the input dtype is followed, defaulting
to bf16 for fp32/other) while keeping the optional gate ``g`` as float32, and feeds
``actual_seq_lengths`` directly (T = sum(lengths)).

ascend_a2 op spec (TND, bf16): query/key [T,Nk,Dk], value [T,Nv,Dv], beta [T,Nv],
initial_state [B,Nv,Dv,Dk], actual_seq_lengths [B] int32, g(optional) [T,Nv] float32,
out [T,Nv,Dv], final_state [B,Nv,Dv,Dk], attr scale_value (default 1.0). No
cu_seqlens / ssm_state_indices / chunk_size — those belong to the ascend_300iduo op.
"""

import torch
import torch_npu


def _ensure_nd_format(tensor):
    """Cast an NPU tensor to ND format (format id 2) if it is not already ND."""
    if tensor.is_npu and torch_npu.get_npu_format(tensor) != 2:
        return torch_npu.npu_format_cast(tensor, 2)
    return tensor


def chunk_gated_delta_rule(
    query,
    key,
    value,
    beta,
    initial_state,
    actual_seq_lengths,
    g=None,
    scale_value=1.0,
):
    r"""
    Chunked (prefill) Gated Delta Rule operator on NPU (ascend_a2 op).

    Wraps ``torch.ops.lite_boost.chunk_gated_delta_rule``, which maps to
    the CANN AscendC operator ``aclnnChunkGatedDeltaRule`` on A2.  The
    user-facing BNSD layout is converted to the TND layout expected by
    the CANN op; ``query``/``key``/``value``/``beta``/``initial_state``
    are cast to the low dtype (bfloat16 or float16, following the input
    dtype; fp32/other inputs default to bfloat16) while the optional
    gate ``g`` stays float32, and ``actual_seq_lengths`` is passed
    through directly (T = sum(actual_seq_lengths)).

    At each time step :math:`t` the Gated Delta Rule computes the new
    recurrent state and the attention output as

    .. math::

        S_t = \alpha_t S_{t-1} + \beta_t (v_t - \alpha_t S_{t-1} k_t) k_t^{\top}

    .. math::

        o_t = S_t q_t \cdot scale

    where :math:`\alpha_t = \exp(g_t)` is the decay factor (with ``g``
    omitted the decay is disabled, i.e. :math:`\alpha_t = 1`), and
    :math:`\beta_t` is the delta update step size. This operator is the
    chunked (blocked-parallel) implementation of the recurrence above,
    which is more efficient than the token-by-token form on long
    sequences and thus suited for the prefill phase; it produces the
    output at every step as well as the final state.

    Supported only on A2; 300I Duo is not supported.

    Args:
        query (Tensor): Query tensor with shape :math:`(B, N_k, T, D_k)`.
            Cast to the low dtype before computation.
        key (Tensor): Key tensor with shape :math:`(B, N_k, T, D_k)`.
            Cast to the low dtype before computation.
        value (Tensor): Value tensor with shape :math:`(B, N_v, T, D_v)`.
            Cast to the low dtype before computation.
        beta (Tensor): Delta update step size
            with shape :math:`(B, N_v, T)`, in the range (0, 1).  Cast
            to the low dtype before computation.
        initial_state (Tensor): Incoming recurrent state
            with shape :math:`(B, N_v, D_k, D_v)` (transposed to the op's
            value-first ``[B, N_v, D_v, D_k]`` layout).
        actual_seq_lengths (Tensor): Per-batch token counts
            with shape :math:`(B)`, dtype int32; the total sequence
            length is ``T = sum(actual_seq_lengths)``.  A uniform T per
            batch is assumed by the BNSD-to-TND flatten.
        g (Tensor, optional): Global decay gate
            with shape :math:`(B, N_v, T)`, dtype float32, must be
            negative.  ``None`` disables the decay gate
            (hasGamma=0 path). Default: ``None``.
        scale_value (float, optional): Attention scale applied to
            `query`. Default: ``1.0``.

    Returns:
        tuple[Tensor, Tensor]

        - **out** (Tensor) — Attention output
          with shape :math:`(B, N_v, T, D_v)`, same dtype as the
          low-dtype cast of the inputs (bfloat16 by default).
        - **final_state** (Tensor) — Updated recurrent state
          with shape :math:`(B, N_v, D_k, D_v)`, same dtype as `out`.

    Raises:
        RuntimeError: If the input tensor shapes, dtypes or devices are
            invalid, or if the CANN operator execution fails.

    Note:
        - This operator is supported on A2 only; 300I Duo is not
          supported (there the registered vendor exposes a different
          aclnn signature, the ``ascend_300iduo`` op).
        - All input tensors must reside on the same NPU device.
        - The CANN op accepts both bfloat16 and float16 for
          q/k/v/beta/state via DataTypeList; the input dtype is followed
          and fp32/other inputs default to bfloat16.  The optional gate
          ``g`` is always float32.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import torch
        >>> import lite_boost.ops as lite_ops
        >>> device = torch.device("npu:0")
        >>> B, N, T, Dk, Dv = 1, 8, 16, 32, 64
        >>> query = torch.randn(B, N, T, Dk, device=device, dtype=torch.bfloat16)
        >>> key = torch.randn(B, N, T, Dk, device=device, dtype=torch.bfloat16)
        >>> value = torch.randn(B, N, T, Dv, device=device, dtype=torch.bfloat16)
        >>> beta = torch.rand(B, N, T, device=device, dtype=torch.bfloat16) * 0.9 + 0.05
        >>> initial_state = torch.zeros(B, N, Dk, Dv, device=device, dtype=torch.bfloat16)
        >>> actual_seq_lengths = torch.tensor([T], dtype=torch.int32, device=device)
        >>> out, final_state = lite_ops.chunk_gated_delta_rule(
        ...     query, key, value, beta, initial_state, actual_seq_lengths)
        >>> print(out.shape)
        torch.Size([1, 8, 16, 64])
        >>> print(final_state.shape)
        torch.Size([1, 8, 32, 64])
        >>> print(out.dtype)
        torch.bfloat16
    """
    if any(not torch.is_tensor(t) for t in
           (query, key, value, beta, initial_state, actual_seq_lengths)):
        raise RuntimeError(
            "query, key, value, beta, initial_state and actual_seq_lengths "
            "must be tensors.")
    device = query.device
    if any(t.device != device for t in
           (key, value, beta, initial_state, actual_seq_lengths)):
        raise RuntimeError(
            "all input tensors must be on the same device as query.")
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise RuntimeError(
            "query, key and value must be 4-D [B, N, T, D], but got "
            f"{[tuple(t.shape) for t in (query, key, value)]}.")
    if beta.dim() != 3:
        raise RuntimeError(
            f"beta must be 3-D [B, Nv, T], but got {tuple(beta.shape)}.")
    if initial_state.dim() != 4:
        raise RuntimeError(
            f"initial_state must be 4-D [B, Nv, Dk, Dv], but got "
            f"{tuple(initial_state.shape)}.")
    if actual_seq_lengths.dim() != 1:
        raise RuntimeError(
            f"actual_seq_lengths must be 1-D [B], but got "
            f"{tuple(actual_seq_lengths.shape)}.")
    if query.size(1) != key.size(1):
        raise RuntimeError(
            "query and key must have the same number of heads, but got "
            f"{query.size(1)} and {key.size(1)}.")
    batch_size = query.shape[0]
    num_heads_q = query.shape[1]
    seq_len = query.shape[2]
    dk = query.shape[3]
    num_heads_v = value.shape[1]
    dv = value.shape[3]

    # ---- BNSD -> TND (Time-first). Assumes equal seq_len per batch. ----
    query_tnd = query.transpose(1, 2).reshape(-1, num_heads_q, dk).contiguous()
    key_tnd = key.transpose(1, 2).reshape(-1, num_heads_q, dk).contiguous()
    value_tnd = value.transpose(1, 2).reshape(-1, num_heads_v, dv).contiguous()
    beta_tnd = beta.transpose(1, 2).reshape(-1, num_heads_v).contiguous()  # [T, Nv]

    # ---- actual_seq_lengths [B] (int32) is passed directly: T = sum(lengths) ----
    seq_lengths_int = actual_seq_lengths.int().contiguous()

    # ---- state layout: user [B,Nv,Dk,Dv] -> op [B,Nv,Dv,Dk] ----
    state_cann = initial_state.transpose(-1, -2).contiguous()

    # ---- low dtype: the A2 op accepts BOTH bf16 and fp16 for q/k/v/beta/state. Follow the
    #      input dtype (bf16 or fp16); default to bf16 for fp32/other. g stays float32. ----
    low_dtype = torch.bfloat16
    if query.dtype in (torch.bfloat16, torch.float16):
        low_dtype = query.dtype
    query_tnd = _ensure_nd_format(query_tnd).to(low_dtype)
    key_tnd = _ensure_nd_format(key_tnd).to(low_dtype)
    value_tnd = _ensure_nd_format(value_tnd).to(low_dtype)
    beta_tnd = _ensure_nd_format(beta_tnd).to(low_dtype)
    state_cann = _ensure_nd_format(state_cann).to(low_dtype)

    # ---- g is OPTIONAL and stays float32 (op dtype for g is FLOAT); None -> omit ----
    g_tnd = None
    if g is not None:
        g_tnd = _ensure_nd_format(g.transpose(1, 2).reshape(-1, num_heads_v).contiguous()).to(torch.float32)

    out_tnd, final_state_cann = torch.ops.lite_boost.chunk_gated_delta_rule(
        query_tnd, key_tnd, value_tnd, beta_tnd, state_cann,
        seq_lengths_int, g_tnd, float(scale_value),
    )

    # ---- TND -> BNSD reverse ----
    out_bnsd = out_tnd.reshape(batch_size, seq_len, num_heads_v, dv).transpose(1, 2).contiguous()
    final_state = final_state_cann.transpose(-1, -2).contiguous()  # CANN [B,Nv,Dv,Dk] -> [B,Nv,Dk,Dv]
    return out_bnsd, final_state
