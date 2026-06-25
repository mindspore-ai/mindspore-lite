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

Wraps ``torch.ops.lite_boost.chunk_gated_delta_rule`` (-> CANN
``aclnnChunkGatedDeltaRule``). Converts user-friendly BNSD layout to the TND
layout the CANN op expects, casts to float16 (the op's only supported dtype),
and builds cu_seqlens from per-batch sequence lengths.
"""

import torch
import torch_npu


def _ensure_nd_format(tensor):
    if tensor.is_npu and torch_npu.get_npu_format(tensor) != 2:
        return torch_npu.npu_format_cast(tensor, 2)
    return tensor


def chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    initial_state,
    actual_seq_lengths,
    ssm_state_indices,
    chunk_size=64,
    scale_value=None,
):
    """Chunked (prefill) Gated Delta Rule on NPU.

    Args (BNSD layout; any float dtype is cast to float16 for the op):
        query (Tensor):         [B, H_q, T, D_k]  (L2-normalized, range [0,1]).
        key (Tensor):           [B, H_q, T, D_k]  (L2-normalized).
        value (Tensor):         [B, H_v, T, D_v].
        g (Tensor):             [B, H_v, T]  global decay gate, must be < 0.
        beta (Tensor):          [B, H_v, T]  delta step size, in (0, 1).
        initial_state (Tensor): [B, H_v, D_k, D_v]  incoming recurrent state.
        actual_seq_lengths (Tensor): [B] (int32) per-batch token count.
        ssm_state_indices (Tensor):  [B] (int32) state-pool index per batch.
        chunk_size (int):       chunk length (attr). Default 64. T is padded up
                                to a multiple of chunk_size internally.
        scale_value (float):    attention scale; default 1/sqrt(D_k).

    Returns:
        tuple[Tensor, Tensor]

        - **out** (Tensor) — Attention output of shape ``[B, H_v, T, D_v]``, dtype=float16.
        - **final_state** (Tensor) — Updated recurrent state of shape ``[B, H_v, D_k, D_v]``, dtype=float16.
    """
    batch_size = query.shape[0]
    num_heads_q = query.shape[1]
    seq_len = query.shape[2]
    dk = query.shape[3]
    num_heads_v = value.shape[1]
    dv = value.shape[3]

    if scale_value is None:
        scale_value = 1.0 / (dk ** 0.5)

    # ---- BNSD -> TND (Time-first). Assumes equal seq_len per batch. ----
    query_tnd = query.transpose(1, 2).reshape(-1, num_heads_q, dk).contiguous()
    key_tnd = key.transpose(1, 2).reshape(-1, num_heads_q, dk).contiguous()
    value_tnd = value.transpose(1, 2).reshape(-1, num_heads_q, dv).contiguous()
    g_tnd = g.transpose(1, 2).reshape(-1, num_heads_v).contiguous()        # [T, Hv]
    beta_tnd = beta.transpose(1, 2).reshape(-1, num_heads_v).contiguous()  # [T, Hv]

    # ---- cu_seqlens [B+1] (cumulative) from per-batch lengths ----
    seq_lengths_int = actual_seq_lengths.int().contiguous()
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=query.device)
    cu_seqlens[1:] = torch.cumsum(seq_lengths_int, dim=0)
    ssm_state_indices = ssm_state_indices.int().contiguous()

    # ---- state layout: Python [B,Hv,Dk,Dv] -> CANN [B,Hv,Dv,Dk] ----
    state_cann = initial_state.transpose(-1, -2).contiguous()

    # ---- ND format + float16 (op requires fp16 for q/k/v/g/beta/state) ----
    query_tnd = _ensure_nd_format(query_tnd).to(torch.float16)
    key_tnd = _ensure_nd_format(key_tnd).to(torch.float16)
    value_tnd = _ensure_nd_format(value_tnd).to(torch.float16)
    g_tnd = _ensure_nd_format(g_tnd).to(torch.float16)
    beta_tnd = _ensure_nd_format(beta_tnd).to(torch.float16)
    state_cann = _ensure_nd_format(state_cann).to(torch.float16)

    out_tnd, final_state_cann = torch.ops.lite_boost.chunk_gated_delta_rule(
        query_tnd, key_tnd, value_tnd, g_tnd, beta_tnd, state_cann,
        cu_seqlens, ssm_state_indices, int(chunk_size), float(scale_value),
    )

    # ---- TND -> BNSD reverse ----
    out_bnsd = out_tnd.reshape(batch_size, seq_len, num_heads_v, dv).transpose(1, 2).contiguous()
    final_state = final_state_cann.transpose(-1, -2).contiguous()  # CANN [B,Hv,Dv,Dk] -> [B,Hv,Dk,Dv]
    return out_bnsd, final_state
