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

Wraps ``torch.ops.lite_boost.chunk_gated_delta_rule`` (-> the ascend_a2 / ascend910b
AscendC op ``aclnnChunkGatedDeltaRule``). Converts the user-friendly BNSD layout to
the TND layout the CANN op expects, casts q/k/v/beta/state to the low dtype (bf16 OR
fp16 — the 910B op accepts both via DataTypeList; the input dtype is followed, defaulting
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
    """Chunked (prefill) Gated Delta Rule on NPU (ascend910b / ascend_a2 op).

    Args (BNSD layout; q/k/v/beta/state are cast to the low dtype — bf16 or fp16,
    following the input dtype; default bf16 for fp32/other):
        query (Tensor):         [B, Nk, T, Dk].
        key (Tensor):           [B, Nk, T, Dk].
        value (Tensor):         [B, Nv, T, Dv].
        beta (Tensor):          [B, Nv, T]  delta step size, in (0, 1).
        initial_state (Tensor): [B, Nv, Dk, Dv]  incoming recurrent state
                                (transposed to the op's Dv-first [B,Nv,Dv,Dk]).
        actual_seq_lengths (Tensor): [B] (int32) per-batch token count;
                                T = sum(actual_seq_lengths). A uniform T per batch is
                                assumed by the BNSD->TND flatten.
        g (Tensor, optional):   [B, Nv, T]  global decay gate (float32, < 0).
                                ``None`` disables the decay gate (hasGamma=0 path).
        scale_value (float):    attention scale applied to the query (op default 1.0).

    Returns:
        tuple[Tensor, Tensor]

        - **out** (Tensor) — Attention output of shape ``[B, Nv, T, Dv]``, dtype=bfloat16.
        - **final_state** (Tensor) — Updated recurrent state of shape ``[B, Nv, Dk, Dv]``,
          dtype=bfloat16.

    Note:
        Targets ascend910b only. On ascend310p the registered vendor exposes a different
        aclnn signature (ascend_300iduo op), so this binding must not be expected to run there.
    """
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

    # ---- low dtype: the 910B op accepts BOTH bf16 and fp16 for q/k/v/beta/state. Follow the
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
