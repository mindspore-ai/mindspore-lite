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
RecurrentGatedDeltaRule operator Python binding
"""

import torch
import torch_npu


def _ensure_nd_format(tensor):
    if tensor.is_npu and torch_npu.get_npu_format(tensor) != 2:
        return torch_npu.npu_format_cast(tensor, 2)
    return tensor


def recurrent_gated_delta_rule(
    query,
    key,
    value,
    beta,
    state,
    actual_seq_lengths,
    ssm_state_indices,
    g,
    gk,
    num_accepted_tokens,
    scale_value=1.0,
):  # pylint: disable=too-many-arguments,too-many-locals,too-many-statements
    r"""
    Recurrent GatedDeltaRule operator — CANN aclnn-backed recurrent linear attention decode.

    Implements the token-by-token recurrent forward pass of the Gated Delta Rule,
    updating the recurrent state matrix and producing the attention output.
    Primarily used for decode-phase inference acceleration in hybrid linear attention
    models such as Qwen3.5.

    Algorithm flow (executed sequentially for each token in each batch):

    1. State decay:   S = S * exp(g) * exp(gk)
    2. Memory retrieval: kv_mem = S^T @ k
    3. Delta update:  S = S + k^T @ ((v - kv_mem) * beta)
    4. Output:        o = S^T @ q

    where S is the recurrent state matrix ``[H, D_k, D_v]`` storing the key-value
    associations of linear attention.

    Args:
        query (torch.Tensor): Query tensor of shape ``[B, N_k, T, D_k]``, dtype=bfloat16.
            Must be L2-normalized (L2 norm of each head vector is 1, value range [0, 1]).
            B=batch_size, N_k=num_key_heads, T=seq_len, D_k=key_dim.
        key (torch.Tensor): Key tensor of shape ``[B, N_k, T, D_k]``, dtype=bfloat16.
            Must be L2-normalized (same as query).
        value (torch.Tensor): Value tensor of shape ``[B, N_v, T, D_v]``, dtype=bfloat16.
            N_v=num_value_heads, D_v=value_dim. N_v must be divisible by N_k.
        beta (torch.Tensor): Delta update step size of shape ``[B, N_v, T]``, dtype=bfloat16.
            Value range (0, 1). Controls the magnitude of each delta update: a larger beta
            causes new information to overwrite old memory more aggressively; a smaller
            beta tends to preserve existing memory.
        state (torch.Tensor): Recurrent state pool of shape
            ``[state_slots, N_v, D_k, D_v]``,
            dtype=bfloat16. Stores the cumulative key-value associations for linear attention.
            D_k is the key dimension (rows), D_v is the value dimension (columns).
            Can be initialized to zeros for the first call.
        actual_seq_lengths (torch.Tensor): Actual sequence lengths of shape ``[B]``, dtype=int32.
            Used for variable-length sequence inference. Each element represents the number
            of valid tokens in the corresponding batch.
            E.g., ``[4, 3, 5]`` means 3 batches with sequence lengths 4, 3, and 5.
        ssm_state_indices (torch.Tensor): State-slot indices of shape ``[T_total]``,
            dtype=int32. Each flattened token selects one entry in the global state pool.
        g (torch.Tensor): Global decay gate of shape ``[B, N_v, T]``, dtype=float32.
            **Must be negative**. ``exp(g)`` serves as the state decay factor with range (0, 1).
            The more negative ``g`` is, the faster historical information is forgotten.
            E.g., when g=-1, approximately 37% of the historical state is retained per step.
        gk (torch.Tensor): Key-dimension gate of shape ``[B, N_v, T, D_k]``, dtype=float32.
            **Must be negative**. ``exp(gk)`` applies per-dimension decay independently along
            the key dimension, enabling finer-grained memory control. Unlike the global gate g,
            gk operates element-wise along the D_k dimension.
        num_accepted_tokens (torch.Tensor): Number of accepted tokens of shape ``[B]``, dtype=int32.
            Used in speculative decoding and similar scenarios to mark the number of actually
            accepted (non-rejected) tokens. For standard inference, this is the same as
            ``actual_seq_lengths``.
        scale_value (float, optional): Attention scale factor, default 1.0.
            Typically set to ``1.0 / sqrt(D_k)``, consistent with standard attention scaling.
            The query is multiplied by this scale factor before computation.

    Returns:
        tuple[Tensor, Tensor]

        - **out** (Tensor) — Attention output of shape ``[B, N_v, T, D_v]``, dtype=bfloat16.
          The linear attention result at each token position.
        - **state_out** (Tensor) — Updated recurrent state pool with the same shape as ``state``,
          dtype=bfloat16. Must be passed as ``state`` input in the next recurrent step to
          form a state-passing chain.

    Raises:
        RuntimeError: If input tensor shapes, dtypes, or devices are invalid, or if the
            CANN operator execution fails.

    Note:
        - This operator only supports the **decode phase** (token-by-token inference),
          with sequence length T not exceeding 8. For parallel prefill computation,
          use the chunk-level operator.
        - Supports grouped recurrent heads where N_v is an integer multiple of N_k.
        - All input tensors must reside on the same NPU device.
        - The CANN operator stores state internally as
          ``[state_slots, N_v, D_v, D_k]`` layout
          (value dimension first). This function automatically performs the layout conversion.

    Examples:
        >>> import torch
        >>> import lite_boost.ops as lite_ops
        >>> device = torch.device("npu:0")
        >>> B, N, T, Dk, Dv = 1, 64, 1, 64, 512
        >>> query  = torch.randn(B, N, T, Dk, device=device, dtype=torch.bfloat16)
        >>> key    = torch.randn(B, N, T, Dk, device=device, dtype=torch.bfloat16)
        >>> value  = torch.randn(B, N, T, Dv, device=device, dtype=torch.bfloat16)
        >>> beta   = torch.rand(B, N, T, device=device, dtype=torch.bfloat16) * 0.9 + 0.05
        >>> state  = torch.zeros(B, N, Dk, Dv, device=device, dtype=torch.bfloat16)
        >>> g      = -(torch.rand(B, N, T, device=device) + 0.01)
        >>> gk     = -(torch.rand(B, N, T, Dk, device=device) + 0.01)
        >>> actual_seq_lengths  = torch.tensor([T], dtype=torch.int32, device=device)
        >>> ssm_state_indices   = torch.tensor([0], dtype=torch.int32, device=device)
        >>> num_accepted_tokens = torch.tensor([T], dtype=torch.int32, device=device)
        >>> output, state_out = lite_ops.recurrent_gated_delta_rule(
        ...     query, key, value, beta, state,
        ...     actual_seq_lengths, ssm_state_indices,
        ...     g, gk, num_accepted_tokens,
        ...     scale_value=1.0 / (Dk ** 0.5))
    """
    # =========================================================================
    # 1. Extract dimensions from the BNSD (Batch, Num_heads, Seq_len, Dim) layout
    # =========================================================================
    # BNSD layout: [Batch, Num_heads, Seq_len, Dim]
    # - B (batch_size):    batch size
    # - H_k (num_heads_q): number of key/query heads
    # - T (seq_len):       sequence length (typically 1~8 in decode phase)
    # - D_k (dk):          Key/Query attention head dimension
    # - H_v (num_heads_v): number of value heads (a multiple of H_k)
    # - D_v (dv):          Value attention head dimension
    batch_size = query.shape[0]
    num_heads_q = query.shape[1]
    seq_len = query.shape[2]
    dk = query.shape[3]
    # Value head count may differ from query (GQA/MQA mode)
    num_heads_v = value.shape[1]
    dv = value.shape[3]
    if num_heads_v < num_heads_q or num_heads_v % num_heads_q != 0:
        raise ValueError("value heads must be an integer multiple of query/key heads")
    if ssm_state_indices.numel() != batch_size * seq_len:
        raise ValueError("ssm_state_indices must contain one state-slot index per token")

    # =========================================================================
    # 2. BNSD -> TND layout conversion
    # =========================================================================
    # The CANN operator requires TND (Time-first) layout: the sequence dimension
    # is flattened and placed first.
    # T_total = B * T (when all batches have equal length) or
    #           sum(actual_seq_lengths) (for variable-length sequences).
    #
    # Conversion rule for 4D tensors:
    #   [B, H, T, D] --transpose(1,2)--> [B, T, H, D] --reshape(-1,H,D)--> [B*T, H, D]
    #
    # Conversion rule for 3D tensors:
    #   [B, H, T] --transpose(1,2)--> [B, T, H] --reshape(-1,H)--> [B*T, H]
    #
    # =========================================================================

    # query: [B, H_q, T, D_k] -> [T_total, H_q, D_k]
    query_tnd = query.transpose(1, 2).reshape(-1, num_heads_q, dk).contiguous()

    # key: [B, H_q, T, D_k] -> [T_total, H_q, D_k]
    # Shares the same head count and dimension as query (key-query symmetry in linear attention)
    key_tnd = key.transpose(1, 2).reshape(-1, num_heads_q, dk).contiguous()

    # value: [B, H_v, T, D_v] -> [T_total, H_v, D_v]
    # Recurrent GDR maps each key/query head to Nv/Nk value heads in the
    # kernel. Preserve H_v here; reshaping with H_q corrupts T_total for GQA.
    value_tnd = value.transpose(1, 2).reshape(-1, num_heads_v, dv).contiguous()

    # beta: [B, H_v, T] -> [T_total, H_v]
    # Delta update step size, controls how much new information overwrites old memory
    beta_tnd = beta.transpose(1, 2).reshape(-1, num_heads_v).contiguous()

    # g: [B, H_v, T] -> [T_total, H_v]
    # Global decay gate, exp(g) ∈ (0, 1) controls state decay rate
    g_tnd = g.transpose(1, 2).reshape(-1, num_heads_v).contiguous()

    # gk: [B, H_v, T, D_k] -> [T_total, H_v, D_k]
    # Per-element gate along the key dimension, providing finer-grained memory control
    # than the global gate g
    gk_tnd = gk.transpose(1, 2).reshape(-1, num_heads_v, dk).contiguous()

    # =========================================================================
    # 3. Prepare actual sequence lengths
    # =========================================================================
    # The ACLNN interface accepts one length per batch rather than a cumulative
    # prefix vector.
    # =========================================================================
    seq_lengths_int = actual_seq_lengths.int().contiguous()
    # =========================================================================
    # 4. Recurrent state matrix layout conversion
    # =========================================================================
    # Python-side convention: state[..., D_k, D_v] (key dimension first, value dimension second)
    # CANN-side convention:   state[..., D_v, D_k] (value dimension first, key dimension second)
    # Therefore, swap the last two dimensions.
    # =========================================================================
    state_cann = state.transpose(-1, -2).contiguous()

    query_tnd = _ensure_nd_format(query_tnd)
    key_tnd = _ensure_nd_format(key_tnd)
    value_tnd = _ensure_nd_format(value_tnd)
    beta_tnd = _ensure_nd_format(beta_tnd)
    g_tnd = _ensure_nd_format(g_tnd)
    gk_tnd = _ensure_nd_format(gk_tnd)
    state_cann = _ensure_nd_format(state_cann)

    # =========================================================================
    # 5. Invoke the CANN aclnnRecurrentGatedDeltaRule operator
    # =========================================================================
    # Calls the C++ registered operator via PyTorch custom op mechanism (torch.ops.lite_boost).
    # The C++ layer invokes the CANN backend via the EXEC_NPU_CMD macro, which automatically
    # handles workspace allocation and asynchronous execution.
    #
    # Input tensor summary (all in TND layout):
    #   query_tnd:  [T_total, H_k, D_k]      - L2-normalized query
    #   key_tnd:    [T_total, H_k, D_k]      - L2-normalized key
    #   value_tnd:  [T_total, H_v, D_v]      - value
    #   beta_tnd:   [T_total, H_v]            - Delta update step size (0, 1)
    #   state_cann: [state_slots, H_v, D_v, D_k] - recurrent state pool
    #   seq_lengths_int: [B]                 - actual sequence lengths
    #   ssm_state_indices: [T_total]          - state pool indices
    #   g_tnd:      [T_total, H_v]            - global decay gate (negative)
    #   gk_tnd:     [T_total, H_v, D_k]      - key gate (negative)
    #   num_accepted_tokens: [B]              - accepted token counts
    #   scale_value: float                    - scale factor
    #
    # Output tensors:
    #   out_tnd:        [T_total, H_v, D_v]   - attention output
    #   state_out_cann: [state_slots, H_v, D_v, D_k] - updated recurrent state
    # =========================================================================
    out_tnd, state_out_cann = torch.ops.lite_boost.recurrent_gated_delta_rule(
        query_tnd,
        key_tnd,
        value_tnd,
        beta_tnd,
        state_cann,
        seq_lengths_int,
        ssm_state_indices,
        g_tnd,
        gk_tnd,
        num_accepted_tokens,
        scale_value,
    )

    # =========================================================================
    # 6. TND -> BNSD reverse layout conversion
    # =========================================================================
    # Convert the CANN operator outputs from TND layout back to the user-friendly
    # BNSD layout.
    # =========================================================================

    # state_out: [state_slots, H_v, D_v, D_k] -> [state_slots, H_v, D_k, D_v]
    # Restore Python-side convention: key dimension first, value dimension second
    state_out = state_out_cann.transpose(-1, -2).contiguous()
    state_out = _ensure_nd_format(state_out)

    # out: [T_total, H_v, D_v] -> [B, T, H_v, D_v] -> [B, H_v, T, D_v]
    # Reverse: reshape back to 4D, then transpose sequence and head dimensions
    out_bnsd = (
        out_tnd.reshape(batch_size, seq_len, num_heads_v, dv)
        .transpose(1, 2)
        .contiguous()
    )

    return out_bnsd, state_out
