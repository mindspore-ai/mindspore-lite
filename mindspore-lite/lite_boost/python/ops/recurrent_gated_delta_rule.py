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
):
    """Recurrent GatedDeltaRule operator — CANN aclnn-backed recurrent linear attention decode.

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
        query (torch.Tensor): Query tensor of shape ``[B, H_q, T, D_k]``, dtype=bfloat16.
            Must be L2-normalized (L2 norm of each head vector is 1, value range [0, 1]).
            B=batch_size, H_q=num_query_heads, T=seq_len, D_k=key_dim.
        key (torch.Tensor): Key tensor of shape ``[B, H_q, T, D_k]``, dtype=bfloat16.
            Must be L2-normalized (same as query). In GQA/MQA scenarios, multiple query
            heads share the same set of keys.
        value (torch.Tensor): Value tensor of shape ``[B, H_v, T, D_v]``, dtype=bfloat16.
            H_v=num_value_heads (can differ from H_q to support GQA/MQA), D_v=value_dim.
        beta (torch.Tensor): Delta update step size of shape ``[B, H_v, T]``, dtype=bfloat16.
            Value range (0, 1). Controls the magnitude of each delta update: a larger beta
            causes new information to overwrite old memory more aggressively; a smaller
            beta tends to preserve existing memory.
        state (torch.Tensor): Recurrent state matrix of shape ``[B, H_v, D_k, D_v]``,
            dtype=bfloat16. Stores the cumulative key-value associations for linear attention.
            D_k is the key dimension (rows), D_v is the value dimension (columns).
            Can be initialized to zeros for the first call.
        actual_seq_lengths (torch.Tensor): Actual sequence lengths of shape ``[B]``, dtype=int32.
            Used for variable-length sequence inference. Each element represents the number
            of valid tokens in the corresponding batch.
            E.g., ``[4, 3, 5]`` means 3 batches with sequence lengths 4, 3, and 5.
        ssm_state_indices (torch.Tensor): SSM state indices of shape ``[B]``, dtype=int32.
            Indicates the index position of each batch item in the global state pool,
            used for state management in multi-batch scenarios. Typically ``[0, 1, 2, ..., B-1]``.
        g (torch.Tensor): Global decay gate of shape ``[B, H_v, T]``, dtype=float32.
            **Must be negative**. ``exp(g)`` serves as the state decay factor with range (0, 1).
            The more negative ``g`` is, the faster historical information is forgotten.
            E.g., when g=-1, approximately 37% of the historical state is retained per step.
        gk (torch.Tensor): Key-dimension gate of shape ``[B, H_v, T, D_k]``, dtype=float32.
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
        tuple[torch.Tensor, torch.Tensor]:
            - **out** (torch.Tensor): Attention output of shape ``[B, H_v, T, D_v]``, dtype=bfloat16.
              The linear attention result at each token position.
            - **state_out** (torch.Tensor): Updated recurrent state of shape ``[B, H_v, D_k, D_v]``,
              dtype=bfloat16. Must be passed as ``state`` input in the next recurrent step to
              form a state-passing chain.

    Raises:
        RuntimeError: If input tensor shapes, dtypes, or devices are invalid, or if the
            CANN operator execution fails.

    Note:
        - This operator only supports the **decode phase** (token-by-token inference),
          with sequence length T not exceeding 8. For parallel prefill computation,
          use the chunk-level operator.
        - Supports GQA (Grouped Query Attention) / MQA (Multi-Query Attention) modes,
          i.e., H_q can be greater than H_v, with multiple query heads sharing the same
          set of key/value heads.
        - All input tensors must reside on the same NPU device.
        - The CANN operator stores state internally as ``[B, H_v, D_v, D_k]`` layout
          (value dimension first). This function automatically performs the layout conversion.

    Example::

        import torch
        import lite_boost.ops as lite_ops

        # Qwen3.5-2B decode configuration
        B, H, T, Dk, Dv = 1, 64, 1, 64, 512

        # Initialize inputs (must satisfy CANN operator constraints)
        query  = torch.randn(B, H, T, Dk, device="npu:0", dtype=torch.bfloat16)
        key    = torch.randn(B, H, T, Dk, device="npu:0", dtype=torch.bfloat16)
        value  = torch.randn(B, H, T, Dv, device="npu:0", dtype=torch.bfloat16)
        beta   = torch.rand(B, H, T, device="npu:0", dtype=torch.bfloat16) * 0.9 + 0.05
        state  = torch.zeros(B, H, Dk, Dv, device="npu:0", dtype=torch.bfloat16)
        g      = -(torch.rand(B, H, T, device="npu:0") + 0.01)       # negative
        gk     = -(torch.rand(B, H, T, Dk, device="npu:0") + 0.01)   # negative

        actual_seq_lengths    = torch.tensor([T], dtype=torch.int32, device="npu:0")
        ssm_state_indices     = torch.tensor([0], dtype=torch.int32, device="npu:0")
        num_accepted_tokens   = torch.tensor([T], dtype=torch.int32, device="npu:0")

        # Execute recurrent inference
        output, state_out = lite_ops.recurrent_gated_delta_rule(
            query, key, value, beta, state,
            actual_seq_lengths, ssm_state_indices,
            g, gk, num_accepted_tokens,
            scale_value=1.0 / (Dk ** 0.5),
        )
        # output:    [1, 64, 1, 512]  -- attention output for the current token
        # state_out: [1, 64, 64, 512] -- updated recurrent state, passed to next step
    """
    # =========================================================================
    # 1. Extract dimensions from the BNSD (Batch, Num_heads, Seq_len, Dim) layout
    # =========================================================================
    # BNSD layout: [Batch, Num_heads, Seq_len, Dim]
    # - B (batch_size):    batch size
    # - H_q (num_heads_q): number of query heads (H_q >= H_v in GQA scenarios)
    # - T (seq_len):       sequence length (typically 1~8 in decode phase)
    # - D_k (dk):          Key/Query attention head dimension
    # - H_v (num_heads_v): number of value heads (can be less than H_q in GQA)
    # - D_v (dv):          Value attention head dimension
    batch_size = query.shape[0]
    num_heads_q = query.shape[1]
    seq_len = query.shape[2]
    dk = query.shape[3]
    # Value head count may differ from query (GQA/MQA mode)
    num_heads_v = value.shape[1]
    dv = value.shape[3]

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
    # Note: value uses num_heads_q instead of num_heads_v for reshape,
    # because the CANN operator internally handles GQA mapping via the Nv/Nk ratio.
    # =========================================================================

    # query: [B, H_q, T, D_k] -> [T_total, H_q, D_k]
    query_tnd = query.transpose(1, 2).reshape(-1, num_heads_q, dk).contiguous()

    # key: [B, H_q, T, D_k] -> [T_total, H_q, D_k]
    # Shares the same head count and dimension as query (key-query symmetry in linear attention)
    key_tnd = key.transpose(1, 2).reshape(-1, num_heads_q, dk).contiguous()

    # value: [B, H_v, T, D_v] -> [T_total, H_q, D_v]
    # Note: reshape uses num_heads_q (query head count) instead of num_heads_v (value head count),
    # because the CANN operator expects consistent head dimensions for query/key/value in TND layout.
    # The GQA head mapping is handled internally by the CANN operator via the Nv/Nk ratio.
    value_tnd = value.transpose(1, 2).reshape(-1, num_heads_q, dv).contiguous()

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
    # 3. Build cumulative sequence length vector cu_seqlen
    # =========================================================================
    # cu_seqlen locates the start and end positions of each batch within the
    # flattened T_total dimension.
    #
    # Example: actual_seq_lengths = [4, 3, 5] -> cu_seqlen = [0, 4, 7, 12]
    #   - batch 0: tokens [0, 4)
    #   - batch 1: tokens [4, 7)
    #   - batch 2: tokens [7, 12)
    # =========================================================================
    seq_lengths_int = actual_seq_lengths.int().contiguous()
    cu_seqlen = torch.zeros(batch_size + 1, dtype=torch.int32, device=query.device)
    cu_seqlen[1:] = torch.cumsum(seq_lengths_int, dim=0)

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
    #   query_tnd:  [T_total, H_q, D_k]      - L2-normalized query
    #   key_tnd:    [T_total, H_q, D_k]      - L2-normalized key
    #   value_tnd:  [T_total, H_q, D_v]      - value
    #   beta_tnd:   [T_total, H_v]            - Delta update step size (0, 1)
    #   state_cann: [B, H_v, D_v, D_k]       - recurrent state matrix
    #   cu_seqlen:  [B+1]                     - cumulative sequence lengths
    #   ssm_state_indices: [B]                - state pool indices
    #   g_tnd:      [T_total, H_v]            - global decay gate (negative)
    #   gk_tnd:     [T_total, H_v, D_k]      - key gate (negative)
    #   num_accepted_tokens: [B]              - accepted token counts
    #   scale_value: float                    - scale factor
    #
    # Output tensors:
    #   out_tnd:        [T_total, H_v, D_v]   - attention output
    #   state_out_cann: [B, H_v, D_v, D_k]    - updated recurrent state
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

    # state_out: CANN layout [B, H_v, D_v, D_k] -> Python layout [B, H_v, D_k, D_v]
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
