/**
 * Copyright 2026 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file recurrent_gated_delta_rule.h
 * @brief RecurrentGatedDeltaRule operator — CANN aclnn-backed recurrent linear attention.
 *
 * This header declares the NPU implementation of the RecurrentGatedDeltaRule operator,
 * which wraps the Ascend CANN backend `aclnnRecurrentGatedDeltaRule` for high-performance
 * token-by-token recurrent inference (decode phase) in hybrid linear attention models
 * such as Qwen3.5.
 *
 * == Algorithm Background ==
 *
 * The Gated Delta Rule originates from "Gated Delta Networks: Improving Mamba2 with
 * Delta Rule" (Yang et al., ICLR 2025, arXiv:2412.06464). It combines recurrent state
 * updates from linear attention with two complementary mechanisms:
 *
 * - **Gating**: Uses exponential decay factors `exp(g)` for rapid erasure/forgetting of
 *   recurrent memory. `g` is negative, making `exp(g) ∈ (0, 1)`, achieving per-token
 *   state decay. Additionally, `exp(gk)` applies per-dimension decay along the key
 *   dimension for finer-grained memory control.
 *
 * - **Delta Rule**: Borrowed from the Delta learning rule, it does not overwrite memory
 *   directly. Instead, it computes the error (delta) between the current value and the
 *   stored memory, then applies a targeted correction along the key direction with a
 *   step size controlled by `beta`. This makes memory updates more precise and
 *   significantly improves retrieval and long-context task performance.
 *
 * Recurrence formulas (decode phase, computed per token):
 *
 *   For the i-th token:
 *     1. Decay + key gating:
 *        S_i = S_{i-1} * exp(g_i) * exp(gk_i)
 *        where S is the recurrent state matrix [H, D_k, D_v].
 *
 *     2. Memory retrieval:
 *        kv_mem_i = S_i @ k_i
 *
 *     3. Delta update:
 *        delta_i = (v_i - kv_mem_i) * beta_i
 *        S_i = S_i + k_i^T @ delta_i
 *
 *     4. Query readout:
 *        o_i = S_i^T @ q_i
 *
 * == Implementation Architecture ==
 *
 * The operator is structured in three layers:
 *
 *   1. **Python binding layer** (`python/ops/recurrent_gated_delta_rule.py`):
 *      Converts between user-friendly BNSD layout [B, H, T, D] and the TND layout
 *      [T_total, H, D] required by the CANN operator. Also handles cu_seqlen
 *      construction and state matrix layout adaptation (D_k,D_v <-> D_v,D_k).
 *
 *   2. **C++ operator layer** (this file + `recurrent_gated_delta_rule.cc`):
 *      Registered via `TORCH_LIBRARY(lite_boost, ...)` with `PrivateUse1` (NPU) dispatch.
 *      The implementation allocates output tensors and invokes the CANN backend via
 *      the `EXEC_NPU_CMD` macro.
 *
 *   3. **CANN backend** (`aclnnRecurrentGatedDeltaRule`):
 *      Hardware-accelerated execution on Ascend NPU, with automatic workspace
 *      allocation and asynchronous execution.
 *
 * == TND Layout Convention ==
 *
 * All tensor inputs to this function use TND (Time-first) layout:
 *   - query:            [T_total, H_q, D_k]
 *   - key:              [T_total, H_q, D_k]
 *   - value:            [T_total, H_q, D_v]
 *   - beta:             [T_total, H_v]
 *   - state (CANN):     [B, H_v, D_v, D_k]   (value dim first, key dim second)
 *   - actual_seq_lengths: [B+1] (cumulative, i.e. cu_seqlen)
 *   - g:                [T_total, H_v]
 *   - gk:               [T_total, H_v, D_k]
 *
 * Outputs:
 *   - out:              [T_total, H_v, D_v]
 *   - state_out:        [B, H_v, D_v, D_k]
 *
 * where T_total = sum(actual_seq_lengths_per_batch), and H_q >= H_v for GQA/MQA support.
 *
 * == CANN Operator Constraints ==
 *
 *   - Per-batch sequence length: 0 < Li <= 8
 *   - Number of attention heads: 0 < Nk <= 256, Nk <= Nv <= 256, Nv % Nk == 0
 *   - Attention head dimensions: 0 < Dk <= 512, 0 < Dv <= 512
 *   - query/key must be L2-normalized (value range [0, 1])
 *   - g < 0  (decay gate, ensuring exp(g) ∈ (0, 1))
 *   - gk < 0 (key gate, ensuring exp(gk) ∈ (0, 1])
 *   - 0 < beta < 1 (Delta update step size)
 *
 * == Related Files ==
 *
 *   - Python binding:      `lite_boost/python/ops/recurrent_gated_delta_rule.py`
 *   - C++ implementation:  `lite_boost/src/ops/plugin/recurrent_gated_delta_rule.cc`
 *   - Operator registration: `lite_boost/src/ops/register_ops.cc`
 *   - Test cases:           `lite_boost/test/ops/test_recurrent_gated_delta_rule.py`
 *   - Operator documentation: `lite_boost/docs/ops/RecurrentGatedDeltaRule.md`
 *
 * == References ==
 *
 *   - Yang, S., Kautz, J., & Hatamizadeh, A. (2025). Gated Delta Networks:
 *     Improving Mamba2 with Delta Rule. ICLR 2025. arXiv:2412.06464.
 *     https://arxiv.org/abs/2412.06464
 */

#ifndef LITE_BOOST_OPS_PLUGIN_RECURRENT_GATED_DELTA_RULE_H_
#define LITE_BOOST_OPS_PLUGIN_RECURRENT_GATED_DELTA_RULE_H_

#include <string>
#include <tuple>
#include "ATen/Tensor.h"
#include "c10/util/Optional.h"

/**
 * @brief NPU implementation of RecurrentGatedDeltaRule operator.
 *
 * Invokes the CANN `aclnnRecurrentGatedDeltaRule` backend to perform token-by-token
 * recurrent forward pass of the Gated Delta Rule, updating the recurrent state and
 * producing the attention output.
 *
 * All tensor inputs use TND (Time-first) layout. The Python binding layer handles
 * BNSD <-> TND conversion before calling this function.
 *
 * @param query               [T_total, H_q, D_k]      L2-normalized query tensor (bfloat16).
 * @param key                 [T_total, H_q, D_k]      L2-normalized key tensor (bfloat16).
 * @param value               [T_total, H_q, D_v]      Value tensor (bfloat16).
 * @param beta                [T_total, H_v]            Delta update step size in (0,1) (bfloat16).
 * @param state               [B, H_v, D_v, D_k]       Recurrent state matrix (bfloat16).
 *                                                   CANN convention: value dim first, key dim second.
 * @param actual_seq_lengths  [B+1]                     Cumulative sequence lengths (int32).
 *                                                   cu_seqlen[i] is the offset into T_total where
 *                                                   batch item i begins.
 * @param ssm_state_indices   [B]                       State pool indices for each batch item (int32).
 * @param g                   [T_total, H_v]            Global decay gate, must be negative (float32).
 *                                                   exp(g) ∈ (0, 1) controls state decay rate.
 * @param gk                  [T_total, H_v, D_k]       Key-dimension gate, must be negative (float32).
 *                                                   exp(gk) applies per-dimension decay along D_k.
 * @param num_accepted_tokens [B]                       Number of accepted tokens per batch (int32).
 *                                                   For speculative decoding; same as actual_seq_lengths
 *                                                   in standard inference.
 * @param scale_value         Attention scale factor. Typically 1.0 / sqrt(D_k).
 *
 * @return std::tuple<at::Tensor, at::Tensor>
 *   - out:        [T_total, H_v, D_v]   Attention output (bfloat16).
 *   - state_out:  [B, H_v, D_v, D_k]   Updated recurrent state (bfloat16).
 */
std::tuple<at::Tensor, at::Tensor> RecurrentGatedDeltaRuleLiteBoostImplNPU(
  const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &beta,
  const at::Tensor &state, const at::Tensor &actual_seq_lengths, const at::Tensor &ssm_state_indices,
  const at::Tensor &g, const at::Tensor &gk, const at::Tensor &num_accepted_tokens, double scale_value);

#endif  // LITE_BOOST_OPS_PLUGIN_RECURRENT_GATED_DELTA_RULE_H_
