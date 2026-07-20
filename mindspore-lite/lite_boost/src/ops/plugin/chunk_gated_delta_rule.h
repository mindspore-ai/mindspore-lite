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
 * @file chunk_gated_delta_rule.h
 * @brief ChunkGatedDeltaRule operator — CANN aclnn-backed chunked (prefill) Gated Delta Rule.
 *
 * Wraps the AscendC `aclnnChunkGatedDeltaRule` shipped under ascend_a2 (ascend910b). The op
 * implements the chunked Gated Delta Rule (the prefill counterpart of the token-by-token
 * recurrent op). Interface mirrors the ascend_a2 op prototype verbatim (TND layout):
 *
 *   query             [T, Nk, Dk]   bf16
 *   key               [T, Nk, Dk]   bf16
 *   value             [T, Nv, Dv]   bf16
 *   beta              [T, Nv]       bf16
 *   initial_state     [B, Nv, Dv, Dk]  bf16   (Dv-first, matches op storage)
 *   actual_seq_lengths [B]          int32   (T = sum(actual_seq_lengths))
 *   g (optional)      [T, Nv]       float32 (< 0); absent -> no decay gate (hasGamma=0)
 *   out               [T, Nv, Dv]   bf16
 *   final_state       [B, Nv, Dv, Dk] bf16
 *   attr scale_value  float (default 1.0)
 *
 * The Python binding handles BNSD<->TND conversion and the (Dk,Dv)<->(Dv,Dk) state transpose.
 * Targets ascend910b only — the op-def is single-SoC and tiling rejects fp16 / FP32-state.
 */

#ifndef LITE_BOOST_OPS_PLUGIN_CHUNK_GATED_DELTA_RULE_H_
#define LITE_BOOST_OPS_PLUGIN_CHUNK_GATED_DELTA_RULE_H_

#include <tuple>
#include "ATen/Tensor.h"
#include "c10/util/Optional.h"

/**
 * @brief NPU implementation of ChunkGatedDeltaRule (ascend910b / ascend_a2 op spec).
 *
 * Tensors are passed in TND layout (the Python binding converts from BNSD). `g` is optional:
 * pass c10::nullopt (or an empty at::Tensor) to take the no-decay-gate path.
 *
 * @returns (out [T,Nv,Dv] bf16, final_state [B,Nv,Dv,Dk] bf16)
 */
std::tuple<at::Tensor, at::Tensor> ChunkGatedDeltaRuleLiteBoostImplNPU(const at::Tensor &query, const at::Tensor &key,
                                                                       const at::Tensor &value, const at::Tensor &beta,
                                                                       const at::Tensor &initial_state,
                                                                       const at::Tensor &actual_seq_lengths,
                                                                       const c10::optional<at::Tensor> &g,
                                                                       double scale_value);

#endif  // LITE_BOOST_OPS_PLUGIN_CHUNK_GATED_DELTA_RULE_H_
