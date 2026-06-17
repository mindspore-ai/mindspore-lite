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
 * @brief ChunkGatedDeltaRule operator — CANN aclnn-backed chunked (prefill) linear attention.
 *
 * Wraps the AscendC `aclnnChunkGatedDeltaRule` for parallel chunk-level computation of the
 * Gated Delta Rule (the prefill counterpart of the token-by-token recurrent op). Inputs use
 * TND (Time-first) layout; the Python binding handles BNSD<->TND conversion.
 *
 * TND layout:
 *   query         [T, Hqk, Dk]      value         [T, Hv, Dv]
 *   key           [T, Hqk, Dk]      g             [T, Hv]          (float16, < 0)
 *   beta          [T, Hv]           initial_state [B, Hv, Dv, Dk]  (CANN: Dv first)
 *   cu_seqlens    [B+1] (int32)     ssm_state_indices [B] (int32)
 * Outputs:
 *   out           [T, Hv, Dv]       final_state   [B, Hv, Dv, Dk]
 * Attrs: chunk_size (int, default 64), scale_value (float, default 1/sqrt(Dk))
 */

#ifndef LITE_BOOST_OPS_PLUGIN_CHUNK_GATED_DELTA_RULE_H_
#define LITE_BOOST_OPS_PLUGIN_CHUNK_GATED_DELTA_RULE_H_

#include <tuple>
#include "ATen/Tensor.h"

/**
 * @brief NPU implementation of ChunkGatedDeltaRule.
 * @returns (out [T,Hv,Dv], final_state [B,Hv,Dv,Dk]) — both float16.
 */
std::tuple<at::Tensor, at::Tensor> ChunkGatedDeltaRuleLiteBoostImplNPU(
  const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &g, const at::Tensor &beta,
  const at::Tensor &initial_state, const at::Tensor &cu_seqlens, const at::Tensor &ssm_state_indices,
  int64_t chunk_size, double scale_value);

#endif  // LITE_BOOST_OPS_PLUGIN_CHUNK_GATED_DELTA_RULE_H_
