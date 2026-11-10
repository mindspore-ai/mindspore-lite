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

#include "plugin/fused_qk_l2norm_chunk_gated_delta_rule.h"
#include <string_view>
#include "torch/library.h"
#include "plugin/chunk_gated_delta_rule_common.h"
#include "plugin/pytorch_npu_helper.h"

namespace {
constexpr std::string_view kOpNameFusedQkL2NormChunkGatedDeltaRule = "aclnnFusedQkL2NormChunkGatedDeltaRule";
}  // namespace

namespace lite_boost {

std::tuple<at::Tensor, at::Tensor> FusedQkL2NormChunkGatedDeltaRuleLiteBoostImplNPU(
  const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &beta,
  const at::Tensor &initial_state, const at::Tensor &actual_seq_lengths, const c10::optional<at::Tensor> &g,
  double scale_value, bool out_zero_pad_rows) {
  // Outputs and the optional gate are prepared by the shared helper; see
  // chunk_gated_delta_rule_common.h for the dtype/format contract.
  ChunkGatedDeltaRuleTensors tensors = PrepareChunkGatedDeltaRuleTensors(value, initial_state, g);

  // aclnn arg order (ascend_a2 op prototype): query, key, value, beta, initial_state,
  // actual_seq_lengths, g(optional), then the attrs in op-def declaration order
  // (scale_value, out_zero_pad_rows), then out, final_state.
  EXEC_NPU_CMD<kOpNameFusedQkL2NormChunkGatedDeltaRule>(query, key, value, beta, initial_state, actual_seq_lengths,
                                                        tensors.g, scale_value, out_zero_pad_rows, tensors.out,
                                                        tensors.final_state);

  return std::tuple<at::Tensor, at::Tensor>(tensors.out, tensors.final_state);
}

}  // namespace lite_boost
