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

#include "plugin/recurrent_gated_delta_rule.h"
#include <string_view>
#include "torch/library.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "plugin/pytorch_npu_helper.h"

namespace {
constexpr std::string_view kOpNameRecurrentGatedDeltaRule = "aclnnRecurrentGatedDeltaRule";
constexpr std::string_view kOpNameRecurrentGatedDeltaRule310P = "aclnnRecurrentGatedDeltaRule310P";

bool IsAscend310P() {
  const char *soc_name = aclrtGetSocName();
  return soc_name != nullptr && std::string_view(soc_name).find("310P") != std::string_view::npos;
}
}  // namespace

std::tuple<at::Tensor, at::Tensor> RecurrentGatedDeltaRuleLiteBoostImplNPU(
  const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &beta,
  const at::Tensor &state, const at::Tensor &actual_seq_lengths, const at::Tensor &ssm_state_indices,
  const at::Tensor &g, const at::Tensor &gk, const at::Tensor &num_accepted_tokens, double scale_value) {
  // Create output tensor with the same shape, dtype and NPU format as query
  auto out_size = value.sizes().vec();
  at::Tensor out = at_npu::native::empty_with_format(out_size, value.options(), at_npu::native::get_npu_format(value));
  at::Tensor state_out;
  if (IsAscend310P()) {
    state_out = state.clone();
    EXEC_NPU_CMD<kOpNameRecurrentGatedDeltaRule310P>(query, key, value, beta, state_out, actual_seq_lengths,
                                                     ssm_state_indices, g, gk, num_accepted_tokens, scale_value, out);
  } else {
    // The built-in CANN operator updates its state reference input in place.
    state_out = state.clone();
    EXEC_NPU_CMD<kOpNameRecurrentGatedDeltaRule>(query, key, value, beta, state_out, actual_seq_lengths,
                                                 ssm_state_indices, g, gk, num_accepted_tokens,
                                                 static_cast<float>(scale_value), out);
  }

  return std::tuple<at::Tensor, at::Tensor>(out, state_out);
}
