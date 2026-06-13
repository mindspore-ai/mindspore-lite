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

#include "plugin/chunk_gated_delta_rule.h"
#include <string_view>
#include "torch/library.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "plugin/pytorch_npu_helper.h"

namespace {
constexpr std::string_view kOpNameChunkGatedDeltaRule = "aclnnChunkGatedDeltaRule";
}  // namespace

std::tuple<at::Tensor, at::Tensor> ChunkGatedDeltaRuleLiteBoostImplNPU(
  const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &g, const at::Tensor &beta,
  const at::Tensor &initial_state, const at::Tensor &cu_seqlens, const at::Tensor &ssm_state_indices,
  int64_t chunk_size, double scale_value) {
  // Outputs: out mirrors value's shape [T,Hv,Dv]; final_state mirrors initial_state
  // [B,Hv,Dv,Dk]. Both are float16 (per op def / InferDataType).
  at::Tensor out =
    at_npu::native::empty_with_format(value.sizes().vec(), value.options(), at_npu::native::get_npu_format(value));
  at::Tensor final_state = at_npu::native::empty_with_format(initial_state.sizes().vec(), initial_state.options(),
                                                             at_npu::native::get_npu_format(initial_state));

  EXEC_NPU_CMD<kOpNameChunkGatedDeltaRule>(query, key, value, g, beta, initial_state, cu_seqlens, ssm_state_indices,
                                           chunk_size, static_cast<float>(scale_value), out, final_state);

  return std::tuple<at::Tensor, at::Tensor>(out, final_state);
}
