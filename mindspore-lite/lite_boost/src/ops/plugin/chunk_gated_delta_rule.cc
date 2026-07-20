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

std::tuple<at::Tensor, at::Tensor> ChunkGatedDeltaRuleLiteBoostImplNPU(const at::Tensor &query, const at::Tensor &key,
                                                                       const at::Tensor &value, const at::Tensor &beta,
                                                                       const at::Tensor &initial_state,
                                                                       const at::Tensor &actual_seq_lengths,
                                                                       const c10::optional<at::Tensor> &g,
                                                                       double scale_value) {
  // Outputs: out mirrors value's TND shape [T, Nv, Dv]; final_state mirrors initial_state
  // [B, Nv, Dv, Dk]. Both bf16 — InferDataType sets out<-query, final_state<-initial_state,
  // and the ascend_a2 op rejects fp16 in tiling (AnalyzeDtype). value/initial_state arrive
  // already cast to bf16 by the Python binding.
  at::Tensor out =
    at_npu::native::empty_with_format(value.sizes().vec(), value.options(), at_npu::native::get_npu_format(value));
  at::Tensor final_state = at_npu::native::empty_with_format(initial_state.sizes().vec(), initial_state.options(),
                                                             at_npu::native::get_npu_format(initial_state));

  // Optional g: materialize to an empty at::Tensor when absent, matching the rain_fusion
  // convention — the CANN op-api treats an empty aclTensor as "optional input not provided"
  // (hasGamma=0 in the kernel). When present, g must be float32 (op dtype for g is FLOAT).
  const at::Tensor &g_tensor = c10::value_or_else(g, [] { return at::Tensor(); });

  // aclnn arg order (ascend_a2 op prototype): query, key, value, beta, initial_state,
  // actual_seq_lengths, g(optional), scale_value(attr), out, final_state. (beta precedes g;
  // no cu_seqlens / ssm_state_indices / chunk_size — those belong to the ascend_300iduo op.)
  EXEC_NPU_CMD<kOpNameChunkGatedDeltaRule>(query, key, value, beta, initial_state, actual_seq_lengths, g_tensor,
                                           scale_value, out, final_state);

  return std::tuple<at::Tensor, at::Tensor>(out, final_state);
}
