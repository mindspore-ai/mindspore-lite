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

#include "plugin/rain_fusion_attention.h"
#include <string_view>
#include "torch/library.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "plugin/pytorch_npu_helper.h"

namespace {
constexpr std::string_view kOpNameRainFusionAttention = "aclnnRainFusionAttention";
}  // namespace
std::tuple<at::Tensor, at::Tensor> RainFusionAttentionLiteBoostImplNPU(
  const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &select_idx,
  const at::Tensor &select_num_idx, at::IntArrayRef block_shape, const c10::optional<at::Tensor> &attn_mask,
  c10::OptionalIntArrayRef actual_seq_lengths, c10::OptionalIntArrayRef actual_seq_lengths_kv,
  const c10::optional<at::Tensor> &block_table, std::string q_input_layout, std::string kv_input_layout,
  int64_t num_key_value_heads, int64_t mask_type, double scale_value, int64_t inner_precise, int64_t block_size) {
  // Check the input layout
  TORCH_CHECK(q_input_layout == "TND" || q_input_layout == "BNSD",
              "q_input_layout only supports 'TND' and 'BNSD' now.");
  TORCH_CHECK(kv_input_layout == "TND" || kv_input_layout == "BNSD",
              "kv_input_layout only supports 'TND' and 'BNSD' now.");

  const at::Tensor &attn_mask_tensor = c10::value_or_else(attn_mask, [] { return at::Tensor(); });
  auto actual_seq_lengths_tensor = actual_seq_lengths.value_or(at::IntArrayRef{});
  auto actual_seq_lengths_kv_tensor = actual_seq_lengths_kv.value_or(at::IntArrayRef{});
  const at::Tensor &block_table_tensor = c10::value_or_else(block_table, [] { return at::Tensor(); });

  const char *q_input_layout_ptr = q_input_layout.data();
  const char *kv_input_layout_ptr = kv_input_layout.data();

  // Create an attention_out tensor with the same shape, dtype, and NPU format as the query tensor to store the
  // attention output Create a softmax_lse tensor with the same shape, dtype, and NPU format as the query tensor to
  // store the logsumexp values from the softmax operation
  at::Tensor attention_out =
    at_npu::native::empty_with_format(query.sizes(), query.options(), at_npu::native::get_npu_format(query));
  at::Tensor softmax_lse = at_npu::native::empty_with_format({query.sizes()[0], query.sizes()[1], query.sizes()[2]},
                                                             query.options(), at_npu::native::get_npu_format(query));

  EXEC_NPU_CMD<kOpNameRainFusionAttention>(query, key, value, select_idx, select_num_idx, block_shape, attn_mask_tensor,
                                           actual_seq_lengths_tensor, actual_seq_lengths_kv_tensor, block_table_tensor,
                                           q_input_layout_ptr, kv_input_layout_ptr, num_key_value_heads, mask_type,
                                           scale_value, inner_precise, block_size, attention_out, softmax_lse);
  return std::tuple<at::Tensor, at::Tensor>(attention_out, softmax_lse);
}
