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

#include "plugin/qkv_rms_norm_rope_cache.h"
#include <string_view>
#include "torch/library.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "plugin/pytorch_npu_helper.h"

namespace {
constexpr std::string_view kOpNameQkvRmsNormRopeCache = "aclnnQkvRmsNormRopeCache";
constexpr int64_t kMinHeadNumsSize = 3;
constexpr int64_t kCacheBlockSize = 32;
}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> QkvRmsNormRopeCacheLiteBoostImplNPU(
  const at::Tensor &qkv, const at::Tensor &q_gamma, const at::Tensor &k_gamma, const at::Tensor &cos,
  const at::Tensor &sin, const at::Tensor &index, at::IntArrayRef qkv_size, at::IntArrayRef head_nums, double epsilon,
  std::string cache_mode) {
  TORCH_CHECK(head_nums.size() >= kMinHeadNumsSize, "head_nums must be [Nq, Nk, Nv]");
  const int64_t num_head_q = head_nums[0];
  const int64_t num_head_k = head_nums[1];
  const int64_t num_head_v = head_nums[2];
  const int64_t num_token = qkv.size(0);
  const int64_t head_dim = q_gamma.size(0);
  const char *cache_mode_ptr = cache_mode.c_str();

  auto format = at_npu::native::get_npu_format(qkv);
  auto options = qkv.options();
  at::Tensor q_out = at_npu::native::empty_with_format({num_token, num_head_q * head_dim}, options, format);
  at::Tensor q_proto = at_npu::native::empty_with_format({num_token, num_head_q * head_dim}, options, format);
  at::Tensor k_out = at_npu::native::empty_with_format({num_token, num_head_k * head_dim}, options, format);
  at::Tensor v_out = at_npu::native::empty_with_format({num_token, num_head_v * head_dim}, options, format);
  constexpr int64_t kNzD0 = 16;
  const int64_t nz_d1 = head_dim / kNzD0;
  const int64_t block_num = (num_token + kCacheBlockSize - 1) / kCacheBlockSize;
  at::Tensor k_cache =
    at_npu::native::empty_with_format({block_num, num_head_k * nz_d1, kCacheBlockSize, kNzD0}, options, format);
  at::Tensor v_cache =
    at_npu::native::empty_with_format({block_num, num_head_v * nz_d1, kCacheBlockSize, kNzD0}, options, format);

  EXEC_NPU_CMD<kOpNameQkvRmsNormRopeCache>(qkv, q_gamma, k_gamma, cos, sin, index, q_out, k_cache, v_cache,
                                           c10::optional<at::Tensor>(), c10::optional<at::Tensor>(),
                                           c10::optional<at::Tensor>(), c10::optional<at::Tensor>(), qkv_size,
                                           head_nums, epsilon, cache_mode_ptr, q_proto, k_out, v_out);
  return std::tuple<at::Tensor, at::Tensor, at::Tensor>(q_out, k_out, v_out);
}
