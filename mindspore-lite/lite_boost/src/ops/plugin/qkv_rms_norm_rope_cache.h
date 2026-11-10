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

#ifndef LITE_BOOST_OPS_PLUGIN_QKV_RMS_NORM_ROPE_CACHE_H_
#define LITE_BOOST_OPS_PLUGIN_QKV_RMS_NORM_ROPE_CACHE_H_

#include <string>
#include <tuple>
#include "ATen/Tensor.h"
#include "c10/util/Optional.h"

// aclnnQkvRmsNormRopeCache: fused q_norm + k_norm + rotate-half RoPE + paged KV cache
std::tuple<at::Tensor, at::Tensor, at::Tensor> QkvRmsNormRopeCacheLiteBoostImplNPU(
  const at::Tensor &qkv, const at::Tensor &q_gamma, const at::Tensor &k_gamma, const at::Tensor &cos,
  const at::Tensor &sin, const at::Tensor &index, at::IntArrayRef qkv_size, at::IntArrayRef head_nums, double epsilon,
  std::string cache_mode);

#endif
