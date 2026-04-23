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

#ifndef LITE_BOOST_OPS_PLUGIN_RAIN_FUSION_ATTENTION_H_
#define LITE_BOOST_OPS_PLUGIN_RAIN_FUSION_ATTENTION_H_

#include <string>
#include <tuple>
#include "ATen/Tensor.h"
#include "c10/util/Optional.h"

std::tuple<at::Tensor, at::Tensor> RainFusionAttentionLiteBoostImplNPU(
  const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &select_idx,
  const at::Tensor &select_num_idx, at::IntArrayRef block_shape, const c10::optional<at::Tensor> &attn_mask,
  c10::OptionalIntArrayRef actual_seq_lengths, c10::OptionalIntArrayRef actual_seq_lengths_kv,
  const c10::optional<at::Tensor> &block_table, std::string q_input_layout, std::string kv_input_layout,
  int64_t num_key_value_heads, int64_t mask_type, double scale_value, int64_t inner_precise, int64_t block_size);

#endif  // LITE_BOOST_OPS_PLUGIN_RAIN_FUSION_ATTENTION_H_
