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

#include <torch/library.h>
#include "plugin/rain_fusion_attention.h"

// RainFusionAttention
TORCH_LIBRARY(lite_boost, m) {
  m.def(R"str(
    rain_fusion_attention(
      Tensor query,
      Tensor key,
      Tensor value,
      Tensor select_idx,
      Tensor select_num_idx,
      int[] block_shape,
      Tensor? attn_mask=None,
      int[]? actual_seq_lengths=None,
      int[]? actual_seq_lengths_kv=None,
      Tensor? block_table=None,
      str q_input_layout="TND",
      str kv_input_layout="TND",
      int num_key_value_heads=1,
      int mask_type=0,
      float scale_value=1.0,
      int inner_precise=1,
      int block_size=0
    ) -> (Tensor, Tensor)
  )str");
}

TORCH_LIBRARY_IMPL(lite_boost, PrivateUse1, m) {
  m.impl("rain_fusion_attention", &RainFusionAttentionLiteBoostImplNPU);
}
