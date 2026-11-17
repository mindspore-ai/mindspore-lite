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

#ifndef FUSED_QK_NORM_ROPE_BSH_TILING_DATA_H
#define FUSED_QK_NORM_ROPE_BSH_TILING_DATA_H

#include "kernel_operator.h"  // NOLINT(build/include_subdir)

// FusedQKNormRopeBSHTilingData MUST stay in the global namespace. CANN's
// REGISTER_TILING_DEFAULT / GET_TILING_DATA autogen references this struct
// from global scope. Mirrored by op_host/fused_qk_norm_rope_bsh_tiling.h
// (byte-identical).
constexpr int kFusedQKNormRopeBSHTilingAlign = 8;
#pragma pack(push, 8)
struct alignas(kFusedQKNormRopeBSHTilingAlign) FusedQKNormRopeBSHTilingData {
  uint32_t coreNum;    // number of cores used (blockDim)
  uint32_t m;          // flattened token rows
  uint32_t hq;         // query heads
  uint32_t hkv;        // key/value heads
  uint32_t d;          // head dim (pow2, 32..128)
  uint32_t rowBlock;   // rows per block tile
  float epsilon;       // rms norm epsilon
  float invD;          // 1/d precomputed on host
  uint32_t reserved0;  // reserved
  uint32_t reserved1;  // reserved
};
#pragma pack(pop)

#endif  // FUSED_QK_NORM_ROPE_BSH_TILING_DATA_H
