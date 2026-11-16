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

#ifndef OP_TILING_FUSED_QK_NORM_ROPE_BSH_TILING_H
#define OP_TILING_FUSED_QK_NORM_ROPE_BSH_TILING_H

#include <stdint.h>

// The tiling struct must live in the global namespace and stay byte-identical
// to op_kernel/fused_qk_norm_rope_bsh_tiling_data.h (REGISTER_TILING_DEFAULT
// maps the GM tiling block onto it verbatim). Same 40-byte layout as
// fused_qk_norm_rope: keeping the field set identical makes cross-op tiling
// audits diffable.
#pragma pack(push, 8)
struct FusedQKNormRopeBSHTilingData {
  uint32_t coreNum = 0;   // block dim actually launched
  uint32_t m = 0;         // flattened token rows
  uint32_t hq = 0;        // query heads
  uint32_t hkv = 0;       // key/value heads
  uint32_t d = 0;         // head dim (pow2, 32..128)
  uint32_t rowBlock = 0;  // rows per block tile
  float epsilon = 1e-6;   // rms norm epsilon
  float invD = 0.0f;      // 1/d precomputed on host
  uint32_t reserved0 = 0;
  uint32_t reserved1 = 0;
};
#pragma pack(pop)
static_assert(sizeof(FusedQKNormRopeBSHTilingData) == 40, "tiling struct size");

#endif  // OP_TILING_FUSED_QK_NORM_ROPE_BSH_TILING_H
