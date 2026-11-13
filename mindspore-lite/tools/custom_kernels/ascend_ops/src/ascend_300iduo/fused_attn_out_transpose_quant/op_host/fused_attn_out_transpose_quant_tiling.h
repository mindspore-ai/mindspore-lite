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

#ifndef OP_TILING_FUSED_ATTN_OUT_TRANSPOSE_QUANT_TILING_H
#define OP_TILING_FUSED_ATTN_OUT_TRANSPOSE_QUANT_TILING_H

#include <stdint.h>

// The tiling struct must live in the global namespace and stay byte-identical
// to op_kernel/fused_attn_out_transpose_quant_tiling_data.h
// (REGISTER_TILING_DEFAULT maps the GM tiling block onto it verbatim).
constexpr int kFusedAttnOutTransposeQuantTilingAlign = 8;
#pragma pack(push, 8)
struct alignas(kFusedAttnOutTransposeQuantTilingAlign) FusedAttnOutTransposeQuantTilingData {
  uint32_t coreNum = 0;    // block dim actually launched
  uint32_t m = 0;          // flattened token rows
  uint32_t h = 0;          // heads (flattened leading dims of attn)
  uint32_t d = 0;          // head dim (kHeadDim, 128)
  uint32_t n = 0;          // row length h * d
  uint32_t rowBlock = 0;   // rows per block R (1..16, multiple/divisor of 8)
  uint32_t hasSmooth = 0;  // 1 when smooth_scale is present
  uint32_t fold2d = 0;     // 1 = 2D DataCopyParams fold load (m <= 8192 at d=128)
};
#pragma pack(pop)
static_assert(sizeof(FusedAttnOutTransposeQuantTilingData) == 32, "tiling struct size");
#endif  // OP_TILING_FUSED_ATTN_OUT_TRANSPOSE_QUANT_TILING_H
