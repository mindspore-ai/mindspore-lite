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

#ifndef OP_TILING_FUSED_ADD_RMS_NORM_SMOOTH_QUANT_TILING_H
#define OP_TILING_FUSED_ADD_RMS_NORM_SMOOTH_QUANT_TILING_H

#include <stdint.h>

// The tiling struct must live in the global namespace and stay byte-identical
// to op_kernel/fused_add_rms_norm_smooth_quant_tiling_data.h
// (REGISTER_TILING_DEFAULT maps the GM tiling block onto it verbatim).
constexpr int kFusedAddRmsNormSmoothQuantTilingAlign = 8;
#pragma pack(push, 8)
struct alignas(kFusedAddRmsNormSmoothQuantTilingAlign) FusedAddRmsNormSmoothQuantTilingData {
  uint32_t coreNum = 0;    // block dim actually launched
  uint32_t m = 0;          // flattened row count (LOGICAL rows; NZ pad rows
                           // excluded -- token_scale is [M] ND and must never
                           // be written past the logical end)
  uint32_t n = 0;          // column count (hidden)
  uint32_t colChunk = 0;   // columns per pass-1 chunk (multiple of 32)
  uint32_t rowBlock = 0;   // rows per block R; tiling emits a divisor of 8 (8/4/2/1)
  uint32_t hasSmooth = 0;  // 1 when smooth_scale is present
  float epsilon = 1e-6;    // rmsnorm epsilon
  // NZ layout info, bit-packed to keep the struct 32B so stale pre-NZ
  // legacy tilings (which bake 0 here) safely decode as ND:
  //   bit 0     isNz: 1 = x1/x2/res_out/y are FRACTAL_NZ storage
  //   bits 1..12 m1t: NZ row blocks (16 rows) per prefix dim (<= 4096,
  //             guarded by the uint16 stride limit m1t*16-1 <= 65535)
  uint32_t nzInfo = 0;
};
#pragma pack(pop)
static_assert(sizeof(FusedAddRmsNormSmoothQuantTilingData) == 32, "tiling struct size");
#endif  // OP_TILING_FUSED_ADD_RMS_NORM_SMOOTH_QUANT_TILING_H
