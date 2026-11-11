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

#ifndef FUSED_ADD_RMS_NORM_SMOOTH_QUANT_TILING_DATA_H
#define FUSED_ADD_RMS_NORM_SMOOTH_QUANT_TILING_DATA_H

#include "kernel_operator.h"  // NOLINT(build/include_subdir)

// FusedAddRmsNormSmoothQuantTilingData MUST stay in the global namespace.
// CANN's REGISTER_TILING_DEFAULT / GET_TILING_DATA autogen reference this
// struct from global scope. Mirrored by
// op_host/fused_add_rms_norm_smooth_quant_tiling.h (byte-identical).
constexpr int kFusedAddRmsNormSmoothQuantTilingAlign = 8;
#pragma pack(push, 8)
struct alignas(kFusedAddRmsNormSmoothQuantTilingAlign) FusedAddRmsNormSmoothQuantTilingData {
  uint32_t coreNum;    // number of AIV cores used (blockDim)
  uint32_t m;          // flattened token rows (LOGICAL; NZ pad rows excluded)
  uint32_t n;          // hidden columns (multiple of 32)
  uint32_t colChunk;   // columns per UB chunk (multiple of 32)
  uint32_t rowBlock;   // rows per block R; host emits a divisor of 8 (8/4/2/1)
  uint32_t hasSmooth;  // 1 when the optional smooth_scale input is present
  float epsilon;       // rms epsilon
  // NZ layout info, bit-packed to keep the struct 32B (legacy pre-NZ
  // tilings bake 0 here and decode as ND):
  //   bit 0     isNz: 1 = x1/x2/res_out/y are FRACTAL_NZ storage
  //   bits 1..12 m1t: NZ row blocks (16 rows) per prefix dim
  uint32_t nzInfo;
};
#pragma pack(pop)

static_assert(sizeof(FusedAddRmsNormSmoothQuantTilingData) == 32, "tiling struct size");

#endif  // FUSED_ADD_RMS_NORM_SMOOTH_QUANT_TILING_DATA_H
