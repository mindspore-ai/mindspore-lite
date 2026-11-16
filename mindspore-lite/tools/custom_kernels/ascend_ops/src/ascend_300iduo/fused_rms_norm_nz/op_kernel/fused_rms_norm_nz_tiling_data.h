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

#ifndef FUSED_RMS_NORM_NZ_TILING_DATA_H
#define FUSED_RMS_NORM_NZ_TILING_DATA_H

#include "kernel_operator.h"  // NOLINT(build/include_subdir)

// FusedRmsNormNZTilingData MUST stay in the global namespace.
// CANN's REGISTER_TILING_DEFAULT / GET_TILING_DATA autogen reference this
// struct from global scope. Mirrored by
// op_host/fused_rms_norm_nz_tiling.h (byte-identical).
constexpr int kFusedRmsNormNZTilingAlign = 8;
#pragma pack(push, 8)
struct alignas(kFusedRmsNormNZTilingAlign) FusedRmsNormNZTilingData {
  uint32_t coreNum;   // number of cores used (blockDim)
  uint32_t m;         // flattened rows (NZ: prefix*m1t*16, pad rows included)
  uint32_t n;         // row length (multiple of 32)
  uint32_t colChunk;  // RETIRED by v2 whole-row residency; kept for v1 MindIR byte-compat (kernel ignores it)
  uint32_t rowBlock;  // rows per block group R (1..16)
  uint32_t isNz;      // 1 = x/y are FRACTAL_NZ storage, 0 = ND
  float epsilon;      // rms epsilon
  uint32_t m1t;       // NZ fractal row blocks (storage dim d-3); 0 for ND
};
#pragma pack(pop)
// Mirror of the host-side assert in op_host/fused_rms_norm_nz_tiling.h: a
// one-sided field change must fail here instead of silently shifting the
// GET_TILING_DATA layout.
static_assert(sizeof(FusedRmsNormNZTilingData) == 32, "tiling struct size");

#endif  // FUSED_RMS_NORM_NZ_TILING_DATA_H
