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

#ifndef OP_TILING_FUSED_RMS_NORM_NZ_TILING_H
#define OP_TILING_FUSED_RMS_NORM_NZ_TILING_H

#include <stdint.h>

// The tiling struct must live in the global namespace and stay byte-identical
// to op_kernel/fused_rms_norm_nz_tiling_data.h
// (REGISTER_TILING_DEFAULT maps the GM tiling block onto it verbatim).
#pragma pack(push, 8)
struct FusedRmsNormNZTilingData {
  uint32_t coreNum = 0;   // block dim actually launched
  uint32_t m = 0;         // flattened row count (NZ: prefix*m1t*16, pad rows included)
  uint32_t n = 0;         // row length (== gamma length, multiple of 32)
  uint32_t colChunk = 0;  // RETIRED by v2 whole-row residency; kept for v1 MindIR
                          // byte-compat (kernel ignores it, tiling writes n)
  uint32_t rowBlock = 0;  // rows per block group R (1..16)
  uint32_t isNz = 0;      // 1 = x/y are FRACTAL_NZ storage, 0 = ND
  float epsilon = 1e-6;   // rmsnorm epsilon
  uint32_t m1t = 0;       // NZ fractal row blocks (storage dim d-3); 0 for ND
};
#pragma pack(pop)
static_assert(sizeof(FusedRmsNormNZTilingData) == 32, "tiling struct size");

#endif  // OP_TILING_FUSED_RMS_NORM_NZ_TILING_H
