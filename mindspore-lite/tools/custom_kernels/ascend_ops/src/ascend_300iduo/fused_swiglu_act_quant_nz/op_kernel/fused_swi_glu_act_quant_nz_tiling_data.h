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

#ifndef FUSED_SWI_GLU_ACT_QUANT_NZ_TILING_DATA_H
#define FUSED_SWI_GLU_ACT_QUANT_NZ_TILING_DATA_H

#include "kernel_operator.h"  // NOLINT(build/include_subdir)

// FusedSwiGLUActQuantNZTilingData MUST stay in the global namespace. CANN's
// REGISTER_TILING_DEFAULT / GET_TILING_DATA autogen reference this struct from
// global scope. Mirrored by op_host/fused_swi_glu_act_quant_nz_tiling.h
// (byte-identical).
constexpr int kFusedSwiGLUActQuantNZTilingAlign = 8;
#pragma pack(push, 8)
struct alignas(kFusedSwiGLUActQuantNZTilingAlign) FusedSwiGLUActQuantNZTilingData {
  uint32_t coreNum;    // number of AIV cores used (blockDim)
  uint32_t m;          // flattened token rows (LOGICAL rows; NZ pad rows
                       // excluded)
  uint32_t n;          // per-branch columns (inter size, multiple of 32)
  uint32_t colChunk;   // columns per UB chunk (multiple of 32)
  uint32_t hasSmooth;  // 1 when the optional smooth_scale input is present
  uint32_t rowBlock;   // rows per block R (8/4/2/1, divides the 8-row group)
  // NZ layout info in the ex-reserved fields (struct stays 32B so stale
  // pre-NZ tilings, which bake 0s here, decode as ND):
  uint32_t isNz;  // 1 = gate/up/y are FRACTAL_NZ storage
  uint32_t m1t;   // NZ row blocks (16 rows) per prefix dim (<= 4096,
                  // guarded by the uint16 stride limit m1t*16-1 <= 65535)
};
#pragma pack(pop)

static_assert(sizeof(FusedSwiGLUActQuantNZTilingData) == 32, "tiling data must stay 32B to match the host-side struct");

#endif  // FUSED_SWI_GLU_ACT_QUANT_NZ_TILING_DATA_H
