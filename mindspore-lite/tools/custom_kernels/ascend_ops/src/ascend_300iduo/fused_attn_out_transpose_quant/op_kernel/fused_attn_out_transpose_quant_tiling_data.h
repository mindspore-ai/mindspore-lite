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

#ifndef FUSED_ATTN_OUT_TRANSPOSE_QUANT_TILING_DATA_H
#define FUSED_ATTN_OUT_TRANSPOSE_QUANT_TILING_DATA_H

#include "kernel_operator.h"  // NOLINT(build/include_subdir)

// FusedAttnOutTransposeQuantTilingData MUST stay in the global namespace.
// CANN's REGISTER_TILING_DEFAULT / GET_TILING_DATA autogen reference this
// struct from global scope. Mirrored by
// op_host/fused_attn_out_transpose_quant_tiling.h (byte-identical).
constexpr int kFusedAttnOutTransposeQuantTilingAlign = 8;
#pragma pack(push, 8)
struct alignas(kFusedAttnOutTransposeQuantTilingAlign) FusedAttnOutTransposeQuantTilingData {
  uint32_t coreNum;    // number of AI cores used (blockDim)
  uint32_t m;          // flattened token rows
  uint32_t h;          // heads (flattened leading dims of attn)
  uint32_t d;          // head dim (128)
  uint32_t n;          // row length h * d
  uint32_t rowBlock;   // rows per block R (1..16; 16/8 = whole 8-row groups, 4/2/1 = divisors of 8)
  uint32_t hasSmooth;  // 1 when the optional smooth_scale input is present
  uint32_t fold2d;     // 1 = 2D DataCopyParams fold load (m <= 8192 at d=128)
};
#pragma pack(pop)

#endif  // FUSED_ATTN_OUT_TRANSPOSE_QUANT_TILING_DATA_H
