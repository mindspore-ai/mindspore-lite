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

#ifndef FUSED_GELU_TANH_NZ_TILING_DATA_H
#define FUSED_GELU_TANH_NZ_TILING_DATA_H

#include "kernel_operator.h"  // NOLINT(build/include_subdir)

// FusedGeluTanhNZTilingData MUST stay in the global namespace. CANN's
// REGISTER_TILING_DEFAULT / GET_TILING_DATA autogen reference this struct from
// global scope. Mirrored by op_host/fused_gelu_tanh_nz_tiling.h
// (byte-identical, locked by the same static_assert on both sides).
constexpr int kFusedGeluTanhNZTilingAlign = 8;
#pragma pack(push, 8)
struct alignas(kFusedGeluTanhNZTilingAlign) FusedGeluTanhNZTilingData {
  uint64_t totalElems;  // flattened element count (product of storage dims)
  uint32_t coreNum;     // number of cores used (blockDim)
  uint32_t chunkElems;  // elements per UB chunk (multiple of 256, fits UB)
  uint32_t reserved0;   // reserved, keeps 8-byte packing
  uint32_t reserved1;   // reserved
  uint32_t reserved2;   // reserved
};
#pragma pack(pop)
static_assert(sizeof(FusedGeluTanhNZTilingData) == 32, "tiling struct size");

#endif  // FUSED_GELU_TANH_NZ_TILING_DATA_H
