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

/* fused_causal_conv_silu_tiling_data.h Host/device tiling-data contract for FusedCausalConvSilu. */

#ifndef FUSED_CAUSAL_CONV_SILU_TILING_DATA_H
#define FUSED_CAUSAL_CONV_SILU_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

constexpr int64_t FUSED_CONV_TILE_C = 512;

constexpr int64_t FUSED_CONV_TR = 16;

constexpr uint64_t FUSED_CAUSAL_CONV_SILU_STRUCT_ALIGNAS = 8;
#pragma pack(push, 8)
struct alignas(FUSED_CAUSAL_CONV_SILU_STRUCT_ALIGNAS) FusedCausalConvSiluTilingData {
  int64_t T;
  int64_t C;
  int64_t S;
  int64_t K;
  int64_t usedCoreNum;
};
#pragma pack(pop)

#endif
