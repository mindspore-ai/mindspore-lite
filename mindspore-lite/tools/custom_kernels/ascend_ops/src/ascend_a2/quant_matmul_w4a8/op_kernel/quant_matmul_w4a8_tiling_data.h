/**
 * modified from
 * https://gitcode.com/cann/ops-nn/blob/master/matmul/quant_batch_matmul_v4/op_kernel/quant_batch_matmul_v4_tiling_data.h
 * Copyright 2026 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file quant_matmul_w4a8_tiling_data.h
 * @brief V5 MSD tiling data — exact copy from QuantBatchMatmulV4MsdTilingData.
 */

#ifndef QUANT_MATMUL_W4A8_TILING_DATA_H
#define QUANT_MATMUL_W4A8_TILING_DATA_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

#pragma pack(push, 8)
struct alignas(8) QuantMatmulW4a8TilingData {
  uint8_t coreNum;
  uint32_t vBaseM;
  uint32_t ubRestBytes;
  uint32_t parallNum;
  uint32_t ubCalSize;
  uint32_t mSize;
  uint32_t kSize;
  uint32_t nSize;
  uint32_t groupSize;
  TCubeTiling matmulTiling;
};
#pragma pack(pop)

#endif  // QUANT_MATMUL_W4A8_TILING_DATA_H
