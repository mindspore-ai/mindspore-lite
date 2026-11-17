/**
 * Copyright 2026 Huawei Technologies Co., Ltd
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 * You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License file in the root of the software repository for the full text of the License.
 */

#ifndef INNER_QUANT_BATCH_MATMUL_V3_KERNEL_TILING_DATA_H
#define INNER_QUANT_BATCH_MATMUL_V3_KERNEL_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

// Field-for-field copy of CANN built-in QuantMatmulPertokenTilingDataArch20
// (cann_ops/ops-nn/matmul/quant_batch_matmul_v3/op_kernel/
// quant_batch_matmul_v3_kernel_tiling_data.h:94-112). The vendored kernel
// class reads this struct verbatim via GET_TILING_DATA (vendor-op
// wrapper-generated accessor, REGISTER_TILING_DEFAULT contract -- see
// inner_quant_batch_matmul_v3.cpp), so the
// layout (and the #pragma pack(8) + alignas(8) discipline) must not drift
// from the built-in. Sizes: 16 * uint32 + 2 * bool = 68 bytes, padded to 72.
constexpr size_t kTilingAlignBytes = 8;
#pragma pack(push, 8)
struct alignas(kTilingAlignBytes) QuantMatmulPertokenTilingDataArch20 {
  uint32_t batchSize;
  uint32_t m;
  uint32_t k;
  uint32_t n;
  uint32_t m0;
  uint32_t k0;
  uint32_t n0;
  uint32_t mLoop;
  uint32_t kLoop;
  uint32_t nLoop;
  uint32_t coreLoop;
  uint32_t blockDim;
  uint32_t swizzleDirect;
  uint32_t swizzleCount;
  uint32_t biasDtype;
  bool withBias;
  bool biasWithBatch;
};
#pragma pack(pop)

#endif
