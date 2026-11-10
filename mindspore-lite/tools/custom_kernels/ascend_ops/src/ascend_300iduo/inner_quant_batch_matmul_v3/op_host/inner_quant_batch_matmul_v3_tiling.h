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

#ifndef INNER_QUANT_BATCH_MATMUL_V3_TILING_H
#define INNER_QUANT_BATCH_MATMUL_V3_TILING_H

#include <cstdint>

// QuantMatmulPertokenTilingDataArch20 MUST stay in the global namespace and
// MUST stay byte-identical to the kernel-side mirror in
// op_kernel/inner_quant_batch_matmul_v3_tiling_data.h (itself a copy of the
// CANN built-in struct, cann_ops/ops-nn/matmul/quant_batch_matmul_v3/
// op_kernel/quant_batch_matmul_v3_kernel_tiling_data.h:94-112): the host
// writes the byte stream here, the device reads it via
// GET_TILING_DATA (vendor-op wrapper-generated; the built-in's
// GET_TILING_DATA_WITH_STRUCT form is not generated for vendor ops), and
// the vendored arch20 kernel class reads the fields verbatim. 16 * uint32 + 2 * bool = 68 bytes padded to 72.
constexpr int kInnerQuantBatchMatmulV3TilingAlign = 8;
#pragma pack(push, 8)
struct alignas(kInnerQuantBatchMatmulV3TilingAlign) QuantMatmulPertokenTilingDataArch20 {
  uint32_t batchSize;      // flattened leading dims of x1/x2
  uint32_t m;              // x1 origin [.., m, k]
  uint32_t k;              // contraction dim (multiple of 32)
  uint32_t n;              // x2 origin [.., n, k] (transpose_x2=true)
  uint32_t m0;             // m tile (16..256, multiple of 16)
  uint32_t k0;             // k chunk per L1 fill (multiple of 256 on 310P)
  uint32_t n0;             // n tile (16..256, multiple of 16)
  uint32_t mLoop;          // ceil(m / m0)
  uint32_t kLoop;          // ceil(k / k0)
  uint32_t nLoop;          // ceil(n / n0)
  uint32_t coreLoop;       // batchSize * mLoop * nLoop (nLoop after spread)
  uint32_t blockDim;       // min(coreLoop, aic core num)
  uint32_t swizzleDirect;  // 0 = Zn, 1 = Nz (tile visit order)
  uint32_t swizzleCount;   // swizzle group size
  uint32_t biasDtype;      // ge dtype int of the bias (int32 or float)
  bool withBias;           // bias present AND offset absent
  bool biasWithBatch;      // bias origin is [b, 1, n]
};
#pragma pack(pop)

#endif  // INNER_QUANT_BATCH_MATMUL_V3_TILING_H
