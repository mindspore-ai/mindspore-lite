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

#ifndef INNER_QUANT_BATCH_MATMUL_V3_CUBE_FORMAT_H
#define INNER_QUANT_BATCH_MATMUL_V3_CUBE_FORMAT_H

// CubeFormat, vendored VERBATIM (global scope, every enumerator, comments
// included) from CANN 9.2.0:
//   aarch64-linux/asc/include/adv_api/matmul/matmul_config.h:23-37
//
// The vendored arch20 pertoken kernel (inner_quant_batch_matmul_v3_
// pertoken_arch20.h, origin opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/
// ascendc/quant_batch_matmul_v3/arch20/) uses CubeFormat as template
// parameters. The built-in resolved the enum from its own build
// environment (atcos -I over the adv_api headers); the te online-build
// include chain (tikcfw kernel_operator.h) does not define CubeFormat
// anywhere. This enum is the minimal closure: no other symbol of
// matmul_config.h is referenced by the kernel (matmul_config.h itself
// pulls kernel_tiling/kernel_tiling.h, which is not needed here).
//
// Collision note: if a future CANN kernel_operator.h chain starts
// defining CubeFormat at global scope, this header redefines it and the
// build fails loudly -- the fix is then to delete this vendored include
// (and the include line in the arch header).
/**
 * @enum class CubeFormat
 * @brief physical data layout formats
 */
enum class CubeFormat {
  ND = 0,            // Standard format, row-major
  NZ,                // Colum-major between blocks, row-major within blocks
  ZN,                // row-major between blocks, Colum-major within blocks
  ZZ,                // row-major between blocks, row-major within blocks
  NN,                // Colum-major between blocks, Colum-major within blocks
  ND_ALIGN,          // Aligned ND format, the output is aligned to 32-byte boundaries along the N dimension
  SCALAR,            // A single numerical value with no dimensions
  VECTOR,            // 1D array format for vector data
  ROW_MAJOR = ND,    // Row-major format (equivalent to ND format)
  COLUMN_MAJOR = 8,  // Column-major format (DN format)
};

#endif  // INNER_QUANT_BATCH_MATMUL_V3_CUBE_FORMAT_H
