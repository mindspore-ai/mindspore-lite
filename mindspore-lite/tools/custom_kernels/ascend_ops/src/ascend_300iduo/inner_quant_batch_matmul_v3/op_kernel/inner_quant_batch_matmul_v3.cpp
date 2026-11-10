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

// InnerQuantBatchMatmulV3 kernel entry (Atlas 300I Duo / Ascend310P3,
// dav-m200 aicore 200).
//
// Vendor of the CANN built-in QuantBatchMatmulV3 310P per-token path
// (arch20/quant_batch_matmul_v3_pertoken_arch20.h, raw CCE -- this IS the
// kernel that the CANN 9.2 built-in dispatches for the NZ_NZ int8
// combo on 310P, i.e. the exact baseline this op must beat). The kernel
// class is copied verbatim; only the tiling differs (see
// op_host/inner_quant_batch_matmul_v3_tiling.cpp): shape-specialized
// k0 (512 instead of the built-in 256 for the suffix_1536 W8A8 GEMM
// shapes) which is legal because the built-in host tiling budgets L1
// as 131072/(m0+n0) int8 elements per ping set while the kernel actually
// allocates four separate 131072-element L1 buffers (a-ping/a-pong/
// b-ping/b-pong), so the per-buffer budget is 131072/max(m0,n0).
// Bit-exactness holds by construction: the epilogue math chain and the
// int32 L0C accumulation (exact, associative) are unchanged; rechunking k
// only regroups exact integer adds.
//
// Dispatch contract (fixed, validated in tiling):
//   x1 int8 FRACTAL_NZ [.., M, K]           transpose_x1=false
//   x2 int8 FRACTAL_NZ [.., N, K]           transpose_x2=true
//   scale      float32 ND [N]     (per-channel, a.k.a. x2Scale)
//   offset     float32 ND [N] opt (accepted; pertoken path ignores it,
//                                  same as the built-in -- when offset is
//                                  present the built-in also drops bias)
//   bias       int32/float ND [N] or [B,1,N] opt (int32 added in the int32
//                                  L0C domain via BroadCastVecToMM, so it
//                                  is exact for |acc| beyond fp16/fp24 range)
//   pertoken_scale float32 ND [M] opt (per-row; when present, the output
//                                  offset term is dropped, per the
//                                  aclnnQuantMatmulWeightNz doc formulas)
//   y          float16 FRACTAL_NZ [.., M, N]
//
// out = cast_fp16( cast_fp32(acc_i32 [+ bias_i32]) * scale[n]
//                  * pertoken_scale[m] )     -- mul order fixed for
//                                          bit-exactness vs the built-in

// Include order mirrors the built-in quant_batch_matmul_v3.cpp (kernel_
// operator.h at its line 30, BEFORE arch20/..._pertoken_arch20.h at 34).
// The arch20 header's template params use CubeFormat, which the built-in
// resolved from its atcos -I build environment; the CANN tikcfw
// kernel_operator.h chain does NOT define it, so the arch header vendors
// the enum verbatim from CANN's adv_api/matmul/matmul_config.h via
// inner_quant_batch_matmul_v3_cube_format.h.
#include <kernel_operator.h>
#include "./inner_quant_batch_matmul_v3_pertoken_arch20.h"

using namespace AscendC;     // NOLINT(build/namespaces)
using namespace PpMatMulNS;  // NOLINT(build/namespaces)

// Same runtime dispatch as the built-in INVOKE_QUANT_BATCH_MATMUL_PERTOKEN_
// ARCH20_IMPL: one binary, six template instantiations selected by tiling
// fields (swizzleDirect x withBias x biasDtype). Formats/dtypes are fixed
// by the op def (NZ/NZ/NZ, int8 x int8 -> fp16, fp32 descale), so the
// format templates are constants here.
namespace {
// Port formats are fixed by the op def; kept file-local.
constexpr CubeFormat FORMAT_X1_NZ = CubeFormat::NZ;
constexpr CubeFormat FORMAT_X2_NZ = CubeFormat::NZ;
constexpr CubeFormat FORMAT_Y_NZ = CubeFormat::NZ;
}  // namespace

#define INVOKE_INNER_QMM_PERTOKEN_ARCH20(swizzle, biasFlag, biasType)                                         \
  do {                                                                                                        \
    QuantBatchMatMulPertokenArch20<swizzle, biasFlag, FORMAT_X1_NZ, FORMAT_X2_NZ, FORMAT_Y_NZ, int8_t, float, \
                                   biasType, half>                                                            \
      op;                                                                                                     \
    op.Init(x1, x2, bias, scale, pertokenScale, y, &tilingData);                                              \
    op.Process();                                                                                             \
  } while (0)

extern "C" __global__ __aicore__ void inner_quant_batch_matmul_v3(GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR offset,
                                                                  GM_ADDR bias, GM_ADDR pertokenScale, GM_ADDR y,
                                                                  GM_ADDR workSpace, GM_ADDR tiling) {
  if (workSpace == nullptr) {
    return;
  }
  // Mirror the built-in entry: require a user workspace slot even though
  // the pertoken path never touches it (the built-in does the same).
  GM_ADDR user1 = GetUserWorkspace(workSpace);
  if (user1 == nullptr) {
    return;
  }
  // Vendor-op tiling contract (fused_qk_norm_rope_bsh.cpp:26 precedent):
  // REGISTER_TILING_DEFAULT marks the tiling struct for te's kernel-source
  // scan (kernel_info_infer.py find_tiling_struct_and_expression,
  // "__enable_custom_tiling <struct> = default;") and GET_TILING_DATA is
  // the wrapper-generated accessor that bakes the tiling bytes produced by
  // this op's tiling .so (vendor op_tiling/liboptiling.so). The built-in's
  // GET_TILING_DATA_WITH_STRUCT form is only generated for ops registered
  // in the built-in op_host .so registry -- vendor ops are not, so the
  // vendor wrapper uses GET_TILING_DATA instead. Struct layout is untouched (hand-written,
  // byte-identical on both sides -- BSH discipline).
  REGISTER_TILING_DEFAULT(QuantMatmulPertokenTilingDataArch20);
  GET_TILING_DATA(tilingData, tiling);
  const bool isFp32Bias = (tilingData.biasDtype == static_cast<uint32_t>(DT_FLOAT));
  // offset is not consumed by the per-token path (built-in contract);
  // its presence is resolved on the host side (bias dropped).
  (void)offset;
  (void)user1;
  if (tilingData.swizzleDirect == 0 && tilingData.withBias == false) {
    INVOKE_INNER_QMM_PERTOKEN_ARCH20(0, false, int32_t);
  } else if (tilingData.swizzleDirect == 0 && tilingData.withBias == true && isFp32Bias) {
    INVOKE_INNER_QMM_PERTOKEN_ARCH20(0, true, float);
  } else if (tilingData.swizzleDirect == 0 && tilingData.withBias == true) {
    INVOKE_INNER_QMM_PERTOKEN_ARCH20(0, true, int32_t);
  } else if (tilingData.swizzleDirect == 1 && tilingData.withBias == false) {
    INVOKE_INNER_QMM_PERTOKEN_ARCH20(1, false, int32_t);
  } else if (tilingData.swizzleDirect == 1 && tilingData.withBias == true && isFp32Bias) {
    INVOKE_INNER_QMM_PERTOKEN_ARCH20(1, true, float);
  } else if (tilingData.swizzleDirect == 1 && tilingData.withBias == true) {
    INVOKE_INNER_QMM_PERTOKEN_ARCH20(1, true, int32_t);
  }
}
