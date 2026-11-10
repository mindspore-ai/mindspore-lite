/** * Derived from https://gitcode.com/cann/ops-transformer/tree/master/attention/chunk_gated_delta_rule
 *
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

/*!
 * \file fused_qk_l2norm_chunk_gated_delta_rule_tiling_data.h
 * \brief Host/device tiling-data contract for FusedQkL2NormChunkGatedDeltaRule (ascend910b / arch22).
 *
 * Mirror of the ChunkGatedDeltaRule tiling data plus one field: `outZeroPadRows`
 * (1 = the kernel zeroes output rows [actual_seq_len, T) itself). The q/k L2-normalization
 * needs no field — this operator always normalizes, so the flag is not part of the contract.
 *
 * The one structural change required by this repo's CANN "customize" build: the
 * FusedQkL2NormChunkGatedDeltaRuleTilingData struct is wrapped in namespace fused_tiling_detail
 * with a using-declaration that re-exports it to global scope. REGISTER_TILING_DEFAULT /
 * GET_TILING_DATA expanded inside the device-kernel entry require a globally-visible
 * tiling-data type here (a namespace-qualified struct makes the CANN autogen emit
 * <ns>::-qualified tiling symbols that fail to resolve); the upstream ops-transformer module
 * framework does not have this constraint.
 * ChunkGroup stays in the FusedQkL2NormChunkGatedDeltaRule namespace (only the kernel classes reference it).
 */

#ifndef FUSED_QK_L2NORM_CHUNK_GATED_DELTA_RULE_TILING_DATA_H
#define FUSED_QK_L2NORM_CHUNK_GATED_DELTA_RULE_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

// Tiling-data struct alignment (bytes) and the struct itself are wrapped in
// namespace fused_tiling_detail to satisfy clangTidy G.INC.11-CPP; the using-declaration
// below re-exports the type to global scope because the CANN tiling macros
// (REGISTER_TILING_DEFAULT / GET_TILING_DATA) expanded in the device-kernel entry
// require a globally-visible tiling-data type (see file header).
namespace fused_tiling_detail {
constexpr uint64_t kFusedTilingDataAlign = 8;
#pragma pack(push, 8)
struct alignas(kFusedTilingDataAlign) FusedQkL2NormChunkGatedDeltaRuleTilingData {
  int64_t aiCoreNum;
  int64_t t;
  int64_t nk;
  int64_t dk;
  int64_t nv;
  int64_t dv;
  int64_t b;
  int64_t hasGamma;
  int64_t chunkSize;
  int64_t maxGroupLength;  // maxGroupLength = p * aiCoreNum * chunkSize
  int64_t interWorkspaceSz;
  int64_t stageWorkspaceSz;
  int64_t stageOneParaNum;
  float scale;
  AscendC::tiling::TCubeTiling matmulTilingFp32;   // BF16 C matmul tiling (arch22 / 910b path)
  AscendC::tiling::TCubeTiling matmulTilingFp32C;  // FP32 C matmul tiling (ascend950 FP32-state path; unused on 910b)
  int64_t stateIsFp32;
  int64_t isFp16;  // 1 = q/k/v/beta/state/out/final_state are FP16; 0 = BF16 (910b path)
  int64_t stateStride0;
  int64_t stateStride1;
  int64_t outZeroPadRows;  // 1 = zero output rows [actual_seq_len, T) in-kernel
                           // (replaces the caller's masked_fill); 0 = leave untouched
};
#pragma pack(pop)
}  // namespace fused_tiling_detail
using fused_tiling_detail::FusedQkL2NormChunkGatedDeltaRuleTilingData;

namespace FusedQkL2NormChunkGatedDeltaRule {

struct ChunkGroup {
  int64_t startPos = 0;   // start position of this ChunkGroup on T
  int64_t length = 0;     // length of this ChunkGroup
  int64_t chunkSize = 0;  // length of each chunk
  int64_t coreStart = 0;  // reserved
  int64_t coreEnd = 0;    // reserved
};

}  // namespace FusedQkL2NormChunkGatedDeltaRule

#endif  // FUSED_QK_L2NORM_CHUNK_GATED_DELTA_RULE_TILING_DATA_H
