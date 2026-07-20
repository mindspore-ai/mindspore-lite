/** * copy from https://gitcode.com/cann/ops-transformer/tree/master/attention/chunk_gated_delta_rule
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
 * \file chunk_gated_delta_rule_tiling_data.h
 * \brief Host/device tiling-data contract for ChunkGatedDeltaRule (ascend910b / arch22).
 *
 * Ported from ops-transformer attention/chunk_gated_delta_rule/op_kernel. The one structural
 * change required by this repo's CANN "customize" build: the ChunkGatedDeltaRuleTilingData
 * struct is kept in the GLOBAL namespace. REGISTER_TILING_DEFAULT / GET_TILING_DATA expanded
 * inside the device-kernel entry require a globally-scoped tiling-data type here (a namespace-
 * qualified struct makes the CANN autogen emit <ns>::-qualified tiling symbols that fail to
 * resolve); the upstream ops-transformer module framework does not have this constraint.
 * ChunkGroup stays in the ChunkGatedDeltaRule namespace (only the kernel classes reference it).
 */

#ifndef CHUNK_GATED_DELTA_RULE_TILING_DATA_H
#define CHUNK_GATED_DELTA_RULE_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

constexpr uint64_t STRUCT_ALIGNAS = 8;
#pragma pack(push, 8)
// MUST stay global for the CANN tiling macros in this build (see file header).
struct alignas(STRUCT_ALIGNAS) ChunkGatedDeltaRuleTilingData {
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
};
#pragma pack(pop)

namespace ChunkGatedDeltaRule {

struct ChunkGroup {
  int64_t startPos = 0;   // start position of this ChunkGroup on T
  int64_t length = 0;     // length of this ChunkGroup
  int64_t chunkSize = 0;  // length of each chunk
  int64_t coreStart = 0;  // reserved
  int64_t coreEnd = 0;    // reserved
};

}  // namespace ChunkGatedDeltaRule

#endif  // CHUNK_GATED_DELTA_RULE_TILING_DATA_H
