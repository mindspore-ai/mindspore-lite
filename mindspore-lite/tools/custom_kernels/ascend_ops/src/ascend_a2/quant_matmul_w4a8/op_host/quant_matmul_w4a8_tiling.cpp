/**
 * modified from
 * https://gitcode.com/cann/ops-nn/blob/master/matmul/quant_batch_matmul_v4/op_host/quant_batch_matmul_v4_tiling.cpp
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

#include "quant_matmul_w4a8_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "tiling/matrix/matmul_tiling.h"
#include "tiling/matrix/bmm_tiling.h"

namespace {
constexpr uint32_t A_INDEX = 0;
constexpr uint32_t W_INDEX = 1;
constexpr uint32_t Y_INDEX = 0;
constexpr int64_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024;

// V5 MSD constants
constexpr uint32_t UB_CALSIZE = 32U * 256U;  // = 8192
constexpr uint32_t UB_BUFFER_NUM = 4;
constexpr uint32_t INT4_SIZE = 2;
constexpr uint32_t CV_PARALL_NUM = 4;

static inline uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
}  // namespace

namespace Ops {
namespace NN {
namespace QuantMatmulW4a8 {

void QuantMatmulW4a8Tiling::InitCompileInfo() {
  if (context_ == nullptr) {
    return;
  }
  auto platformInfoPtr = context_->GetPlatformInfo();
  if (platformInfoPtr == nullptr) {
    return;
  }
  const auto &ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo_.ubSize);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfo_.l1Size);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfo_.l0ASize);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfo_.l0BSize);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfo_.l0CSize);
  compileInfo_.aicNum = ascendcPlatform.GetCoreNumAic();
}

ge::graphStatus QuantMatmulW4a8Tiling::GetShapeAttrsInfo() {
  tilingDataSize_ = sizeof(QuantMatmulW4a8TilingData);
  if (inputParams_.initFlag) {
    return ge::GRAPH_SUCCESS;
  }
  inputParams_.opName = context_->GetNodeName();
  if (CheckContext() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }

  auto actShape = context_->GetInputShape(A_INDEX);
  auto wtShape = context_->GetInputShape(W_INDEX);
  const auto &aDims = actShape->GetStorageShape();
  const auto &wDims = wtShape->GetStorageShape();
  inputParams_.M = aDims.GetDim(0);
  inputParams_.K = aDims.GetDim(1);
  inputParams_.N = wDims.GetDim(0);

  inputParams_.initFlag = true;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMatmulW4a8Tiling::CheckContext() {
  if (context_ == nullptr) {
    return ge::GRAPH_FAILED;
  }
  if (context_->GetInputShape(A_INDEX) == nullptr) {
    return ge::GRAPH_FAILED;
  }
  if (context_->GetInputShape(W_INDEX) == nullptr) {
    return ge::GRAPH_FAILED;
  }
  if (context_->GetOutputShape(Y_INDEX) == nullptr) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

bool QuantMatmulW4a8Tiling::SetMatmulTiling() {
  auto &mt = tilingData_.matmulTiling;
  memset_s(&mt, sizeof(mt), 0, sizeof(mt));

  uint32_t M = static_cast<uint32_t>(inputParams_.M);
  uint32_t N = static_cast<uint32_t>(inputParams_.N);
  uint32_t Kpad = ((static_cast<uint32_t>(inputParams_.K) + 31) / 32) * 32;

  if (static_cast<uint32_t>(inputParams_.K) % 32 != 0 || N < 16) {
    return false;
  }

  // V5 tiling: baseM=120, baseN=128, like V5's per-channel path
  uint32_t bm = (2 * M < 120) ? (2 * M) : 120;
  uint32_t bn = (N < 128) ? N : 128;
  if (bn >= 16) bn = (bn / 16) * 16;
  // UB budget: V5 uses ubCalSize*4*float ≈ 128KB for tmpBuf_
  // rest for queues: ubCalSize*2*half + ubCalSize*bf16 ≈ 32KB
  // total ~160KB < 192KB UB
  uint32_t maxBm = (static_cast<uint32_t>(compileInfo_.ubSize) - 16u * bn) / (12u * bn);
  if (bm > maxBm) bm = maxBm;
  if (bm < 2) bm = 2;
  uint32_t bk = (Kpad < 64) ? Kpad : 64;

  matmul_tiling::PlatformInfo platformInfo;
  platformInfo.socVersion = platform_ascendc::SocVersion::ASCEND910B;
  platformInfo.ubSize = compileInfo_.ubSize;
  platformInfo.l1Size = compileInfo_.l1Size;
  platformInfo.l0CSize = compileInfo_.l0CSize;
  platformInfo.l0ASize = compileInfo_.l0ASize;
  platformInfo.l0BSize = compileInfo_.l0BSize;

  matmul_tiling::MultiCoreMatmulTiling mm(platformInfo);
  mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT4, false);
  mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT4, true);
  mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
  mm.SetBias(false);
  // Use full problem shape (2*M for interleaved rows) to get correct
  // baseM/baseN/stepM/stepN/Ka/Kb from MultiCoreMatmulTiling.
  // usedCoreNum is overridden below (V5 pattern: computed from tiles).
  mm.SetOrgShape(2 * M, N, Kpad);
  mm.SetShape(2 * M, N, Kpad);
  mm.SetFixSplit(bm, bn, bk);

  if (mm.GetTiling(mt) == -1) {
    return false;
  }

  // V5 pattern: MultiCoreMatmulTiling may return usedCoreNum=1 regardless
  // of org shape.  Compute it directly from tile counts like V5's
  // QuantBatchMatmulV3BasicTiling::SetBase does.
  uint32_t apiBaseM = static_cast<uint32_t>(mt.baseM);
  uint32_t apiBaseN = static_cast<uint32_t>(mt.baseN);
  if (apiBaseM > 2 * M) apiBaseM = 2 * M;
  if (apiBaseM == 0) apiBaseM = 8;
  uint32_t blockDimM = CeilDiv(2 * M, apiBaseM);
  uint32_t blockDimN = CeilDiv(N, apiBaseN);
  uint32_t totalBlocks = blockDimM * blockDimN;
  uint32_t usedCoreNum = (totalBlocks < static_cast<uint32_t>(compileInfo_.aicNum))
                           ? totalBlocks
                           : static_cast<uint32_t>(compileInfo_.aicNum);
  if (usedCoreNum < 1) usedCoreNum = 1;
  mt.usedCoreNum = usedCoreNum;  // override API's value (V5 pattern)

  return true;
}

ge::graphStatus QuantMatmulW4a8Tiling::DoOpTiling() {
  uint32_t M = static_cast<uint32_t>(inputParams_.M);
  uint32_t K = ((static_cast<uint32_t>(inputParams_.K) + 31) / 32) * 32;
  uint32_t N = static_cast<uint32_t>(inputParams_.N);

  if (!SetMatmulTiling()) {
    return ge::GRAPH_FAILED;
  }

  uint32_t bm = static_cast<uint32_t>(tilingData_.matmulTiling.baseM);
  uint32_t bn = static_cast<uint32_t>(tilingData_.matmulTiling.baseN);
  if (bm > 2 * M) bm = 2 * M;
  if (bm == 0) bm = 8;
  uint32_t usedCoreNum = static_cast<uint32_t>(tilingData_.matmulTiling.usedCoreNum);
  if (usedCoreNum < 1) usedCoreNum = 1;

  // ── V5 tiling data fields ──
  tilingData_.coreNum = static_cast<uint8_t>(usedCoreNum);
  tilingData_.mSize = M;
  tilingData_.kSize = K;
  tilingData_.nSize = N;
  tilingData_.groupSize = 0;  // K_C: per-channel, not per-group
  tilingData_.ubCalSize = UB_CALSIZE;
  tilingData_.ubRestBytes = UB_CALSIZE * sizeof(float) * UB_BUFFER_NUM;
  tilingData_.parallNum = CV_PARALL_NUM;
  // vBaseM: max rows per VEC chunk = min(ubCalSize / baseN, baseM / 2)
  uint32_t vBaseMMax = UB_CALSIZE / bn;
  uint32_t vBaseMOne = bm / INT4_SIZE;
  tilingData_.vBaseM = (vBaseMMax < vBaseMOne) ? vBaseMMax : vBaseMOne;

  // ── V5 workspace: per-core ping-pong × parallNum ──
  uint64_t baseSize = static_cast<uint64_t>(bm) * bn;
  workspaceSize_ = SYS_WORKSPACE_SIZE + static_cast<size_t>(CV_PARALL_NUM) * usedCoreNum * baseSize * sizeof(uint16_t);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMatmulW4a8Tiling::PostTiling() {
  uint32_t blockDim = tilingData_.coreNum;
  if (blockDim < 1) blockDim = 1;
  context_->SetBlockDim(blockDim);
  context_->SetScheduleMode(1);  // exclusive cores (required for CrossCoreSetFlag)

  errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                         static_cast<void *>(&tilingData_), tilingDataSize_);
  if (ret != EOK) {
    return ge::GRAPH_FAILED;
  }
  context_->GetRawTilingData()->SetDataSize(tilingDataSize_);

  size_t *workspaces = context_->GetWorkspaceSizes(1);
  workspaces[0] = workspaceSize_;
  return ge::GRAPH_SUCCESS;
}

// ── CANN-standard tiling entry point ──
// Replaces the ops-nn REGISTER_TILING_TEMPLATE + TilingRegistry pattern.
static ge::graphStatus QuantMatmulW4a8TilingFunc(gert::TilingContext *context) {
  if (context == nullptr) {
    return ge::GRAPH_FAILED;
  }
  QuantMatmulW4a8Tiling tiling(context);
  if (tiling.GetShapeAttrsInfo() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  if (tiling.DoOpTiling() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  if (tiling.PostTiling() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForQuantMatmulW4a8(gert::TilingParseContext *context) {
  if (context == nullptr) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(QuantMatmulW4a8)
  .Tiling(QuantMatmulW4a8TilingFunc)
  .TilingParse<V4MCompileInfo>(TilingParseForQuantMatmulW4a8);

}  // namespace QuantMatmulW4a8
}  // namespace NN
}  // namespace Ops
