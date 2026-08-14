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

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "recurrent_gated_delta_rule_tiling.h"

using namespace ge;
using namespace gert;

namespace {

constexpr uint32_t MAX_MTP = 8;
constexpr uint32_t FP16_NUM_PER_BLOCK = 16;
constexpr uint32_t MAX_TQUE_BUFFER_NUM_310P = 8;

uint32_t CeilAlign(uint32_t val, uint32_t align) { return (val + align - 1) / align * align; }

uint32_t CeilDiv(uint32_t val, uint32_t div) { return (val + div - 1) / div; }

int64_t CalcFixedUbBytes(int64_t aNv, int64_t aDv, int64_t aDk, bool hasGama, bool hasGamaK, bool gamaKScalar) {
  int64_t usedUbBytes = MAX_MTP * (4 * aDk + 2 * aDv);
  usedUbBytes += 128;
  if (hasGamaK) {
    if (gamaKScalar) {
      usedUbBytes += MAX_MTP * 4 * aNv;  // scalar per head, same size as gama
    } else {
      usedUbBytes += MAX_MTP * 4 * aDk;
    }
  }
  usedUbBytes += MAX_MTP * 2 * aNv;
  return usedUbBytes;
}

int64_t CalcWorkingUbBytes(int64_t aNv, int64_t aDv, int64_t aDk, bool hasGama, bool hasGamaK, bool gamaKScalar) {
  int64_t usedUbBytes = CalcFixedUbBytes(aNv, aDv, aDk, hasGama, hasGamaK, gamaKScalar);
  usedUbBytes += MAX_MTP * (8 * aDk + 4 * aDv + 4 * aNv);
  if (hasGama) {
    usedUbBytes += MAX_MTP * 4 * aNv;
  }
  return usedUbBytes;
}

int64_t CalcVStepCoeff(int64_t aDk, uint32_t stateOutBufferNum, uint32_t attnOutBufferNum) {
  int64_t coeff = static_cast<int64_t>(2 * stateOutBufferNum) * aDk + static_cast<int64_t>(2 * attnOutBufferNum);
  coeff += (4 + 4) * aDk + 4;
  return coeff;
}

struct BufferProfile {
  uint32_t stateOutBufferNum;
  uint32_t attnOutBufferNum;
  uint32_t vStep;
  uint32_t repeatTime;
  bool valid;
};

bool EvaluateBufferProfile(int64_t ubSize, int64_t usedUbBytes, int64_t aDk, uint32_t dv, uint32_t stateOutBufferNum,
                           uint32_t attnOutBufferNum, BufferProfile &profile) {
  int64_t coeff = CalcVStepCoeff(aDk, stateOutBufferNum, attnOutBufferNum);
  int64_t vStep = (ubSize - usedUbBytes) / coeff / static_cast<int64_t>(FP16_NUM_PER_BLOCK) *
                  static_cast<int64_t>(FP16_NUM_PER_BLOCK);
  if (vStep < static_cast<int64_t>(FP16_NUM_PER_BLOCK)) {
    return false;
  }
  int64_t repeatTime = CeilDiv(dv, static_cast<uint32_t>(vStep));
  vStep = CeilAlign(CeilDiv(dv, static_cast<uint32_t>(repeatTime)), FP16_NUM_PER_BLOCK);
  if (vStep < static_cast<int64_t>(FP16_NUM_PER_BLOCK)) {
    return false;
  }
  profile.stateOutBufferNum = stateOutBufferNum;
  profile.attnOutBufferNum = attnOutBufferNum;
  profile.vStep = static_cast<uint32_t>(vStep);
  profile.repeatTime = static_cast<uint32_t>(repeatTime);
  profile.valid = true;
  return true;
}

bool IsBetterProfile(const BufferProfile &candidate, const BufferProfile &current) {
  if (!current.valid) {
    return true;
  }
  if (candidate.repeatTime != current.repeatTime) {
    return candidate.repeatTime < current.repeatTime;
  }
  uint32_t candidateDepth = candidate.stateOutBufferNum + candidate.attnOutBufferNum;
  uint32_t currentDepth = current.stateOutBufferNum + current.attnOutBufferNum;
  if (candidateDepth != currentDepth) {
    return candidateDepth > currentDepth;
  }
  return candidate.vStep > current.vStep;
}

}  // namespace

static uint32_t RecurrentGatedDeltaRuleTilingFunc(TilingContext *context) {
  if (context == nullptr) {
    return GRAPH_FAILED;
  }

  // Platform info
  auto platformInfo = context->GetPlatformInfo();
  if (platformInfo == nullptr) {
    return GRAPH_FAILED;
  }
  platform_ascendc::PlatformAscendC platform(platformInfo);
  // The 310P unified AICore launch still executes one RGDR vector task per
  // AIV slot.  GetCoreNum() reports the logical vector lanes (32 here),
  // while only GetCoreNumAiv() gives the eight schedulable task blocks.
  uint32_t coreNum = platform.GetCoreNumAiv();
  if (coreNum == 0) {
    coreNum = platform.GetCoreNum();
  }
  if (coreNum == 0) coreNum = 1;
  constexpr uint32_t MAX_SCHEDULABLE_AICORE_310P = 8;
  if (coreNum > MAX_SCHEDULABLE_AICORE_310P) {
    coreNum = MAX_SCHEDULABLE_AICORE_310P;
  }
  uint64_t ubSize64 = 0;
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize64);
  int64_t ubSize = static_cast<int64_t>(ubSize64);

  // Input shapes
  auto queryShape = context->GetInputShape(0);
  auto valueShape = context->GetInputShape(2);
  auto stateShape = context->GetInputShape(4);
  auto cuSeqlensShape = context->GetInputShape(5);
  if (queryShape == nullptr || valueShape == nullptr || stateShape == nullptr || cuSeqlensShape == nullptr) {
    return GRAPH_FAILED;
  }

  const auto &qDims = queryShape->GetStorageShape();
  const auto &vDims = valueShape->GetStorageShape();
  const auto &sDims = stateShape->GetStorageShape();
  const auto &cDims = cuSeqlensShape->GetStorageShape();
  uint32_t t = qDims.GetDim(0);
  uint32_t nk = qDims.GetDim(1);
  uint32_t dk = qDims.GetDim(2);
  uint32_t nv = vDims.GetDim(1);
  uint32_t dv = vDims.GetDim(2);
  uint32_t sBlockNum = sDims.GetDim(0);
  uint32_t cLen = cDims.GetDim(0);
  uint32_t cuSeqlensIsPrefix = 0;
  uint32_t b = cLen;
  if (b == 0) {
    b = 1;
  }
  // Scale attribute
  float scale = 1.0f;
  auto attrs = context->GetAttrs();
  if (attrs != nullptr && attrs->GetAttrNum() > 0) {
    const float *scalePtr = attrs->GetAttrPointer<float>(0);
    if (scalePtr != nullptr) {
      scale = *scalePtr;
    }
  }
  // Optional inputs
  uint32_t hasGama = 0;
  uint32_t hasGamaK = 0;
  uint32_t hasAcceptedTokens = 0;
  uint32_t gamaKScalar = 0;
  auto gShape = context->GetOptionalInputShape(7);
  if (gShape != nullptr) hasGama = 1;
  auto gkShape = context->GetOptionalInputShape(8);
  if (gkShape != nullptr) {
    hasGamaK = 1;
    // Detect gamaK mode: scalar per head [T, NV] vs vector per head [T, NV*DK]
    const auto &gkDims = gkShape->GetStorageShape();
    int64_t gkTotal = 1;
    for (size_t d = 0; d < gkDims.GetDimNum(); d++) {
      gkTotal *= gkDims.GetDim(d);
    }
    // If total elements < T * NV * DK, treat as scalar mode
    int64_t expectedVector = static_cast<int64_t>(t) * nv * dk;
    if (gkTotal < expectedVector) {
      gamaKScalar = 1;
    }
  }
  auto natShape = context->GetOptionalInputShape(9);
  if (natShape != nullptr) hasAcceptedTokens = 1;

  if (hasGama == 1 && hasGamaK == 1 && gamaKScalar == 1) {
    hasGamaK = 0;
    gamaKScalar = 0;
  }

  uint32_t cuSeqlensIsInt64 = 0;
  uint32_t ssmStateIndicesIsInt64 = 0;
  auto cuTensor = context->GetInputTensor(5);
  if (cuTensor != nullptr && cuTensor->GetDataType() == ge::DT_INT64) {
    cuSeqlensIsInt64 = 1;
  }
  auto ssmTensor = context->GetInputTensor(6);
  if (ssmTensor != nullptr && ssmTensor->GetDataType() == ge::DT_INT64) {
    ssmStateIndicesIsInt64 = 1;
  }

  // Aligned dimensions
  uint32_t aNv = CeilAlign(nv, FP16_NUM_PER_BLOCK);
  uint32_t aDv = CeilAlign(dv, FP16_NUM_PER_BLOCK);
  uint32_t aDk = CeilAlign(dk, FP16_NUM_PER_BLOCK);

  // UB calculation
  int64_t fixedUbBytes = CalcFixedUbBytes(aNv, aDv, aDk, hasGama == 1, hasGamaK == 1, gamaKScalar == 1);
  int64_t workingUbBytes = CalcWorkingUbBytes(aNv, aDv, aDk, hasGama == 1, hasGamaK == 1, gamaKScalar == 1);

  // Evaluate buffer profiles
  BufferProfile selected = {0, 0, 0, 0, false};
  const uint32_t candidates[][2] = {{1, 1}, {1, 2}, {2, 2}};
  // beta and gama have disjoint lifetimes and share one input queue in the
  // kernel, keeping the 310P total within its eight-TQue limit.
  uint32_t fixedQueueBufferNum = 5U + ((hasGamaK == 1U && gamaKScalar == 0U) ? 1U : 0U);
  for (auto &c : candidates) {
    if (fixedQueueBufferNum + c[0] + c[1] > MAX_TQUE_BUFFER_NUM_310P) {
      continue;
    }
    BufferProfile profile;
    if (!EvaluateBufferProfile(ubSize, workingUbBytes, aDk, dv, c[0], c[1], profile)) {
      continue;
    }
    if (IsBetterProfile(profile, selected)) {
      selected = profile;
    }
  }
  if (!selected.valid) {
    return GRAPH_FAILED;
  }

  // Calculate rest UB bytes
  int64_t queueCoeff = (2 + static_cast<int64_t>(2 * selected.stateOutBufferNum)) * aDk +
                       static_cast<int64_t>(2 * selected.attnOutBufferNum);
  int64_t ubRestBytes = ubSize - fixedUbBytes - queueCoeff * static_cast<int64_t>(selected.vStep);
  if (ubRestBytes < 0) {
    return GRAPH_FAILED;
  }

  // Dynamic blockDim based on task units
  uint64_t taskUnits = static_cast<uint64_t>(b) * static_cast<uint64_t>(nv);
  if (taskUnits == 0) taskUnits = 1;
  uint32_t blockDim = (taskUnits < coreNum) ? static_cast<uint32_t>(taskUnits) : coreNum;
  if (blockDim == 0) blockDim = 1;

  // Fill tiling data
  auto tilingData = context->GetTilingData<RecurrentGatedDeltaRuleTilingData>();
  if (tilingData == nullptr) {
    return GRAPH_FAILED;
  }
  tilingData->vectorCoreNum = coreNum;
  tilingData->ubCalSize = static_cast<uint32_t>(ubSize);
  tilingData->ubRestBytes = static_cast<uint32_t>(ubRestBytes);
  tilingData->t = t;
  tilingData->nk = nk;
  tilingData->dk = dk;
  tilingData->nv = nv;
  tilingData->dv = dv;
  tilingData->sBlockNum = sBlockNum;
  tilingData->b = b;
  tilingData->vStep = selected.vStep;
  tilingData->stateOutBufferNum = selected.stateOutBufferNum;
  tilingData->attnOutBufferNum = selected.attnOutBufferNum;
  tilingData->scale = scale;
  tilingData->hasGama = hasGama;
  tilingData->hasGamaK = hasGamaK;
  tilingData->hasAcceptedTokens = hasAcceptedTokens;
  tilingData->gamaKScalar = gamaKScalar;
  tilingData->cuSeqlensIsPrefix = cuSeqlensIsPrefix;
  tilingData->cuSeqlensIsInt64 = cuSeqlensIsInt64;
  tilingData->ssmStateIndicesIsInt64 = ssmStateIndicesIsInt64;
  tilingData->reserved = 0;
  context->SetBlockDim(blockDim);
  context->SetTilingKey(0);

  // Workspace: 16MB system workspace
  size_t workspaceSize = 16ULL << 20;
  size_t *ws = context->GetWorkspaceSizes(1);
  if (ws != nullptr) {
    ws[0] = workspaceSize;
  }

  return GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RecurrentGatedDeltaRule310P)
  .Tiling(RecurrentGatedDeltaRuleTilingFunc, sizeof(RecurrentGatedDeltaRuleTilingData));
