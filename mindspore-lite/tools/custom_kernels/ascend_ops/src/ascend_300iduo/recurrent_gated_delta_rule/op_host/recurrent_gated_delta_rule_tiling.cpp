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
#include "recurrent_gated_delta_rule_tiling.h"  // NOLINT(build/include_subdir)

using namespace ge;    // NOLINT(build/namespaces)
using namespace gert;  // NOLINT(build/namespaces)

namespace {

constexpr uint32_t MAX_MTP = 8;
constexpr uint32_t FP16_NUM_PER_BLOCK = 16;
constexpr uint32_t MAX_TQUE_BUFFER_NUM_310P = 8;
constexpr uint32_t MAX_SCHEDULABLE_AICORE_310P = 8;
constexpr size_t SYSTEM_WORKSPACE_BYTES = 16ULL << 20;
// Optional-input indices in the op definition (after the 7 required inputs:
// query, key, value, beta, state, actual_seq_lengths, ssm_state_indices).
constexpr uint32_t OPTIONAL_INPUT_G = 7;
constexpr uint32_t OPTIONAL_INPUT_GK = 8;
constexpr uint32_t OPTIONAL_INPUT_NUM_ACCEPTED_TOKENS = 9;
// Bytes per fp32 / fp16 element in the UB-size formulas below.
constexpr int64_t FP32_BYTES = 4;
constexpr int64_t FP16_BYTES = 2;
// Number of fp32 dk working buffers per token/head in the working-UB formula.
constexpr int64_t WORKING_DK_FP32_BUFFER_NUM = 2;
// Small fixed UB allowance (bytes) for event/scheduling bookkeeping.
constexpr int64_t UB_FIXED_ALLOWANCE_BYTES = 128;

uint32_t CeilAlign(uint32_t val, uint32_t align) { return (val + align - 1) / align * align; }

uint32_t CeilDiv(uint32_t val, uint32_t div) { return (val + div - 1) / div; }

int64_t CalcFixedUbBytes(int64_t aNv, int64_t aDv, int64_t aDk, bool hasGama, bool hasGamaK, bool gamaKScalar) {
  int64_t usedUbBytes = MAX_MTP * (FP32_BYTES * aDk + FP16_BYTES * aDv);
  usedUbBytes += UB_FIXED_ALLOWANCE_BYTES;
  if (hasGamaK) {
    if (gamaKScalar) {
      usedUbBytes += MAX_MTP * FP32_BYTES * aNv;  // scalar per head, same size as gama
    } else {
      usedUbBytes += MAX_MTP * FP32_BYTES * aDk;
    }
  }
  usedUbBytes += MAX_MTP * FP16_BYTES * aNv;
  return usedUbBytes;
}

int64_t CalcWorkingUbBytes(int64_t aNv, int64_t aDv, int64_t aDk, bool hasGama, bool hasGamaK, bool gamaKScalar) {
  int64_t usedUbBytes = CalcFixedUbBytes(aNv, aDv, aDk, hasGama, hasGamaK, gamaKScalar);
  usedUbBytes += MAX_MTP * (WORKING_DK_FP32_BUFFER_NUM * FP32_BYTES * aDk + FP32_BYTES * aDv + FP32_BYTES * aNv);
  if (hasGama) {
    usedUbBytes += MAX_MTP * FP32_BYTES * aNv;
  }
  return usedUbBytes;
}

int64_t CalcVStepCoeff(int64_t aDk, uint32_t stateOutBufferNum, uint32_t attnOutBufferNum) {
  int64_t coeff = static_cast<int64_t>(2 * stateOutBufferNum) * aDk + static_cast<int64_t>(2 * attnOutBufferNum);
  coeff += (FP32_BYTES + FP32_BYTES) * aDk + FP32_BYTES;
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

struct ShapeDims {
  uint32_t t;
  uint32_t nk;
  uint32_t dk;
  uint32_t nv;
  uint32_t dv;
  uint32_t sBlockNum;
  uint32_t b;
};

struct OptionalInputs {
  uint32_t hasGama;
  uint32_t hasGamaK;
  uint32_t hasAcceptedTokens;
  uint32_t gamaKScalar;
};

struct IndexTypes {
  uint32_t cuSeqlensIsInt64;
  uint32_t ssmStateIndicesIsInt64;
};

bool GetPlatformResources(TilingContext *context, uint32_t &coreNum, int64_t &ubSize) {
  auto platformInfo = context->GetPlatformInfo();
  if (platformInfo == nullptr) {
    return false;
  }
  platform_ascendc::PlatformAscendC platform(platformInfo);
  coreNum = platform.GetCoreNumAiv();
  if (coreNum == 0) {
    coreNum = platform.GetCoreNum();
  }
  if (coreNum == 0) {
    coreNum = 1;
  }
  if (coreNum > MAX_SCHEDULABLE_AICORE_310P) {
    coreNum = MAX_SCHEDULABLE_AICORE_310P;
  }
  uint64_t ubSize64 = 0;
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize64);
  ubSize = static_cast<int64_t>(ubSize64);
  return true;
}

bool GetShapeDims(TilingContext *context, ShapeDims &dims) {
  constexpr uint32_t INPUT_QUERY = 0;
  constexpr uint32_t INPUT_VALUE = 2;
  constexpr uint32_t INPUT_STATE = 4;
  constexpr uint32_t INPUT_CU_SEQLENS = 5;
  // Dimension indices of the [T, N, D] query/value storage shapes.
  constexpr size_t DIM_TOKEN_T = 0;
  constexpr size_t DIM_HEAD_N = 1;
  constexpr size_t DIM_HEAD_D = 2;
  auto queryShape = context->GetInputShape(INPUT_QUERY);
  auto valueShape = context->GetInputShape(INPUT_VALUE);
  auto stateShape = context->GetInputShape(INPUT_STATE);
  auto cuSeqlensShape = context->GetInputShape(INPUT_CU_SEQLENS);
  if (queryShape == nullptr || valueShape == nullptr || stateShape == nullptr || cuSeqlensShape == nullptr) {
    return false;
  }
  const auto &qDims = queryShape->GetStorageShape();
  const auto &vDims = valueShape->GetStorageShape();
  const auto &sDims = stateShape->GetStorageShape();
  const auto &cDims = cuSeqlensShape->GetStorageShape();
  dims.t = qDims.GetDim(DIM_TOKEN_T);
  dims.nk = qDims.GetDim(DIM_HEAD_N);
  dims.dk = qDims.GetDim(DIM_HEAD_D);
  dims.nv = vDims.GetDim(DIM_HEAD_N);
  dims.dv = vDims.GetDim(DIM_HEAD_D);
  dims.sBlockNum = sDims.GetDim(0);
  dims.b = cDims.GetDim(0);
  if (dims.b == 0) {
    dims.b = 1;
  }
  return true;
}

float GetScale(TilingContext *context) {
  auto attrs = context->GetAttrs();
  if (attrs == nullptr || attrs->GetAttrNum() == 0) {
    return 1.0f;
  }
  const float *scale = attrs->GetAttrPointer<float>(0);
  return scale == nullptr ? 1.0f : *scale;
}

OptionalInputs GetOptionalInputs(TilingContext *context, const ShapeDims &dims) {
  OptionalInputs inputs = {0, 0, 0, 0};
  inputs.hasGama = context->GetOptionalInputShape(OPTIONAL_INPUT_G) == nullptr ? 0 : 1;
  auto gamaKShape = context->GetOptionalInputShape(OPTIONAL_INPUT_GK);
  if (gamaKShape != nullptr) {
    inputs.hasGamaK = 1;
    const auto &gamaKDims = gamaKShape->GetStorageShape();
    int64_t gamaKTotal = 1;
    for (size_t i = 0; i < gamaKDims.GetDimNum(); ++i) {
      gamaKTotal *= gamaKDims.GetDim(i);
    }
    int64_t expectedVector = static_cast<int64_t>(dims.t) * dims.nv * dims.dk;
    inputs.gamaKScalar = gamaKTotal < expectedVector ? 1 : 0;
  }
  inputs.hasAcceptedTokens = context->GetOptionalInputShape(OPTIONAL_INPUT_NUM_ACCEPTED_TOKENS) == nullptr ? 0 : 1;
  if (inputs.hasGama == 1 && inputs.hasGamaK == 1 && inputs.gamaKScalar == 1) {
    inputs.hasGamaK = 0;
    inputs.gamaKScalar = 0;
  }
  return inputs;
}

IndexTypes GetIndexTypes(TilingContext *context) {
  IndexTypes types = {0, 0};
  auto cuTensor = context->GetInputTensor(5);
  if (cuTensor != nullptr && cuTensor->GetDataType() == ge::DT_INT64) {
    types.cuSeqlensIsInt64 = 1;
  }
  auto ssmTensor = context->GetInputTensor(6);
  if (ssmTensor != nullptr && ssmTensor->GetDataType() == ge::DT_INT64) {
    types.ssmStateIndicesIsInt64 = 1;
  }
  return types;
}

bool SelectBufferProfile(int64_t ubSize, const ShapeDims &dims, const OptionalInputs &inputs, BufferProfile &selected,
                         int64_t &ubRestBytes) {
  uint32_t alignedNv = CeilAlign(dims.nv, FP16_NUM_PER_BLOCK);
  uint32_t alignedDv = CeilAlign(dims.dv, FP16_NUM_PER_BLOCK);
  uint32_t alignedDk = CeilAlign(dims.dk, FP16_NUM_PER_BLOCK);
  int64_t fixedUbBytes = CalcFixedUbBytes(alignedNv, alignedDv, alignedDk, inputs.hasGama == 1, inputs.hasGamaK == 1,
                                          inputs.gamaKScalar == 1);
  int64_t workingUbBytes = CalcWorkingUbBytes(alignedNv, alignedDv, alignedDk, inputs.hasGama == 1,
                                              inputs.hasGamaK == 1, inputs.gamaKScalar == 1);
  selected = {0, 0, 0, 0, false};
  const uint32_t candidates[][2] = {{1, 1}, {1, 2}, {2, 2}};
  uint32_t fixedQueueBufferNum = 5U + ((inputs.hasGamaK == 1U && inputs.gamaKScalar == 0U) ? 1U : 0U);
  for (const auto &candidate : candidates) {
    if (fixedQueueBufferNum + candidate[0] + candidate[1] > MAX_TQUE_BUFFER_NUM_310P) {
      continue;
    }
    BufferProfile profile = {0, 0, 0, 0, false};
    if (EvaluateBufferProfile(ubSize, workingUbBytes, alignedDk, dims.dv, candidate[0], candidate[1], profile) &&
        IsBetterProfile(profile, selected)) {
      selected = profile;
    }
  }
  if (!selected.valid) {
    return false;
  }
  int64_t queueCoeff = (2 + static_cast<int64_t>(2 * selected.stateOutBufferNum)) * alignedDk +
                       static_cast<int64_t>(2 * selected.attnOutBufferNum);
  ubRestBytes = ubSize - fixedUbBytes - queueCoeff * static_cast<int64_t>(selected.vStep);
  return ubRestBytes >= 0;
}

uint32_t GetBlockDim(const ShapeDims &dims, uint32_t coreNum) {
  uint64_t taskUnits = static_cast<uint64_t>(dims.b) * dims.nv;
  if (taskUnits == 0) {
    taskUnits = 1;
  }
  uint32_t blockDim = taskUnits < coreNum ? static_cast<uint32_t>(taskUnits) : coreNum;
  return blockDim == 0 ? 1 : blockDim;
}

bool FillTilingData(TilingContext *context, const ShapeDims &dims, const OptionalInputs &inputs,
                    const IndexTypes &indexTypes, const BufferProfile &profile, uint32_t coreNum, int64_t ubSize,
                    int64_t ubRestBytes) {
  auto tilingData = context->GetTilingData<RecurrentGatedDeltaRuleTilingData>();
  if (tilingData == nullptr) {
    return false;
  }
  tilingData->vectorCoreNum = coreNum;
  tilingData->ubCalSize = static_cast<uint32_t>(ubSize);
  tilingData->ubRestBytes = static_cast<uint32_t>(ubRestBytes);
  tilingData->t = dims.t;
  tilingData->nk = dims.nk;
  tilingData->dk = dims.dk;
  tilingData->nv = dims.nv;
  tilingData->dv = dims.dv;
  tilingData->sBlockNum = dims.sBlockNum;
  tilingData->b = dims.b;
  tilingData->vStep = profile.vStep;
  tilingData->stateOutBufferNum = profile.stateOutBufferNum;
  tilingData->attnOutBufferNum = profile.attnOutBufferNum;
  tilingData->scale = GetScale(context);
  tilingData->hasGama = inputs.hasGama;
  tilingData->hasGamaK = inputs.hasGamaK;
  tilingData->hasAcceptedTokens = inputs.hasAcceptedTokens;
  tilingData->gamaKScalar = inputs.gamaKScalar;
  tilingData->cuSeqlensIsPrefix = 0;
  tilingData->cuSeqlensIsInt64 = indexTypes.cuSeqlensIsInt64;
  tilingData->ssmStateIndicesIsInt64 = indexTypes.ssmStateIndicesIsInt64;
  tilingData->reserved = 0;
  return true;
}

}  // namespace

static uint32_t RecurrentGatedDeltaRuleTilingFunc(TilingContext *context) {
  if (context == nullptr) {
    return GRAPH_FAILED;
  }

  uint32_t coreNum = 0;
  int64_t ubSize = 0;
  ShapeDims dims = {};
  if (!GetPlatformResources(context, coreNum, ubSize) || !GetShapeDims(context, dims)) {
    return GRAPH_FAILED;
  }
  OptionalInputs optionalInputs = GetOptionalInputs(context, dims);
  IndexTypes indexTypes = GetIndexTypes(context);
  BufferProfile selected = {0, 0, 0, 0, false};
  int64_t ubRestBytes = 0;
  if (!SelectBufferProfile(ubSize, dims, optionalInputs, selected, ubRestBytes) ||
      !FillTilingData(context, dims, optionalInputs, indexTypes, selected, coreNum, ubSize, ubRestBytes)) {
    return GRAPH_FAILED;
  }
  context->SetBlockDim(GetBlockDim(dims, coreNum));
  context->SetTilingKey(0);
  size_t *ws = context->GetWorkspaceSizes(1);
  if (ws != nullptr) {
    ws[0] = SYSTEM_WORKSPACE_BYTES;
  }
  return GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RecurrentGatedDeltaRule)
  .Tiling(RecurrentGatedDeltaRuleTilingFunc, sizeof(RecurrentGatedDeltaRuleTilingData));
