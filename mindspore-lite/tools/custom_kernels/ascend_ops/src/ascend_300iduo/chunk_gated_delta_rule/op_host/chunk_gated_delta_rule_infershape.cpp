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
#include "exe_graph/runtime/infer_datatype_context.h"

using namespace ge;    // NOLINT(build/namespaces)
using namespace gert;  // NOLINT(build/namespaces)

namespace {
// out is rank-3 [T, Hv, Dv]; the head-dim axis is the last of the [T, H, D] layouts.
constexpr uint32_t kOutputRank = 3;
constexpr int32_t kHeadDimAxis = 2;
// Input indices for the CGDR operator.
constexpr uint32_t kQueryInput = 0;
constexpr uint32_t kValueInput = 2;
constexpr uint32_t kInitialStateInput = 4;
// Shape dimension indices within the [T, H, D] layout.
constexpr uint32_t kDimT = 0;
constexpr uint32_t kDimH = 1;
// Output indices.
constexpr uint32_t kOutShapeIndex = 0;
constexpr uint32_t kFinalStateOutIndex = 1;

static uint32_t ChunkGatedDeltaRuleInferShape(InferShapeContext *context) {
  auto queryShape = context->GetInputShape(kQueryInput);
  if (queryShape == nullptr) {
    return ge::GRAPH_FAILED;
  }
  auto valueShape = context->GetInputShape(kValueInput);
  if (valueShape == nullptr) {
    return ge::GRAPH_FAILED;
  }
  auto initialStateShape = context->GetInputShape(kInitialStateInput);
  if (initialStateShape == nullptr) {
    return ge::GRAPH_FAILED;
  }

  // query: [T, Hqk, Dk], value: [T, Hv, Dv]
  int64_t T = queryShape->GetDim(kDimT);
  int64_t Hv = valueShape->GetDim(kDimH);
  int64_t Dv = valueShape->GetDim(kHeadDimAxis);

  // out: [T, Hv, Dv]
  auto outShape = context->GetOutputShape(kOutShapeIndex);
  outShape->SetDimNum(kOutputRank);
  outShape->SetDim(kDimT, T);
  outShape->SetDim(kDimH, Hv);
  outShape->SetDim(kHeadDimAxis, Dv);

  // final_state: [B, Hv, Dv, Dk] — same shape as initial_state
  auto finalStateShape = context->GetOutputShape(kFinalStateOutIndex);
  uint32_t stateDimNum = initialStateShape->GetDimNum();
  finalStateShape->SetDimNum(stateDimNum);
  for (uint32_t i = 0; i < stateDimNum; i++) {
    finalStateShape->SetDim(i, initialStateShape->GetDim(i));
  }

  return ge::GRAPH_SUCCESS;
}

uint32_t ChunkGatedDeltaRuleInferDataType(InferDataTypeContext *context) {
  context->SetOutputDataType(kOutShapeIndex, context->GetInputDataType(kQueryInput));
  context->SetOutputDataType(kFinalStateOutIndex, context->GetInputDataType(kInitialStateInput));
  return ge::GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_INFERSHAPE(ChunkGatedDeltaRule)
  .InferShape(ChunkGatedDeltaRuleInferShape)
  .InferDataType(ChunkGatedDeltaRuleInferDataType);
