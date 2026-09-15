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
constexpr uint32_t OUTPUT_OUT = 0;
constexpr uint32_t OUTPUT_STATE = 1;

static uint32_t RecurrentGatedDeltaRuleInferShape(InferShapeContext *context) {
  constexpr uint32_t INPUT_QUERY = 0;
  constexpr uint32_t INPUT_VALUE = 2;
  constexpr uint32_t INPUT_STATE = 4;
  constexpr size_t OUT_RANK = 3;    // out: [T, NV, DV]
  constexpr size_t STATE_RANK = 4;  // state: [S, NV, DK, DV]
  constexpr int64_t kIndex0 = 0;
  constexpr int64_t kIndex1 = 1;
  constexpr int64_t kIndex2 = 2;

  auto queryShape = context->GetInputShape(INPUT_QUERY);  // [T, NK, DK]
  auto valueShape = context->GetInputShape(INPUT_VALUE);  // [T, NV, DV]
  if (queryShape == nullptr || valueShape == nullptr) {
    return ge::GRAPH_FAILED;
  }

  int64_t t = queryShape->GetDim(kIndex0);
  int64_t nv = valueShape->GetDim(kIndex1);
  int64_t dv = valueShape->GetDim(kIndex2);

  // Output "out": [T, NV, DV]
  auto outShape = context->GetOutputShape(OUTPUT_OUT);
  outShape->SetDimNum(OUT_RANK);
  outShape->SetDim(kIndex0, t);
  outShape->SetDim(kIndex1, nv);
  outShape->SetDim(kIndex2, dv);

  auto stateShape = context->GetInputShape(INPUT_STATE);
  auto stateOutShape = context->GetOutputShape(OUTPUT_STATE);
  if (stateShape == nullptr || stateOutShape == nullptr) {
    return ge::GRAPH_FAILED;
  }
  stateOutShape->SetDimNum(STATE_RANK);
  for (size_t i = 0; i < STATE_RANK; ++i) {
    stateOutShape->SetDim(i, stateShape->GetDim(i));
  }

  return ge::GRAPH_SUCCESS;
}

static uint32_t RecurrentGatedDeltaRuleInferDataType(InferDataTypeContext *context) {
  context->SetOutputDataType(OUTPUT_OUT, ge::DT_FLOAT16);
  context->SetOutputDataType(OUTPUT_STATE, ge::DT_FLOAT16);
  return ge::GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_INFERSHAPE(RecurrentGatedDeltaRule)
  .InferShape(RecurrentGatedDeltaRuleInferShape)
  .InferDataType(RecurrentGatedDeltaRuleInferDataType);
