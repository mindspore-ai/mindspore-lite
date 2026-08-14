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

using namespace ge;
using namespace gert;

namespace {
static uint32_t RecurrentGatedDeltaRuleInferShape(InferShapeContext *context) {
  auto queryShape = context->GetInputShape(0);  // [T, NK, DK]
  auto valueShape = context->GetInputShape(2);  // [T, NV, DV]
  if (queryShape == nullptr || valueShape == nullptr) {
    return ge::GRAPH_FAILED;
  }

  int64_t t = queryShape->GetDim(0);
  int64_t nv = valueShape->GetDim(1);
  int64_t dv = valueShape->GetDim(2);

  // Output "out": [T, NV, DV]
  auto outShape = context->GetOutputShape(0);
  outShape->SetDimNum(3);
  outShape->SetDim(0, t);
  outShape->SetDim(1, nv);
  outShape->SetDim(2, dv);

  auto stateShape = context->GetInputShape(4);
  auto stateOutShape = context->GetOutputShape(1);
  if (stateShape == nullptr || stateOutShape == nullptr) {
    return ge::GRAPH_FAILED;
  }
  stateOutShape->SetDimNum(4);
  for (size_t i = 0; i < 4; ++i) {
    stateOutShape->SetDim(i, stateShape->GetDim(i));
  }

  return ge::GRAPH_SUCCESS;
}

static uint32_t RecurrentGatedDeltaRuleInferDataType(InferDataTypeContext *context) {
  context->SetOutputDataType(0, ge::DT_FLOAT16);
  context->SetOutputDataType(1, ge::DT_FLOAT16);
  return ge::GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_INFERSHAPE(RecurrentGatedDeltaRule310P)
  .InferShape(RecurrentGatedDeltaRuleInferShape)
  .InferDataType(RecurrentGatedDeltaRuleInferDataType);
