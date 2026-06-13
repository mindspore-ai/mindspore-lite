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

using namespace ge;   // NOLINT(build/namespaces)
using namespace gert;  // NOLINT(build/namespaces)

namespace {
static uint32_t ChunkGatedDeltaRuleInferShape(InferShapeContext *context) {
    auto queryShape = context->GetInputShape(0);
    auto valueShape = context->GetInputShape(2);
    auto initialStateShape = context->GetInputShape(5);

    if (queryShape == nullptr || valueShape == nullptr || initialStateShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // query: [T, Hqk, Dk], value: [T, Hv, Dv]
    int64_t T = queryShape->GetDim(0);
    int64_t Hv = valueShape->GetDim(1);
    int64_t Dv = valueShape->GetDim(2);

    // out: [T, Hv, Dv]
    auto outShape = context->GetOutputShape(0);
    outShape->SetDimNum(3);
    outShape->SetDim(0, T);
    outShape->SetDim(1, Hv);
    outShape->SetDim(2, Dv);

    // final_state: [B, Hv, Dv, Dk] — same shape as initial_state
    auto finalStateShape = context->GetOutputShape(1);
    uint32_t stateDimNum = initialStateShape->GetDimNum();
    finalStateShape->SetDimNum(stateDimNum);
    for (uint32_t i = 0; i < stateDimNum; i++) {
        finalStateShape->SetDim(i, initialStateShape->GetDim(i));
    }

    return ge::GRAPH_SUCCESS;
}
}  // namespace

static uint32_t ChunkGatedDeltaRuleInferDataType(InferDataTypeContext *context) {
    context->SetOutputDataType(0, ge::DT_FLOAT16);
    context->SetOutputDataType(1, ge::DT_FLOAT16);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ChunkGatedDeltaRule)
    .InferShape(ChunkGatedDeltaRuleInferShape)
    .InferDataType(ChunkGatedDeltaRuleInferDataType);
