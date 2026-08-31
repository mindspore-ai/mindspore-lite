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
 * \file chunk_gated_delta_rule_infershape.cpp
 * \brief InferShape/InferDataType for ChunkGatedDeltaRule — logic ported verbatim from
 *        ops-transformer. Error handling adapted to this repo's header-light style (the
 *        upstream OP_LOGE / OP_CHECK_IF macros live in err/ops_err.h which is not shipped in
 *        this CANN install, so plain null-checks + GRAPH_FAILED returns are used instead).
 *
 *   out         shape = value[:3]              (T, Nv, Dv)     dtype = query dtype
 *   final_state shape = initial_state          (B, Nv, Dv, Dk) dtype = initial_state dtype
 */

#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"

namespace ops {

// Input indices (kept consistent with the operator prototype definition)
const size_t QUERY_INDEX = 0;
const size_t KEY_INDEX = 1;
const size_t VALUE_INDEX = 2;
const size_t BETA_INDEX = 3;
const size_t STATE_INDEX = 4;
const size_t ACTUAL_SEQ_LENGTHS_INDEX = 5;
const size_t G_INDEX = 6;

// Output indices
const size_t OUTPUT_OUT_IDX = 0;
const size_t OUTPUT_FINAL_STATE_IDX = 1;
const size_t VALUE_DIM = 3;
const size_t STATE_DIM = 4;

const size_t DIM_0 = 0;
const size_t DIM_1 = 1;
const size_t DIM_2 = 2;
const size_t DIM_3 = 3;

static ge::graphStatus InferShapeChunkGatedDeltaRule(gert::InferShapeContext *context) {
  if (context == nullptr) {
    return ge::GRAPH_FAILED;
  }

  auto shapeValue = context->GetInputShape(VALUE_INDEX);
  auto shapeInitialState = context->GetInputShape(STATE_INDEX);
  auto shapeOut = context->GetOutputShape(DIM_0);
  auto shapeFinalState = context->GetOutputShape(DIM_1);
  if (shapeValue == nullptr || shapeInitialState == nullptr || shapeOut == nullptr || shapeFinalState == nullptr) {
    return ge::GRAPH_FAILED;
  }

  // Validate DimNum of value and initialState before GetDim
  if (shapeValue->GetDimNum() != VALUE_DIM) {
    return ge::GRAPH_FAILED;
  }
  if (shapeInitialState->GetDimNum() != STATE_DIM) {
    return ge::GRAPH_FAILED;
  }

  // out shape comes from the first three dims of value (T, Nv, Dv)
  shapeOut->SetDimNum(VALUE_DIM);
  int64_t outDim0 = shapeValue->GetDim(DIM_0);
  int64_t outDim1 = shapeValue->GetDim(DIM_1);
  int64_t outDim2 = shapeValue->GetDim(DIM_2);
  shapeOut->SetDim(DIM_0, outDim0);
  shapeOut->SetDim(DIM_1, outDim1);
  shapeOut->SetDim(DIM_2, outDim2);

  // final_state shape follows initial_state
  shapeFinalState->SetDimNum(STATE_DIM);
  int64_t stateDim0 = shapeInitialState->GetDim(DIM_0);
  int64_t stateDim1 = shapeInitialState->GetDim(DIM_1);
  int64_t stateDim2 = shapeInitialState->GetDim(DIM_2);
  int64_t stateDim3 = shapeInitialState->GetDim(DIM_3);
  shapeFinalState->SetDim(DIM_0, stateDim0);
  shapeFinalState->SetDim(DIM_1, stateDim1);
  shapeFinalState->SetDim(DIM_2, stateDim2);
  shapeFinalState->SetDim(DIM_3, stateDim3);

  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeChunkGatedDeltaRule(gert::InferDataTypeContext *context) {
  if (context == nullptr) {
    return ge::GRAPH_FAILED;
  }
  // Infer out dtype from the input dtype of query; finalState follows initialState
  auto queryDtype = context->GetInputDataType(QUERY_INDEX);
  auto stateDtype = context->GetInputDataType(STATE_INDEX);
  context->SetOutputDataType(OUTPUT_OUT_IDX, queryDtype);
  context->SetOutputDataType(OUTPUT_FINAL_STATE_IDX, stateDtype);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ChunkGatedDeltaRule)
  .InferShape(InferShapeChunkGatedDeltaRule)
  .InferDataType(InferDataTypeChunkGatedDeltaRule);
}  // namespace ops
