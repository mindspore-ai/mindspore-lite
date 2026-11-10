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

/* fused_causal_conv_silu_infershape.cpp InferShape/InferDataType for FusedCausalConvSilu. */

#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"

namespace ops {

namespace {
constexpr int64_t INPUT_DIM_NUM = 3;   // x: [T, S] tokens × in-channels (+batch slot)
constexpr int64_t WEIGHT_DIM_NUM = 2;  // w: [K, C] taps × out-channels
constexpr int64_t OUTPUT_DIM_NUM = 3;  // y/state keep the 3-D layout
constexpr int64_t DIM_IDX_2 = 2;
}  // namespace

static ge::graphStatus InferShapeFusedCausalConvSilu(gert::InferShapeContext *context) {
  if (context == nullptr) {
    return ge::GRAPH_FAILED;
  }
  auto shapeX = context->GetInputShape(0);
  auto shapeW = context->GetInputShape(1);
  auto shapeY = context->GetOutputShape(0);
  auto shapeSt = context->GetOutputShape(1);
  if (shapeX == nullptr || shapeW == nullptr || shapeY == nullptr || shapeSt == nullptr) {
    return ge::GRAPH_FAILED;
  }
  if (shapeX->GetDimNum() != INPUT_DIM_NUM || shapeW->GetDimNum() != WEIGHT_DIM_NUM) {
    return ge::GRAPH_FAILED;
  }
  shapeY->SetDimNum(OUTPUT_DIM_NUM);
  shapeY->SetDim(0, shapeX->GetDim(0));
  shapeY->SetDim(1, shapeX->GetDim(1));
  shapeY->SetDim(DIM_IDX_2, shapeW->GetDim(1));
  shapeSt->SetDimNum(OUTPUT_DIM_NUM);
  shapeSt->SetDim(0, shapeX->GetDim(0));
  shapeSt->SetDim(1, shapeW->GetDim(0));
  shapeSt->SetDim(DIM_IDX_2, shapeW->GetDim(1));
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeFusedCausalConvSilu(gert::InferDataTypeContext *context) {
  if (context == nullptr) {
    return ge::GRAPH_FAILED;
  }
  context->SetOutputDataType(0, context->GetInputDataType(0));
  context->SetOutputDataType(1, context->GetInputDataType(0));
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FusedCausalConvSilu)
  .InferShape(InferShapeFusedCausalConvSilu)
  .InferDataType(InferDataTypeFusedCausalConvSilu);

}  // namespace ops
