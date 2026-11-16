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
// Input/output indices of FusedAddRmsNormSmoothQuant (see
// fused_add_rms_norm_smooth_quant.cpp).
constexpr uint32_t kX1Input = 0;
constexpr uint32_t kResOutOutput = 0;
constexpr uint32_t kYOutput = 1;
constexpr uint32_t kTokenScaleOutput = 2;

uint32_t FusedAddRmsNormSmoothQuantInferShape(InferShapeContext *context) {
  auto x1Shape = context->GetInputShape(kX1Input);
  if (x1Shape == nullptr) {
    return ge::GRAPH_FAILED;
  }
  size_t dimNum = x1Shape->GetDimNum();
  if (dimNum == 0) {
    return ge::GRAPH_FAILED;
  }
  // res_out and y mirror the full x1 shape; token_scale is [M] with M the
  // product of the leading dims (rows), matching the kernel's flattened
  // row-major view.
  // Dynamic dims (-1) can reach InferShape: fold only concrete dims and use the
  // unknown marker for the flattened row count instead of negative arithmetic.
  int64_t rows = 1;
  bool rowsUnknown = false;
  for (size_t d = 0; d + 1 < dimNum; ++d) {
    int64_t dim = x1Shape->GetDim(d);
    if (dim < 0) {
      rowsUnknown = true;
      break;
    }
    rows *= dim;
  }
  if (rowsUnknown) {
    rows = -1;
  }
  for (uint32_t out : {kResOutOutput, kYOutput}) {
    auto outShape = context->GetOutputShape(out);
    if (outShape == nullptr) {
      return ge::GRAPH_FAILED;
    }
    outShape->SetDimNum(dimNum);
    for (size_t d = 0; d < dimNum; ++d) {
      outShape->SetDim(d, x1Shape->GetDim(d));
    }
  }
  auto scaleShape = context->GetOutputShape(kTokenScaleOutput);
  if (scaleShape == nullptr) {
    return ge::GRAPH_FAILED;
  }
  scaleShape->SetDimNum(1);
  scaleShape->SetDim(0, rows);
  return ge::GRAPH_SUCCESS;
}

uint32_t FusedAddRmsNormSmoothQuantInferDataType(InferDataTypeContext *context) {
  context->SetOutputDataType(kResOutOutput, ge::DT_FLOAT16);
  context->SetOutputDataType(kYOutput, ge::DT_INT8);
  context->SetOutputDataType(kTokenScaleOutput, ge::DT_FLOAT);
  return ge::GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_INFERSHAPE(FusedAddRmsNormSmoothQuant)
  .InferShape(FusedAddRmsNormSmoothQuantInferShape)
  .InferDataType(FusedAddRmsNormSmoothQuantInferDataType);
