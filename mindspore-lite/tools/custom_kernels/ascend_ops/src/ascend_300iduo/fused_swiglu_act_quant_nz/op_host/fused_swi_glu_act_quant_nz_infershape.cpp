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
// Input/output indices of FusedSwiGLUActQuantNZ (see fused_swi_glu_act_quant_nz.cpp).
constexpr uint32_t kGateInput = 0;
constexpr uint32_t kYOutput = 0;
constexpr uint32_t kTokenScaleOutput = 1;

uint32_t FusedSwiGLUActQuantNZInferShape(InferShapeContext *context) {
  auto gateShape = context->GetInputShape(kGateInput);
  if (gateShape == nullptr) {
    return ge::GRAPH_FAILED;
  }
  size_t dimNum = gateShape->GetDimNum();
  if (dimNum == 0) {
    return ge::GRAPH_FAILED;
  }
  // y mirrors the full gate shape; token_scale is [M] with M the product of
  // the leading dims (rows), matching the kernel's flattened row-major view.
  auto yShape = context->GetOutputShape(kYOutput);
  if (yShape == nullptr) {
    return ge::GRAPH_FAILED;
  }
  yShape->SetDimNum(dimNum);
  int64_t rows = 1;
  for (size_t d = 0; d < dimNum; ++d) {
    yShape->SetDim(d, gateShape->GetDim(d));
    if (d + 1 < dimNum) {
      rows *= gateShape->GetDim(d);
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

uint32_t FusedSwiGLUActQuantNZInferDataType(InferDataTypeContext *context) {
  context->SetOutputDataType(kYOutput, ge::DT_INT8);
  context->SetOutputDataType(kTokenScaleOutput, ge::DT_FLOAT);
  return ge::GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_INFERSHAPE(FusedSwiGLUActQuantNZ)
  .InferShape(FusedSwiGLUActQuantNZInferShape)
  .InferDataType(FusedSwiGLUActQuantNZInferDataType);
