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
// Input/output indices of FusedGeluTanhNZ (see fused_gelu_tanh_nz.cpp).
constexpr uint32_t kXInput = 0;
constexpr uint32_t kYOutput = 0;

uint32_t FusedGeluTanhNZInferShape(InferShapeContext *context) {
  auto xShape = context->GetInputShape(kXInput);
  if (xShape == nullptr) {
    return ge::GRAPH_FAILED;
  }
  size_t dimNum = xShape->GetDimNum();
  if (dimNum == 0) {
    return ge::GRAPH_FAILED;
  }
  // y mirrors the full x shape; shape inference works on logical dims and the
  // framework derives the storage (NZ fractal) view from the selected format
  // combination on its own.
  auto yShape = context->GetOutputShape(kYOutput);
  if (yShape == nullptr) {
    return ge::GRAPH_FAILED;
  }
  yShape->SetDimNum(dimNum);
  for (size_t d = 0; d < dimNum; ++d) {
    yShape->SetDim(d, xShape->GetDim(d));
  }
  return ge::GRAPH_SUCCESS;
}

uint32_t FusedGeluTanhNZInferDataType(InferDataTypeContext *context) {
  context->SetOutputDataType(kYOutput, ge::DT_FLOAT16);
  return ge::GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_INFERSHAPE(FusedGeluTanhNZ).InferShape(FusedGeluTanhNZInferShape).InferDataType(FusedGeluTanhNZInferDataType);
