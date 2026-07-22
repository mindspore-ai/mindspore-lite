/**
 * Copyright 2026 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "register/op_impl_registry.h"

namespace ops {

static ge::graphStatus QuantMatmulW4a8InferShape(gert::InferShapeContext *context) {
  if (context == nullptr || context->GetInputShape(0) == nullptr || context->GetInputShape(1) == nullptr ||
      context->GetOutputShape(0) == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const auto &aDims = context->GetInputShape(0);
  int64_t M = aDims->GetDim(0);
  const auto &wDims = context->GetInputShape(1);
  int64_t N = wDims->GetDim(0);

  auto *outShape = context->GetOutputShape(0);
  outShape->SetDimNum(2);
  outShape->SetDim(0, M);
  outShape->SetDim(1, N);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus QuantMatmulW4a8InferDataType(gert::InferDataTypeContext *context) {
  if (context == nullptr) {
    return ge::GRAPH_FAILED;
  }
  context->SetOutputDataType(0, ge::DT_BF16);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QuantMatmulW4a8).InferShape(QuantMatmulW4a8InferShape).InferDataType(QuantMatmulW4a8InferDataType);

}  // namespace ops
