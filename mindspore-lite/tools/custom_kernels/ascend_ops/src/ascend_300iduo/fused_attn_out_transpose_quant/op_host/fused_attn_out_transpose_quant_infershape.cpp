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
// Input/output indices of FusedAttnOutTransposeQuant (see
// fused_attn_out_transpose_quant.cpp).
constexpr uint32_t kAttnInput = 0;
constexpr uint32_t kYOutput = 0;
constexpr uint32_t kTokenScaleOutput = 1;
// attn [..., M, D]: the trailing (M, D) dims are always present and the
// leading dims fold into H; y is the rank-2 token-major fold [M, H*D].
constexpr size_t kAttnTrailingDims = 2;
constexpr size_t kYRank = 2;

uint32_t FusedAttnOutTransposeQuantInferShape(InferShapeContext *context) {
  auto attnShape = context->GetInputShape(kAttnInput);
  if (attnShape == nullptr) {
    return ge::GRAPH_FAILED;
  }
  size_t dimNum = attnShape->GetDimNum();
  if (dimNum < kAttnTrailingDims) {
    return ge::GRAPH_FAILED;
  }
  // attn [..., M, D]: leading dims fold into H (e.g. [H,M,D] or [B,H,S,D] ->
  // H = B*H), M = tokens, D = head dim. y is the token-major fold [M, H*D];
  // token_scale is [M] (the kernel's flattened row-major view).
  // Dynamic dims (-1) can reach InferShape: fold only concrete dims and fall
  // back to the unknown marker instead of propagating negative arithmetic.
  int64_t heads = 1;
  bool headsUnknown = false;
  for (size_t d = 0; d + kAttnTrailingDims < dimNum; ++d) {
    int64_t headDim = attnShape->GetDim(d);
    if (headDim < 0) {
      headsUnknown = true;
      break;
    }
    heads *= headDim;
  }
  int64_t rows = attnShape->GetDim(dimNum - kAttnTrailingDims);
  int64_t colDim = attnShape->GetDim(dimNum - 1);
  int64_t cols = (headsUnknown || colDim < 0) ? -1 : heads * colDim;

  auto yShape = context->GetOutputShape(kYOutput);
  if (yShape == nullptr) {
    return ge::GRAPH_FAILED;
  }
  yShape->SetDimNum(kYRank);
  yShape->SetDim(0, rows);
  yShape->SetDim(1, cols);
  auto scaleShape = context->GetOutputShape(kTokenScaleOutput);
  if (scaleShape == nullptr) {
    return ge::GRAPH_FAILED;
  }
  scaleShape->SetDimNum(1);
  scaleShape->SetDim(0, rows);
  return ge::GRAPH_SUCCESS;
}

uint32_t FusedAttnOutTransposeQuantInferDataType(InferDataTypeContext *context) {
  context->SetOutputDataType(kYOutput, ge::DT_INT8);
  context->SetOutputDataType(kTokenScaleOutput, ge::DT_FLOAT);
  return ge::GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_INFERSHAPE(FusedAttnOutTransposeQuant)
  .InferShape(FusedAttnOutTransposeQuantInferShape)
  .InferDataType(FusedAttnOutTransposeQuantInferDataType);
