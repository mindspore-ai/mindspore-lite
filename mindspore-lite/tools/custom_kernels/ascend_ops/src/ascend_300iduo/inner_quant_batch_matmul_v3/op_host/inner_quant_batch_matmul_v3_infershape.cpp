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

// Matmul rank layout: [..., M, K] x [..., N, K] -> [..., M, N].
constexpr size_t kMatmulRankTail = 2;
// Input/output indices of InnerQuantBatchMatmulV3 (see the op def).
constexpr uint32_t kX1 = 0;
constexpr uint32_t kX2 = 1;
constexpr uint32_t kY = 0;

// y origin = [batch..., M, N]: the leading dims mirror x1 (validated equal
// to x2's in the tiling), M = x1 dim[-2], N = x2 dim[-2] (transpose_x2=true
// contract: x2 is [.., N, K]). The NZ storage layout is derived by GE from
// the def's FRACTAL_NZ port declaration (fused_swiglu/rms_norm_nz
// precedent: origin-only infershape).
uint32_t InnerQuantBatchMatmulV3InferShape(InferShapeContext *context) {
  // InferShapeContext::GetInputShape returns the ORIGIN shape directly
  // (gert::Shape: GetDim/GetDimNum only; there is no storage/origin split at
  // the infer-shape stage -- swiglu/IPFA infershape precedent).
  auto x1Shape = context->GetInputShape(kX1);
  auto x2Shape = context->GetInputShape(kX2);
  if (x1Shape == nullptr || x2Shape == nullptr) {
    return ge::GRAPH_FAILED;
  }
  size_t aDims = x1Shape->GetDimNum();
  size_t bDims = x2Shape->GetDimNum();
  if (aDims < kMatmulRankTail || bDims < kMatmulRankTail || aDims != bDims) {
    return ge::GRAPH_FAILED;
  }
  for (size_t d = 0; d + kMatmulRankTail < aDims; ++d) {
    if (x1Shape->GetDim(d) != x2Shape->GetDim(d)) {
      return ge::GRAPH_FAILED;
    }
  }
  auto yShape = context->GetOutputShape(kY);
  if (yShape == nullptr) {
    return ge::GRAPH_FAILED;
  }
  yShape->SetDimNum(aDims);
  for (size_t d = 0; d + kMatmulRankTail < aDims; ++d) {
    yShape->SetDim(d, x1Shape->GetDim(d));
  }
  yShape->SetDim(aDims - kMatmulRankTail, x1Shape->GetDim(aDims - kMatmulRankTail));
  yShape->SetDim(aDims - 1, x2Shape->GetDim(bDims - kMatmulRankTail));
  return ge::GRAPH_SUCCESS;
}

uint32_t InnerQuantBatchMatmulV3InferDataType(InferDataTypeContext *context) {
  context->SetOutputDataType(kY, ge::DT_FLOAT16);
  return ge::GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_INFERSHAPE(InnerQuantBatchMatmulV3)
  .InferShape(InnerQuantBatchMatmulV3InferShape)
  .InferDataType(InnerQuantBatchMatmulV3InferDataType);
