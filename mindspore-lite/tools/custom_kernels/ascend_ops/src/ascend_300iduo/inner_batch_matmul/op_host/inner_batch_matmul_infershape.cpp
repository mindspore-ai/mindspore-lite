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
 * ============================================================================
 */
// InnerBatchMatMul infershape: y = x1 @ x2 with right-aligned (numpy) batch broadcast.
// GE's InferShapePass runs "InferShapeAndType", which requires the shape to be derived dim by dim
// and InferDataType to be registered as well (copying the whole shape or omitting InferDataType
// makes GE report "Call InferShapeAndType for node ... failed").
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "ref_shape_rules.h"  // NOLINT(build/include_subdir)

using namespace ge;                      // NOLINT(build/namespaces)
using namespace inner_batch_matmul_ref;  // NOLINT(build/namespaces)

namespace {
constexpr size_t kAttrAdjX1Idx = 0;
constexpr size_t kAttrAdjX2Idx = 1;

static uint32_t InnerBatchMatMulInferShape(gert::InferShapeContext *context) {
  if (context == nullptr) {
    return ge::GRAPH_FAILED;
  }
  auto x1Shape = context->GetInputShape(0);
  auto x2Shape = context->GetInputShape(1);
  auto outShape = context->GetOutputShape(0);
  auto attrs = context->GetAttrs();
  if (x1Shape == nullptr || x2Shape == nullptr || outShape == nullptr || attrs == nullptr) {
    return ge::GRAPH_FAILED;
  }
  // adj_x1/adj_x2 are INT in the op def (ONNX has no BOOL attribute type).
  const int64_t *adjX1 = attrs->GetAttrPointer<int64_t>(kAttrAdjX1Idx);
  const int64_t *adjX2 = attrs->GetAttrPointer<int64_t>(kAttrAdjX2Idx);
  // The op does not implement transposed multiply; reject it here as well so that shape inference and
  // tiling agree (otherwise a transposed node would get an inferred shape and fail only in tiling).
  if ((adjX1 != nullptr && *adjX1 != 0) || (adjX2 != nullptr && *adjX2 != 0)) {
    return ge::GRAPH_FAILED;
  }

  const size_t x1Rank = x1Shape->GetDimNum();
  const size_t x2Rank = x2Shape->GetDimNum();
  if (x1Rank < kMinRank || x2Rank < kMinRank || x1Rank > kMaxRank || x2Rank > kMaxRank) {
    return ge::GRAPH_FAILED;
  }
  int64_t x1Dims[kMaxRank] = {0};
  int64_t x2Dims[kMaxRank] = {0};
  for (size_t i = 0; i < x1Rank; ++i) {
    x1Dims[i] = x1Shape->GetDim(i);
  }
  for (size_t i = 0; i < x2Rank; ++i) {
    x2Dims[i] = x2Shape->GetDim(i);
  }

  MatMulDims dims;
  if (!GetMatMulDims(x1Dims, x1Rank, x2Dims, x2Rank, dims)) {
    return ge::GRAPH_FAILED;
  }
  if (!BroadcastBatchDims(x1Dims, dims.x1BatchRank, x2Dims, dims.x2BatchRank, dims)) {
    return ge::GRAPH_FAILED;
  }

  outShape->SetDimNum(dims.outBatchRank + kMatDims);
  for (size_t i = 0; i < dims.outBatchRank; ++i) {
    outShape->SetDim(i, dims.outBatch[i]);
  }
  outShape->SetDim(dims.outBatchRank, dims.m);
  outShape->SetDim(dims.outBatchRank + 1, dims.n);
  return ge::GRAPH_SUCCESS;
}

static uint32_t InnerBatchMatMulInferDataType(gert::InferDataTypeContext *context) {
  if (context == nullptr) {
    return ge::GRAPH_FAILED;
  }
  return context->SetOutputDataType(0, context->GetInputDataType(0));
}
}  // namespace

// The registration object is only needed for its static initializer, so it can stay file-local.
static IMPL_OP_INFERSHAPE(InnerBatchMatMul)
  .InferShape(InnerBatchMatMulInferShape)
  .InferDataType(InnerBatchMatMulInferDataType);
