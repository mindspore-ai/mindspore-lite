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
// Input/output indices of FusedQKNormRopeBSH (see fused_qk_norm_rope_bsh.cpp).
constexpr uint32_t kQInput = 0;
constexpr uint32_t kKInput = 1;
constexpr uint32_t kQOutOutput = 0;
constexpr uint32_t kKOutOutput = 1;
constexpr uint32_t kMaxDimNum = 8;

// q [M..., Hq*D] / k [M..., Hkv*D] -> q_out / k_out mirroring their input
// shapes exactly (BSH layout preserved end to end; the only shape checks
// that matter -- D | Hq*D, cross-input M -- live in the tiling func).
uint32_t FusedQKNormRopeBSHInferShape(InferShapeContext *context) {
  auto qShape = context->GetInputShape(kQInput);
  auto kShape = context->GetInputShape(kKInput);
  if (qShape == nullptr || kShape == nullptr) {
    return ge::GRAPH_FAILED;
  }
  size_t qDimNum = qShape->GetDimNum();
  size_t kDimNum = kShape->GetDimNum();
  // Both ranks feed SetDimNum; validate the same upper bound on q and k so
  // a dynamic-rank k can never write past the output shape buffer.
  if (qDimNum == 0 || kDimNum == 0 || qDimNum > kMaxDimNum || kDimNum > kMaxDimNum) {
    return ge::GRAPH_FAILED;
  }
  auto qOutShape = context->GetOutputShape(kQOutOutput);
  auto kOutShape = context->GetOutputShape(kKOutOutput);
  if (qOutShape == nullptr || kOutShape == nullptr) {
    return ge::GRAPH_FAILED;
  }
  qOutShape->SetDimNum(qDimNum);
  for (size_t i = 0; i < qDimNum; ++i) {
    qOutShape->SetDim(i, qShape->GetDim(i));
  }
  kOutShape->SetDimNum(kDimNum);
  for (size_t i = 0; i < kDimNum; ++i) {
    kOutShape->SetDim(i, kShape->GetDim(i));
  }
  return ge::GRAPH_SUCCESS;
}

uint32_t FusedQKNormRopeBSHInferDataType(InferDataTypeContext *context) {
  context->SetOutputDataType(kQOutOutput, ge::DT_FLOAT16);
  context->SetOutputDataType(kKOutOutput, ge::DT_FLOAT16);
  return ge::GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_INFERSHAPE(FusedQKNormRopeBSH)
  .InferShape(FusedQKNormRopeBSHInferShape)
  .InferDataType(FusedQKNormRopeBSHInferDataType);
