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

#include "src/litert/delegate/pnna/op/exp_pnna.h"

namespace mindspore {
namespace lite {

namespace {
constexpr float kScaleDefault = 1.0f;
constexpr float kShiftDefault = 0.0f;
constexpr float kBaseDefault = -1.0f;
}  // namespace

bool PNNAExp::IsSupport() { return true; }

int PNNAExp::InitParams() {
  auto exp = op_primitive_->value_as_ExpFusion();
  MS_CHECK_TRUE_RET(exp != nullptr, RET_ERROR);
  auto scale = exp->scale();
  auto shift = exp->shift();
  auto base = exp->base();
  MS_CHECK_TRUE_RET(scale == kScaleDefault, RET_ERROR);
  MS_CHECK_TRUE_RET(shift == kShiftDefault, RET_ERROR);
  MS_CHECK_TRUE_RET(base == kBaseDefault, RET_ERROR);
  return RET_OK;
}

int PNNAExp::AddOpToPNNAModel(PNNASubGraph *graph) {
  MS_CHECK_TRUE_RET(graph != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(graph->graph() != nullptr, RET_ERROR);
  auto input_tensor = graph->GetMappedTensor(&in_tensors_[kInputIndex]);
  if (!input_tensor) {
    input_tensor = graph->ConvertOperand(&in_tensors_[kInputIndex]);
  }
  auto output_tensor = graph->ConvertOperand(&out_tensors_[kOutputIndex]);
  auto exp = graph->graph()->CreateOperation<pnna::ops::Exp>();
  exp->BindInputs({input_tensor});
  exp->BindOutputs({output_tensor});
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
