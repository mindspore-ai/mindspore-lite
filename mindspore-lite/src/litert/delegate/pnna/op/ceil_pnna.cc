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

#include "src/litert/delegate/pnna/op/ceil_pnna.h"

namespace mindspore {
namespace lite {
bool PNNACeil::IsSupport() { return true; }

int PNNACeil::InitParams() {
  auto ceil = op_primitive_->value_as_Ceil();
  MS_CHECK_TRUE_RET(ceil != nullptr, RET_ERROR);
  return RET_OK;
}

int PNNACeil::AddOpToPNNAModel(PNNASubGraph *graph) {
  MS_CHECK_TRUE_RET(graph->graph() != nullptr, RET_ERROR);
  auto input_tensor = graph->GetMappedTensor(&in_tensors_[kInputIndex]);
  if (!input_tensor) {
    input_tensor = graph->ConvertOperand(&in_tensors_[kInputIndex]);
  }
  auto output_tensor = graph->ConvertOperand(&out_tensors_[kOutputIndex]);
  auto ceil = graph->graph()->CreateOperation<pnna::ops::Ceil>();
  ceil->BindInputs({input_tensor});
  ceil->BindOutputs({output_tensor});
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
