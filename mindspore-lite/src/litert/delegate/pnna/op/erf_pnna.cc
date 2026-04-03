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

#include "src/litert/delegate/pnna/op/erf_pnna.h"

namespace mindspore {
namespace lite {
bool PNNAErf::IsSupport() { return true; }

int PNNAErf::InitParams() {
  auto erf = op_primitive_->value_as_Erf();
  MS_CHECK_TRUE_RET(erf != nullptr, RET_ERROR);
  return RET_OK;
}

int PNNAErf::AddOpToPNNAModel(PNNASubGraph *graph) {
  MS_CHECK_TRUE_RET(graph != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(graph->graph() != nullptr, RET_ERROR);
  auto input_tensor = graph->GetMappedTensor(&in_tensors_[kInputIndex]);
  if (!input_tensor) {
    input_tensor = graph->ConvertOperand(&in_tensors_[kInputIndex]);
  }
  auto output_tensor = graph->ConvertOperand(&out_tensors_[kOutputIndex]);
  auto erf_op = graph->graph()->CreateOperation<pnna::ops::Erf>();
  erf_op->BindInputs({input_tensor});
  erf_op->BindOutputs({output_tensor});
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
