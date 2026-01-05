/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "src/litert/delegate/pnna/op/addn_pnna.h"
#include <vector>

namespace mindspore {
namespace lite {
bool PNNAAddN::IsSupport() { return true; }

int PNNAAddN::InitParams() {
  auto addn = op_primitive_->value_as_AddN();
  MS_CHECK_TRUE_RET(addn != nullptr, RET_ERROR);
  return RET_OK;
}

int PNNAAddN::AddOpToPNNAModel(PNNASubGraph *graph) {
  MS_CHECK_TRUE_RET(graph->graph() != nullptr, RET_ERROR);
  std::vector<std::shared_ptr<pnna::Tensor>> input_tensors(in_tensors_.size());
  for (size_t i = 0; i < in_tensors_.size(); i++) {
    auto input_tensor = graph->GetMappedTensor(&in_tensors_[i]);
    if (!input_tensor) {
      input_tensor = graph->ConvertOperand(&in_tensors_[i]);
    }
    input_tensors[i] = input_tensor;
  }
  auto output_tensor = graph->ConvertOperand(&out_tensors_[kOutputIndex]);
  auto addn = graph->graph()->CreateOperation<pnna::ops::AddN>(in_tensors_.size());
  addn->BindInputs(input_tensors);
  addn->BindOutputs({output_tensor});
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
