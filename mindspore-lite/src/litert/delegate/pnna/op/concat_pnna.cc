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

#include "src/litert/delegate/pnna/op/concat_pnna.h"

namespace mindspore {
namespace lite {
bool PNNAConcat::IsSupport() {
  auto input = in_tensors_.front();
  return input.Shape().size() <= DIMENSION_4D;
}

int PNNAConcat::InitParams() {
  auto concat = op_primitive_->value_as_Concat();
  MS_CHECK_TRUE_RET(concat != nullptr, RET_ERROR);
  axis_ = concat->axis();
  if (axis_ < 0) {
    axis_ += static_cast<int>(in_tensors_.front().Shape().size());
  }
  return RET_OK;
}

int PNNAConcat::AddOpToPNNAModel(PNNASubGraph *graph) {
  MS_CHECK_TRUE_RET(graph != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(graph->graph() != nullptr, RET_ERROR);

  std::vector<std::shared_ptr<pnna::Tensor>> input_tensors;
  for (size_t i = 0; i < in_tensors_.size(); i++) {
    auto input_tensor = graph->GetMappedTensor(&in_tensors_[i]);
    if (!input_tensor) {
      input_tensor = graph->ConvertOperand(&in_tensors_[i]);
    }
    input_tensors.push_back(input_tensor);
  }
  auto output_tensor = graph->ConvertOperand(&out_tensors_[0]);
  auto concat_op = graph->graph()->CreateOperation<pnna::ops::Concat>(
    ConvertToPnnaAxis(axis_, in_tensors_.front().Shape().size()) /* WHCN */, input_tensors.size());
  concat_op->BindInputs(input_tensors);
  concat_op->BindOutputs({output_tensor});
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
