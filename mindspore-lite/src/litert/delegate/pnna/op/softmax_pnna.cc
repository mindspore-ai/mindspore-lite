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

#include "src/litert/delegate/pnna/op/softmax_pnna.h"

namespace mindspore {
namespace lite {
bool PNNASoftmax::IsSupport() {
  auto input = in_tensors_.front();
  return input.Shape().size() == DIMENSION_2D || input.Shape().size() == DIMENSION_4D;
}

int PNNASoftmax::InitParams() {
  auto softmax = op_primitive_->value_as_Softmax();
  MS_CHECK_TRUE_RET(softmax != nullptr, RET_ERROR);
  auto axis_data = softmax->axis();
  MS_CHECK_TRUE_RET(axis_data != nullptr && axis_data->size() == 1, RET_ERROR);
  axis_ = ConvertToPnnaAxis(axis_data->data()[Index0], in_tensors_.front().Shape().size());
  return RET_OK;
}

int PNNASoftmax::AddOpToPNNAModel(PNNASubGraph *graph) {
  MS_CHECK_TRUE_RET(graph != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(graph->graph() != nullptr, RET_ERROR);
  auto input_tensor = graph->GetMappedTensor(&in_tensors_[kInputIndex]);
  if (!input_tensor) {
    input_tensor = graph->ConvertOperand(&in_tensors_[kInputIndex]);
  }
  auto output_tensor = graph->ConvertOperand(&out_tensors_[kOutputIndex]);
  auto softmax = graph->graph()->CreateOperation<pnna::ops::Softmax>(1.0 /* beta */, axis_);
  (*softmax).BindInput(input_tensor).BindOutput(output_tensor);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
