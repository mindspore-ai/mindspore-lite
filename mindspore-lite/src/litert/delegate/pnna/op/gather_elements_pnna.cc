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

#include "src/litert/delegate/pnna/op/gather_elements_pnna.h"

namespace mindspore {
namespace lite {

namespace {
constexpr int kAxisNumDefault = 1;
constexpr int kAxisIndex = 1;
constexpr int kIndicesIndex = 2;
}  // namespace

bool PNNAGatherElements::IsSupport() { return true; }

int PNNAGatherElements::InitParams() {
  MS_CHECK_TRUE_RET(in_tensors_.size() == kInputSize2, RET_ERROR);
  auto axis_tensor = in_tensors_.at(kAxisIndex);
  MS_CHECK_TRUE_RET(axis_tensor.IsConst() && axis_tensor.DataType() == DataType::kNumberTypeInt32, RET_ERROR);
  MS_CHECK_TRUE_RET(axis_tensor.ElementNum() == kAxisNumDefault, RET_ERROR);
  auto axis_data = axis_tensor.Data().get();
  if (axis_data == nullptr) {
    MS_LOG(ERROR) << "Axis data is null.";
    return RET_ERROR;
  }
  int axis = *(reinterpret_cast<const int *>(axis_data));
  axis_ = ConvertToPnnaAxis(axis, in_tensors_.front().Shape().size());
  return RET_OK;
}

int PNNAGatherElements::AddOpToPNNAModel(PNNASubGraph *graph) {
  MS_CHECK_TRUE_RET(graph != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(graph->graph() != nullptr, RET_ERROR);
  auto input_tensor = graph->GetMappedTensor(&in_tensors_[kInputIndex]);
  if (!input_tensor) {
    input_tensor = graph->ConvertOperand(&in_tensors_[kInputIndex]);
  }
  auto indices_tensor = graph->GetMappedTensor(&in_tensors_[kIndicesIndex]);
  if (!indices_tensor) {
    indices_tensor = graph->ConvertOperand(&in_tensors_[kIndicesIndex]);
  }
  auto output_tensor = graph->ConvertOperand(&out_tensors_[kOutputIndex]);
  auto gather_op = graph->graph()->CreateOperation<pnna::ops::GatherElements>(axis_);
  gather_op->BindInputs({input_tensor, indices_tensor});
  gather_op->BindOutputs({output_tensor});
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
