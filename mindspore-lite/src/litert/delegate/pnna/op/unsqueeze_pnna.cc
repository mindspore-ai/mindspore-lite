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

#include "src/litert/delegate/pnna/op/unsqueeze_pnna.h"
#include <algorithm>
#include <vector>

namespace mindspore {
namespace lite {
bool PNNAUnsqueeze::IsSupport() {
  auto input = in_tensors_.front();
  return input.Shape().size() <= DIMENSION_4D;
}

int PNNAUnsqueeze::InitParams() {
  auto unsqueeze = op_primitive_->value_as_Unsqueeze();
  MS_CHECK_TRUE_RET(unsqueeze != nullptr, RET_ERROR);
  auto axes = unsqueeze->axis();
  MS_CHECK_TRUE_RET(axes != nullptr, RET_ERROR);
  (void)std::transform(axes->begin(), axes->end(), std::back_inserter(axis_), [](int x) { return x; });
  return RET_OK;
}

int PNNAUnsqueeze::AddOpToPNNAModel(PNNASubGraph *graph) {
  MS_CHECK_TRUE_RET(graph != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(graph->graph() != nullptr, RET_ERROR);
  auto input_tensor = graph->GetMappedTensor(&in_tensors_[kInputIndex]);
  if (!input_tensor) {
    input_tensor = graph->ConvertOperand(&in_tensors_[kInputIndex]);
  }
  auto output_tensor = graph->ConvertOperand(&out_tensors_[kOutputIndex]);
  // WHCN
  std::vector<uint32_t> size;
  for (int i = out_tensors_[kOutputIndex].Shape().size() - 1; i >= 0; i--) {
    size.push_back(out_tensors_[kOutputIndex].Shape()[i]);
  }
  auto reshape_op = graph->graph()->CreateOperation<pnna::ops::Reshape>(size);
  reshape_op->BindInputs({input_tensor});
  reshape_op->BindOutputs({output_tensor});
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
