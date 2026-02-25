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

#include "src/litert/delegate/pnna/op/data_convert_pnna.h"

namespace mindspore {
namespace lite {
bool PNNADataConvert::IsSupport() { return true; }

int PNNADataConvert::InitParams() {
  auto data_convert = op_primitive_->value_as_QuantDTypeCast();
  MS_CHECK_TRUE_RET(data_convert != nullptr, RET_ERROR);
  return RET_OK;
}

int PNNADataConvert::AddOpToPNNAModel(PNNASubGraph *graph) {
  MS_CHECK_TRUE_RET(graph != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(graph->graph() != nullptr, RET_ERROR);

  auto input_tensor = graph->GetMappedTensor(&in_tensors_[Index0]);
  if (!input_tensor) {
    input_tensor = graph->ConvertOperand(&in_tensors_[Index0]);
  }
  auto output_tensor = graph->ConvertOperand(&out_tensors_[kOutputIndex]);
  auto data_convert_op = graph->graph()->CreateOperation<pnna::ops::DataConvert>();
  data_convert_op->BindInputs({input_tensor});
  data_convert_op->BindOutputs({output_tensor});
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
