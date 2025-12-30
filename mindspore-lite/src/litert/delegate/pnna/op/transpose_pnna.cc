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

#include "src/litert/delegate/pnna/op/transpose_pnna.h"
#include "src/litert/delegate/pnna/pnna_utils.h"

namespace mindspore {
namespace lite {
int PNNATranspose::InitParams() {
  if (!perm_.empty()) {
    return RET_OK;
  }
  if (in_tensors_.size() != kInputSize1) {
    MS_LOG(ERROR) << "Transpose op expects " << kInputSize1 << " inputs, but got " << in_tensors_.size();
    return RET_ERROR;
  }
  auto perm_tensor = in_tensors_.at(1);
  if (perm_tensor.Data() == nullptr) {
    MS_LOG(ERROR) << "Transpose perm tensor data is nullptr";
    return RET_ERROR;
  }
  auto perm_num = perm_tensor.ElementNum();
  auto perm_data = reinterpret_cast<const int *>(perm_tensor.Data().get());
  for (size_t i = 0; i < perm_num; i++) {
    perm_.push_back(perm_data[i]);
  }
  return RET_OK;
}

bool PNNATranspose::IsSupport() {
  MS_CHECK_GE(in_tensors_.size(), kInputSize1, false);
  auto perm_tensor = in_tensors_.at(1);
  if (!perm_tensor.IsConst()) {
    MS_LOG(WARNING) << "PNNA transpose must get fixed axis values.";
    return false;
  }
  return true;
}

int PNNATranspose::AddOpToPNNAModel(PNNASubGraph *graph) {
  MS_CHECK_TRUE_RET(graph->graph() != nullptr, RET_ERROR);

  auto input_tensor = graph->GetMappedTensor(&in_tensors_[0]);
  if (!input_tensor) {
    input_tensor = graph->ConvertOperand(&in_tensors_[0]);
  }
  auto output_tensor = graph->ConvertOperand(&out_tensors_[0]);
  auto transpose_op =
    graph->graph()->CreateOperation<pnna::ops::Transpose>(ConvertToPnnaPerm(perm_.data(), perm_.size()));
  transpose_op->BindInputs({input_tensor});
  transpose_op->BindOutputs({output_tensor});
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
