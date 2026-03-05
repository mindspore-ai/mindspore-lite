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

#include "src/litert/delegate/pnna/op/depth_to_space_pnna.h"
#include "src/litert/delegate/pnna/pnna_utils.h"

namespace mindspore {
namespace lite {

bool PNNADepthToSpace::IsSupport() {
  auto input = in_tensors_.front();
  return input.Shape().size() == DIMENSION_4D;
}

int PNNADepthToSpace::InitParams() {
  auto depthtospace = op_primitive_->value_as_DepthToSpace();
  MS_CHECK_TRUE_RET(depthtospace != nullptr, RET_ERROR);
  block_size_ = depthtospace->block_size();
  MS_CHECK_TRUE_RET(block_size_ >= 2, RET_ERROR);
  auto mode_str = depthtospace->mode();
  if (mode_str != nullptr) {
    mode_str_ = mode_str->data();
    if (mode_str_ == "DCR") {
      mode_ = pnna::ops::DepthToSpace::DCR_mode;
    } else if (mode_str_ == "CRD") {
      mode_ = pnna::ops::DepthToSpace::CRD_mode;
    }
  }
  return RET_OK;
}

int PNNADepthToSpace::AddOpToPNNAModel(PNNASubGraph *graph) {
  MS_CHECK_TRUE_RET(graph->graph() != nullptr, RET_ERROR);
  auto input_tensor = graph->GetMappedTensor(&in_tensors_[kInputIndex]);
  if (!input_tensor) {
    input_tensor = graph->ConvertOperand(&in_tensors_[kInputIndex]);
  }
  auto output_tensor = graph->ConvertOperand(&out_tensors_[kOutputIndex]);
  std::shared_ptr<pnna::Operation> depthtospace_op;
  if (mode_str_.empty()) {
    depthtospace_op = graph->graph()->CreateOperation<pnna::ops::DepthToSpace>(block_size_);
  } else {
    depthtospace_op = graph->graph()->CreateOperation<pnna::ops::DepthToSpace>(block_size_, mode_);
  }
  depthtospace_op->BindInputs({input_tensor});
  depthtospace_op->BindOutputs({output_tensor});
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
