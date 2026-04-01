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

#include "src/litert/delegate/pnna/op/fully_connected_pnna.h"
#include "src/litert/delegate/delegate_utils.h"
#include "src/common/common.h"
#include "src/litert/cxx_api/tensor/tensor_impl.h"

namespace mindspore {
namespace lite {

bool PNNAFullyConnected::IsSupport() {
  if (in_tensors_.size() < DIMENSION_2D) {
    return false;
  }
  auto weight = in_tensors_[kWeightIndex];
  return weight.Shape().size() == DIMENSION_2D;
}

int PNNAFullyConnected::InitParams() {
  auto fc = op_primitive_->value_as_FullConnection();
  MS_CHECK_TRUE_RET(fc != nullptr, RET_ERROR);
  if (fc->use_axis()) {
    axis_ = ConvertToPnnaAxis(fc->axis(), in_tensors_.front().Shape().size());
  } else {
    auto format = in_tensors_.front().format();
    int axis = 0;
    if (format == NCHW) {
      axis = NCHW_C;
    } else if (format == NHWC) {
      axis = NHWC_C;
    } else {
      axis = -1;
    }
    axis_ = ConvertToPnnaAxis(axis, in_tensors_.front().Shape().size());
  }
  return RET_OK;
}

int PNNAFullyConnected::AddOpToPNNAModel(PNNASubGraph *graph) {
  MS_CHECK_TRUE_RET(graph != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(graph->graph() != nullptr, RET_ERROR);
  std::vector<std::shared_ptr<pnna::Tensor>> input_tensors;
  auto input_tensor = graph->GetMappedTensor(&in_tensors_[kInputIndex]);
  if (!input_tensor) {
    input_tensor = graph->ConvertOperand(&in_tensors_[kInputIndex]);
  }
  input_tensors.emplace_back(input_tensor);

  int ret = HandleConstantInputs(graph, &in_tensors_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "handle constant inputs failed.";
    return RET_ERROR;
  }

  auto filter_tensor = graph->ConvertOperand(&in_tensors_[kWeightIndex]);
  input_tensors.emplace_back(filter_tensor);
  if (in_tensors_.size() >= kInputSize2) {
    auto bias_tensor = graph->GetMappedTensor(&in_tensors_[kBiasIndex]);
    if (!bias_tensor) {
      bias_tensor = graph->ConvertOperand(&in_tensors_[kBiasIndex]);
    }
    input_tensors.emplace_back(bias_tensor);
  }
  auto output_tensor = graph->ConvertOperand(&out_tensors_[OUTPUT_INDEX]);

  auto fc_op = graph->graph()->CreateOperation<pnna::ops::FullyConnected>(axis_);
  fc_op->BindInputs(input_tensors);
  fc_op->BindOutputs({output_tensor});
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
