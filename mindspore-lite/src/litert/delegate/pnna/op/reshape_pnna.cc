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

#include "src/litert/delegate/pnna/op/reshape_pnna.h"
#include "src/litert/delegate/pnna/pnna_utils.h"

namespace mindspore {
namespace lite {
bool PNNAReshape::IsSupport() {
  if (in_tensors_.size() != kInputSize1) {
    MS_LOG(WARNING) << "Reshape op should have 2 input tensors.";
    return false;
  }
  auto shape_tensor = in_tensors_.at(1);
  if (shape_tensor.Data() == nullptr) {
    MS_LOG(WARNING) << "Reshape op only supports const shape.";
    return false;
  }
  if (shape_tensor.Shape().size() > 1 || shape_tensor.ElementNum() > DIMENSION_4D) {
    MS_LOG(WARNING) << "For PNNA Reshape op, the shape tensor should be a one-dimension tensor and its element number "
                       "should be less than 4.";
    return false;
  }
  return true;
}

int PNNAReshape::AddOpToPNNAModel(PNNASubGraph *graph) {
  MS_CHECK_TRUE_RET(graph != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(graph->graph() != nullptr, RET_ERROR);

  auto input_tensor = graph->GetMappedTensor(&in_tensors_[0]);
  if (!input_tensor) {
    input_tensor = graph->ConvertOperand(&in_tensors_[0]);
  }
  auto output_tensor = graph->ConvertOperand(&out_tensors_[0]);
  std::vector<uint32_t> size;
  auto num = in_tensors_[1].ElementNum();
  auto shape_data = reinterpret_cast<const int *>(in_tensors_[1].Data().get());
  for (int i = num - 1; i >= 0; i--) {
    size.push_back(shape_data[i]);
  }
  auto reshape_op = graph->graph()->CreateOperation<pnna::ops::Reshape>(size);
  reshape_op->BindInputs({input_tensor});
  reshape_op->BindOutputs({output_tensor});
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
