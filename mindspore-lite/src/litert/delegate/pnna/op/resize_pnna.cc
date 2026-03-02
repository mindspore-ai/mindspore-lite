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

#include "src/litert/delegate/pnna/op/resize_pnna.h"
#include <cmath>

namespace mindspore {
namespace lite {
bool PNNAResize::IsSupport() {
  auto input = in_tensors_.front();
  bool valid_input = input.Shape().size() == DIMENSION_4D;
  bool valid_method = (method_ == static_cast<int>(schema::ResizeMethod_LINEAR) ||
                       method_ == static_cast<int>(schema::ResizeMethod_NEAREST));
  return valid_input && valid_method;
}

int PNNAResize::InitParams() {
  auto resize = op_primitive_->value_as_Resize();
  MS_CHECK_TRUE_RET(resize != nullptr, RET_ERROR);
  method_ = static_cast<int>(resize->method());
  height_.ix_ = resize->new_height();
  width_.ix_ = resize->new_width();
  if (in_tensors_.size() == DIMENSION_2D) {
    auto new_size_tensor = in_tensors_.at(1);
    if (!new_size_tensor.IsConst() ||
        (new_size_tensor.ElementNum() != DIMENSION_2D && new_size_tensor.ElementNum() != DIMENSION_4D)) {
      MS_LOG(ERROR) << "The new size of resize must be const value.";
      return RET_ERROR;
    }
    data_type_ = static_cast<int>(new_size_tensor.DataType());
    auto new_size = new_size_tensor.MutableData();
    MS_CHECK_TRUE_RET(new_size != nullptr, RET_ERROR);
    int height_idx = new_size_tensor.ElementNum() == DIMENSION_2D ? 0 : 2;
    int width_idx = new_size_tensor.ElementNum() == DIMENSION_2D ? 1 : 3;
    switch (static_cast<DataType>(data_type_)) {
      case DataType::kNumberTypeInt32:
        height_.ix_ = reinterpret_cast<int *>(new_size)[height_idx];
        width_.ix_ = reinterpret_cast<int *>(new_size)[width_idx];
        break;
      case DataType::kNumberTypeFloat32:
        height_.fx_ = reinterpret_cast<float *>(new_size)[height_idx];
        width_.fx_ = reinterpret_cast<float *>(new_size)[width_idx];
        break;
      default:
        MS_LOG(ERROR) << "The new size should be an int value or a float value.";
        return RET_ERROR;
    }
  }
  return RET_OK;
}

int PNNAResize::AddOpToPNNAModel(PNNASubGraph *graph) {
  MS_CHECK_TRUE_RET(graph != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(graph->graph() != nullptr, RET_ERROR);
  auto node_type = method_ == static_cast<int>(schema::ResizeMethod_LINEAR) ? pnna::ResizeType::BILINEAR
                                                                            : pnna::ResizeType::NEAREST_NEIGHBOR;

  auto input_tensor = graph->GetMappedTensor(&in_tensors_[0]);
  if (!input_tensor) {
    input_tensor = graph->ConvertOperand(&in_tensors_[0]);
  }
  auto output_tensor = graph->ConvertOperand(&out_tensors_[0]);
  switch (static_cast<DataType>(data_type_)) {
    case DataType::kNumberTypeInt32: {
      auto resize_op =
        graph->graph()->CreateOperation<pnna::ops::Resize>(node_type, 0, false, false, height_.ix_, width_.ix_);
      resize_op->BindInputs({input_tensor});
      resize_op->BindOutputs({output_tensor});
    } break;
    case DataType::kNumberTypeFloat32: {
      int target_height = static_cast<int>(std::lround(height_.fx_ * in_tensors_[0].Shape()[2]));
      int target_width = static_cast<int>(std::lround(width_.fx_ * in_tensors_[0].Shape()[3]));
      auto resize_op =
        graph->graph()->CreateOperation<pnna::ops::Resize>(node_type, 0.0f, false, false, target_height, target_width);
      resize_op->BindInputs({input_tensor});
      resize_op->BindOutputs({output_tensor});
    } break;
    default:
      MS_LOG(ERROR) << "The new size should be an int value or a float value.";
      return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
