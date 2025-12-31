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

#include "src/litert/delegate/pnna/op/arg_pnna.h"

namespace mindspore {
namespace lite {
bool PNNAArg::IsSupport() {
  if (type_ != schema::PrimitiveType_ArgMinFusion && type_ != schema::PrimitiveType_ArgMaxFusion) {
    MS_LOG(WARNING) << "Unsupported arg type for operation " << static_cast<int>(type_) << " when running pnna";
    return false;
  }
  return true;
}

int PNNAArg::InitParams() {
  switch (type_) {
    case schema::PrimitiveType_ArgMinFusion: {
      auto arg_min = op_primitive_->value_as_ArgMinFusion();
      MS_CHECK_TRUE_RET(arg_min != nullptr, RET_ERROR);
      axis_ = arg_min->axis();
      if (axis_ < 0) {
        axis_ += static_cast<int>(in_tensors_.front().Shape().size());
      }
    } break;
    case schema::PrimitiveType_ArgMaxFusion: {
      auto arg_max = op_primitive_->value_as_ArgMaxFusion();
      MS_CHECK_TRUE_RET(arg_max != nullptr, RET_ERROR);
      axis_ = arg_max->axis();
      if (axis_ < 0) {
        axis_ += static_cast<int>(in_tensors_.front().Shape().size());
      }
    } break;
    default:
      MS_LOG(WARNING) << "Unsupported arg type for operation " << static_cast<int>(type_) << " when running pnna";
      return RET_NOT_SUPPORT;
  }
  return RET_OK;
}

int PNNAArg::AddOpToPNNAModel(PNNASubGraph *graph) {
  MS_CHECK_TRUE_RET(graph->graph() != nullptr, RET_ERROR);
  // Convert to pnna tensors and operators
  auto input0_tensor = graph->GetMappedTensor(&in_tensors_[Index0]);
  if (!input0_tensor) {
    input0_tensor = graph->ConvertOperand(&in_tensors_[Index0]);
  }
  auto output_tensor = graph->ConvertOperand(&out_tensors_[kOutputIndex]);
  switch (type_) {
#define CONVERT_ELEMENTWISE(TYPE, CLASS_NAME)                             \
  case schema::PrimitiveType_##TYPE: {                                    \
    auto arg_op = graph->graph()->CreateOperation<pnna::ops::CLASS_NAME>( \
      ConvertToPnnaAxis(axis_, in_tensors_.front().Shape().size()));      \
    arg_op->BindInputs({input0_tensor});                                  \
    arg_op->BindOutput({output_tensor});                                  \
  } break;
    CONVERT_ELEMENTWISE(ArgMinFusion, ArgMin);
    CONVERT_ELEMENTWISE(ArgMaxFusion, ArgMax);
#undef CONVERT_ELEMENTWISE
    default:
      MS_LOG(ERROR) << "Unsupported arg type " << static_cast<int>(type_) << " is found.";
      return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
