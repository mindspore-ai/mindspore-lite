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

#include "src/litert/delegate/pnna/op/arithmetic_pnna.h"

namespace mindspore {
namespace lite {
#define CONVERT_PNNA_ELEMENTWISE(TYPE, CLASS_NAME)                                 \
  case schema::PrimitiveType_##TYPE: {                                             \
    auto arithmetic_op = graph->graph()->CreateOperation<pnna::ops::CLASS_NAME>(); \
    arithmetic_op->BindInputs({input0_tensor, input1_tensor});                     \
    arithmetic_op->BindOutput({output_tensor});                                    \
  } break;

bool PNNAArithmetic::IsSupport() {
  static const std::set<schema::PrimitiveType> supported_types = {
    schema::PrimitiveType_AddFusion, schema::PrimitiveType_SubFusion, schema::PrimitiveType_MulFusion,
    schema::PrimitiveType_DivFusion, schema::PrimitiveType_Maximum,   schema::PrimitiveType_Minimum,
    schema::PrimitiveType_PowFusion, schema::PrimitiveType_FloorDiv};
  if (supported_types.find(type_) == supported_types.end()) {
    MS_LOG(WARNING) << "Unsupported arithmetic type for operation " << static_cast<int>(type_) << " when running pnna";
    return false;
  }
  return true;
}

int PNNAArithmetic::InitParams() {
  switch (type_) {
    case schema::PrimitiveType_AddFusion: {
      auto add = op_primitive_->value_as_AddFusion();
      MS_CHECK_TRUE_RET(add != nullptr, RET_ERROR);
      act_type_ = add->activation_type();
    } break;
    case schema::PrimitiveType_SubFusion: {
      auto sub = op_primitive_->value_as_SubFusion();
      MS_CHECK_TRUE_RET(sub != nullptr, RET_ERROR);
      act_type_ = sub->activation_type();
    } break;
    case schema::PrimitiveType_MulFusion: {
      auto mul = op_primitive_->value_as_MulFusion();
      MS_CHECK_TRUE_RET(mul != nullptr, RET_ERROR);
      act_type_ = mul->activation_type();
    } break;
    case schema::PrimitiveType_DivFusion: {
      auto div = op_primitive_->value_as_DivFusion();
      MS_CHECK_TRUE_RET(div != nullptr, RET_ERROR);
      act_type_ = div->activation_type();
    } break;
    default:
      act_type_ = schema::ActivationType_NO_ACTIVATION;
      break;
  }
  return RET_OK;
}

int PNNAArithmetic::AddOpToPNNAModel(PNNASubGraph *graph) {
  MS_CHECK_TRUE_RET(graph != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(graph->graph() != nullptr, RET_ERROR);

  auto input0_tensor = graph->GetMappedTensor(&in_tensors_[Index0]);
  if (!input0_tensor) {
    input0_tensor = graph->ConvertOperand(&in_tensors_[Index0]);
  }
  auto input1_tensor = graph->GetMappedTensor(&in_tensors_[Index1]);
  if (!input1_tensor) {
    input1_tensor = graph->ConvertOperand(&in_tensors_[Index1]);
  }
  auto output_tensor = graph->ConvertOperand(&out_tensors_[kOutputIndex]);
  switch (type_) {
    CONVERT_PNNA_ELEMENTWISE(AddFusion, Add);
    CONVERT_PNNA_ELEMENTWISE(DivFusion, Div);
    CONVERT_PNNA_ELEMENTWISE(MulFusion, Multiply);
    CONVERT_PNNA_ELEMENTWISE(SubFusion, Sub);
    CONVERT_PNNA_ELEMENTWISE(Maximum, Maximum);
    CONVERT_PNNA_ELEMENTWISE(Minimum, Minimum);
    CONVERT_PNNA_ELEMENTWISE(PowFusion, Pow);
    CONVERT_PNNA_ELEMENTWISE(FloorDiv, FloorDiv);
    default:
      MS_LOG(ERROR) << "Unsupported arithmetic type " << static_cast<int>(type_) << " is found.";
      break;
  }
  return RET_OK;
}
#undef CONVERT_PNNA_ELEMENTWISE
}  // namespace lite
}  // namespace mindspore
