/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "src/common/ops/operator_populate/operator_populate_register.h"
#include "nnacl_c/constant_of_shape_parameter.h"
#include "infer/constant_of_shape.h"
#include "mindapi/ir/value.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_c.h"
using mindspore::ops::kNameConstantOfShape;
using mindspore::schema::PrimitiveType_ConstantOfShape;
namespace mindspore {
namespace lite {
OpParameter *PopulateConstantOfShapeOpParameter(const BaseOperatorPtr &base_operator) {
  auto param =
    reinterpret_cast<ConstantOfShapeParameter *>(PopulateOpParameter<ConstantOfShapeParameter>(base_operator));
  if (param == nullptr) {
    MS_LOG(ERROR) << "new ConstantOfShapeParameter failed.";
    return nullptr;
  }
  auto op = dynamic_cast<ops::ConstantOfShape *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "base_operator cast to ConstantOfShape failed";
    free(param);
    return nullptr;
  }
  auto value = op->get_value();
  param->data_type_ = static_cast<int>(op->get_data_type());
  std::vector<int64_t> shape;
  auto shape_attr = op->GetAttr("shape");
  if (shape_attr != nullptr) {
    shape = GetValue<std::vector<int64_t>>(shape_attr);
  }
  if (!shape.empty() && shape.size() <= MAX_SHAPE_SIZE) {
    param->shape_size_ = static_cast<int>(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
      param->shape_[i] = static_cast<int>(shape[i]);
    }
  }
  switch (param->data_type_) {
    case kNumberTypeFloat32:
      param->value_.f32_value_ = value[0];
      break;
    case kNumberTypeInt32:
      param->value_.int32_value_ = static_cast<int32_t>(value[0]);
      break;
    case kNumberTypeBool:
      param->value_.bool_value_ = static_cast<bool>(value[0]);
      break;
    case kNumberTypeInt8:
      // value attr is always float; the int8 fill byte is quantized from it at runtime
      // with the output tensor's scale/zeroPoint, so keep the real value here.
      param->value_.f32_value_ = value[0];
      break;
    default:
      MS_LOG(ERROR) << "The value of constant of shape is invalid";
      free(param);
      return nullptr;
  }
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameConstantOfShape, PrimitiveType_ConstantOfShape, PopulateConstantOfShapeOpParameter)
}  // namespace lite
}  // namespace mindspore
