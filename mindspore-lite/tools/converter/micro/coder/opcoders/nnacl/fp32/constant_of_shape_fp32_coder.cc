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

#include "coder/opcoders/nnacl/fp32/constant_of_shape_fp32_coder.h"
#include <string>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include "coder/log.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "nnacl_c/constant_of_shape_parameter.h"

using mindspore::schema::PrimitiveType_ConstantOfShape;

namespace {
// Emit a float literal that round-trips exactly (std::to_string only keeps 6 digits).
std::string FloatLiteral(float value) {
  std::ostringstream oss;
  oss << std::setprecision(std::numeric_limits<float>::max_digits10) << value;
  return oss.str();
}
}  // namespace

namespace mindspore::lite::micro::nnacl {

int ConstantOfShapeFP32Coder::Prepare(CoderContext *const context) {
  element_num_ = static_cast<int>(output_tensor_->ElementsNum());
  // Try computing from output shape (may be empty for dynamic)
  if (element_num_ <= 0) {
    auto shape = output_tensor_->shape();
    if (!shape.empty()) {
      element_num_ = 1;
      for (auto dim : shape) {
        if (dim > 0) element_num_ *= static_cast<int>(dim);
      }
    }
  }
  auto param = reinterpret_cast<ConstantOfShapeParameter *>(parameter_);
  if (param != nullptr) {
    data_type_ = param->data_type_;
    fill_value_ = param->value_.f32_value_;
    fill_value_int_ = param->value_.int32_value_;
    fill_value_bool_ = param->value_.bool_value_;
    if (param->element_size_ > 0) {
      element_num_ = param->element_size_;
    }
    if (param->shape_size_ > 0) {
      std::vector<int> out_shape;
      for (int i = 0; i < param->shape_size_; ++i) {
        out_shape.push_back(param->shape_[i]);
      }
      output_tensor_->set_shape(out_shape);
    }
  }
  auto ret = ResolveQuantFillValue();
  if (ret != RET_OK) {
    return ret;
  }
  // If still no element count, try reading from the shape input tensor's data
  if (element_num_ <= 0 && input_tensors().size() > 0) {
    auto shape_tensor = input_tensors().at(0);
    if (shape_tensor != nullptr && shape_tensor->data() != nullptr && shape_tensor->ElementsNum() > 0) {
      auto shape_data = static_cast<const int64_t *>(shape_tensor->data());
      element_num_ = 1;
      for (int i = 0; i < shape_tensor->ElementsNum(); ++i) {
        element_num_ *= static_cast<int>(shape_data[i]);
      }
    }
  }
  if (element_num_ <= 0) {
    MS_LOG(ERROR) << "ConstantOfShape output element count is unknown.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConstantOfShapeFP32Coder::ResolveQuantFillValue() {
  if (data_type_ != kNumberTypeInt8 || output_tensor_ == nullptr) {
    return RET_OK;
  }
  auto quant_params = output_tensor_->quant_params();
  if (quant_params.empty()) {
    MS_LOG(ERROR) << "int8 output quant_params cannot be empty.";
    return RET_ERROR;
  }
  if (quant_params.front().scale <= 0.0f) {
    MS_LOG(ERROR) << "int8 output quant_param scale must be positive.";
    return RET_ERROR;
  }
  int32_t q =
    static_cast<int32_t>(std::round(fill_value_ / quant_params.front().scale)) + quant_params.front().zeroPoint;
  q = q > INT8_MAX ? INT8_MAX : (q < INT8_MIN ? INT8_MIN : q);
  fill_value_int8_ = static_cast<int8_t>(q);
  return RET_OK;
}

int ConstantOfShapeFP32Coder::DoCode(CoderContext *const context) {
  Collect(context, {"nnacl_c/fp32/constant_of_shape_fp32.h"}, {"constant_of_shape_fp32.c"});

  NNaclFp32Serializer code;
  auto output_str = allocator_->GetRuntimeAddr(output_tensor_);
  int count = element_num_;

  switch (data_type_) {
    case kNumberTypeFloat32:
    case kNumberTypeFloat:
      code << "  for (int i = 0; i < " << count << "; ++i) {\n"
           << "    ((" << output_str << "))[i] = " << FloatLiteral(fill_value_) << "f;\n"
           << "  }\n";
      break;
    case kNumberTypeInt32:
      code << "  for (int i = 0; i < " << count << "; ++i) {\n"
           << "    ((" << output_str << "))[i] = " << fill_value_int_ << ";\n"
           << "  }\n";
      break;
    case kNumberTypeBool:
      code << "  for (int i = 0; i < " << count << "; ++i) {\n"
           << "    ((" << output_str << "))[i] = " << (fill_value_bool_ ? "true" : "false") << ";\n"
           << "  }\n";
      break;
    case kNumberTypeInt8:
      code << "  ConstantOfShapeInt8((" << output_str << "), 0, " << count << ", " << static_cast<int>(fill_value_int8_)
           << ");\n";
      break;
    default:
      code << "  for (int i = 0; i < " << count << "; ++i) {\n"
           << "    ((" << output_str << "))[i] = " << FloatLiteral(fill_value_) << "f;\n"
           << "  }\n";
      break;
  }
  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_ConstantOfShape,
                   CPUOpCoderCreator<ConstantOfShapeFP32Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt32, PrimitiveType_ConstantOfShape,
                   CPUOpCoderCreator<ConstantOfShapeFP32Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeBool, PrimitiveType_ConstantOfShape,
                   CPUOpCoderCreator<ConstantOfShapeFP32Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_ConstantOfShape,
                   CPUOpCoderCreator<ConstantOfShapeFP32Coder>)
}  // namespace mindspore::lite::micro::nnacl
