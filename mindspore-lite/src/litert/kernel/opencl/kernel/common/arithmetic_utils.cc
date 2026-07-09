/**
 * Copyright 2023-2026 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/opencl/kernel/common/arithmetic_utils.h"

#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "src/litert/kernel/opencl/utils.h"

namespace mindspore::kernel {
int ValidateArithmeticSpecs(const std::vector<lite::Tensor *> &in_tensors,
                            const std::vector<lite::Tensor *> &out_tensors, const void *op_parameter,
                            schema::PrimitiveType primitive_type) {
  constexpr int kInputTensorSize = 2;
  constexpr int kOutputTensorSize = 1;
  if (in_tensors.size() != kInputTensorSize || out_tensors.size() != kOutputTensorSize) {
    MS_LOG(WARNING) << "in size: " << in_tensors.size() << ", out size: " << out_tensors.size();
    return lite::RET_ERROR;
  }

  auto *param = reinterpret_cast<const ArithmeticParameter *>(op_parameter);
  if (param == nullptr) {
    MS_LOG(WARNING) << "arithmetic op_parameter is nullptr.";
    return lite::RET_ERROR;
  }
  if (!IsArithmetic(primitive_type)) {
    MS_LOG(WARNING) << "UnSupported Operator: " << schema::EnumNamePrimitiveType(primitive_type);
    return lite::RET_ERROR;
  }

  if (primitive_type == schema::PrimitiveType_Eltwise) {
    auto mode = param->eltwise_mode_;
    if (mode != schema::EltwiseMode_PROD && mode != schema::EltwiseMode_SUM && mode != schema::EltwiseMode_MAXIMUM) {
      MS_LOG(WARNING) << "Eltwise mode not support, mode:" << mode;
      return lite::RET_ERROR;
    }
  }

  if (!(param->activation_type_ == schema::ActivationType_NO_ACTIVATION ||
        param->activation_type_ == schema::ActivationType_RELU ||
        param->activation_type_ == schema::ActivationType_RELU6)) {
    MS_LOG(WARNING) << "Unsupported activation type " << param->activation_type_;
    return lite::RET_ERROR;
  }

  return lite::RET_OK;
}
}  // namespace mindspore::kernel
