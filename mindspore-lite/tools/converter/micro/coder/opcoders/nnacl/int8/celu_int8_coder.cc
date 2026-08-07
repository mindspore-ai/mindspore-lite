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

#include "coder/opcoders/nnacl/int8/celu_int8_coder.h"
#include "coder/log.h"
#include "include/errorcode.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.h"
#include "nnacl_c/int8/celu_int8.h"

namespace mindspore::lite::micro::nnacl {
constexpr auto kInt8Range = 256;

int CeluInt8Coder::Prepare(CoderContext *const context) {
  table_list_ = static_cast<int8_t *>(allocator_->Malloc(kNumberTypeInt8, kInt8Range, kOfflinePackWeight));
  MS_CHECK_TRUE_MSG(table_list_ != nullptr, RET_PARAM_INVALID, "Celu table list is nullptr.");
  MS_CHECK_TRUE_MSG(input_tensor_ != nullptr, RET_PARAM_INVALID, "input_tensor_ is nullptr.");
  MS_CHECK_TRUE_MSG(!input_tensor_->quant_params().empty(), RET_PARAM_INVALID,
                    "input_tensor_->quant_params() is empty.");
  MS_CHECK_TRUE_MSG(output_tensor_ != nullptr, RET_PARAM_INVALID, "output_tensor_ is nullptr.");
  MS_CHECK_TRUE_MSG(!output_tensor_->quant_params().empty(), RET_PARAM_INVALID,
                    "output_tensor_->quant_params() is empty.");
  MS_CHECK_TRUE_MSG(parameter_ != nullptr, RET_PARAM_INVALID, "parameter_ is nullptr.");
  auto *activation_param = reinterpret_cast<ActivationParameter *>(parameter_);
  quant_celu_parm_.in_args_.scale_ = input_tensor_->quant_params().at(0).scale;
  quant_celu_parm_.in_args_.zp_ = input_tensor_->quant_params().at(0).zeroPoint;
  quant_celu_parm_.out_args_.scale_ = output_tensor_->quant_params().at(0).scale;
  quant_celu_parm_.out_args_.zp_ = output_tensor_->quant_params().at(0).zeroPoint;
  quant_celu_parm_.alpha_ = activation_param->alpha_;
  auto ret = CeluInt8InitLUT(&quant_celu_parm_, table_list_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "CeluInt8InitLUT failed.";
    return ret;
  }
  return RET_OK;
}

int CeluInt8Coder::DoCode(CoderContext *const context) {
  Collect(context,
          {
            "nnacl_c/int8/celu_int8.h",
          },
          {
            "celu_int8.c",
          });

  NNaclInt8Serializer code;
  const int length = static_cast<int>(input_tensor_->ElementsNum());
  code.CodeFunction("CeluInt8", input_tensor_, length, output_tensor_, table_list_);
  context->AppendCode(code.str());
  return RET_OK;
}
}  // namespace mindspore::lite::micro::nnacl
