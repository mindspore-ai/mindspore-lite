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

#include "coder/opcoders/nnacl/int8/hsigmoid_int8_coder.h"
#include "coder/log.h"
#include "include/errorcode.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.h"
#include "nnacl_c/activation_parameter.h"
#include "nnacl_c/int8/hsigmoid_int8.h"

namespace mindspore::lite::micro::nnacl {
constexpr auto kInt8Range = 256;

int HardSigmoidInt8Coder::Prepare(CoderContext *const context) {
  table_list_ = static_cast<int8_t *>(allocator_->Malloc(kNumberTypeInt8, kInt8Range, kOfflinePackWeight));
  MS_CHECK_TRUE_MSG(table_list_ != nullptr, RET_PARAM_INVALID, "HardSigmoid int8 table list is nullptr.");
  MS_CHECK_TRUE_MSG(input_tensor_ != nullptr, RET_PARAM_INVALID, "HardSigmoid int8 input_tensor_ is nullptr.");
  auto in_quant_args = input_tensor_->quant_params();
  MS_CHECK_TRUE_MSG(!in_quant_args.empty(), RET_PARAM_INVALID, "HardSigmoid int8 input quant param cannot be empty.");
  MS_CHECK_TRUE_MSG(output_tensor_ != nullptr, RET_PARAM_INVALID, "HardSigmoid int8 output_tensor_ is nullptr.");
  auto out_quant_args = output_tensor_->quant_params();
  MS_CHECK_TRUE_MSG(!out_quant_args.empty(), RET_PARAM_INVALID, "HardSigmoid int8 output quant param cannot be empty.");
  MS_CHECK_TRUE_MSG(parameter_ != nullptr, RET_PARAM_INVALID, "HardSigmoid int8 parameter_ is nullptr.");
  auto *activation_param = reinterpret_cast<ActivationParameter *>(parameter_);
  auto ret = HardSigmoidInt8InitLUT(static_cast<float>(in_quant_args.front().scale), in_quant_args.front().zeroPoint,
                                    static_cast<float>(out_quant_args.front().scale), out_quant_args.front().zeroPoint,
                                    activation_param->alpha_, activation_param->beta_, table_list_);
  MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "HardSigmoid int8 LUT init failed.");
  return RET_OK;
}

int HardSigmoidInt8Coder::DoCode(CoderContext *const context) {
  Collect(context,
          {
            "nnacl_c/int8/hsigmoid_int8.h",
          },
          {
            "hsigmoid_int8.c",
          });

  NNaclInt8Serializer code;
  int length = static_cast<int>(input_tensor_->ElementsNum());
  code.CodeFunction("HardSigmoidInt8", input_tensor_, length, output_tensor_, table_list_);
  context->AppendCode(code.str());
  return RET_OK;
}
}  // namespace mindspore::lite::micro::nnacl
