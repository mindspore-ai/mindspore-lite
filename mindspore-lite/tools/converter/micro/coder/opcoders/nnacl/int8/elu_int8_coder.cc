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
#include "coder/opcoders/nnacl/int8/elu_int8_coder.h"
#include <limits>
#include <algorithm>
#include "coder/log.h"
#include "include/errorcode.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.h"

namespace mindspore::lite::micro::nnacl {
constexpr auto kInt8Range = 256;

void CalculateEluTableList(int8_t *table, const float input_scale, const int32_t input_zp, const float output_scale,
                           const int32_t output_zp, const float alpha) {
  constexpr int32_t min_value = std::numeric_limits<int8_t>::min();
  constexpr int32_t max_value = std::numeric_limits<int8_t>::max();

  for (int i = min_value; i <= max_value; ++i) {
    const float real_input = input_scale * (i - input_zp);
    // Apply ELu transform: y = x > 0 ? x : alpha * (exp(x) - 1)
    float transformed;
    if (real_input > 0.0f) {
      transformed = real_input;
    } else {
      transformed = alpha * std::expm1(real_input);
    }
    const int32_t quantized = static_cast<int32_t>(std::round(transformed / output_scale) + output_zp);
    auto out_value = static_cast<int8_t>(std::max(std::min(quantized, max_value), min_value));
    auto index = static_cast<uint8_t>(i);
    table[index] = out_value;
  }
}

int EluInt8Coder::Prepare(CoderContext *const context) {
  table_list_ = static_cast<int8_t *>(allocator_->Malloc(kNumberTypeInt8, kInt8Range, kOfflinePackWeight));
  MS_CHECK_TRUE_MSG(table_list_ != nullptr, RET_PARAM_INVALID, "Elu table list is nullptr.");
  MS_CHECK_TRUE_MSG(input_tensor_ != nullptr, RET_PARAM_INVALID, "input_tensor_ is nullptr.");
  MS_CHECK_TRUE_MSG(!input_tensor_->quant_params().empty(), RET_PARAM_INVALID,
                    "input_tensor_->quant_params() is empty.");
  MS_CHECK_TRUE_MSG(output_tensor_ != nullptr, RET_PARAM_INVALID, "output_tensor_ is nullptr.");
  MS_CHECK_TRUE_MSG(!output_tensor_->quant_params().empty(), RET_PARAM_INVALID,
                    "output_tensor_->quant_params() is empty.");
  const float input_scale = input_tensor_->quant_params().at(0).scale;
  const int32_t input_zp = input_tensor_->quant_params().at(0).zeroPoint;
  const float output_scale = output_tensor_->quant_params().at(0).scale;
  const int32_t output_zp = output_tensor_->quant_params().at(0).zeroPoint;
  MS_CHECK_TRUE_MSG(parameter_ != nullptr, RET_PARAM_INVALID, "parameter_ is nullptr.");
  auto *activation_param = reinterpret_cast<ActivationParameter *>(parameter_);
  const float alpha = activation_param->alpha_;
  CalculateEluTableList(table_list_, input_scale, input_zp, output_scale, output_zp, alpha);
  return RET_OK;
}

int EluInt8Coder::DoCode(CoderContext *const context) {
  Collect(context,
          {
            "nnacl_c/int8/elu_int8.h",
          },
          {
            "elu_int8.c",
          });

  NNaclInt8Serializer code;

  int length = static_cast<int>(input_tensor_->ElementsNum());
  code.CodeFunction("EluInt8", input_tensor_, length, output_tensor_, table_list_);

  context->AppendCode(code.str());

  return RET_OK;
}
}  // namespace mindspore::lite::micro::nnacl
