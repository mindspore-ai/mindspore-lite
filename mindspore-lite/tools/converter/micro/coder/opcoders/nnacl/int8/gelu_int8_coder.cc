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
#include "coder/opcoders/nnacl/int8/gelu_int8_coder.h"
#include <limits>
#include <algorithm>
#include <cmath>
#include "coder/log.h"
#include "include/errorcode.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.h"

namespace mindspore::lite::micro::nnacl {
constexpr auto kInt8Range = 256;

// Gelu has two modes selected by the `approximate` flag:
//   approximate=false (erf):  0.5 * x * (1 + erf(x / sqrt(2)))
//   approximate=true  (tanh): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Both must stay numerically identical to ComputeGeluFloat in nnacl_c/int8/gelu_int8.c
// so the codegen-time LUT equals the runtime LUT for the same mode.
constexpr float kGeluSqrtTwoOverPi = 0.7978845608f;
constexpr float kGeluCubicCoeff = 0.044715f;
constexpr float kGeluInvSqrt2 = 0.70710678118654752f;  // 1/sqrt(2), matches the fp32 erf form

namespace {
float ComputeGeluActivation(float real_input, bool approximate) {
  const float x = real_input;
  if (approximate) {
    const float x_cubic = x * x * x;
    const float tanh_arg = kGeluSqrtTwoOverPi * (x + kGeluCubicCoeff * x_cubic);
    return 0.5f * x * (1.0f + std::tanh(tanh_arg));
  }
  return 0.5f * x * (1.0f + std::erf(x * kGeluInvSqrt2));
}

int8_t ClampToInt8(int32_t quantized) {
  constexpr int32_t min_value = std::numeric_limits<int8_t>::min();
  constexpr int32_t max_value = std::numeric_limits<int8_t>::max();
  return static_cast<int8_t>(std::max(std::min(quantized, max_value), min_value));
}

int8_t QuantizeGeluValue(int input, const float input_scale, const int32_t input_zp, const float output_scale,
                         const int32_t output_zp, bool approximate) {
  const float real_input = input_scale * (input - input_zp);
  const float transformed = ComputeGeluActivation(real_input, approximate);
  const int32_t quantized = static_cast<int32_t>(std::round(transformed / output_scale) + output_zp);
  return ClampToInt8(quantized);
}
}  // namespace

void CalculateGeluTableList(int8_t *table, const float input_scale, const int32_t input_zp, const float output_scale,
                            const int32_t output_zp, bool approximate) {
  constexpr int32_t min_value = std::numeric_limits<int8_t>::min();
  constexpr int32_t max_value = std::numeric_limits<int8_t>::max();
  for (int i = min_value; i <= max_value; ++i) {
    table[static_cast<uint8_t>(i)] = QuantizeGeluValue(i, input_scale, input_zp, output_scale, output_zp, approximate);
  }
}

int GeluInt8Coder::Prepare(CoderContext *const context) {
  table_list_ = static_cast<int8_t *>(allocator_->Malloc(kNumberTypeInt8, kInt8Range, kOfflinePackWeight));
  MS_CHECK_TRUE_MSG(table_list_ != nullptr, RET_PARAM_INVALID, "Gelu table list is nullptr.");
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
  // Honour the approximate flag so the codegen-time LUT matches the model's erf/tanh mode.
  bool approximate = false;
  if (parameter_ != nullptr) {
    approximate = reinterpret_cast<ActivationParameter *>(parameter_)->approximate_;
  }
  CalculateGeluTableList(table_list_, input_scale, input_zp, output_scale, output_zp, approximate);
  return RET_OK;
}

int GeluInt8Coder::DoCode(CoderContext *const context) {
  Collect(context,
          {
            "nnacl_c/int8/gelu_int8.h",
          },
          {
            "gelu_int8.c",
          });

  NNaclInt8Serializer code;

  int length = static_cast<int>(input_tensor_->ElementsNum());
  code.CodeFunction("GeluInt8", input_tensor_, length, output_tensor_, table_list_);

  context->AppendCode(code.str());

  return RET_OK;
}

}  // namespace mindspore::lite::micro::nnacl
