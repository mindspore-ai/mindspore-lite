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

#include "nnacl_c/int8/gelu_int8.h"
#include <math.h>
#include <limits.h>
#include "nnacl_c/errorcode.h"

#define GELU_SQRT_2_OVER_PI 0.7978845608f
#define GELU_COEFF 0.044715f
// 1/sqrt(2), kept in sync with the fp32 runtime erf form erf(x / sqrt(2)).
#define GELU_INV_SQRT_2 0.70710678118654752f

static float ComputeGeluFloat(float x, bool approximate) {
  if (approximate) {
    // tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float x_cubic = x * x * x;
    const float tanh_arg = GELU_SQRT_2_OVER_PI * (x + GELU_COEFF * x_cubic);
    const float tanh_val = tanhf(tanh_arg);
    return 0.5f * x * (1.0f + tanh_val);
  }
  // exact erf form: 0.5 * x * (1 + erf(x / sqrt(2)))
  const float erf_val = erff(x * GELU_INV_SQRT_2);
  return 0.5f * x * (1.0f + erf_val);
}

int GeluInt8InitLUT(const GeluQuantArg *quant_gelu_param, int8_t *table, bool approximate) {
  if (quant_gelu_param == NULL || table == NULL) {
    return NNACL_NULL_PTR;
  }

  const int32_t min_val = INT8_MIN;
  const int32_t max_val = INT8_MAX;

  const float in_scale = quant_gelu_param->in_args_.scale_;
  const int32_t in_zp = quant_gelu_param->in_args_.zp_;
  const float out_scale = quant_gelu_param->out_args_.scale_;
  const int32_t out_zp = quant_gelu_param->out_args_.zp_;
  const float output_inverse_scale = 1.0f / out_scale;

  for (int32_t i = min_val; i <= max_val; ++i) {
    const float real_input = in_scale * (i - in_zp);
    const float transformed = ComputeGeluFloat(real_input, approximate);
    const float rescaled = roundf(transformed * output_inverse_scale);
    int32_t quantized_value = (int32_t)(rescaled + out_zp);
    quantized_value = quantized_value > max_val ? max_val : quantized_value;
    quantized_value = quantized_value < min_val ? min_val : quantized_value;
    table[(uint8_t)i] = (int8_t)quantized_value;
  }

  return NNACL_OK;
}

int GeluInt8(const int8_t *src, int length, int8_t *dst, const int8_t *table) {
  if (src == NULL || dst == NULL || table == NULL) {
    return NNACL_NULL_PTR;
  }

  for (int i = 0; i < length; ++i) {
    const int8_t input_value = src[i];
    const uint8_t index = (uint8_t)input_value;
    dst[i] = table[index];
  }

  return NNACL_OK;
}
