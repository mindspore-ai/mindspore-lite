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

#include "nnacl_c/int8/elu_int8.h"
#include <math.h>
#include <limits.h>
#include "nnacl_c/errorcode.h"

int EluInt8InitLUT(const EluQuantArg *quant_elu_param, int8_t *table) {
  if (quant_elu_param == NULL || table == NULL) {
    return NNACL_NULL_PTR;
  }
  int32_t min_val = INT8_MIN;
  int32_t max_val = INT8_MAX;
  const float in_scale = quant_elu_param->in_args_.scale_;
  const int32_t in_zp = quant_elu_param->in_args_.zp_;
  const float out_scale = quant_elu_param->out_args_.scale_;
  const int32_t out_zp = quant_elu_param->out_args_.zp_;
  const float alpha = quant_elu_param->alpha_;
  const float output_inverse_scale = 1.0f / out_scale;

  for (int32_t i = min_val; i <= max_val; ++i) {
    const float real_input = in_scale * (i - in_zp);
    float transformed;
    // elu(x) = x if x > 0 else alpha*(exp(x)-1)
    if (real_input > 0.0f) {
      transformed = real_input;
    } else {
      transformed = alpha * (expf(real_input) - 1.0f);
    }
    const float rescaled = roundf(transformed * output_inverse_scale);
    int32_t quantized_value = (int32_t)(rescaled + out_zp);
    quantized_value = quantized_value > max_val ? max_val : quantized_value;
    quantized_value = quantized_value < min_val ? min_val : quantized_value;
    table[(uint8_t)i] = (int8_t)quantized_value;
  }
  return NNACL_OK;
}

int EluInt8(const int8_t *src, int length, int8_t *dst, const int8_t *table) {
  if (src == NULL || dst == NULL || table == NULL) {
    return NNACL_NULL_PTR;
  }
  for (int i = 0; i < length; ++i) {
    const int8_t input_value = src[i];
    uint8_t index = (uint8_t)input_value;
    dst[i] = table[index];
  }
  return NNACL_OK;
}
