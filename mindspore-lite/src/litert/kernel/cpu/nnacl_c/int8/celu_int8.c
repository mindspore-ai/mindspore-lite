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

#include "nnacl_c/int8/celu_int8.h"
#include <math.h>
#include <limits.h>
#include "nnacl_c/errorcode.h"

int CeluInt8InitLUT(const CeluQuantArg *quant_celu_param, int8_t *table) {
  if (quant_celu_param == NULL || table == NULL) {
    return NNACL_NULL_PTR;
  }
  if (quant_celu_param->alpha_ <= 0.0f) {
    return NNACL_PARAM_INVALID;
  }
  const int32_t min_val = INT8_MIN;
  const int32_t max_val = INT8_MAX;
  const float in_scale = quant_celu_param->in_args_.scale_;
  const int32_t in_zp = quant_celu_param->in_args_.zp_;
  const float out_scale = quant_celu_param->out_args_.scale_;
  const int32_t out_zp = quant_celu_param->out_args_.zp_;
  const float alpha = quant_celu_param->alpha_;
  const float output_inverse_scale = 1.0f / out_scale;

  for (int32_t i = min_val; i <= max_val; ++i) {
    const float real_input = in_scale * (i - in_zp);
    float transformed;
    if (real_input > 0.0f) {
      transformed = real_input;
    } else {
      transformed = alpha * expm1f(real_input / alpha);
    }
    const float rescaled = roundf(transformed * output_inverse_scale);
    int32_t quantized_value = (int32_t)(rescaled + out_zp);
    quantized_value = quantized_value > max_val ? max_val : quantized_value;
    quantized_value = quantized_value < min_val ? min_val : quantized_value;
    table[(uint8_t)i] = (int8_t)quantized_value;
  }
  return NNACL_OK;
}

int CeluInt8(const int8_t *src, int length, int8_t *dst, const int8_t *table) {
  if (src == NULL || dst == NULL || table == NULL) {
    return NNACL_NULL_PTR;
  }
  for (int i = 0; i < length; ++i) {
    dst[i] = table[(uint8_t)src[i]];
  }
  return NNACL_OK;
}
