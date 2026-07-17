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

#include "nnacl_c/int8/hsigmoid_int8.h"
#include <math.h>
#include <limits.h>
#include "nnacl_c/errorcode.h"

int HardSigmoidInt8InitLUT(float input_scale, int32_t input_zp, float output_scale, int32_t output_zp, float alpha,
                           float beta, int8_t *table) {
  if (table == NULL) {
    return NNACL_NULL_PTR;
  }
  int32_t min_val = INT8_MIN;
  int32_t max_val = INT8_MAX;
  const float output_inverse_scale = 1.0f / output_scale;
  for (int32_t i = min_val; i <= max_val; ++i) {
    const float real_input = input_scale * (i - input_zp);
    float transformed = alpha * real_input + beta;
    if (transformed < 0.0f) {
      transformed = 0.0f;
    } else if (transformed > 1.0f) {
      transformed = 1.0f;
    }
    const float rescaled = roundf(transformed * output_inverse_scale);
    int32_t quantized_value = (int32_t)(rescaled + output_zp);
    quantized_value = quantized_value > max_val ? max_val : quantized_value;
    quantized_value = quantized_value < min_val ? min_val : quantized_value;
    table[(uint8_t)i] = (int8_t)quantized_value;
  }
  return NNACL_OK;
}

int HardSigmoidInt8(const int8_t *src, int length, int8_t *dst, const int8_t *table) {
  if (src == NULL || dst == NULL || table == NULL) {
    return NNACL_NULL_PTR;
  }
  for (int i = 0; i < length; ++i) {
    dst[i] = table[(uint8_t)src[i]];
  }
  return NNACL_OK;
}
