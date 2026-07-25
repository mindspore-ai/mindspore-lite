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

#include "nnacl_c/int8/trilu_int8.h"
#include <math.h>
#include "nnacl_c/errorcode.h"

// fp32 Triu/Tril now uses the upstream TriuByte4/TrilByte4 from triu_tril_fp32.c.
// This file keeps only the int8 variant (upstream TriuByte1 writes literal 0 for masked
// elements instead of out_zp).

int TrilInt8(const int8_t *input, int height, int width, int diagonal, int8_t *output, int num, float in_scale,
             int in_zp, float out_scale, int out_zp) {
  if (input == NULL || output == NULL) {
    return NNACL_ERR;
  }
  if (height <= 0 || width <= 0 || num <= 0) {
    return NNACL_ERR;
  }
  const float ratio = in_scale / out_scale;
  for (int m = 0; m < num; m++) {
    const int plane = m * height * width;
    for (int h = 0; h < height; h++) {
      const int row_base = plane + h * width;
      for (int w = 0; w < width; w++) {
        if (h + diagonal >= w) {
          int q = (int)roundf(ratio * (input[row_base + w] - in_zp)) + out_zp;
          q = q > 127 ? 127 : q;
          q = q < -128 ? -128 : q;
          output[row_base + w] = (int8_t)q;
        } else {
          output[row_base + w] = (int8_t)out_zp;
        }
      }
    }
  }
  return NNACL_OK;
}

int TriuInt8(const int8_t *input, int height, int width, int diagonal, int8_t *output, int num, float in_scale,
             int in_zp, float out_scale, int out_zp) {
  if (input == NULL || output == NULL) {
    return NNACL_ERR;
  }
  if (height <= 0 || width <= 0 || num <= 0) {
    return NNACL_ERR;
  }
  const float ratio = in_scale / out_scale;
  for (int m = 0; m < num; m++) {
    const int plane = m * height * width;
    for (int h = 0; h < height; h++) {
      const int row_base = plane + h * width;
      for (int w = 0; w < width; w++) {
        if (h + diagonal <= w) {
          int q = (int)roundf(ratio * (input[row_base + w] - in_zp)) + out_zp;
          q = q > 127 ? 127 : q;
          q = q < -128 ? -128 : q;
          output[row_base + w] = (int8_t)q;
        } else {
          output[row_base + w] = (int8_t)out_zp;
        }
      }
    }
  }
  return NNACL_OK;
}
