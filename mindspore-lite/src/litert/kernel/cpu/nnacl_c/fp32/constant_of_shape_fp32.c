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

#include "nnacl_c/fp32/constant_of_shape_fp32.h"

int ConstantOfShapeInt32(int32_t *output, int start, int end, int32_t value) {
  for (int i = start; i < end; i++) {
    output[i] = value;
  }
  return NNACL_OK;
}

int ConstantOfShapeFp32(float *output, int start, int end, float value) {
  for (int i = start; i < end; i++) {
    output[i] = value;
  }
  return NNACL_OK;
}

int ConstantOfShapeBool(bool *output, int start, int end, bool value) {
  for (int i = start; i < end; i++) {
    output[i] = value;
  }
  return NNACL_OK;
}

int ConstantOfShapeInt8(int8_t *output, int start, int end, int8_t value) {
  for (int i = start; i < end; i++) {
    output[i] = value;
  }
  return NNACL_OK;
}
