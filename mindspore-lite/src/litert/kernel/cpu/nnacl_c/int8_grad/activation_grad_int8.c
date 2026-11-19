/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#include "nnacl_c/int8_grad/activation_grad_int8.h"

void ReluXInt8Grad(float *grad, const int8_t *relu_input, int length, int zero_point) {
  for (int i = 0; i < length; ++i) {
    if (relu_input[i] <= zero_point) {
      grad[i] = 0.0f;
    }
  }
}
