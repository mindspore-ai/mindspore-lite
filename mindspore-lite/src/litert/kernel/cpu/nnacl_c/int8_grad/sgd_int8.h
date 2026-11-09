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
#ifndef NNACL_INT8_GRAD_SGD_INT8_H_
#define NNACL_INT8_GRAD_SGD_INT8_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Momentum SGD over quantized integer weights/biases (QAS). `grad` is the
// integer-domain gradient produced by the backward pass (already multiplied by
// effective_scale). `inv_scale_square[o]` rescales it so an integer step with
// `learning_rate` is equivalent to a real-domain SGD step:
//   weight (int8, real = w_scale * q):  inv_scale_square = 1 / w_scale^2
//   bias   (int32, real = x_scale*w_scale*q): inv_scale_square = 1 / (x_scale*w_scale)^2
// Precompute these per output channel to avoid dividing at runtime.
void SgdMomentumUpdateInt8(int8_t *weight, float *momentum, const float *grad, const float *inv_scale_square,
                           int output_size, int inner_size, float learning_rate, float momentum_value);
void SgdMomentumUpdateInt32(int32_t *weight, float *momentum, const float *grad, const float *inv_scale_square,
                            int output_size, int inner_size, float learning_rate, float momentum_value);

#ifdef __cplusplus
}
#endif

#endif  // NNACL_INT8_GRAD_SGD_INT8_H_
