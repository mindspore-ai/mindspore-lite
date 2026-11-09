/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_NNACL_FP32_CONV_COMMON_LOW_MEMORY_H_
#define MINDSPORE_NNACL_FP32_CONV_COMMON_LOW_MEMORY_H_

#include "nnacl_c/conv_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif

void ConvFp32LowMemory(const float *input_data, float *row2col_input, const float *weight, const float *bias_data,
                       float *output_data, const ConvParameter *conv_param);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP32_CONV_COMMON_LOW_MEMORY_H_
