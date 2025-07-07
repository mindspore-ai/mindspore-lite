/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_WRAPPER_FP32_CONV_WINOGRAD_FP32_WRAPPER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_WRAPPER_FP32_CONV_WINOGRAD_FP32_WRAPPER_H_
#include "nnacl/fp32/winograd_utils.h"
#include "nnacl/fp32/conv_winograd_fp32.h"
#ifdef __cplusplus
#include <string>
typedef struct TransFuncStr {
  std::string in_func_;
  std::string out_func_;
} TransFuncStr;
extern "C" {
#endif

typedef struct {
  const float *input_data_;
  const float *trans_weight_;
  const float *bias_data_;
  float *output_data_;
  TmpBufferAddress *buffer_list_;
  const ConvParameter *conv_param_;
  TransFuncList trans_func_;
} ConvWinogradFp32Args;

int ConvWinogradFp32Run(void *cdata, int task_id, float lhs_scale, float rhs_scale);

#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_WRAPPER_FP32_CONV_WINOGRAD_FP32_WRAPPER_H_
