/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef NNACL_FP16_WINOGRAD_UTILS_H_
#define NNACL_FP16_WINOGRAD_UTILS_H_

#include "nnacl/conv_parameter.h"
#include "nnacl/op_base.h"
#include "nnacl/fp16/winograd_utils_fp16_macro.h"

#define MAX_LEN 256

#ifdef __cplusplus
extern "C" {
#endif
typedef void (*InputTransFp16Func)(const float16_t *src_data, float16_t *dst_data, int src_step, int dst_step,
                                   int real_c);

typedef void (*InputTransStepFp16Func)(const float16_t *src_data, float16_t *dst_data, int src_step, int dst_step,
                                       int dst_row_step);

typedef void (*InputTransPackFp16Func)(float16_t *src_data, float16_t *dst_data, int src_step, int dst_step,
                                       int real_c);

typedef void (*OutputTransFp16Func)(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);

typedef struct TransFp16FuncList {
  InputTransFp16Func in_func_;
  InputTransStepFp16Func in_step_func_;
  InputTransPackFp16Func in_pack_func_;
  OutputTransFp16Func out_func_;
} TransFp16FuncList;

InputTransFp16Func GetInputTransFp16Func(int input_unit);

#ifdef ENABLE_ARM64
InputTransStepFp16Func GetInputTransStepFp16Func(int input_unit);

InputTransPackFp16Func GetInputTransPackFp16Func(int input_unit);
#endif

void InputTransform4x4UnitFp16(const float16_t *src_data, float16_t *dst_data, int src_step, int dst_step, int real_c);

void InputTransform6x6UnitFp16(const float16_t *src_data, float16_t *dst_data, int src_step, int dst_step, int real_c);

void InputTransform8x8UnitFp16(const float16_t *src_data, float16_t *dst_data, int src_step, int dst_step, int real_c);

void InputTransform4x4StepFp16(const float16_t *src_data, float16_t *dst_data, int src_step, int dst_step,
                               int dst_row_step);

void InputTransform6x6StepFp16(const float16_t *src_data, float16_t *dst_data, int src_step, int dst_step,
                               int dst_row_step);

void InputTransform8x8StepFp16(const float16_t *src_data, float16_t *dst_data, int src_step, int dst_step,
                               int dst_row_step);

#ifdef ENABLE_ARM64
void InputTransform4x4Pack16Fp16(float16_t *src_data, float16_t *dst_data, int src_step, int dst_step, int real_c);

void InputTransform6x6Pack16Fp16(float16_t *src_data, float16_t *dst_data, int src_step, int dst_step, int real_c);

void InputTransform8x8Pack16Fp16(float16_t *src_data, float16_t *dst_data, int src_step, int dst_step, int real_c);
#endif

OutputTransFp16Func GetOutputTransFp16Func(int input_unit, int output_unit, ActType act_type);

void OutputTransform4x2UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform4x2ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform4x2Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform4x3UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform4x3ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform4x3Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);

void OutputTransform6x2UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x2ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x2Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x3UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x3ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x3Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x4UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x4ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x4Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x5UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x5ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x5Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);

void OutputTransform8x2UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x2ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x2Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x3UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x3ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x3Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x4UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x4ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x4Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x5UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x5ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x5Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x6ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x6Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x7UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x7ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x7Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);

int SelectOutputUnitFp16(const ConvParameter *conv_param);

void CheckIfUseWinogradFp16(bool *use_winograd, int *output_unit, const ConvParameter *conv_param);
#ifdef __cplusplus
}
#endif

#endif  //  NNACL_FP16_WINOGRAD_UTILS_H_
