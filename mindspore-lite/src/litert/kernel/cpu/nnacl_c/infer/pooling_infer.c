/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "nnacl_c/infer/pooling_infer.h"
#include <math.h>
#include "nnacl_c/infer/infer_register.h"

const int kOriginDefault = 0;
const int kStrideDefault = 1;

int ComputePadList(PoolingParameter *param, int input_h, int input_w, int output_h, int output_w) {
  if (param == NULL) {
    return NNACL_NULL_PTR;
  }
  int pad_h_all = ((output_h - 1) * param->stride_h_ + (param->window_h_ - 1) + 1 - input_h);
  int pad_w_all = ((output_w - 1) * param->stride_w_ + (param->window_w_ - 1) + 1 - input_w);
  if (pad_h_all < 0) {
    param->pad_u_ = param->pad_d_ = 0;
  } else {
    param->pad_u_ = pad_h_all / 2;
    param->pad_d_ = pad_h_all - param->pad_u_;
  }
  if (pad_w_all < 0) {
    param->pad_l_ = param->pad_r_ = 0;
  } else {
    param->pad_l_ = pad_w_all / 2;
    param->pad_r_ = pad_w_all - param->pad_l_;
  }
  return NNACL_OK;
}

int PoolingHandleDefault(PoolingParameter *param) {
  if (param == NULL) {
    return NNACL_NULL_PTR;
  }
  if (param->stride_h_ == kOriginDefault) {
    param->stride_h_ = kStrideDefault;
  }
  if (param->stride_w_ == kOriginDefault) {
    param->stride_w_ = kStrideDefault;
  }
  return NNACL_OK;
}

void HandleInputHW(const TensorC *input, int input_shape_size, int *input_h, int *input_w) {
  // if shape is 3 then Handle MaxPool1D else Handle MaxPool2D
  if (input_shape_size == DIMENSION_3D) {
    *input_h = 1;
    *input_w = input->shape_[1];
  } else {
    *input_h = input->shape_[1];
    *input_w = input->shape_[C2NUM];
  }
}

int HandleOutputHW(PoolingParameter *param, int *input_h, int *input_w, int *output_h, int *output_w) {
  int window_h = param->window_h_;
  int window_w = param->window_w_;
  if (param->global_) {
    param->window_h_ = window_h = *input_h;
    param->window_w_ = window_w = *input_w;
  }
  if ((param->stride_h_ == 0 || param->stride_w_ == 0) && !param->global_) {
    return NNACL_PARAM_INVALID;
  }
  if (param->pad_mode_ == Pad_same) {
    *output_w = ceil((float)(*input_w) / (float)(param->stride_w_));
    *output_h = ceil((float)(*input_h) / (float)(param->stride_h_));
    if (ComputePadList(param, *input_h, *input_w, *output_h, *output_w) != NNACL_OK) {
      return NNACL_NULL_PTR;
    }
    return NNACL_OK;
  } else {
    int round_mode = (RoundType)param->round_type_;
    if (round_mode == RoundType_Floor) {
      *output_h = floor((float)(*input_h + param->pad_u_ + param->pad_d_ - window_h) / param->stride_h_) + 1;
      *output_w = floor((float)(*input_w + param->pad_l_ + param->pad_r_ - window_w) / param->stride_w_) + 1;
      return NNACL_OK;
    } else if (round_mode == RoundType_Ceil) {
      *output_h = ceil((float)(*input_h + param->pad_u_ + param->pad_d_ - window_h) / param->stride_h_) + 1;
      *output_w = ceil((float)(*input_w + param->pad_l_ + param->pad_r_ - window_w) / param->stride_w_) + 1;
      return NNACL_OK;
    } else {
      return NNACL_ERR;
    }
  }
}

void HaneleOutputShape(int input_shape_size, int *output_shape, int *output_h, int *output_w) {
  // if MaxPool1D, output_shape[1]=output_w
  if (input_shape_size == DIMENSION_3D) {
    output_shape[1] = *output_w > 0 ? *output_w : 1;
  } else {
    output_shape[1] = *output_h > 0 ? *output_h : 1;
    output_shape[C2NUM] = *output_w > 0 ? *output_w : 1;
  }
}

int PoolingInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                      OpParameter *parameter) {
  int check_ret = CheckAugmentWithMinSize(inputs, inputs_size, outputs, outputs_size, parameter, 1, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  const TensorC *input = inputs[0];
  NNACL_CHECK_TRUE_RET(input->format_ == Format_NHWC || input->format_ == Format_NWC, NNACL_FORMAT_ERROR);
  for (size_t i = 0; i < outputs_size; i++) {
    TensorC *output = outputs[i];
    SetDataTypeFormat(output, input);
  }
  PoolingParameter *param = (PoolingParameter *)parameter;
  check_ret = PoolingHandleDefault(param);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }
  // MaxPool Only support 3D or 4D input
  if (input->shape_size_ < DIMENSION_3D || input->shape_size_ > DIMENSION_4D) {
    return NNACL_INPUT_TENSOR_ERROR;
  }

  int input_h = 0;
  int input_w = 0;
  // if 4D input, h,w = shape[1], shape[2]; if 3D input, h,w = 1, shape[1]
  if (input->shape_size_ != DIMENSION_3D && input->shape_size_ != DIMENSION_4D) {
    return NNACL_INPUT_TENSOR_ERROR;
  }
  HandleInputHW(input, input->shape_size_, &input_h, &input_w);
  int output_h = 0;
  int output_w = 0;
  check_ret = HandleOutputHW(param, &input_h, &input_w, &output_h, &output_w);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
  int output_shape[MAX_SHAPE_SIZE];
  size_t input_shape_size = 0;
  ShapeSet(output_shape, &input_shape_size, input->shape_, input->shape_size_);
  HaneleOutputShape(input_shape_size, output_shape, &output_h, &output_w);
  for (size_t i = 0; i < outputs_size; i++) {
    TensorC *output = outputs[i];
    SetShapeArray(output, output_shape, input_shape_size);
  }
  return NNACL_OK;
}

REG_INFER(MaxPool, PrimType_MaxPoolFusion, PoolingInferShape)
REG_INFER(AvgPool, PrimType_AvgPoolFusion, PoolingInferShape)
