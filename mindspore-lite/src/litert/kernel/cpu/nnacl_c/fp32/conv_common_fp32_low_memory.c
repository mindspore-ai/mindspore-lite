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

#include <string.h>
#include "nnacl_c/fp32/conv_common_fp32_low_memory.h"
#include "nnacl_c/fp32/matmul_fp32.h"

void Im2ColPackUnitFp32Lowmemory(const float *input_data, const ConvParameter *conv_param, float *packed_input,
                                 int real_cal_num, int block_index) {
  // input format : nhwc
  int kernel_h = conv_param->kernel_h_;
  int kernel_w = conv_param->kernel_w_;
  int kernel_plane = kernel_h * kernel_w;
  int dilation_h = conv_param->dilation_h_;
  int dilation_w = conv_param->dilation_w_;
  int out_w = conv_param->output_w_;
  if (dilation_h == 0 || dilation_w == 0 || out_w == 0) {
    return;
  }
  int in_channel = conv_param->input_channel_;
  int in_w = conv_param->input_w_;
  for (int i = 0; i < real_cal_num; i++) {
    int block_start = block_index + i;
    int input_h = block_start / out_w * conv_param->stride_h_ - conv_param->pad_u_;
    int input_w = block_start % out_w * conv_param->stride_w_ - conv_param->pad_l_;
    if (conv_param->input_h_ - input_h < 0 || in_w - input_w < 0) {
      continue;
    }
    int input_stride = (input_h * in_w + input_w) * in_channel;
    int kh_s = MSMAX(0, UP_DIV(-input_h, dilation_h));
    int kh_e = MSMIN(kernel_h, UP_DIV(conv_param->input_h_ - input_h, dilation_h));
    int kw_s = MSMAX(0, UP_DIV(-input_w, dilation_w));
    int kw_e = MSMIN(kernel_w, UP_DIV(in_w - input_w, dilation_w));
    if (kw_e <= kw_s) {
      continue;
    }
    if (dilation_w == 1 && dilation_h == 1) {
      for (int j = kh_s; j < kh_e; j++) {
        int input_y_stride = j * in_w * in_channel + input_stride;
        int input_x_stride = input_y_stride + kw_s * in_channel;
        int input_plane_offset = (j * kernel_w + kw_s) * in_channel + i * in_channel * kernel_plane;
        memcpy(packed_input + input_plane_offset, input_data + input_x_stride,
               (kw_e - kw_s) * in_channel * sizeof(float));
      }  // kernel_h loop
    } else {
      for (int j = kh_s; j < kh_e; j++) {
        int input_y_stride = j * dilation_h * in_w * in_channel + input_stride;
        for (int k = kw_s; k < kw_e; ++k) {
          int input_x_stride = input_y_stride + k * dilation_w * in_channel;
          int input_plane_offset = (j * kernel_w + k) * in_channel + i * in_channel * kernel_plane;
          memcpy(packed_input + input_plane_offset, input_data + input_x_stride, in_channel * sizeof(float));
        }
      }  // kernel_h loop
    }
  }  // tile num loop
}

void ConvFp32LowMemory(const float *input_data, float *row2col_input, const float *weight, const float *bias_data,
                       float *output_data, const ConvParameter *conv_param) {
  int cal_num = 12;
  int output_hw = conv_param->output_h_ * conv_param->output_w_;
  int end_hw = output_hw;
  if (end_hw <= 0) {
    return;
  }
  int out_stride = conv_param->output_channel_ * cal_num;
  int deep = conv_param->kernel_h_ * conv_param->kernel_w_ * conv_param->input_channel_;
  size_t input_size = deep * cal_num * sizeof(float);
  for (int b = 0; b < conv_param->input_batch_; b++) {
    int out_channel = conv_param->output_channel_;
    int in_offset = b * conv_param->input_channel_ * conv_param->input_h_ * conv_param->input_w_;
    int out_offset = b * out_channel * output_hw;
    for (int i = 0; i < end_hw; i += cal_num, out_offset += out_stride) {
      int real_cal_row = MSMIN(output_hw - i, cal_num);
      memset(row2col_input, 0, input_size);
      Im2ColPackUnitFp32Lowmemory(input_data + in_offset, conv_param, row2col_input, real_cal_row, i);
      float *gemm_output = output_data + out_offset;
      MatmulLowMemory(row2col_input, weight, gemm_output, bias_data, (int)conv_param->act_type_, deep, real_cal_row,
                      out_channel, false, true);
    }
  }
}
