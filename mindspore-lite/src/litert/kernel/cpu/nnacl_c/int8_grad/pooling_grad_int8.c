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
#include "nnacl_c/int8_grad/pooling_grad_int8.h"

void MaxPoolInt8Grad(const float *dy, const int8_t *pool_input, const PoolingParameter *param,
                     const PoolingComputeParam *compute, float *dx) {
  int output_h = compute->output_h_;
  int output_w = compute->output_w_;
  for (int oh = 0; oh < output_h; ++oh) {
    for (int ow = 0; ow < output_w; ++ow) {
      for (int c = 0; c < compute->input_channel_; ++c) {
        int base_h = oh * param->stride_h_ - param->pad_u_;
        int base_w = ow * param->stride_w_ - param->pad_l_;
        int max_index = -1;
        int8_t max_value = INT8_MIN;
        for (int kh = 0; kh < compute->window_h_; ++kh) {
          int ih = base_h + kh;
          if (ih < 0 || ih >= compute->input_h_) {
            continue;
          }
          for (int kw = 0; kw < compute->window_w_; ++kw) {
            int iw = base_w + kw;
            if (iw < 0 || iw >= compute->input_w_) {
              continue;
            }
            int index = (ih * compute->input_w_ + iw) * compute->input_channel_ + c;
            if (max_index < 0 || pool_input[index] > max_value) {
              max_value = pool_input[index];
              max_index = index;
            }
          }
        }
        if (max_index >= 0) {
          dx[max_index] += dy[(oh * output_w + ow) * compute->output_channel_ + c];
        }
      }
    }
  }
}
