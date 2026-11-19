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
#include "nnacl_c/int8_grad/convolution_grad_int8.h"

// This backward path expects conv weights and writes dw in the same OIHW layout:
//   [output_channel][input_channel][kernel_h][kernel_w].
// The caller owns gradient buffer initialization. dx/dw/db are accumulated into,
// so callers must zero them before this function when starting a fresh gradient.
void Conv2dInt8Grad(const float *dy, const int8_t *input, const int8_t *weight, const ConvParameter *param,
                    const float *effective_scale, int input_zp, float *dx, float *dw, float *db) {
  for (int oh = 0; oh < param->output_h_; ++oh) {
    for (int ow = 0; ow < param->output_w_; ++ow) {
      for (int oc = 0; oc < param->output_channel_; ++oc) {
        float dlinear_out = dy[(oh * param->output_w_ + ow) * param->output_channel_ + oc] * effective_scale[oc];
        db[oc] += dlinear_out;
        for (int kh = 0; kh < param->kernel_h_; ++kh) {
          int ih = oh * param->stride_h_ + kh * param->dilation_h_ - param->pad_u_;
          if (ih < 0 || ih >= param->input_h_) {
            continue;
          }
          for (int kw = 0; kw < param->kernel_w_; ++kw) {
            int iw = ow * param->stride_w_ + kw * param->dilation_w_ - param->pad_l_;
            if (iw < 0 || iw >= param->input_w_) {
              continue;
            }
            for (int ic = 0; ic < param->input_channel_; ++ic) {
              int input_index = (ih * param->input_w_ + iw) * param->input_channel_ + ic;
              int weight_index = ((oc * param->input_channel_ + ic) * param->kernel_h_ + kh) * param->kernel_w_ + kw;
              int input_centered = (int)input[input_index] - input_zp;
              int weight_centered = (int)weight[weight_index];
              dw[weight_index] += dlinear_out * (float)input_centered;
              dx[input_index] += dlinear_out * (float)weight_centered;
            }
          }
        }
      }
    }
  }
}
