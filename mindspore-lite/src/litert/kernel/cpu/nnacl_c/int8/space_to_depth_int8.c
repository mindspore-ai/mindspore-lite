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
#include "nnacl_c/int8/space_to_depth_int8.h"
#include "nnacl_c/common_func.h"
#include "nnacl_c/errorcode.h"

int SpaceToDepthForNHWCInt8(const int8_t *input, int8_t *output, const int32_t *in_shape, const int32_t *out_shape,
                            int shape_size, SpaceToDepthParameter *param, QuantArg *in_quant_arg,
                            QuantArg *out_quant_arg, int task_id) {
  if (param->op_parameter_.thread_num_ == 0) {
    return NNACL_ERR;
  }
  int output_h = out_shape[kNHWC_H];
  int unit_per_thread = UP_DIV(output_h, param->op_parameter_.thread_num_);
  int h_start = unit_per_thread * task_id;
  int h_end = MSMIN(h_start + unit_per_thread, output_h);

  int block_size = param->block_size_;
  int in_strides[C4NUM];
  int out_strides[C4NUM];
  ComputeStrides(in_shape, in_strides, shape_size);
  ComputeStrides(out_shape, out_strides, shape_size);

  const float output_inverse_scale = 1.f / out_quant_arg->scale_;
  float scale = in_quant_arg->scale_ * output_inverse_scale;
  float bias = -in_quant_arg->zp_ * scale;
  int32_t output_zp = out_quant_arg->zp_;
  int copy_size = block_size * in_strides[DIMENSION_2D];

  for (int i = 0; i < out_shape[0]; ++i) {
    int64_t in_offset_n = i * in_strides[0];
    int64_t out_offset_n = i * out_strides[0];
    for (int j = h_start; j < h_end; ++j) {
      int64_t in_offset_h = in_offset_n + j * block_size * in_strides[1];
      int64_t out_offset_h = out_offset_n + j * out_strides[1];
      for (int k = 0; k < out_shape[2]; ++k) {
        int64_t in_offset_w = in_offset_h + k * block_size * in_strides[2];
        int64_t out_offset_w = out_offset_h + k * out_strides[2];
        for (int l = 0; l < block_size; ++l) {
          int64_t out_base = out_offset_w + l * block_size * in_strides[DIMENSION_2D];
          int64_t in_base = in_offset_w + l * in_strides[DIMENSION_1D];
          for (int m = 0; m < copy_size; ++m) {
            int32_t output_tmp = round(input[in_base + m] * scale + bias) + output_zp;
            output_tmp = output_tmp > 127 ? 127 : output_tmp;
            output_tmp = output_tmp < -128 ? -128 : output_tmp;
            output[out_base + m] = (int8_t)output_tmp;
          }
        }
      }
    }
  }
  return NNACL_OK;
}
