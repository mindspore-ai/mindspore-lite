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
#ifndef NNACL_INT8_GRAD_MATMUL_GRAD_INT8_H_
#define NNACL_INT8_GRAD_MATMUL_GRAD_INT8_H_

#include <stdint.h>
#include "nnacl_c/matmul_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif

// QAS (quantization-aware) matmul / FC backward. The weight is symmetric
// (zero-point 0, used raw); only the input x may be asymmetric, so only x is
// centered. Forward (per output column out_col, reduction index deep_idx):
//   x_centered = x_q[row_idx][deep_idx] - input_zp   // per-tensor input zero-point
//   acc = sum_i x_centered * weight_q[out_col][deep_idx] + bias[out_col]   // weight used raw
//   out_q = clamp(round(acc * effective_scale[out_col]) + zero_y)
//                  effective_scale = x_scale * weight_scale / out_scale
//
// `input_zp` is a per-tensor input zero-point.
//
// Gradient scale transforms (split across backward + optimizer):
//   1) backward: grad is pulled from the out_q domain back to the int32 accumulator
//      domain by multiplying dy by effective_scale (the only scale op here).
//   2) optimizer (sgd_int8): the integer-domain grad is rescaled by 1/weight_scale^2
//      (weight) / 1/(x_scale*weight_scale)^2 (bias) so an integer step with `lr` equals
//      a real-domain SGD step.
//
// `dx`, `dw_tile`, `db_tile` must be zeroed by the caller before starting a
// fresh gradient. dx accumulates across output-column tiles.
void MatMulInt8GradOutColTile(const float *dy, const int8_t *x, const int8_t *weight, const MatMulParameter *param,
                              const float *effective_scale, int input_zp, int out_col_start, int tile_size, float *dx,
                              float *dw_tile, float *db_tile);

#ifdef __cplusplus
}
#endif

#endif  // NNACL_INT8_GRAD_MATMUL_GRAD_INT8_H_
