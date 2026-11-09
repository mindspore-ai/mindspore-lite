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
#include "nnacl_c/int8_grad/matmul_grad_int8.h"

static int InputOffset(const MatMulParameter *param, int row_idx, int deep_idx) {
  return param->a_transpose_ ? deep_idx * param->row_ + row_idx : row_idx * param->deep_ + deep_idx;
}

static int WeightOffset(const MatMulParameter *param, int out_col, int deep_idx) {
  return param->b_transpose_ ? out_col * param->deep_ + deep_idx : deep_idx * param->col_ + out_col;
}

static int WeightGradTileOffset(const MatMulParameter *param, int local_col, int deep_idx, int tile_size) {
  return param->b_transpose_ ? local_col * param->deep_ + deep_idx : deep_idx * tile_size + local_col;
}

// The caller owns gradient buffer initialization. dx/dw_tile/db_tile are accumulated into,
// so callers must zero the needed ranges before starting a fresh gradient or tile.
void MatMulInt8GradOutColTile(const float *dy, const int8_t *x, const int8_t *weight, const MatMulParameter *param,
                              const float *effective_scale, int input_zp, int out_col_start, int tile_size, float *dx,
                              float *dw_tile, float *db_tile) {
  for (int row_idx = 0; row_idx < param->row_; ++row_idx) {
    for (int local_col = 0; local_col < tile_size; ++local_col) {
      int out_col = out_col_start + local_col;
      // pull grad back from the out_q domain to the int32 accumulator domain
      float dacc = dy[row_idx * param->col_ + out_col] * effective_scale[out_col];
      if (param->has_bias_) {
        db_tile[local_col] += dacc;
      }
      for (int deep_idx = 0; deep_idx < param->deep_; ++deep_idx) {
        int x_offset = InputOffset(param, row_idx, deep_idx);
        int w_offset = WeightOffset(param, out_col, deep_idx);
        int dw_tile_idx = WeightGradTileOffset(param, local_col, deep_idx, tile_size);
        int x_centered = (int)x[x_offset] - input_zp;
        dw_tile[dw_tile_idx] += dacc * (float)x_centered;
        dx[x_offset] += dacc * (float)weight[w_offset];  // weight is symmetric (zp=0), used raw
      }
    }
  }
}
