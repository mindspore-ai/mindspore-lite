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

#include <riscv_vector.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nnacl_c/pack.h"
#include "nnacl_c/op_base.h"
#include "nnacl_c/common_func.h"
#include "nnacl_c/conv_parameter.h"

const int K_ROW_BLOCK = 12;
const int K_COL_BLOCK = 4;
const float K_RELU6_MAX = 6.0f;
const float K_ZERO = 0.0f;

/**
 * @brief RVV optimized implementation of Adder12x4.
 * @param a Input tensor A (blocked NC12).
 * @param b Input tensor B (blocked NC4).
 * @param dst Output tensor.
 * @param bias Bias vector.
 * @param act_type Activation type.
 * @param deep Depth dimension.
 * @param row Number of rows.
 * @param col Number of columns.
 * @param stride Stride for dst.
 */
void AdderFloatRVV(const float *a, const float *b, float *dst, const float *bias, ActType act_type, int deep, int row,
                   int col, int stride) {
  for (int r = 0; r < row; ++r) {
    int r_block = r / K_ROW_BLOCK;
    int r_offset = r % K_ROW_BLOCK;
    const float *a_row_base = a + (r_block * deep * K_ROW_BLOCK) + r_offset;

    // Iterate over columns with vector length management
    for (int c = 0; c < col;) {
      // Set vector length based on remaining columns
      size_t vl = __riscv_vsetvl_e32m1(col - c);
      int c_end = c + (int)vl;

      // Initialize accumulator to 0.0
      vfloat32m1_t v_acc = __riscv_vfmv_v_f_f32m1(K_ZERO, vl);

      // Pre-calculate column block indices and base offset for 'b'
      int c_block = c / K_COL_BLOCK;
      int c_offset = c % K_COL_BLOCK;
      const float *b_col_base = b + (c_block * deep * K_COL_BLOCK) + c_offset;

      // Depth-wise accumulation
      for (int d = 0; d < deep; ++d) {
        // Load 'a': Since 'a' index does not depend on vector lane 'i'
        float a_val = a_row_base[d * K_ROW_BLOCK];
        vfloat32m1_t v_a = __riscv_vfmv_v_f_f32m1(a_val, vl);

        // Load 'b': Load contiguous vector from memory
        vfloat32m1_t v_b = __riscv_vle32_v_f32m1(b_col_base + d * K_COL_BLOCK, vl);

        // Compute |a - b|
        vfloat32m1_t v_diff = __riscv_vfsub_vv_f32m1(v_a, v_b, vl);
        vfloat32m1_t v_abs = __riscv_vfabs_v_f32m1(v_diff, vl);

        // Accumulate sum of absolute differences
        v_acc = __riscv_vfadd_vv_f32m1(v_acc, v_abs, vl);
      }

      // AdderNet Logic: Output = -Sum(|a - b|) + Bias
      v_acc = __riscv_vfneg_v_f32m1(v_acc, vl);

      // Add Bias if provided
      if (bias != NULL) {
        vfloat32m1_t v_bias = __riscv_vle32_v_f32m1(bias + c, vl);
        v_acc = __riscv_vfadd_vv_f32m1(v_acc, v_bias, vl);
      }

      // Apply Activation Functions
      if (act_type == ActType_Relu6) {
        // Clip upper bound at 6.0
        v_acc = __riscv_vfmin_vf_f32m1(v_acc, K_RELU6_MAX, vl);
      }

      // Apply ReLU (max(0, x)) for both Relu and Relu6
      if (act_type != ActType_No) {
        v_acc = __riscv_vfmax_vf_f32m1(v_acc, K_ZERO, vl);
      }

      // Store result to destination
      __riscv_vse32_v_f32m1(dst + r * stride + c, v_acc, vl);

      // Move to next column block
      c = c_end;
    }
  }
}
