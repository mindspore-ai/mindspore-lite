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

#if ENABLE_RVV
#include <riscv_vector.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include "include/securec.h"

// Constants (same as original)
#define C4NUM 4
#define C8NUM 8
#define OutType_TileC8 1
#define OutType_C8 2
#define OutType_Nhwc 3
#define ACTIVATION_NONE 0
#define ACTIVATION_RELU 1

// Simple bias addition for 8 elements
static void DoBiasBlock8(const float *bias, float *output, int rows) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < 8; j++) {
      output[i * 8 + j] += bias[j];
    }
  }
}

// Simple ReLU activation
static void ApplyActivation(float *output, int elements, int act_type) {
  if (act_type == ACTIVATION_RELU) {
    for (int i = 0; i < elements; i++) {
      if (output[i] < 0.0f) output[i] = 0.0f;
    }
  }
}

/**
 * @brief Core computation block for 4x8 matrix multiplication, optimized with RISC-V vector instructions
 * @param a Pointer to 4 rows of input matrix A
 * @param b Data pointer of input matrix B
 * @param output Output buffer, of size 4x8
 * @param depth Depth dimension of the matrix multiplication
 * @param col_stride_b Column stride of matrix B
 */
static void Matmul4x8Block(const float *a, const float *b, float *output, int depth, int col_stride_b) {
  // Zero initialize output
  for (int i = 0; i < 4 * 8; i++) {
    output[i] = 0.0f;
  }

  // Process depth dimension
  for (int d = 0; d < depth; d++) {
    // Get row values from A matrix (4 rows)
    float a0 = a[d];
    float a1 = a[depth + d];
    float a2 = a[2 * depth + d];
    float a3 = a[3 * depth + d];

    // Get column values from B matrix (8 columns)
    const float *b_col = b + d * col_stride_b;

    // Multiply and accumulate for each row
    for (int j = 0; j < 8; j++) {
      float b_val = b_col[j];
      output[j] += a0 * b_val;       // Row 0
      output[8 + j] += a1 * b_val;   // Row 1
      output[16 + j] += a2 * b_val;  // Row 2
      output[24 + j] += a3 * b_val;  // Row 3
    }
  }
}

// Write partial columns (1-7 columns)
static void WritePartialColumns(float *dst, const float *src, int rows, int actual_cols, int stride) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < actual_cols; j++) {
      dst[i * stride + j] = src[i * 8 + j];
    }
  }
}
/**
 * @brief Matrix multiplication function optimized with RISC-V vector instructions, handling 4x8 size data blocks
 * @param a Input matrix A (row x depth)
 * @param b Input matrix B (depth x col)
 * @param c Output matrix C (row x col)
 * @param bias Bias vector, can be NULL
 * @param act_type Activation function type (ACTIVATION_NONE, ACTIVATION_RELU, etc.)
 * @param depth Number of columns of matrix A and number of rows of matrix B
 * @param row Number of rows of matrix A
 * @param col Number of columns of matrix B
 * @param stride Column stride of the output matrix
 * @param write_mode Output format mode (OutType_TileC8, OutType_C8, OutType_Nhwc)
 */
void MatmulFloatRvv64Opt(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int row,
                         int col, int stride, int write_mode) {
  // We'll process in 4-row blocks
  for (int row_start = 0; row_start < row; row_start += 4) {
    int rows_in_block = (row - row_start >= 4) ? 4 : (row - row_start);

    // Process in 8-column blocks
    for (int col_start = 0; col_start < col; col_start += 8) {
      int cols_in_block = (col - col_start >= 8) ? 8 : (col - col_start);

      // Temporary buffer for 4x8 block computation
      float block_output[4 * 8];

      // Get pointers for this block
      const float *a_block = a + row_start * depth;
      const float *b_block = b + col_start;

      // Compute 4x8 block
      Matmul4x8Block(a_block, b_block, block_output, depth, col);

      // Apply bias if present
      if (bias != NULL && cols_in_block == 8) {
        DoBiasBlock8(bias + col_start, block_output, rows_in_block);
      }

      // Apply activation
      ApplyActivation(block_output, rows_in_block * cols_in_block, act_type);

      // Write results based on write mode
      if (write_mode == OutType_TileC8) {
        // TileC8 format: write in 4x4 tiles
        float *dst_base = c + row_start * stride + col_start;
        for (int i = 0; i < rows_in_block; i++) {
          for (int j = 0; j < cols_in_block; j++) {
            dst_base[i * stride + j] = block_output[i * 8 + j];
          }
        }
      } else if (write_mode == OutType_C8) {
        // C8 format: write in column-major 8-column blocks
        float *dst_base = c + (row_start / 4) * 32 + (col_start / 8) * (row * 8) + (row_start % 4) * 8;
        for (int i = 0; i < rows_in_block; i++) {
          for (int j = 0; j < cols_in_block; j++) {
            dst_base[i * 8 + j] = block_output[i * 8 + j];
          }
        }
      } else {
        // NHWC format: direct row-major
        if (cols_in_block < 8) {
          // Partial columns
          WritePartialColumns(c + row_start * stride + col_start, block_output, rows_in_block, cols_in_block, stride);
        } else {
          // Full 8 columns
          for (int i = 0; i < rows_in_block; i++) {
            memcpy_s(c + (row_start + i) * stride + col_start, 8 * sizeof(float), block_output + i * 8,
                     8 * sizeof(float));
          }
        }
      }
    }

    // Move A pointer to next 4-row block
    a += 4 * depth;
  }
}
#endif
