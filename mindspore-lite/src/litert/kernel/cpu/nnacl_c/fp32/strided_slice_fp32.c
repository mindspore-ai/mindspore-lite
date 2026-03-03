/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "nnacl_c/fp32/strided_slice_fp32.h"
#include <limits.h>
#include <stdio.h>
#include "nnacl_c/nnacl_common.h"
#include "nnacl_c/errorcode.h"
#include "include/securec.h"

// Safe multiplication check macro to prevent integer overflow
// Returns true if multiplication would overflow, false otherwise
#define CHECK_MUL_OVERFLOW(a, b) (((a) > 0 && (b) > 0 && ((a) > SIZE_MAX / (b))))

// Safe multiplication with overflow check
#define SAFE_MUL(result, a, b)           \
  do {                                   \
    if (CHECK_MUL_OVERFLOW(a, b)) {      \
      return NNACL_ERRCODE_MUL_OVERFLOW; \
    }                                    \
    *(result) = (a) * (b);               \
  } while (0)

// Safe left shift with overflow check
#define SAFE_LSHIFT(result, value, shift)                                            \
  do {                                                                               \
    if ((shift) >= sizeof(size_t) * CHAR_BIT || ((value) > (SIZE_MAX >> (shift)))) { \
      return NNACL_ERRCODE_MUL_OVERFLOW;                                             \
    }                                                                                \
    *(result) = (value) << (shift);                                                  \
  } while (0)

// Safe addition with overflow check for pointer arithmetic
#define SAFE_ADD(result, a, b)                 \
  do {                                         \
    if ((a) > SIZE_MAX - (b)) {                \
      return NNACL_ERRCODE_INDEX_OUT_OF_RANGE; \
    }                                          \
    *(result) = (a) + (b);                     \
  } while (0)

// Get log2 of element size (power of 2 bytes)
// Returns the exponent n such that 2^n = size, or 0 for unsupported sizes
static inline size_t GetLog2ElementSize(int data_type_size) {
  // Use a lookup table for correct log2 values
  // data_type_size must be a power of 2
  switch (data_type_size) {
    case 1:      // int8, uint8, bool, char
      return 0;  // 2^0 = 1
    case 2:      // int16, uint16, float16
      return 1;  // 2^1 = 2
    case 4:      // int32, uint32, float, float32
      return 2;  // 2^2 = 4
    case 8:      // int64, uint64, double, float64
      return 3;  // 2^3 = 8
    case 16:     // Complex128 (2 * double)
      return 4;  // 2^4 = 16
    default:
      // Unsupported data type size, return 0 to avoid shifting
      return 0;
  }
}

// Normalize the slicing parameters in StridedSliceStruct.
// Remove all slice dimensions of size 1 (squeeze) and merge consecutive full-dimensional slices to optimize subsequent
// slicing operations.
int NormalizedSlice(StridedSliceStruct *stride_slice) {
  // Validate input_shape_size to prevent:
  // 1. Negative value overflow when casting to size_t
  // 2. Out-of-bounds array access when exceeding MAX_SHAPE_SIZE
  if (stride_slice->in_shape_size_ < 0) {
    return NNACL_STRIDED_SLICE_INVALID_SHAPE_SIZE;
  }
  if (stride_slice->in_shape_size_ > MAX_SHAPE_SIZE) {
    return NNACL_STRIDED_SLICE_INVALID_SHAPE_SIZE;
  }
  size_t input_shape_size = (size_t)(stride_slice->in_shape_size_);
  for (size_t i = 0; i < DIMENSION_8D; ++i) {
    stride_slice->normalized_begins[i] = 0;
    stride_slice->normalized_input_shape[i] = 1;
    stride_slice->normalized_output_shape[i] = 1;
  }
  // Remove all slices of size 1 (process in reverse order starting from the lowest dimension).
  size_t num_size_one = 0;
  for (size_t i = 0; i < input_shape_size; i++) {
    const size_t begin =
      (size_t)(stride_slice->begins_[input_shape_size - 1 - i]);  // Start index of the current dimension
    const size_t input_dim = (size_t)(stride_slice->in_shape_[input_shape_size - 1 - i]);
    const size_t size =
      (size_t)(stride_slice->ends_[input_shape_size - 1 - i] - stride_slice->begins_[input_shape_size - 1 - i]);
    // If the output size is 1 and it is not the first dimension, then it is squeezed into a higher dimension.
    if (size == 1 && i != 0) {
      stride_slice->normalized_begins[DIMENSION_8D - i + num_size_one] +=
        begin *
        stride_slice
          ->normalized_input_shape[DIMENSION_8D - i + num_size_one];  // Merge Offset: The current offset is accumulated
                                                                      // into the offset of a higher dimension.
      stride_slice->normalized_input_shape[DIMENSION_8D - i + num_size_one] *= input_dim;
      stride_slice->normalized_output_shape[DIMENSION_8D - i + num_size_one] *= size;
      num_size_one++;
    } else {  // No need to merge, just assign directly.
      stride_slice->normalized_begins[DIMENSION_8D - 1 - i + num_size_one] = begin;
      stride_slice->normalized_input_shape[DIMENSION_8D - 1 - i + num_size_one] = input_dim;
      stride_slice->normalized_output_shape[DIMENSION_8D - 1 - i + num_size_one] = size;
    }
  }
  // Merge consecutive full-dimensional slices
  size_t new_num_dims = input_shape_size - num_size_one;
  size_t output_dims = new_num_dims;
  bool merge_previous_dim =
    false;  // Indicates whether the current dimension needs to be merged into the previous dimension.
  size_t num_sliced_dims = 0;  // Record the number of dimensions for non-full slices
  for (size_t i = 0; i < new_num_dims; i++) {
    const size_t begin = stride_slice->normalized_begins[DIMENSION_8D - 1 - i];
    const size_t size = stride_slice->normalized_output_shape[DIMENSION_8D - 1 - i];
    const size_t input_dim = stride_slice->normalized_input_shape[DIMENSION_8D - 1 - i];
    const bool merge_current_dim = (begin == 0 && size == input_dim);
    if (merge_previous_dim) {
      // If the previous dimension has been marked for merging, merge the current dimension into it.
      stride_slice->normalized_begins[DIMENSION_8D - 1 - num_sliced_dims] =
        begin * stride_slice->normalized_input_shape[DIMENSION_8D - 1 - num_sliced_dims];
      stride_slice->normalized_input_shape[DIMENSION_8D - 1 - num_sliced_dims] *= input_dim;
      stride_slice->normalized_output_shape[DIMENSION_8D - 1 - num_sliced_dims] *= size;
      output_dims -= 1;
      if (!merge_current_dim) {
        num_sliced_dims += 1;
      }
    } else {
      // Do not merge
      stride_slice->normalized_begins[DIMENSION_8D - 1 - num_sliced_dims] = begin;
      stride_slice->normalized_input_shape[DIMENSION_8D - 1 - num_sliced_dims] = input_dim;
      stride_slice->normalized_output_shape[DIMENSION_8D - 1 - num_sliced_dims] = size;
      if (!merge_current_dim) {
        num_sliced_dims += 1;
      }
    }
    merge_previous_dim = merge_current_dim;
  }
  for (size_t i = 0; i < DIMENSION_8D - output_dims; i++) {
    stride_slice->normalized_begins[i] = 0;
    stride_slice->normalized_input_shape[i] = 1;
    stride_slice->normalized_output_shape[i] = 1;
  }
  stride_slice->num_normalized_dims = output_dims;
  return NNACL_OK;
}

// Initialize slice context including offsets and strides calculation
static int InitSliceContext(void *input_data, void *output_data, StridedSliceStruct *s, SliceContext *ctx) {
  int data_type_size = (int)DataTypeCSize(s->data_type_);
  size_t log2_element_size = GetLog2ElementSize(data_type_size);
  ctx->num_normalized_dims = s->num_normalized_dims;

  // Calculate offsets
  for (size_t i = 0; i < DIMENSION_8D; i++) {
    ctx->offsets[i] = s->normalized_begins[DIMENSION_8D - 1 - i];
  }
  // Safe left shift to prevent overflow
  size_t offset_0_shifted;
  SAFE_LSHIFT(&offset_0_shifted, ctx->offsets[0], log2_element_size);
  ctx->offsets[0] = offset_0_shifted;

  // Calculate strides
  size_t input_stride = s->normalized_input_shape[DIMENSION_8D - 1];
  size_t output_stride = s->normalized_output_shape[DIMENSION_8D - 1];
  for (size_t i = 1; i < DIMENSION_8D; i++) {
    // Safe left shift for stride calculation
    size_t input_stride_shifted;
    size_t output_stride_shifted;
    SAFE_LSHIFT(&input_stride_shifted, input_stride, log2_element_size);
    SAFE_LSHIFT(&output_stride_shifted, output_stride, log2_element_size);
    ctx->input_stride[i - 1] = input_stride_shifted;
    ctx->output_stride[i - 1] = output_stride_shifted;

    // Safe multiplication to prevent overflow when accumulating strides
    size_t new_input_stride;
    size_t new_output_stride;
    SAFE_MUL(&new_input_stride, input_stride, s->normalized_input_shape[DIMENSION_8D - 1 - i]);
    SAFE_MUL(&new_output_stride, output_stride, s->normalized_output_shape[DIMENSION_8D - 1 - i]);
    input_stride = new_input_stride;
    output_stride = new_output_stride;
  }
  // Safe left shift for contiguous_size
  SAFE_LSHIFT(&ctx->contiguous_size, s->normalized_output_shape[DIMENSION_8D - 1], log2_element_size);

  // Move the pointer to the start point of the slice
  ctx->input = (uint8_t *)input_data + ctx->offsets[0];
  ctx->output = (uint8_t *)output_data;
  for (size_t i = 1; i < ctx->num_normalized_dims; ++i) {
    // Safe multiplication and addition for pointer arithmetic
    size_t offset;
    SAFE_MUL(&offset, ctx->offsets[i], ctx->input_stride[i - 1]);
    ctx->input = ctx->input + offset;
  }
  return NNACL_OK;
}

// Copy data for 1 or 2 dimensional slices
static int CopySlice2D(const SliceContext *ctx, const StridedSliceStruct *s) {
  for (size_t i = 0; i < s->normalized_output_shape[DIMENSION_8D - 2]; ++i) {
    // Safe multiplication for pointer offset
    size_t input_offset, output_offset;
    SAFE_MUL(&input_offset, i, ctx->input_stride[0]);
    SAFE_MUL(&output_offset, i, ctx->output_stride[0]);
    const uint8_t *src = ctx->input + input_offset;
    uint8_t *dst = ctx->output + output_offset;
    memcpy(dst, src, ctx->contiguous_size);
  }
  return NNACL_OK;
}

// Copy data for 3 dimensional slices
static int CopySlice3D(const SliceContext *ctx, const StridedSliceStruct *s) {
  for (size_t i = 0; i < s->normalized_output_shape[DIMENSION_8D - 3]; ++i) {
    for (size_t j = 0; j < s->normalized_output_shape[DIMENSION_8D - 2]; ++j) {
      // Safe multiplication for pointer offset
      size_t input_offset_i, input_offset_j, input_offset_total;
      size_t output_offset_i, output_offset_j, output_offset_total;
      SAFE_MUL(&input_offset_i, i, ctx->input_stride[1]);
      SAFE_MUL(&input_offset_j, j, ctx->input_stride[0]);
      SAFE_ADD(&input_offset_total, input_offset_i, input_offset_j);
      SAFE_MUL(&output_offset_i, i, ctx->output_stride[1]);
      SAFE_MUL(&output_offset_j, j, ctx->output_stride[0]);
      SAFE_ADD(&output_offset_total, output_offset_i, output_offset_j);
      const void *src = ctx->input + input_offset_total;
      void *dst = ctx->output + output_offset_total;
      memcpy(dst, src, ctx->contiguous_size);
    }
  }
  return NNACL_OK;
}

// Copy data for 4 dimensional slices
static int CopySlice4D(const SliceContext *ctx, const StridedSliceStruct *s) {
  for (size_t i = 0; i < s->normalized_output_shape[DIMENSION_8D - 4]; ++i) {
    for (size_t j = 0; j < s->normalized_output_shape[DIMENSION_8D - 3]; ++j) {
      for (size_t l = 0; l < s->normalized_output_shape[DIMENSION_8D - 2]; ++l) {
        // Safe multiplication for pointer offset
        size_t input_offset_i, input_offset_j, input_offset_l, input_offset_total;
        size_t output_offset_i, output_offset_j, output_offset_l, output_offset_total;
        SAFE_MUL(&input_offset_i, i, ctx->input_stride[2]);
        SAFE_MUL(&input_offset_j, j, ctx->input_stride[1]);
        SAFE_ADD(&input_offset_total, input_offset_i, input_offset_j);
        SAFE_MUL(&input_offset_l, l, ctx->input_stride[0]);
        SAFE_ADD(&input_offset_total, input_offset_total, input_offset_l);
        SAFE_MUL(&output_offset_i, i, ctx->output_stride[2]);
        SAFE_MUL(&output_offset_j, j, ctx->output_stride[1]);
        SAFE_ADD(&output_offset_total, output_offset_i, output_offset_j);
        SAFE_MUL(&output_offset_l, l, ctx->output_stride[0]);
        SAFE_ADD(&output_offset_total, output_offset_total, output_offset_l);
        const void *src = ctx->input + input_offset_total;
        void *dst = ctx->output + output_offset_total;
        memcpy(dst, src, ctx->contiguous_size);
      }
    }
  }
  return NNACL_OK;
}

// Build the context for StridedSlice, including normalizing slice parameters, calculating offsets and strides, and
// performing data copying. float data type size being 2^log2_element_size = 4.
int DoStrideSliceCopyOpt(void *input_data, void *output_data, StridedSliceStruct *s) {
  int ret = NormalizedSlice(s);
  if (ret != NNACL_OK) {
    return ret;
  }

  SliceContext ctx;
  ret = InitSliceContext(input_data, output_data, s, &ctx);
  if (ret != NNACL_OK) {
    return ret;
  }

  switch (ctx.num_normalized_dims) {
    // 1/2 dimension needs to be copied
    case 1:
    case 2:
      return CopySlice2D(&ctx, s);
    case 3:
      return CopySlice3D(&ctx, s);
    case 4:
      return CopySlice4D(&ctx, s);
    default:
      break;
  }
  return NNACL_OK;
}

int PadStridedSliceParameterTo8D(StridedSliceStruct *strided_slice) {
  if (strided_slice->in_shape_size_ > DIMENSION_8D) {
    return NNACL_STRIDED_SLICE_UNSUPPORTED_MAX_8D;
  }

  int32_t begins[DIMENSION_8D];
  int32_t ends[DIMENSION_8D];
  int32_t strides[DIMENSION_8D];
  int32_t input_shape[DIMENSION_8D];
  int32_t i;
  for (i = 0; i < strided_slice->in_shape_size_; ++i) {
    begins[i] = strided_slice->begins_[i];
    ends[i] = MSMIN(strided_slice->ends_[i], strided_slice->in_shape_[i]);
    strides[i] = strided_slice->strides_[i];
    input_shape[i] = strided_slice->in_shape_[i];
  }

  int32_t real_index = strided_slice->in_shape_size_ - 1;
  for (i = DIMENSION_8D - 1; i >= 0; --i) {
    if (real_index >= 0) {
      strided_slice->begins_[i] = begins[real_index];
      strided_slice->ends_[i] = ends[real_index];
      strided_slice->strides_[i] = strides[real_index];
      strided_slice->in_shape_[i] = input_shape[real_index--];
    } else {
      strided_slice->begins_[i] = 0;
      strided_slice->ends_[i] = 1;
      strided_slice->strides_[i] = 1;
      strided_slice->in_shape_[i] = 1;
    }
  }
  strided_slice->in_shape_size_ = DIMENSION_8D;
  return NNACL_OK;
}

bool LoopContinue(int stride, int i, int end) { return stride > 0 ? i < end : i > end; }

int DoStridedSliceIn8D(const void *input, void *output, StridedSliceStruct *strided_slice) {
  NNACL_CHECK_NULL_RETURN_ERR(strided_slice);
  NNACL_CHECK_NULL_RETURN_ERR(input);
  NNACL_CHECK_NULL_RETURN_ERR(output);

  const uint8_t *in = (const uint8_t *)input;
  uint8_t *out = (uint8_t *)output;
  int data_type_size = (int)DataTypeCSize(strided_slice->data_type_);

  int32_t *begins = strided_slice->begins_;
  int32_t *ends = strided_slice->ends_;
  int32_t *strides = strided_slice->strides_;
  int32_t *in_shape = strided_slice->in_shape_;

  int dim_offset[DIMENSION_8D - 1];
  dim_offset[6] = in_shape[7];
  dim_offset[5] = in_shape[6] * dim_offset[6];
  dim_offset[4] = in_shape[5] * dim_offset[5];
  dim_offset[3] = in_shape[4] * dim_offset[4];
  dim_offset[2] = in_shape[3] * dim_offset[3];
  dim_offset[1] = in_shape[2] * dim_offset[2];
  dim_offset[0] = in_shape[1] * dim_offset[1];
  size_t out_offset = 0;
  int32_t dim0, dim1, dim2, dim3, dim4, dim5, dim6, dim7;
  for (dim0 = begins[0]; LoopContinue(strides[0], dim0, ends[0]); dim0 += strides[0]) {
    for (dim1 = begins[1]; LoopContinue(strides[1], dim1, ends[1]); dim1 += strides[1]) {
      for (dim2 = begins[2]; LoopContinue(strides[2], dim2, ends[2]); dim2 += strides[2]) {
        for (dim3 = begins[3]; LoopContinue(strides[3], dim3, ends[3]); dim3 += strides[3]) {
          for (dim4 = begins[4]; LoopContinue(strides[4], dim4, ends[4]); dim4 += strides[4]) {
            for (dim5 = begins[5]; LoopContinue(strides[5], dim5, ends[5]); dim5 += strides[5]) {
              for (dim6 = begins[6]; LoopContinue(strides[6], dim6, ends[6]); dim6 += strides[6]) {
                for (dim7 = begins[7]; LoopContinue(strides[7], dim7, ends[7]); dim7 += strides[7]) {
                  int32_t in_offset = dim0 * dim_offset[0] + dim1 * dim_offset[1] + dim2 * dim_offset[2] +
                                      dim3 * dim_offset[3] + dim4 * dim_offset[4] + dim5 * dim_offset[5] +
                                      dim6 * dim_offset[6] + dim7;
                  memcpy(out + out_offset * data_type_size, in + in_offset * data_type_size, data_type_size);
                  out_offset++;
                }
              }
            }
          }
        }
      }
    }
  }
  return NNACL_OK;
}

void FastStride(const uint8_t *input, uint8_t *output, int split_len, int stride, size_t outer, size_t inner_size,
                size_t in_offset) {
  if (stride == 1) {
    size_t unit = split_len * inner_size;
    for (size_t i = 0; i < outer; ++i) {
      memcpy(output, input, unit);
      output += unit;
      input += in_offset;
    }
    return;
  }
  for (size_t i = 0; i < outer; ++i) {
    const uint8_t *input_ptr = input + i * in_offset;
    for (int j = 0; j < split_len; ++j) {
      memcpy(output, input_ptr, inner_size);
      output += inner_size;
      input_ptr += inner_size * stride;
    }
  }
}
