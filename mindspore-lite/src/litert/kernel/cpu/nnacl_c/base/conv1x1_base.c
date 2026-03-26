/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "nnacl_c/base/conv1x1_base.h"

void Conv1x1InputPack(const void *src_ptr, void *dst_ptr, ConvParameter *conv_param, int data_size) {
  /* support nhwc */
  char *src = (char *)src_ptr;
  char *dst = (char *)dst_ptr;
  for (int dst_h = 0; dst_h < conv_param->output_h_; dst_h++) {
    int src_h = dst_h * conv_param->stride_h_ - conv_param->pad_u_;
    if (src_h < 0 || src_h >= conv_param->input_h_) {
      continue;
    }
    const char *src_h_ptr = src + src_h * conv_param->input_w_ * conv_param->input_channel_ * data_size;
    char *dst_h_ptr = dst + dst_h * conv_param->output_w_ * conv_param->input_channel_ * data_size;
    for (int dst_w = 0; dst_w < conv_param->output_w_; dst_w++) {
      int src_w = dst_w * conv_param->stride_w_ - conv_param->pad_l_;
      if (src_w < 0 || src_w >= conv_param->input_w_) {
        continue;
      }
      memcpy(dst_h_ptr + dst_w * conv_param->input_channel_ * data_size,
             src_h_ptr + src_w * conv_param->input_channel_ * data_size, conv_param->input_channel_ * data_size);
    }
  }
  return;
}

/**
 * Conv1x1 input packing function with batch support
 *
 * This function handles multiple batches by flattening the batch dimension
 * into the height dimension. Data layout: [Batch, H, W, C] -> [Batch*H, W, C]
 *
 * @param src_ptr Source data pointer [Batch, H, W, C]
 * @param dst_ptr Destination data pointer [Batch*H, W, C]
 * @param conv_param Convolution parameters
 * @param data_size Size of each data element
 * @param input_batch Number of batches
 */
void Conv1x1InputPackBatch(const void *src_ptr, void *dst_ptr, ConvParameter *conv_param, int data_size,
                           int input_batch) {
  char *src = (char *)src_ptr;
  char *dst = (char *)dst_ptr;

  // Calculate the total output height after flattening batch dimension
  const int total_output_h = conv_param->output_h_ * input_batch;

  // Process each "virtual" height (which includes batch dimension)
  for (int dst_h = 0; dst_h < total_output_h; dst_h++) {
    // Extract batch index and height index
    const int batch_index = dst_h / conv_param->output_h_;
    const int src_h_in_batch = (dst_h % conv_param->output_h_) * conv_param->stride_h_ - conv_param->pad_u_;

    // Check if source height is valid
    if (src_h_in_batch < 0 || src_h_in_batch >= conv_param->input_h_) {
      continue;
    }

    // Calculate source pointer: need to account for batch offset
    // Layout: [Batch, H, W, C] -> offset = batch * H * W * C + h * W * C
    const char *src_h_ptr =
      src + batch_index * conv_param->input_h_ * conv_param->input_w_ * conv_param->input_channel_ * data_size +
      src_h_in_batch * conv_param->input_w_ * conv_param->input_channel_ * data_size;

    // Calculate destination pointer: layout is [Batch*H, W, C]
    char *dst_h_ptr = dst + dst_h * conv_param->output_w_ * conv_param->input_channel_ * data_size;

    // Process width dimension (same as original)
    for (int dst_w = 0; dst_w < conv_param->output_w_; dst_w++) {
      int src_w = dst_w * conv_param->stride_w_ - conv_param->pad_l_;
      if (src_w < 0 || src_w >= conv_param->input_w_) {
        continue;
      }
      memcpy(dst_h_ptr + dst_w * conv_param->input_channel_ * data_size,
             src_h_ptr + src_w * conv_param->input_channel_ * data_size, conv_param->input_channel_ * data_size);
    }
  }
}
