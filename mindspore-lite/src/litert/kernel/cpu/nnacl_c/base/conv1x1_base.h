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

#ifndef NNACL_BASE_CONV1X1_BASE_H_
#define NNACL_BASE_CONV1X1_BASE_H_

#include "nnacl_c/conv_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif

void Conv1x1InputPack(const void *src_ptr, void *dst_ptr, ConvParameter *conv_param, int data_size);

/**
 * @brief Conv1x1 input packing function with batch support
 *
 * This function handles multiple batches by flattening the batch dimension
 * into the height dimension. Data layout: [Batch, H, W, C] -> [Batch*H, W, C]
 *
 * @param src_ptr Source data pointer [Batch, H, W, C]
 * @param dst_ptr Destination data pointer [Batch*H, W, C]
 * @param conv_param Convolution parameters
 * @param data_size Size of each data element (e.g., sizeof(int8_t))
 * @param input_batch Number of batches
 */
void Conv1x1InputPackBatch(const void *src_ptr, void *dst_ptr, ConvParameter *conv_param, int data_size,
                           int input_batch);

#ifdef __cplusplus
}
#endif

#endif  // NNACL_BASE_CONV1X1_BASE_H_
