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

#ifndef NNACL_INT8_CROP_INT8_H_
#define NNACL_INT8_CROP_INT8_H_
#include "nnacl/op_base.h"
#include "nnacl/crop_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif
void Int8Crop(const int8_t *input, int8_t *output, int *input_shape, int *output_shape, int64_t *in_offset,
              int input_dim, int task_id, int thread_count, const CropQuantArg *quant);
#ifdef __cplusplus
}
#endif

#endif  // NNACL_INT8_CROP_INT8_H_
