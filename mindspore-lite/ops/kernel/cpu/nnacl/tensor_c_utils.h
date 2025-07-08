/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef NNACL_TENSORC_UTILS_H_
#define NNACL_TENSORC_UTILS_H_

#include <stddef.h>
#include "nnacl/errorcode.h"
#include "nnacl/op_base.h"
#include "nnacl/tensor_c.h"

#ifdef __cplusplus
extern "C" {
#endif

int NNACLGetBatch(const TensorC *tensor);
int NNACLGetHeight(const TensorC *tensor);
int NNACLGetWidth(const TensorC *tensor);
int NNACLGetChannel(const TensorC *tensor);
void NNACLSetBatch(TensorC *tensor, int batch);
void NNACLSetHeight(TensorC *tensor, int height);
void NNACLSetWidth(TensorC *tensor, int width);
void NNACLSetChannel(TensorC *tensor, int channel);
int NNACLGetElementNum(const TensorC *tensor);
int NNACLGetSize(const TensorC *tensor);
int NNACLGetDimensionSize(const TensorC *tensor, const size_t index);
bool NNACLIsShapeSame(const TensorC *tensor1, const TensorC *tensor2);
bool NNACLIsConst(const TensorC *tensor);

#ifdef __cplusplus
}
#endif

#endif  // NNACL_TENSORC_UTILS_H_
