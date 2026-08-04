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

#ifndef NNACL_INT8_GELU_INT8_H_
#define NNACL_INT8_GELU_INT8_H_

#include "nnacl_c/op_base.h"
#include "nnacl_c/int8/quantize.h"

#ifdef __cplusplus
extern "C" {
#endif

int GeluInt8InitLUT(const GeluQuantArg *quant_gelu_param, int8_t *table, bool approximate);
int GeluInt8(const int8_t *src, int length, int8_t *dst, const int8_t *table);

#ifdef __cplusplus
}
#endif

#endif  // NNACL_INT8_GELU_INT8_H_
