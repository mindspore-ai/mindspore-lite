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

#ifndef MINDSPORE_OPS_KERNEL_CPU_NNACL_INT8_TRILU_INT8_H_
#define MINDSPORE_OPS_KERNEL_CPU_NNACL_INT8_TRILU_INT8_H_

#include <stdint.h>
#include "nnacl_c/op_base.h"

#ifdef __cplusplus
extern "C" {
#endif

// fp32 Triu/Tril uses the upstream TriuByte4/TrilByte4 from triu_tril_fp32.c.
// This file keeps only the int8 variant: upstream TriuByte1/TrilByte1 write literal 0 for
// masked elements, which dequantizes to -zp*scale != 0 when the output zero point != 0 and
// collapses accuracy. The int8 variant requantizes kept elements (input -> output) and
// writes out_zp for masked elements.

int TrilInt8(const int8_t *input, int height, int width, int diagonal, int8_t *output, int num, float in_scale,
             int in_zp, float out_scale, int out_zp);

int TriuInt8(const int8_t *input, int height, int width, int diagonal, int8_t *output, int num, float in_scale,
             int in_zp, float out_scale, int out_zp);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_OPS_KERNEL_CPU_NNACL_INT8_TRILU_INT8_H_
