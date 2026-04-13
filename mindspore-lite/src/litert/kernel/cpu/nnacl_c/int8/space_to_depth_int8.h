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
#ifndef NNACL_INT8_SPACE_TO_DEPTH_INT8_H_
#define NNACL_INT8_SPACE_TO_DEPTH_INT8_H_

#include "nnacl_c/space_to_depth_parameter.h"
#include "nnacl_c/int8/quantize.h"
#include "nnacl_c/op_base.h"

#ifdef __cplusplus
extern "C" {
#endif
int SpaceToDepthForNHWCInt8(const int8_t *input, int8_t *output, const int32_t *in_shape, const int32_t *out_shape,
                            int shape_size, SpaceToDepthParameter *param, QuantArg *in_quant_arg,
                            QuantArg *out_quant_arg, int task_id);
#ifdef __cplusplus
}
#endif

#endif  // NNACL_INT8_SPACE_TO_DEPTH_INT8_H_
