/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_NNACL_C_OP_BASE_H_
#define MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_NNACL_C_OP_BASE_H_
#include "mindspore/mindspore/ops/kernel/cpu/nnacl/op_base.h"
#undef MAX_SHAPE_SIZE
#define MAX_SHAPE_SIZE 16
#endif  // MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_NNACL_C_OP_BASE_H_
