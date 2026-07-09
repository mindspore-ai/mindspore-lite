/**
 * Copyright 2023-2026 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_LITERT_KERNEL_OPENCL_KERNEL_COMMON_ARITHMETIC_UTILS_H_
#define MINDSPORE_LITE_SRC_LITERT_KERNEL_OPENCL_KERNEL_COMMON_ARITHMETIC_UTILS_H_

#include <vector>
#include "src/tensor.h"
#include "nnacl_c/arithmetic_parameter.h"
#include "schema/model_generated.h"

namespace mindspore::kernel {
// Validate arithmetic operator specifications (common logic shared between FP32 and INT8 OpenCL arithmetic).
int ValidateArithmeticSpecs(const std::vector<lite::Tensor *> &in_tensors,
                            const std::vector<lite::Tensor *> &out_tensors, const void *op_parameter,
                            schema::PrimitiveType primitive_type);
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_LITERT_KERNEL_OPENCL_KERNEL_COMMON_ARITHMETIC_UTILS_H_
