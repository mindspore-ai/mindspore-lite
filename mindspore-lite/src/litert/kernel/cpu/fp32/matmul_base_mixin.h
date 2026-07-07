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

#ifndef MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_FP32_MATMUL_BASE_MIXIN_H_
#define MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_FP32_MATMUL_BASE_MIXIN_H_

#include <vector>
#include "src/litert/kernel/cpu/fp32/matmul_fp32_base.h"

namespace mindspore::kernel {
namespace matmul_base_utils {

/**
 * @brief Helper utilities for kernels that delegate to MatmulFp32BaseCPUKernel.
 *
 * These functions provide the common pattern where operations are performed
 * on both the outer kernel (via in_tensors_/out_tensors_) and the internal
 * matmul_base_ member.
 *
 * Usage: Call these functions from your kernel's overridden methods.
 */

inline void SetInTensors(LiteKernel *kernel, MatmulFp32BaseCPUKernel *matmul_base,
                         const std::vector<lite::Tensor *> &in_tensors) {
  kernel->set_in_tensors(in_tensors);
  if (matmul_base != nullptr) {
    matmul_base->set_in_tensors(in_tensors);
  }
}

inline void SetInTensor(LiteKernel *kernel, MatmulFp32BaseCPUKernel *matmul_base, lite::Tensor *in_tensor,
                        size_t index) {
  kernel->set_in_tensor(in_tensor, index);
  if (matmul_base != nullptr) {
    matmul_base->set_in_tensor(in_tensor, index);
  }
}

inline void SetOutTensors(LiteKernel *kernel, MatmulFp32BaseCPUKernel *matmul_base,
                          const std::vector<lite::Tensor *> &out_tensors) {
  kernel->set_out_tensors(out_tensors);
  if (matmul_base != nullptr) {
    matmul_base->set_out_tensors(out_tensors);
  }
}

inline void SetOutTensor(LiteKernel *kernel, MatmulFp32BaseCPUKernel *matmul_base, lite::Tensor *out_tensor,
                         size_t index) {
  kernel->set_out_tensor(out_tensor, index);
  if (matmul_base != nullptr) {
    matmul_base->set_out_tensor(out_tensor, index);
  }
}

// Train API helpers
inline int Train(LiteKernel *kernel, MatmulFp32BaseCPUKernel *matmul_base) {
  if (kernel == nullptr) {
    return matmul_base != nullptr ? matmul_base->Train() : lite::RET_OK;
  }
  int ret = kernel->LiteKernel::Train();
  if (ret != lite::RET_OK) {
    return ret;
  }
  return matmul_base != nullptr ? matmul_base->Train() : lite::RET_OK;
}

inline void SetTrainable(LiteKernel *kernel, MatmulFp32BaseCPUKernel *matmul_base, bool trainable) {
  if (kernel != nullptr) {
    kernel->LiteKernel::SetTrainable(trainable);
  }
  if (matmul_base != nullptr) {
    matmul_base->SetTrainable(trainable);
  }
}

inline size_t WorkspaceSize(LiteKernel *kernel, MatmulFp32BaseCPUKernel *matmul_base) {
  if (kernel != nullptr) {
    size_t kernel_size = kernel->LiteKernel::workspace_size();
    if (matmul_base != nullptr) {
      size_t base_size = matmul_base->workspace_size();
      return kernel_size > base_size ? kernel_size : base_size;
    }
    return kernel_size;
  }
  return matmul_base != nullptr ? matmul_base->workspace_size() : 0;
}

}  // namespace matmul_base_utils
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_FP32_MATMUL_BASE_MIXIN_H_
