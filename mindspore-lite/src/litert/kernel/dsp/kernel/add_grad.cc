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

#include "src/litert/kernel/dsp/kernel/add_grad.h"
#include <vector>
#include "src/litert/kernel_registry.h"
#include "armc/include/operator.h"

using mindspore::kernel::KERNEL_ARCH::kDSP;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_AddGrad;

namespace mindspore::kernel {

namespace {
constexpr int kDyIdx = 0;
constexpr int kX1Idx = 1;
constexpr int kX2Idx = 2;
constexpr int kDx1Idx = 0;
constexpr int kDx2Idx = 1;
constexpr int kOutputSize = 2;
constexpr size_t kMaxDims = 8;
}  // namespace

int AddGradDSPKernel::Prepare() { return RET_OK; }

int AddGradDSPKernel::CheckSpecs() {
  // Lite's AddGrad takes three inputs (dy, x1, x2): the CPU reference kernel
  // (arithmetic_grad.cc) uses the same contract, and a graph built by the converter
  // always carries all three. Requiring them here keeps the DSP kernel from being
  // created for an interface it cannot serve.
  if (in_tensors_.size() != INPUT_TENSOR_SIZE_3) {
    MS_LOG(ERROR) << "AddGrad expects 3 inputs (dy, x1, x2), got " << in_tensors_.size();
    return RET_ERROR;
  }
  if (out_tensors_.size() != kOutputSize) {
    MS_LOG(ERROR) << "AddGrad expects 2 outputs (dx1, dx2), got " << out_tensors_.size();
    return RET_ERROR;
  }
  if (in_tensors_[kDyIdx] == nullptr || in_tensors_[kX1Idx] == nullptr || in_tensors_[kX2Idx] == nullptr ||
      out_tensors_[kDx1Idx] == nullptr || out_tensors_[kDx2Idx] == nullptr) {
    MS_LOG(ERROR) << "AddGrad got null tensor.";
    return RET_ERROR;
  }
  if (in_tensors_[kDyIdx]->shape().size() > kMaxDims || out_tensors_[kDx1Idx]->shape().size() > kMaxDims ||
      out_tensors_[kDx2Idx]->shape().size() > kMaxDims) {
    MS_LOG(ERROR) << "AddGrad supports at most " << kMaxDims << " dims.";
    return RET_ERROR;
  }
  return RET_OK;
}

int AddGradDSPKernel::Run() {
  auto *dy = in_tensors_[kDyIdx];
  auto *dx1 = out_tensors_[kDx1Idx];
  auto *dx2 = out_tensors_[kDx2Idx];
  auto data_type = dy->data_type();
  auto mem_type = GetMemType();
  auto allocator = dsp_runtime_->GetAllocator();
  if (allocator == nullptr) {
    MS_LOG(ERROR) << "AddGradDSPKernel failed to get DSP allocator.";
    return RET_ERROR;
  }
  uint64_t dy_device_ptr = allocator->GetDeviceMemPtr(dy->data());
  if (dy_device_ptr == 0) {
    MS_LOG(ERROR) << "AddGradDSPKernel dy device ptr is null.";
    return RET_ERROR;
  }
  uint64_t dx1_device_ptr = allocator->GetDeviceMemPtr(dx1->data());
  if (dx1_device_ptr == 0) {
    MS_LOG(ERROR) << "AddGradDSPKernel dx1 device ptr is null.";
    return RET_ERROR;
  }
  uint64_t dx2_device_ptr = allocator->GetDeviceMemPtr(dx2->data());
  if (dx2_device_ptr == 0) {
    MS_LOG(ERROR) << "AddGradDSPKernel dx2 device ptr is null.";
    return RET_ERROR;
  }

  // add_grad_func reads the dims arrays on host side and copies them to device memory itself.
  std::vector<int> dx1_dims = dx1->shape();
  std::vector<int> dx2_dims = dx2->shape();
  std::vector<int> dy_dims = dy->shape();
  int ret =
    add_grad_func(dy_device_ptr, dx1_device_ptr, dx2_device_ptr, dx1_dims.data(), static_cast<int>(dx1_dims.size()),
                  dx2_dims.data(), static_cast<int>(dx2_dims.size()), dy_dims.data(), static_cast<int>(dy_dims.size()),
                  core_mask_, static_cast<int>(data_type), static_cast<int>(mem_type));
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "AddGradDSPKernel run failed.";
    return RET_ERROR;
  }

  return RET_OK;
}

REG_KERNEL(kDSP, kNumberTypeFloat32, PrimitiveType_AddGrad, DSPKernelCreator<AddGradDSPKernel>)

#ifdef SUPPORT_FT04
REG_KERNEL(kDSP, kNumberTypeFloat16, PrimitiveType_AddGrad, DSPKernelCreator<AddGradDSPKernel>)
#endif
}  // namespace mindspore::kernel
