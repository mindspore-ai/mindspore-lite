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

#include "src/litert/kernel/dsp/kernel/assign.h"
#include <vector>
#include "src/litert/kernel_registry.h"
#include "armc/include/operator.h"

using mindspore::kernel::KERNEL_ARCH::kDSP;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Assign;

namespace mindspore::kernel {

namespace {
constexpr int kVariableIdx = 0;
constexpr int kValueIdx = 1;
constexpr int kOutIdx = 0;
constexpr size_t kAssignInputNum = 2;
}  // namespace

int AssignDSPKernel::Prepare() { return RET_OK; }

int AssignDSPKernel::CheckSpecs() {
  if (op_parameter_ == nullptr) {
    MS_LOG(ERROR) << "Assign kernel parameter is null.";
    return RET_ERROR;
  }
  // Assign(variable, value) is an in-place update: the CPU reference kernel
  // (cpu/fp32_grad/assign.cc) copies value into the variable tensor and returns it, so the
  // first two inputs are the ones that matter.
  if (in_tensors_.size() < kAssignInputNum) {
    MS_LOG(ERROR) << "Assign expects at least " << kAssignInputNum << " inputs (variable, value), got "
                  << in_tensors_.size();
    return RET_ERROR;
  }
  if (out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(ERROR) << "Assign expects 1 output, got " << out_tensors_.size();
    return RET_ERROR;
  }
  auto *variable = in_tensors_[kVariableIdx];
  auto *value = in_tensors_[kValueIdx];
  auto *output = out_tensors_[kOutIdx];
  if (variable == nullptr || value == nullptr || output == nullptr) {
    MS_LOG(ERROR) << "Assign got null tensor.";
    return RET_ERROR;
  }
  if (variable->data_type() != value->data_type() || variable->data_type() != output->data_type()) {
    MS_LOG(ERROR) << "Assign dtype mismatch: variable " << variable->data_type() << ", value " << value->data_type()
                  << ", output " << output->data_type() << ".";
    return RET_ERROR;
  }
  const auto elements = variable->ElementsNum();
  if (value->ElementsNum() != elements) {
    MS_LOG(ERROR) << "Assign value has " << value->ElementsNum() << " elements while variable has " << elements
                  << "; the DSP kernel copies element by element and needs identical lengths.";
    return RET_ERROR;
  }
  // The assignment is only observable through the variable tensor, so a graph that hands us a
  // separate output buffer cannot be served correctly here: reject it and let the scheduler
  // fall back to the CPU kernel instead of silently leaving the variable untouched.
  // The check is on tensor identity rather than on the data pointers: CheckSpecs runs while the
  // kernel is created, i.e. before DSPKernel::PreProcess allocates the DSP output buffer, so
  // data() is still null at this point and a pointer comparison would pass without checking.
  if (output != variable) {
    MS_LOG(ERROR) << "Assign output must be the same tensor as the variable input, Assign updates in place.";
    return RET_ERROR;
  }
  return RET_OK;
}

int AssignDSPKernel::Run() {
  MS_LOG(DEBUG) << this->name_ << " Running!";
  auto allocator = dsp_runtime_->GetAllocator();
  if (allocator == nullptr) {
    MS_LOG(ERROR) << "Assign DSP allocator is null.";
    return RET_ERROR;
  }
  auto *variable = in_tensors_[kVariableIdx];
  uint64_t dst_device_ptr = allocator->GetDeviceMemPtr(variable->data());
  if (dst_device_ptr == 0) {
    MS_LOG(ERROR) << "Assign variable device ptr is null.";
    return RET_ERROR;
  }
  uint64_t src_device_ptr = allocator->GetDeviceMemPtr(in_tensors_[kValueIdx]->data());
  if (src_device_ptr == 0) {
    MS_LOG(ERROR) << "Assign value device ptr is null.";
    return RET_ERROR;
  }
  // armc takes the element count and device addresses as `unsigned long long`, which is the
  // same type as uint64_t on the 32-bit ARM DSP targets, so plain uint64_t values convert
  // implicitly (the compiler rejects the call if those types ever diverge).
  const uint64_t length = static_cast<uint64_t>(variable->ElementsNum());
  int ret = assign_func(src_device_ptr, dst_device_ptr, length, static_cast<unsigned int>(core_mask_),
                        static_cast<int>(variable->data_type()), static_cast<int>(GetMemType()));
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Assign run failed, ret: " << ret;
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kDSP, kNumberTypeInt16, PrimitiveType_Assign, DSPKernelCreator<AssignDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt32, PrimitiveType_Assign, DSPKernelCreator<AssignDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeFloat32, PrimitiveType_Assign, DSPKernelCreator<AssignDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeComplex64, PrimitiveType_Assign, DSPKernelCreator<AssignDSPKernel>)

#ifdef SUPPORT_FT04
REG_KERNEL(kDSP, kNumberTypeFloat16, PrimitiveType_Assign, DSPKernelCreator<AssignDSPKernel>)
#endif

#ifdef SUPPORT_FT78
REG_KERNEL(kDSP, kNumberTypeInt8, PrimitiveType_Assign, DSPKernelCreator<AssignDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeFloat64, PrimitiveType_Assign, DSPKernelCreator<AssignDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeComplex128, PrimitiveType_Assign, DSPKernelCreator<AssignDSPKernel>)
#endif
}  // namespace mindspore::kernel
