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

#include "src/litert/kernel/dsp/kernel/assign_add.h"
#include <vector>
#include "src/litert/kernel_registry.h"
#include "armc/include/operator.h"

using mindspore::kernel::KERNEL_ARCH::kDSP;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_AssignAdd;

namespace mindspore::kernel {

namespace {
constexpr int kVariableIdx = 0;
constexpr int kValueIdx = 1;
constexpr int kOutIdx = 0;
}  // namespace

int AssignAddDSPKernel::Prepare() { return RET_OK; }

int AssignAddDSPKernel::CheckSpecs() {
  if (op_parameter_ == nullptr) {
    MS_LOG(ERROR) << "AssignAdd kernel parameter is null.";
    return RET_ERROR;
  }
  // AssignAdd(variable, value) is an in-place update: the shape contract comes from the CPU
  // infer function (cpu/nnacl_c/infer/assign_add_infer.c), which takes exactly two inputs and
  // one output that inherits dtype and shape from inputs[0].
  if (in_tensors_.size() != INPUT_TENSOR_SIZE_2) {
    MS_LOG(ERROR) << "AssignAdd expects 2 inputs (variable, value), got " << in_tensors_.size();
    return RET_ERROR;
  }
  if (out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(ERROR) << "AssignAdd expects 1 output, got " << out_tensors_.size();
    return RET_ERROR;
  }
  auto *variable = in_tensors_[kVariableIdx];
  auto *value = in_tensors_[kValueIdx];
  auto *output = out_tensors_[kOutIdx];
  if (variable == nullptr || value == nullptr || output == nullptr) {
    MS_LOG(ERROR) << "AssignAdd got null tensor.";
    return RET_ERROR;
  }
  if (variable->data_type() != value->data_type() || variable->data_type() != output->data_type()) {
    MS_LOG(ERROR) << "AssignAdd dtype mismatch: variable " << variable->data_type() << ", value " << value->data_type()
                  << ", output " << output->data_type() << ".";
    return RET_ERROR;
  }
  const auto elements = variable->ElementsNum();
  if (elements <= 0) {
    MS_LOG(ERROR) << "AssignAdd variable holds " << elements << " elements, there is nothing to update.";
    return RET_ERROR;
  }
  // assign_add_func is given one element count shared by both operands, so only the
  // element-wise case can be served here; a broadcast needs the CPU kernel instead.
  if (value->ElementsNum() != elements) {
    MS_LOG(ERROR) << "AssignAdd value has " << value->ElementsNum() << " elements while variable has " << elements
                  << "; the DSP kernel adds element by element and needs identical lengths.";
    return RET_ERROR;
  }
  // AssignAdd updates the variable in place: a converted Lite graph gives the node an output
  // tensor of its own that is not consumed, so the kernel writes in_tensors_[0] and never touches
  // out_tensors_[0]. A data() alias check cannot be made here either: CheckSpecs runs before
  // DSPKernel::PreProcess allocates the output buffer, so both host pointers are still null.
  return RET_OK;
}

int AssignAddDSPKernel::Run() {
  MS_LOG(DEBUG) << this->name_ << " Running!";
  auto allocator = dsp_runtime_->GetAllocator();
  if (allocator == nullptr) {
    MS_LOG(ERROR) << "AssignAdd DSP allocator is null.";
    return RET_ERROR;
  }
  auto *variable = in_tensors_[kVariableIdx];
  uint64_t dst_device_ptr = allocator->GetDeviceMemPtr(variable->data());
  if (dst_device_ptr == 0) {
    MS_LOG(ERROR) << "AssignAdd variable device ptr is null.";
    return RET_ERROR;
  }
  uint64_t src_device_ptr = allocator->GetDeviceMemPtr(in_tensors_[kValueIdx]->data());
  if (src_device_ptr == 0) {
    MS_LOG(ERROR) << "AssignAdd value device ptr is null.";
    return RET_ERROR;
  }
  // armc adds src into dst in place. `length` counts elements, not components: a complex
  // element carries its real and imaginary part inside one element, which the kernel unpacks.
  const uint64_t length = static_cast<uint64_t>(variable->ElementsNum());
  int ret = assign_add_func(src_device_ptr, dst_device_ptr, length, static_cast<unsigned int>(core_mask_),
                            static_cast<int>(variable->data_type()), static_cast<int>(GetMemType()));
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "AssignAdd run failed, ret: " << ret;
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kDSP, kNumberTypeInt16, PrimitiveType_AssignAdd, DSPKernelCreator<AssignAddDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt32, PrimitiveType_AssignAdd, DSPKernelCreator<AssignAddDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeFloat32, PrimitiveType_AssignAdd, DSPKernelCreator<AssignAddDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeComplex64, PrimitiveType_AssignAdd, DSPKernelCreator<AssignAddDSPKernel>)

#ifdef SUPPORT_FT04
REG_KERNEL(kDSP, kNumberTypeFloat16, PrimitiveType_AssignAdd, DSPKernelCreator<AssignAddDSPKernel>)
#endif

#ifdef SUPPORT_FT78
REG_KERNEL(kDSP, kNumberTypeInt8, PrimitiveType_AssignAdd, DSPKernelCreator<AssignAddDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeFloat64, PrimitiveType_AssignAdd, DSPKernelCreator<AssignAddDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeComplex128, PrimitiveType_AssignAdd, DSPKernelCreator<AssignAddDSPKernel>)
#endif
}  // namespace mindspore::kernel
