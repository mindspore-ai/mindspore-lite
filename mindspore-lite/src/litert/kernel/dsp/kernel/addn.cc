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

#include "src/litert/kernel/dsp/kernel/addn.h"
#include <limits>
#include <vector>
#include "src/litert/kernel_registry.h"
#include "armc/include/operator.h"

using mindspore::kernel::KERNEL_ARCH::kDSP;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_AddN;

namespace mindspore::kernel {

namespace {
constexpr int kOutIdx = 0;
constexpr size_t kScalarElementNum = 1;
}  // namespace

int AddNDSPKernel::Prepare() {
  // The two arrays armc needs are kept as members so that Run() does not allocate on the
  // inference hot path; Run() still resizes them if the input count ever changes.
  input_ptrs_.resize(in_tensors_.size());
  element_nums_.resize(in_tensors_.size());
  return RET_OK;
}

int AddNDSPKernel::CheckSpecs() {
  if (op_parameter_ == nullptr) {
    MS_LOG(ERROR) << "AddN kernel parameter is null.";
    return RET_ERROR;
  }
  // AddN sums two or more tensors, and the CPU infer refuses a smaller list as well
  // (cpu/nnacl_c/infer/addn_infer.c returns NNACL_ERR below two inputs), so accepting a single
  // input here would give the scheduler a kernel no reference implementation agrees with.
  if (in_tensors_.size() < INPUT_TENSOR_SIZE_2) {
    MS_LOG(ERROR) << "AddN expects at least 2 inputs, got " << in_tensors_.size();
    return RET_ERROR;
  }
  if (out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(ERROR) << "AddN expects 1 output, got " << out_tensors_.size();
    return RET_ERROR;
  }
  auto *output = out_tensors_[kOutIdx];
  if (output == nullptr) {
    MS_LOG(ERROR) << "AddN output tensor is null.";
    return RET_ERROR;
  }
  const auto out_dtype = output->data_type();
  const auto out_elements = output->ElementsNum();
  // element_nums_ reaches addn_func as an `int` array, so a count above INT32_MAX would be
  // truncated on the way to the DSP. Reject such a graph here and let the scheduler fall back to
  // the CPU kernel instead of running AddN over the wrong number of elements.
  if (out_elements > std::numeric_limits<int>::max()) {
    MS_LOG(ERROR) << "AddN output has " << out_elements << " elements, more than the DSP kernel supports.";
    return RET_ERROR;
  }
  // The CPU kernel (AddNCompute in cpu/nnacl_c/kernel/addn.c) only handles an input that matches
  // the output shape or holds a single element that is broadcast over it; any other combination
  // falls into ElementOptAdd, which adds one scalar to every element. The DSP kernel mirrors that
  // contract, and because addn_func receives element counts only, tensors that merely share an
  // element count must not be mistaken for the same shape.
  for (size_t i = 0; i < in_tensors_.size(); ++i) {
    auto *input = in_tensors_[i];
    if (input == nullptr) {
      MS_LOG(ERROR) << "AddN input " << i << " is null.";
      return RET_ERROR;
    }
    if (input->data_type() != out_dtype) {
      MS_LOG(ERROR) << "AddN input " << i << " dtype " << input->data_type() << " differs from output dtype "
                    << out_dtype << ".";
      return RET_ERROR;
    }
    const auto in_elements = input->ElementsNum();
    if (in_elements == kScalarElementNum) {
      // The single element operand the CPU kernel broadcasts.
      continue;
    }
    if (input->shape() != output->shape()) {
      MS_LOG(ERROR) << "AddN input " << i << " shape differs from the output shape; only the same shape or a "
                    << "single element input is supported by the DSP kernel.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int AddNDSPKernel::Run() {
  MS_LOG(DEBUG) << this->name_ << " Running!";
  auto allocator = dsp_runtime_->GetAllocator();
  if (allocator == nullptr) {
    MS_LOG(ERROR) << "AddN DSP allocator is null.";
    return RET_ERROR;
  }
  if (input_ptrs_.size() != in_tensors_.size()) {
    // Only a kernel reused on a graph with a different input count reaches this branch.
    input_ptrs_.resize(in_tensors_.size());
    element_nums_.resize(in_tensors_.size());
  }
  for (size_t i = 0; i < in_tensors_.size(); ++i) {
    auto *input = in_tensors_[i];
    uint64_t device_ptr = allocator->GetDeviceMemPtr(input->data());
    if (device_ptr == 0) {
      MS_LOG(ERROR) << "AddN input " << i << " device ptr is null.";
      return RET_ERROR;
    }
    input_ptrs_[i] = device_ptr;
    element_nums_[i] = static_cast<int>(input->ElementsNum());
  }
  auto *output = out_tensors_[kOutIdx];
  uint64_t out_device_ptr = allocator->GetDeviceMemPtr(output->data());
  if (out_device_ptr == 0) {
    MS_LOG(ERROR) << "AddN output device ptr is null.";
    return RET_ERROR;
  }
  // armc takes device addresses as `unsigned long long`, which is the same type as uint64_t
  // on the 32-bit ARM DSP targets, so the pointer array converts implicitly (the compiler
  // rejects the call if those types ever diverge).
  int ret = addn_func(input_ptrs_.data(), out_device_ptr, static_cast<int>(in_tensors_.size()), element_nums_.data(),
                      static_cast<unsigned int>(core_mask_), static_cast<int>(output->data_type()),
                      static_cast<int>(GetMemType()));
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "AddN run failed, ret: " << ret;
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kDSP, kNumberTypeFloat32, PrimitiveType_AddN, DSPKernelCreator<AddNDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt16, PrimitiveType_AddN, DSPKernelCreator<AddNDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt32, PrimitiveType_AddN, DSPKernelCreator<AddNDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeComplex64, PrimitiveType_AddN, DSPKernelCreator<AddNDSPKernel>)

#ifdef SUPPORT_FT04
REG_KERNEL(kDSP, kNumberTypeFloat16, PrimitiveType_AddN, DSPKernelCreator<AddNDSPKernel>)
#endif

#ifdef SUPPORT_FT78
REG_KERNEL(kDSP, kNumberTypeInt8, PrimitiveType_AddN, DSPKernelCreator<AddNDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeFloat64, PrimitiveType_AddN, DSPKernelCreator<AddNDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeComplex128, PrimitiveType_AddN, DSPKernelCreator<AddNDSPKernel>)
#endif
}  // namespace mindspore::kernel
