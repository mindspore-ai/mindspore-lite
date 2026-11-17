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

#include "src/litert/kernel/dsp/kernel/ceil.h"
#include "src/litert/kernel_registry.h"
#include "armc/include/operator.h"

using mindspore::kernel::KERNEL_ARCH::kDSP;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Ceil;

namespace mindspore::kernel {

namespace {
constexpr size_t kInIdx = 0;
constexpr size_t kOutIdx = 0;
}  // namespace

int CeilDSPKernel::Prepare() { return RET_OK; }

int CeilDSPKernel::CheckSpecs() {
  if (op_parameter_ == nullptr) {
    MS_LOG(ERROR) << "Ceil kernel parameter is null.";
    return RET_ERROR;
  }
  if (in_tensors_.size() != INPUT_TENSOR_SIZE_1) {
    MS_LOG(ERROR) << "Ceil expects 1 input, got " << in_tensors_.size();
    return RET_ERROR;
  }
  if (out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(ERROR) << "Ceil expects 1 output, got " << out_tensors_.size();
    return RET_ERROR;
  }
  auto *input = in_tensors_[kInIdx];
  auto *output = out_tensors_[kOutIdx];
  if (input == nullptr) {
    MS_LOG(ERROR) << "Ceil input tensor is null.";
    return RET_ERROR;
  }
  if (output == nullptr) {
    MS_LOG(ERROR) << "Ceil output tensor is null.";
    return RET_ERROR;
  }
  const auto in_elements = input->ElementsNum();
  if (in_elements <= 0) {
    MS_LOG(ERROR) << "Ceil input holds " << in_elements << " elements, nothing to compute.";
    return RET_ERROR;
  }
  if (output->ElementsNum() != in_elements) {
    MS_LOG(ERROR) << "Ceil output holds " << output->ElementsNum() << " elements while the input holds " << in_elements
                  << ".";
    return RET_ERROR;
  }
  // ceil_func keeps the dtype of its input, so a graph that rounds into another dtype cannot be
  // served here and stays on the CPU kernel.
  if (output->data_type() != input->data_type()) {
    MS_LOG(ERROR) << "Ceil output dtype " << output->data_type() << " differs from the input dtype "
                  << input->data_type() << ".";
    return RET_ERROR;
  }
  return RET_OK;
}

int CeilDSPKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  auto allocator = dsp_runtime_->GetAllocator();
  if (allocator == nullptr) {
    MS_LOG(ERROR) << "Ceil DSP allocator is null.";
    return RET_ERROR;
  }
  auto *input = in_tensors_[kInIdx];
  uint64_t x_device_ptr = allocator->GetDeviceMemPtr(input->data());
  if (x_device_ptr == 0) {
    MS_LOG(ERROR) << "Ceil input device ptr is null.";
    return RET_ERROR;
  }
  auto *output = out_tensors_[kOutIdx];
  uint64_t out_device_ptr = allocator->GetDeviceMemPtr(output->data());
  if (out_device_ptr == 0) {
    MS_LOG(ERROR) << "Ceil output device ptr is null.";
    return RET_ERROR;
  }
  const auto length = static_cast<uint64_t>(input->ElementsNum());
  int ret = ceil_func(x_device_ptr, out_device_ptr, length, static_cast<unsigned int>(core_mask_),
                      static_cast<int>(input->data_type()), static_cast<int>(GetMemType()));
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Ceil run failed, ret: " << ret;
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kDSP, kNumberTypeFloat32, PrimitiveType_Ceil, DSPKernelCreator<CeilDSPKernel>)

#ifdef SUPPORT_FT04
REG_KERNEL(kDSP, kNumberTypeFloat16, PrimitiveType_Ceil, DSPKernelCreator<CeilDSPKernel>)
#endif

#ifdef SUPPORT_FT78
REG_KERNEL(kDSP, kNumberTypeFloat64, PrimitiveType_Ceil, DSPKernelCreator<CeilDSPKernel>)
#endif
}  // namespace mindspore::kernel
