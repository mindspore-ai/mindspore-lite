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

#include "src/litert/kernel/dsp/kernel/abs.h"
#include <algorithm>
#include <map>
#include <string>
#include <set>
#include "src/litert/kernel_registry.h"
#include "armc/include/operator.h"

using mindspore::kernel::KERNEL_ARCH::kDSP;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Abs;

namespace mindspore::kernel {

constexpr int kInIdx = 0;
constexpr int kOutIdx = 0;

int AbsDSPKernel::CheckSpecs() {
  if (in_tensors_.size() != INPUT_TENSOR_SIZE_1) {
    MS_LOG(WARNING) << "in size: " << in_tensors_.size();
    return RET_ERROR;
  }
  if (out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(WARNING) << "out size: " << out_tensors_.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int AbsDSPKernel::Prepare() { return RET_OK; }

int AbsDSPKernel::Run() {
  auto *input = in_tensors_[kInIdx];
  auto *output = out_tensors_[kOutIdx];
  auto data_type = input->data_type();
  auto mem_type = GetMemType();
  auto allocator = dsp_runtime_->GetAllocator();
  if (allocator == nullptr) {
    MS_LOG(ERROR) << "AbsDSPKernel failed to get DSP allocator.";
    return RET_ERROR;
  }
  uint64_t input_device_ptr = allocator->GetDeviceMemPtr(input->data());
  if (input_device_ptr == 0) {
    MS_LOG(ERROR) << "AbsDSPKernel input device ptr is null.";
    return RET_ERROR;
  }
  uint64_t output_device_ptr = allocator->GetDeviceMemPtr(output->data());
  if (output_device_ptr == 0) {
    MS_LOG(ERROR) << "AbsDSPKernel output device ptr is null.";
    return RET_ERROR;
  }
  uint64_t length = input->ElementsNum();
  int dtype = static_cast<int>(data_type);
  int mtype = static_cast<int>(mem_type);
  int ret = abs_func(input_device_ptr, output_device_ptr, length, core_mask_, dtype, mtype);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "AbsDSPKernel run failed.";
    return RET_ERROR;
  }

  return RET_OK;
}

REG_KERNEL(kDSP, kNumberTypeFloat32, PrimitiveType_Abs, DSPKernelCreator<AbsDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt16, PrimitiveType_Abs, DSPKernelCreator<AbsDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt32, PrimitiveType_Abs, DSPKernelCreator<AbsDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeComplex64, PrimitiveType_Abs, DSPKernelCreator<AbsDSPKernel>)

#ifdef SUPPORT_FT04
REG_KERNEL(kDSP, kNumberTypeFloat16, PrimitiveType_Abs, DSPKernelCreator<AbsDSPKernel>)
#endif

#ifdef SUPPORT_FT78
REG_KERNEL(kDSP, kNumberTypeFloat64, PrimitiveType_Abs, DSPKernelCreator<AbsDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt8, PrimitiveType_Abs, DSPKernelCreator<AbsDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeComplex128, PrimitiveType_Abs, DSPKernelCreator<AbsDSPKernel>)
#endif
}  // namespace mindspore::kernel
