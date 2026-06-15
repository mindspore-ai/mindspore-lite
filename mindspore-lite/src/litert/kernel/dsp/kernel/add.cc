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

#include "src/litert/kernel/dsp/kernel/add.h"
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
using mindspore::schema::PrimitiveType_AddFusion;

namespace mindspore::kernel {

constexpr int kXIdx = 0;
constexpr int kYIdx = 1;
constexpr int kOutIdx = 0;

int AddDSPKernel::CheckSpecs() {
  if (param_ == nullptr) {
    MS_LOG(ERROR) << "add kernel param is null";
    return RET_ERROR;
  }
  return RET_OK;
}

int AddDSPKernel::Prepare() { return RET_OK; }

int AddDSPKernel::Run() {
  auto *input_x = in_tensors_[kXIdx];
  auto *input_y = in_tensors_[kYIdx];
  auto *output = out_tensors_[kOutIdx];
  auto data_type = input_x->data_type();
  auto mem_type = GetMemType();
  auto allocator = dsp_runtime_->GetAllocator();
  uint64_t x_device_ptr = allocator->GetDeviceMemPtr(input_x->data());
  if (x_device_ptr == 0) {
    MS_LOG(ERROR) << "AddDSPKernel x device ptr is null.";
    return RET_ERROR;
  }
  uint64_t y_device_ptr = allocator->GetDeviceMemPtr(input_y->data());
  if (y_device_ptr == 0) {
    MS_LOG(ERROR) << "AddDSPKernel y device ptr is null.";
    return RET_ERROR;
  }
  uint64_t out_device_ptr = allocator->GetDeviceMemPtr(output->data());
  if (out_device_ptr == 0) {
    MS_LOG(ERROR) << "AddDSPKernel out device ptr is null.";
    return RET_ERROR;
  }
  auto input0_shape = input_x->shape();
  auto input1_shape = input_y->shape();
  auto output_shape = output->shape();
  int ret = add_func(x_device_ptr, y_device_ptr, out_device_ptr, input0_shape.data(),
                     static_cast<int>(input0_shape.size()), input1_shape.data(), static_cast<int>(input1_shape.size()),
                     output_shape.data(), static_cast<int>(output_shape.size()), param_->activation_type_, core_mask_,
                     static_cast<int>(data_type), static_cast<int>(mem_type));
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "AddDSPKernel run failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kDSP, kNumberTypeFloat32, PrimitiveType_AddFusion, DSPKernelCreator<AddDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt16, PrimitiveType_AddFusion, DSPKernelCreator<AddDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt32, PrimitiveType_AddFusion, DSPKernelCreator<AddDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeComplex64, PrimitiveType_AddFusion, DSPKernelCreator<AddDSPKernel>)

#ifdef SUPPORT_FT04
REG_KERNEL(kDSP, kNumberTypeFloat16, PrimitiveType_AddFusion, DSPKernelCreator<AddDSPKernel>)
#endif

#ifdef SUPPORT_FT78
REG_KERNEL(kDSP, kNumberTypeFloat64, PrimitiveType_AddFusion, DSPKernelCreator<AddDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt8, PrimitiveType_AddFusion, DSPKernelCreator<AddDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeComplex128, PrimitiveType_AddFusion, DSPKernelCreator<AddDSPKernel>)
#endif
}  // namespace mindspore::kernel
