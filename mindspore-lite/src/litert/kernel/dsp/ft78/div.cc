
/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/dsp/ft78/div.h"
#include <algorithm>
#include <map>
#include <string>
#include "src/litert/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH::kDSP;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_DivFusion;

namespace mindspore::kernel {
int DivDSPKernel::CheckSpecs() {
  if (in_tensors_.empty() || in_tensors_.size() != INPUT_TENSOR_SIZE_2) {
    MS_LOG(WARNING) << "Invalid input tensors size: " << in_tensors_.size();
    return RET_ERROR;
  }
  auto x_tensor = in_tensors_.front();
  auto y_tensor = in_tensors_.back();
  if (x_tensor->shape() != y_tensor->shape() && x_tensor->ElementsNum() != 1 && y_tensor->ElementsNum() != 1) {
    MS_LOG(WARNING) << "Input shapes must be equal or one must be scalar";
    return RET_ERROR;
  }
  return RET_OK;
}

int DivDSPKernel::Prepare() {
  if (in_tensors_.empty() || in_tensors_.size() != INPUT_TENSOR_SIZE_2) {
    return RET_ERROR;
  }
  auto x_num = in_tensors_[0]->ElementsNum();
  auto y_num = in_tensors_[1]->ElementsNum();
  if (x_num == 1 || y_num == 1) {
    optimize_ = true;
    first_scalar_ = (x_num == 1);
  }
  return RET_OK;
}

int DivDSPKernel::DivRunFp32() {
  kernel_name_ = "fp_div_s";
  core_mask_ = 0xff;
  return dsp_runtime_->RunKernel(kernel_name_, kernel_args_, core_mask_);
}

int DivDSPKernel::DivRunFp64() {
  kernel_name_ = "dp_div_s";
  core_mask_ = 0xff;
  return dsp_runtime_->RunKernel(kernel_name_, kernel_args_, core_mask_);
}

int DivDSPKernel::DivRunInt8() {
  kernel_name_ = "i8_div_s";
  core_mask_ = 0xff;
  return dsp_runtime_->RunKernel(kernel_name_, kernel_args_, core_mask_);
}

int DivDSPKernel::DivRunInt16() {
  kernel_name_ = "i16_div_s";
  core_mask_ = 0xff;
  return dsp_runtime_->RunKernel(kernel_name_, kernel_args_, core_mask_);
}

int DivDSPKernel::DivRunInt32() {
  kernel_name_ = "i32_div_s";
  core_mask_ = 0xff;
  return dsp_runtime_->RunKernel(kernel_name_, kernel_args_, core_mask_);
}

int DivDSPKernel::DivRunComplex64() {
  kernel_name_ = "c64_div_s";
  core_mask_ = 0xff;
  return dsp_runtime_->RunKernel(kernel_name_, kernel_args_, core_mask_);
}

int DivDSPKernel::DivRunComplex128() {
  kernel_name_ = "c128_div_s";
  core_mask_ = 0xff;
  return dsp_runtime_->RunKernel(kernel_name_, kernel_args_, core_mask_);
}

int DivDSPKernel::Run() {
  int ret = -1;
  uint64_t length = in_tensors_[0]->ElementsNum();
  auto allocator = dsp_runtime_->GetAllocator();
  uint64_t x_device_ptr = allocator->GetDeviceMemPtr(in_tensors_[0]->data());
  uint64_t y_device_ptr = allocator->GetDeviceMemPtr(in_tensors_[1]->data());
  uint64_t out_device_ptr = allocator->GetDeviceMemPtr(out_tensors_[0]->data());
  SetKernelArg({x_device_ptr, y_device_ptr, out_device_ptr, length, optimize_, first_scalar_});
  auto data_type = in_tensors_[0]->data_type();
  if (data_type == kNumberTypeFloat32) {
    ret = DivRunFp32();
  } else if (data_type == kNumberTypeFloat64) {
    ret = DivRunFp64();
  } else if (data_type == kNumberTypeInt8) {
    ret = DivRunInt8();
  } else if (data_type == kNumberTypeInt16) {
    ret = DivRunInt16();
  } else if (data_type == kNumberTypeInt32) {
    ret = DivRunInt32();
  } else if (data_type == kNumberTypeComplex64) {
    ret = DivRunComplex64();
  } else if (data_type == kNumberTypeComplex128) {
    ret = DivRunComplex128();
  } else {
    MS_LOG(ERROR) << "unsupported data type: " << static_cast<int>(data_type);
  }
  if (ret != RET_OK) {
    MS_LOG(ERROR) << this->name() << " Run failed! ";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kDSP, kNumberTypeFloat32, PrimitiveType_DivFusion, DSPKernelCreator<DivDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeFloat64, PrimitiveType_DivFusion, DSPKernelCreator<DivDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt8, PrimitiveType_DivFusion, DSPKernelCreator<DivDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt16, PrimitiveType_DivFusion, DSPKernelCreator<DivDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt32, PrimitiveType_DivFusion, DSPKernelCreator<DivDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeComplex64, PrimitiveType_DivFusion, DSPKernelCreator<DivDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeComplex128, PrimitiveType_DivFusion, DSPKernelCreator<DivDSPKernel>)
}  // namespace mindspore::kernel
