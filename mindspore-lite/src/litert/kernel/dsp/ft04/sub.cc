
/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include <algorithm>
#include <map>
#include <string>
#include "src/litert/kernel_registry.h"
#include "src/litert/kernel/dsp/ft04/sub.h"

using mindspore::kernel::KERNEL_ARCH::kDSP;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_SubFusion;

namespace mindspore::kernel {
int SubDSPKernel::CheckSpecs() {
  if (in_tensors_.size() != INPUT_TENSOR_SIZE_2) {
    MS_LOG(WARNING) << "in size: " << in_tensors_.size();
    return RET_ERROR;
  }

  if (in_tensors_.front()->shape() != in_tensors_.back()->shape()) {
    MS_LOG(WARNING) << "input shape must be equal";
    return RET_ERROR;
  }
  return RET_OK;
}

int SubDSPKernel::Prepare() { return RET_OK; }

int SubDSPKernel::SubRunFp32() {
  kernel_name_ = "fp_sub_s";
  core_mask_ = 0xf;
  return dsp_runtime_->RunKernel(kernel_name_, kernel_args_, core_mask_);
}

int SubDSPKernel::SubRunFp16() {
  kernel_name_ = "hp_sub_s";
  core_mask_ = 0xf;
  return dsp_runtime_->RunKernel(kernel_name_, kernel_args_, core_mask_);
}

int SubDSPKernel::SubRunInt16() {
  kernel_name_ = "i16_sub_s";
  core_mask_ = 0xf;
  return dsp_runtime_->RunKernel(kernel_name_, kernel_args_, core_mask_);
}

int SubDSPKernel::SubRunInt32() {
  kernel_name_ = "i32_sub_s";
  core_mask_ = 0xf;
  return dsp_runtime_->RunKernel(kernel_name_, kernel_args_, core_mask_);
}

int SubDSPKernel::SubRunComplex64() {
  kernel_name_ = "c64_sub_s";
  core_mask_ = 0xf;
  return dsp_runtime_->RunKernel(kernel_name_, kernel_args_, core_mask_);
}

int SubDSPKernel::Run() {
  int ret = -1;
  MS_LOG(DEBUG) << this->name() << " Running! ";
  uint64_t length = in_tensors_[0]->ElementsNum();
  auto allocator = dsp_runtime_->GetAllocator();
  uint64_t x_device_ptr = allocator->GetDeviceMemPtr(in_tensors_[0]->data());
  uint64_t y_device_ptr = allocator->GetDeviceMemPtr(in_tensors_[1]->data());
  uint64_t out_device_ptr = allocator->GetDeviceMemPtr(out_tensors_[0]->data());
  SetKernelArg({x_device_ptr, y_device_ptr, out_device_ptr, length});
  auto data_type = in_tensors_[0]->data_type();
  if (data_type == kNumberTypeFloat32) {
    ret = SubRunFp32();
  } else if (data_type == kNumberTypeFloat16) {
    ret = SubRunFp16();
  } else if (data_type == kNumberTypeInt16) {
    ret = SubRunInt16();
  } else if (data_type == kNumberTypeInt32) {
    ret = SubRunInt32();
  } else if (data_type == kNumberTypeComplex64) {
    ret = SubRunComplex64();
  } else {
    MS_LOG(ERROR) << "unsupported data type: " << static_cast<int>(data_type);
  }
  if (ret != RET_OK) {
    MS_LOG(ERROR) << this->name() << " Run failed! ";
    return ret;
  }
  MS_LOG(DEBUG) << this->name() << " Run success! ";
  return RET_OK;
}

REG_KERNEL(kDSP, kNumberTypeFloat32, PrimitiveType_SubFusion, DSPKernelCreator<SubDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt16, PrimitiveType_SubFusion, DSPKernelCreator<SubDSPKernel>);
REG_KERNEL(kDSP, kNumberTypeInt32, PrimitiveType_SubFusion, DSPKernelCreator<SubDSPKernel>);
REG_KERNEL(kDSP, kNumberTypeComplex64, PrimitiveType_SubFusion, DSPKernelCreator<SubDSPKernel>);
REG_KERNEL(kDSP, kNumberTypeFloat16, PrimitiveType_SubFusion, DSPKernelCreator<SubDSPKernel>);
}  // namespace mindspore::kernel
