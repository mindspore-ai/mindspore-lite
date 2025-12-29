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

#include "src/litert/kernel/dsp/base/add_base.h"
#include <algorithm>
#include <map>
#include <string>
#include "src/litert/kernel_registry.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NOT_SUPPORT;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {

int AddBaseDSPKernel::CheckSpecs() {
  if (param_ == nullptr) {
    MS_LOG(ERROR) << "add kernel param is null";
    return RET_ERROR;
  }
  if (param_->broadcasting_) {
    MS_LOG(WARNING) << "add kernel broadcasting is not supported";
    return RET_NOT_SUPPORT;
  }
  if (param_->activation_type_ != schema::ActivationType_NO_ACTIVATION) {
    MS_LOG(WARNING) << "add kernel activation is not supported";
    return RET_NOT_SUPPORT;
  }
  return RET_OK;
}

int AddBaseDSPKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  auto data_type = in_tensors_[0]->data_type();
  auto mem_type = GetMemType();
  if (param_->activation_type_ == schema::ActivationType_NO_ACTIVATION && param_->broadcasting_ == false) {
    auto kernel_name = GenerateKernelName(data_type, mem_type, "add");
    SetKernelName(kernel_name);
  } else {
    MS_LOG(ERROR) << "AddDSPKernel not support activation or broadcasting";
    return RET_ERROR;
  }
  return RET_OK;
}

int AddBaseDSPKernel::Run() {
  uint64_t length = in_tensors_[0]->ElementsNum();
  auto allocator = dsp_runtime_->GetAllocator();
  uint64_t x_device_ptr = allocator->GetDeviceMemPtr(in_tensors_[0]->data());
  if (x_device_ptr == 0) {
    MS_LOG(ERROR) << "AddDSPKernel x device ptr is null.";
    return RET_ERROR;
  }
  uint64_t y_device_ptr = allocator->GetDeviceMemPtr(in_tensors_[1]->data());
  if (y_device_ptr == 0) {
    MS_LOG(ERROR) << "AddDSPKernel y device ptr is null.";
    return RET_ERROR;
  }
  uint64_t out_device_ptr = allocator->GetDeviceMemPtr(out_tensors_[0]->data());
  if (out_device_ptr == 0) {
    MS_LOG(ERROR) << "AddDSPKernel out device ptr is null.";
    return RET_ERROR;
  }
  SetKernelArg({x_device_ptr, y_device_ptr, out_device_ptr, length});
  auto ret = dsp_runtime_->RunKernel(kernel_name_, kernel_args_, core_mask_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "AddDSPKernel run failed.";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
