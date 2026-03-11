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

#include "src/litert/kernel/dsp/kernel/assert.h"
#include <algorithm>
#include <string>
#include "src/litert/kernel_registry.h"
#include "armc/include/operator.h"

using mindspore::kernel::KERNEL_ARCH::kDSP;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Assert;

namespace mindspore::kernel {

int AssertDSPKernel::CheckSpecs() {
  if (in_tensors_.size() != 1 || out_tensors_.size() != 1) {
    MS_LOG(WARNING) << "Assert expects one input and one output";
    return RET_ERROR;
  }
  auto in_dt = in_tensors_[0]->data_type();
  if (in_dt != kNumberTypeInt32) {
    MS_LOG(WARNING) << "Assert DSP only supports int32 input for now. got: " << static_cast<int>(in_dt);
    return RET_ERROR;
  }
  if (out_tensors_[0]->data_type() != kNumberTypeInt32) {
    MS_LOG(WARNING) << "Assert DSP expects int32 output. got: " << static_cast<int>(out_tensors_[0]->data_type());
    return RET_ERROR;
  }
  return RET_OK;
}

int AssertDSPKernel::Prepare() { return RET_OK; }

int AssertDSPKernel::Run() {
  auto allocator = dsp_runtime_->GetAllocator();
  if (allocator == nullptr) {
    MS_LOG(ERROR) << "DSP allocator is null.";
    return RET_ERROR;
  }

  uint64_t cond_device_ptr = allocator->GetDeviceMemPtr(in_tensors_[0]->data());
  uint64_t out_device_ptr = allocator->GetDeviceMemPtr(out_tensors_[0]->data());

  auto data_type = in_tensors_[0]->data_type();
  auto mem_type = GetMemType();
  int dtype = static_cast<int>(data_type);
  int mtype = static_cast<int>(mem_type);

  int ret = assert_func(cond_device_ptr, out_device_ptr, core_mask_, dtype, mtype);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << this->name() << " Run failed!";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kDSP, kNumberTypeInt32, PrimitiveType_Assert, DSPKernelCreator<AssertDSPKernel>)

}  // namespace mindspore::kernel
