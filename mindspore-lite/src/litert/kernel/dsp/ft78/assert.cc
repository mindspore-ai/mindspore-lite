
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

#include "src/litert/kernel/dsp/ft78/assert.h"
#include <algorithm>
#include <string>
#include "src/litert/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH::kDSP;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Assert;

namespace mindspore::kernel {

int AssertDSPKernel::CheckSpecs() {
  // assert takes 1 input (cond) and 1 output
  if (in_tensors_.size() != 1 || out_tensors_.size() != 1) {
    MS_LOG(WARNING) << "Assert expects one input and one output";
    return RET_ERROR;
  }
  return RET_OK;
}

int AssertDSPKernel::Prepare() { return RET_OK; }

int AssertDSPKernel::RunInt() {
  kernel_name_ = "assert_s";
  core_mask_ = 0xff;
  return dsp_runtime_->RunKernel(kernel_name_, kernel_args_, core_mask_);
}

int AssertDSPKernel::Run() {
  if (in_tensors_.empty() || out_tensors_.empty()) {
    MS_LOG(ERROR) << "Assert inputs/outputs are not set.";
    return RET_ERROR;
  }
  auto in_dt = in_tensors_[0]->data_type();
  if (in_dt != kNumberTypeInt32) {
    MS_LOG(ERROR) << "Assert DSP FT78 only supports int32 input for now. got: " << static_cast<int>(in_dt);
    return RET_ERROR;
  }
  auto allocator = dsp_runtime_->GetAllocator();
  if (allocator == nullptr) {
    MS_LOG(ERROR) << "DSP allocator is null.";
    return RET_ERROR;
  }

  // device pointers: cond, out
  uint64_t cond_dev = allocator->GetDeviceMemPtr(in_tensors_[0]->data());
  uint64_t out_dev = allocator->GetDeviceMemPtr(out_tensors_[0]->data());
  SetKernelArg({cond_dev, out_dev});

  if (out_tensors_[0]->data_type() != kNumberTypeInt32) {
    MS_LOG(ERROR) << "Assert DSP FT78 expects int32 output. got: " << static_cast<int>(out_tensors_[0]->data_type());
    return RET_ERROR;
  }
  return RunInt();
}

REG_KERNEL(kDSP, kNumberTypeInt32, PrimitiveType_Assert, DSPKernelCreator<AssertDSPKernel>)

}  // namespace mindspore::kernel
