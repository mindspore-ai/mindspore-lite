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

#include "src/litert/kernel/dsp/ft78/linspace.h"
#include <cstdint>
#include <cstring>
#include <string>
#include "src/litert/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH::kDSP;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_LinSpace;

namespace mindspore::kernel {
int LinspaceDSPKernel::CheckSpecs() {
  if (out_tensors_.size() != 1) {
    MS_LOG(WARNING) << "Linspace out size: " << out_tensors_.size();
    return RET_ERROR;
  }
  if (in_tensors_.size() != 3) {
    MS_LOG(WARNING) << "Linspace in size: " << in_tensors_.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int LinspaceDSPKernel::Prepare() { return RET_OK; }

int LinspaceDSPKernel::LinspaceRunFp32() {
  kernel_name_ = "fp_linspace_s";
  core_mask_ = 0xff;
  return dsp_runtime_->RunKernel(kernel_name_, kernel_args_, core_mask_);
}

int LinspaceDSPKernel::Run() {
  if (in_tensors_[0]->data_type() != kNumberTypeFloat32 || in_tensors_[1]->data_type() != kNumberTypeFloat32) {
    MS_LOG(ERROR) << "Linspace ft78 requires fp32 start/end inputs, got start dtype "
                  << static_cast<int>(in_tensors_[0]->data_type()) << " end dtype "
                  << static_cast<int>(in_tensors_[1]->data_type());
    return RET_ERROR;
  }
  if (in_tensors_[2]->data_type() != kNumberTypeInt32) {
    MS_LOG(ERROR) << "Linspace expects int32 count input, got dtype " << static_cast<int>(in_tensors_[2]->data_type());
    return RET_ERROR;
  }

  int32_t elements = *reinterpret_cast<int32_t *>(in_tensors_[2]->data());
  if (elements <= 0) {
    MS_LOG(ERROR) << "Linspace expects positive num, got " << elements;
    return RET_ERROR;
  }

  uint64_t length = static_cast<uint64_t>(out_tensors_[0]->ElementsNum());
  if (length != static_cast<uint64_t>(elements)) {
    MS_LOG(ERROR) << "Linspace output length " << length << " mismatch with requested num " << elements;
    return RET_ERROR;
  }

  if (out_tensors_[0]->data_type() != kNumberTypeFloat32) {
    MS_LOG(ERROR) << "Linspace ft78 only supports fp32 output, got dtype "
                  << static_cast<int>(out_tensors_[0]->data_type());
    return RET_ERROR;
  }

  float start_v = *reinterpret_cast<float *>(in_tensors_[0]->data());
  float end_v = *reinterpret_cast<float *>(in_tensors_[1]->data());

  auto allocator = dsp_runtime_->GetAllocator();
  uint64_t out_device_ptr = allocator->GetDeviceMemPtr(out_tensors_[0]->data());

  uint64_t start_hex = 0;
  std::memcpy(&start_hex, &start_v, sizeof(float));
  uint64_t end_hex = 0;
  std::memcpy(&end_hex, &end_v, sizeof(float));
  SetKernelArg({out_device_ptr, start_hex, end_hex, length});

  auto ret = LinspaceRunFp32();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << this->name() << " Linspace Run failed!";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kDSP, kNumberTypeFloat32, PrimitiveType_LinSpace, DSPKernelCreator<LinspaceDSPKernel>)
}  // namespace mindspore::kernel
