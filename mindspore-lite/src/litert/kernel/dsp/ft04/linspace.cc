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

#include "src/litert/kernel/dsp/ft04/linspace.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include "src/litert/kernel_registry.h"
#include "schema/inner/model_generated.h"

using mindspore::kernel::KERNEL_ARCH::kDSP;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_LinSpace;

namespace mindspore::kernel {

// Inputs are expected to be fp32 scalars (start, end). No fp16-to-fp32 conversion is needed here.

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
  std::string kernel_name = "fp_linspace_s";
  uint64_t core_mask = 0xF;  // all 4 cores
  return dsp_runtime_->RunKernel(kernel_name, kernel_args_, core_mask);
}

int LinspaceDSPKernel::LinspaceRunFp16() {
  std::string kernel_name = "hp_linspace_s";
  uint64_t core_mask = 0xF;  // all 4 cores
  return dsp_runtime_->RunKernel(kernel_name, kernel_args_, core_mask);
}

int LinspaceDSPKernel::Run() {
  int ret = -1;
  auto allocator = dsp_runtime_->GetAllocator();
  uint64_t out_device_ptr = allocator->GetDeviceMemPtr(out_tensors_[0]->data());
  uint64_t length = static_cast<uint64_t>(out_tensors_[0]->ElementsNum());

  // Read start/end from inputs (scalar tensors)
  float start_v = 0.f;
  float end_v = 0.f;

  auto dt = in_tensors_[0]->data_type();
  if (dt != kNumberTypeFloat32) {
    MS_LOG(ERROR) << "Linspace expects fp32 start/end inputs, got dtype: " << static_cast<int>(dt);
    return RET_ERROR;
  }
  start_v = *(reinterpret_cast<float *>(in_tensors_[0]->data()));
  end_v = *(reinterpret_cast<float *>(in_tensors_[1]->data()));

  // Pack args: output_ptr, start(float bits), end(float bits), length
  uint64_t start_hex = 0;
  std::memcpy(&start_hex, &start_v, sizeof(float));
  uint64_t end_hex = 0;
  std::memcpy(&end_hex, &end_v, sizeof(float));
  SetKernelArg({out_device_ptr, start_hex, end_hex, length});

  auto out_dt = out_tensors_[0]->data_type();
  if (out_dt == kNumberTypeFloat32) {
    ret = LinspaceRunFp32();
  } else if (out_dt == kNumberTypeFloat16) {
    ret = LinspaceRunFp16();
  } else {
    MS_LOG(ERROR) << "Linspace unsupported output dtype: " << static_cast<int>(out_dt);
    return RET_ERROR;
  }

  if (ret != RET_OK) {
    MS_LOG(ERROR) << this->name() << " Linspace Run failed!";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kDSP, kNumberTypeFloat32, PrimitiveType_LinSpace, DSPKernelCreator<LinspaceDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeFloat16, PrimitiveType_LinSpace, DSPKernelCreator<LinspaceDSPKernel>)

}  // namespace mindspore::kernel
