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

#include "src/litert/kernel/dsp/kernel/linspace.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include "src/litert/kernel_registry.h"
#include "schema/inner/model_generated.h"
#include "armc/include/operator.h"

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
  auto dt = in_tensors_[0]->data_type();
  if (dt != kNumberTypeFloat32) {
    MS_LOG(WARNING) << "Linspace expects fp32 start/end inputs, got dtype: " << static_cast<int>(dt);
    return RET_ERROR;
  }
  return RET_OK;
}

int LinspaceDSPKernel::Prepare() { return RET_OK; }

int LinspaceDSPKernel::Run() {
  auto allocator = dsp_runtime_->GetAllocator();
  uint64_t out_device_ptr = allocator->GetDeviceMemPtr(out_tensors_[0]->data());
  uint64_t length = static_cast<uint64_t>(out_tensors_[0]->ElementsNum());

  // Read start/end from inputs (scalar tensors)
  float start = *(reinterpret_cast<float *>(in_tensors_[0]->data()));
  float end = *(reinterpret_cast<float *>(in_tensors_[1]->data()));

  auto data_type = out_tensors_[0]->data_type();
  auto mem_type = GetMemType();
  int dtype = static_cast<int>(data_type);
  int mtype = static_cast<int>(mem_type);

  int ret = linspace_func(out_device_ptr, start, end, length, core_mask_, dtype, mtype);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << this->name() << " Run failed!";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kDSP, kNumberTypeFloat32, PrimitiveType_LinSpace, DSPKernelCreator<LinspaceDSPKernel>)

#ifdef SUPPORT_FT04
REG_KERNEL(kDSP, kNumberTypeFloat16, PrimitiveType_LinSpace, DSPKernelCreator<LinspaceDSPKernel>)
#endif

}  // namespace mindspore::kernel
