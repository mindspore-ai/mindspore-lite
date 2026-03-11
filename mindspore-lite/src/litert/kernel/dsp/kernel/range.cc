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

#include "src/litert/kernel/dsp/kernel/range.h"
#include <cmath>
#include <algorithm>
#include <map>
#include <string>
#include <cstring>
#include "src/litert/kernel_registry.h"
#include "armc/include/operator.h"

using mindspore::kernel::KERNEL_ARCH::kDSP;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Range;

namespace mindspore::kernel {

int RangeDSPKernel::CheckSpecs() {
  if (out_tensors_.size() != 1) {
    MS_LOG(WARNING) << "out size: " << out_tensors_.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int RangeDSPKernel::Prepare() {
  range_param_ = reinterpret_cast<RangeParameter *>(this->op_parameter_);
  return RET_OK;
}

int RangeDSPKernel::Run() {
  auto allocator = dsp_runtime_->GetAllocator();
  uint64_t out_device_ptr = allocator->GetDeviceMemPtr(out_tensors_[0]->data());
  uint64_t length = static_cast<uint64_t>(out_tensors_[0]->ElementsNum());

  auto data_type = out_tensors_[0]->data_type();
  auto mem_type = GetMemType();
  int dtype = static_cast<int>(data_type);
  int mtype = static_cast<int>(mem_type);

  int ret = range_func(out_device_ptr, range_param_->start_, range_param_->delta_, length, core_mask_, dtype, mtype);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << this->name() << " Run failed!";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kDSP, kNumberTypeFloat32, PrimitiveType_Range, DSPKernelCreator<RangeDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt16, PrimitiveType_Range, DSPKernelCreator<RangeDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt32, PrimitiveType_Range, DSPKernelCreator<RangeDSPKernel>)

#ifdef SUPPORT_FT04
REG_KERNEL(kDSP, kNumberTypeFloat16, PrimitiveType_Range, DSPKernelCreator<RangeDSPKernel>)
#endif

#ifdef SUPPORT_FT78
REG_KERNEL(kDSP, kNumberTypeFloat64, PrimitiveType_Range, DSPKernelCreator<RangeDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt8, PrimitiveType_Range, DSPKernelCreator<RangeDSPKernel>)
#endif
}  // namespace mindspore::kernel
