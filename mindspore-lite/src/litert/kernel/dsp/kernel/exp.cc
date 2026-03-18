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

#include "src/litert/kernel/dsp/kernel/exp.h"
#include <cmath>
#include <algorithm>
#include <map>
#include <string>
#include <cstring>
#include "src/litert/kernel/cpu/nnacl_c/exp_parameter.h"
#include "src/litert/kernel_registry.h"
#include "armc/include/operator.h"

using mindspore::kernel::KERNEL_ARCH::kDSP;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ExpFusion;

namespace mindspore::kernel {

int ExpDSPKernel::CheckSpecs() {
  if (in_tensors_.size() != INPUT_TENSOR_SIZE_1) {
    MS_LOG(WARNING) << "in size: " << in_tensors_.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int ExpDSPKernel::Prepare() {
  auto exp_param = reinterpret_cast<const ExpParameter *>(this->op_parameter_);
  scale_ = static_cast<int>(exp_param->scale_);
  float log_base = (exp_param->base_ == -1) ? 1 : std::log(exp_param->base_);
  in_scale_ = exp_param->scale_ * log_base;
  if (exp_param->shift_ == 0) {
    out_scale_ = 1;
  } else {
    if (log_base == 1) {
      out_scale_ = std::exp(exp_param->shift_);
    } else {
      out_scale_ = std::pow(exp_param->base_, exp_param->shift_);
    }
  }
  return RET_OK;
}

int ExpDSPKernel::Run() {
  auto allocator = dsp_runtime_->GetAllocator();
  uint64_t length = in_tensors_[0]->ElementsNum();
  uint64_t x_device_ptr = allocator->GetDeviceMemPtr(in_tensors_[0]->data());
  uint64_t out_device_ptr = allocator->GetDeviceMemPtr(out_tensors_[0]->data());

  auto data_type = in_tensors_[0]->data_type();
  auto mem_type = GetMemType();
  int dtype = static_cast<int>(data_type);
  int mtype = static_cast<int>(mem_type);
  int ret = exp_func(x_device_ptr, out_device_ptr, length, in_scale_, out_scale_, scale_, core_mask_, dtype, mtype);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << this->name() << " Run failed!";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kDSP, kNumberTypeFloat32, PrimitiveType_ExpFusion, DSPKernelCreator<ExpDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt32, PrimitiveType_ExpFusion, DSPKernelCreator<ExpDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeComplex64, PrimitiveType_ExpFusion, DSPKernelCreator<ExpDSPKernel>)

#ifdef SUPPORT_FT04
REG_KERNEL(kDSP, kNumberTypeFloat16, PrimitiveType_ExpFusion, DSPKernelCreator<ExpDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt16, PrimitiveType_ExpFusion, DSPKernelCreator<ExpDSPKernel>)
#endif

#ifdef SUPPORT_FT78
REG_KERNEL(kDSP, kNumberTypeFloat64, PrimitiveType_ExpFusion, DSPKernelCreator<ExpDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt8, PrimitiveType_ExpFusion, DSPKernelCreator<ExpDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeComplex128, PrimitiveType_ExpFusion, DSPKernelCreator<ExpDSPKernel>)
#endif
}  // namespace mindspore::kernel
