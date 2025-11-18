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

#include "src/litert/kernel/dsp/ft04/range.h"
#include <math.h>
#include <algorithm>
#include <map>
#include <string>
#include "src/litert/kernel/cpu/nnacl_c/range_parameter.h"
#include "src/litert/kernel_registry.h"

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
  auto range_param = reinterpret_cast<const RangeParameter *>(this->op_parameter_);
  start_int_ = range_param->start_;
  delta_int_ = range_param->delta_;
  start_ = static_cast<float>(range_param->start_);
  delta_ = static_cast<float>(range_param->delta_);
  return RET_OK;
}

int RangeDSPKernel::RangeRunFp32() {
  kernel_name_ = "fp_range_s";
  core_mask_ = 0xf;
  return dsp_runtime_->RunKernel(kernel_name_, kernel_args_, core_mask_);
}

int RangeDSPKernel::RangeRunFp16() {
  kernel_name_ = "hp_range_s";
  core_mask_ = 0xf;
  return dsp_runtime_->RunKernel(kernel_name_, kernel_args_, core_mask_);
}

int RangeDSPKernel::RangeRunInt16() {
  kernel_name_ = "i16_range_s";
  core_mask_ = 0xf;
  return dsp_runtime_->RunKernel(kernel_name_, kernel_args_, core_mask_);
}

int RangeDSPKernel::RangeRunInt32() {
  kernel_name_ = "i32_range_s";
  core_mask_ = 0xf;
  return dsp_runtime_->RunKernel(kernel_name_, kernel_args_, core_mask_);
}

int RangeDSPKernel::Run() {
  int ret = -1;
  uint64_t length = out_tensors_[0]->ElementsNum();
  auto allocator = dsp_runtime_->GetAllocator();
  uint64_t out_device_ptr = allocator->GetDeviceMemPtr(out_tensors_[0]->data());
  auto data_type = out_tensors_[0]->data_type();

  if (data_type == kNumberTypeFloat32) {
    uint64_t start_hex = 0;
    memcpy(&start_hex, &start_, sizeof(float));
    uint64_t delta_hex = 0;
    memcpy(&delta_hex, &delta_, sizeof(float));
    SetKernelArg({out_device_ptr, start_hex, delta_hex, static_cast<uint64_t>(length)});
    ret = RangeRunFp32();
  } else if (data_type == kNumberTypeFloat16) {
    uint64_t start_hex = 0;
    memcpy(&start_hex, &start_, sizeof(float));
    uint64_t delta_hex = 0;
    memcpy(&delta_hex, &delta_, sizeof(float));
    SetKernelArg({out_device_ptr, start_hex, delta_hex, static_cast<uint64_t>(length)});
    ret = RangeRunFp16();
  } else if (data_type == kNumberTypeInt16) {
    SetKernelArg({out_device_ptr, static_cast<uint64_t>(start_int_), static_cast<uint64_t>(delta_int_),
                  static_cast<uint64_t>(length)});
    ret = RangeRunInt16();
  } else if (data_type == kNumberTypeInt32) {
    SetKernelArg({out_device_ptr, static_cast<uint64_t>(start_int_), static_cast<uint64_t>(delta_int_),
                  static_cast<uint64_t>(length)});
    ret = RangeRunInt32();
  } else {
    MS_LOG(ERROR) << "unsupported data type: " << static_cast<int>(data_type);
  }

  if (ret != RET_OK) {
    MS_LOG(ERROR) << this->name() << " Run failed! ";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kDSP, kNumberTypeFloat32, PrimitiveType_Range, DSPKernelCreator<RangeDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeFloat16, PrimitiveType_Range, DSPKernelCreator<RangeDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt16, PrimitiveType_Range, DSPKernelCreator<RangeDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt32, PrimitiveType_Range, DSPKernelCreator<RangeDSPKernel>)
}  // namespace mindspore::kernel
