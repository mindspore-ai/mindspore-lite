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

#include "src/litert/kernel/dsp/ft78/range.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
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
  start_i32_ = static_cast<int32_t>(range_param->start_);
  delta_i32_ = static_cast<int32_t>(range_param->delta_);
  start_i16_ = static_cast<int16_t>(range_param->start_);
  delta_i16_ = static_cast<int16_t>(range_param->delta_);
  start_i8_ = static_cast<int8_t>(range_param->start_);
  delta_i8_ = static_cast<int8_t>(range_param->delta_);
  start_ = static_cast<float>(range_param->start_);
  delta_ = static_cast<float>(range_param->delta_);
  start_double_ = static_cast<double>(range_param->start_);
  delta_double_ = static_cast<double>(range_param->delta_);
  return RET_OK;
}

int RangeDSPKernel::RangeRunFp32() {
  kernel_name_ = "fp_range_s";
  core_mask_ = 0xff;
  return dsp_runtime_->RunKernel(kernel_name_, kernel_args_, core_mask_);
}

int RangeDSPKernel::RangeRunFp64() {
  kernel_name_ = "dp_range_s";
  core_mask_ = 0xff;
  return dsp_runtime_->RunKernel(kernel_name_, kernel_args_, core_mask_);
}

int RangeDSPKernel::RangeRunInt8() {
  kernel_name_ = "i8_range_s";
  core_mask_ = 0xff;
  return dsp_runtime_->RunKernel(kernel_name_, kernel_args_, core_mask_);
}

int RangeDSPKernel::RangeRunInt16() {
  kernel_name_ = "i16_range_s";
  core_mask_ = 0xff;
  return dsp_runtime_->RunKernel(kernel_name_, kernel_args_, core_mask_);
}

int RangeDSPKernel::RangeRunInt32() {
  kernel_name_ = "i32_range_s";
  core_mask_ = 0xff;
  return dsp_runtime_->RunKernel(kernel_name_, kernel_args_, core_mask_);
}

int RangeDSPKernel::Run() {
  int ret = RET_ERROR;
  uint64_t length = out_tensors_[0]->ElementsNum();
  auto allocator = dsp_runtime_->GetAllocator();
  uint64_t out_device_ptr = allocator->GetDeviceMemPtr(out_tensors_[0]->data());
  auto data_type = out_tensors_[0]->data_type();

  if (data_type == kNumberTypeFloat32) {
    uint64_t start_hex = 0;
    uint64_t delta_hex = 0;
    memcpy(&start_hex, &start_, sizeof(float));
    memcpy(&delta_hex, &delta_, sizeof(float));
    SetKernelArg({out_device_ptr, start_hex, delta_hex, static_cast<uint64_t>(length)});
    ret = RangeRunFp32();
  } else if (data_type == kNumberTypeFloat64) {
    uint64_t start_hex = 0;
    uint64_t delta_hex = 0;
    memcpy(&start_hex, &start_double_, sizeof(double));
    memcpy(&delta_hex, &delta_double_, sizeof(double));
    SetKernelArg({out_device_ptr, start_hex, delta_hex, static_cast<uint64_t>(length)});
    ret = RangeRunFp64();
  } else if (data_type == kNumberTypeInt8) {
    SetKernelArg({out_device_ptr, static_cast<uint64_t>(static_cast<uint8_t>(start_i8_)),
                  static_cast<uint64_t>(static_cast<uint8_t>(delta_i8_)), static_cast<uint64_t>(length)});
    ret = RangeRunInt8();
  } else if (data_type == kNumberTypeInt16) {
    SetKernelArg({out_device_ptr, static_cast<uint64_t>(static_cast<uint16_t>(start_i16_)),
                  static_cast<uint64_t>(static_cast<uint16_t>(delta_i16_)), static_cast<uint64_t>(length)});
    ret = RangeRunInt16();
  } else if (data_type == kNumberTypeInt32) {
    SetKernelArg({out_device_ptr, static_cast<uint64_t>(static_cast<int64_t>(start_i32_)),
                  static_cast<uint64_t>(static_cast<int64_t>(delta_i32_)), static_cast<uint64_t>(length)});
    ret = RangeRunInt32();
  } else {
    MS_LOG(ERROR) << "unsupported data type: " << static_cast<int>(data_type);
    return RET_ERROR;
  }

  if (ret != RET_OK) {
    MS_LOG(ERROR) << this->name() << " Run failed! ";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kDSP, kNumberTypeFloat32, PrimitiveType_Range, DSPKernelCreator<RangeDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeFloat64, PrimitiveType_Range, DSPKernelCreator<RangeDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt8, PrimitiveType_Range, DSPKernelCreator<RangeDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt16, PrimitiveType_Range, DSPKernelCreator<RangeDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt32, PrimitiveType_Range, DSPKernelCreator<RangeDSPKernel>)
}  // namespace mindspore::kernel
