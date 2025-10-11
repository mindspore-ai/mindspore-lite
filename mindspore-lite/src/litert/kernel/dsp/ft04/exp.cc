
/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/dsp/ft04/exp.h"
#include <math.h>
#include <algorithm>
#include <map>
#include <string>
#include "src/litert/kernel/cpu/nnacl_c/exp_parameter.h"
#include "src/litert/kernel_registry.h"

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
  float log_base = (exp_param->base_ == -1) ? 1 : logf(exp_param->base_);
  in_scale_ = exp_param->scale_ * log_base;
  if (exp_param->shift_ == 0) {
    out_scale_ = 1;
  } else {
    if (log_base == 1) {
      out_scale_ = expf(exp_param->shift_);
    } else {
      out_scale_ = powf(exp_param->base_, exp_param->shift_);
    }
  }
  return RET_OK;
}

int ExpDSPKernel::ExpRunFp32() {
  kernel_name_ = "fp_exp_s";
  core_mask_ = 0xf;
  return dsp_runtime_->RunKernel(kernel_name_, kernel_args_, core_mask_);
}

int ExpDSPKernel::ExpRunFp16() {
  kernel_name_ = "hp_exp_s";
  core_mask_ = 0xf;
  return dsp_runtime_->RunKernel(kernel_name_, kernel_args_, core_mask_);
}

int ExpDSPKernel::ExpRunInt16() {
  kernel_name_ = "i16_exp_s";
  core_mask_ = 0xf;
  return dsp_runtime_->RunKernel(kernel_name_, kernel_args_, core_mask_);
}

int ExpDSPKernel::ExpRunInt32() {
  kernel_name_ = "i32_exp_s";
  core_mask_ = 0xf;
  return dsp_runtime_->RunKernel(kernel_name_, kernel_args_, core_mask_);
}

int ExpDSPKernel::ExpRunComplex64() {
  kernel_name_ = "c64_exp_s";
  core_mask_ = 0xf;
  return dsp_runtime_->RunKernel(kernel_name_, kernel_args_, core_mask_);
}

int ExpDSPKernel::Run() {
  int ret = -1;
  MS_LOG(DEBUG) << this->name() << " Running! ";
  uint64_t length = in_tensors_[0]->ElementsNum();
  auto allocator = dsp_runtime_->GetAllocator();
  uint64_t x_device_ptr = allocator->GetDeviceMemPtr(in_tensors_[0]->data());
  uint64_t out_device_ptr = allocator->GetDeviceMemPtr(out_tensors_[0]->data());
  uint64_t in_scale_hex = 0, out_scale_hex = 0;
  memcpy(&in_scale_hex, &in_scale_, sizeof(float));
  memcpy(&out_scale_hex, &out_scale_, sizeof(float));
  SetKernelArg({x_device_ptr, out_device_ptr, length, in_scale_hex, out_scale_hex, static_cast<uint64_t>(scale_)});
  auto data_type = in_tensors_[0]->data_type();
  if (data_type == kNumberTypeFloat32) {
    ret = ExpRunFp32();
  } else if (data_type == kNumberTypeFloat16) {
    ret = ExpRunFp16();
  } else if (data_type == kNumberTypeInt16) {
    ret = ExpRunInt16();
  } else if (data_type == kNumberTypeInt32) {
    ret = ExpRunInt32();
  } else if (data_type == kNumberTypeComplex64) {
    ret = ExpRunComplex64();
  } else {
    MS_LOG(ERROR) << "unsupported data type: " << static_cast<int>(data_type);
  }
  if (ret != RET_OK) {
    MS_LOG(ERROR) << this->name() << " Run failed! ";
    return ret;
  }
  MS_LOG(DEBUG) << this->name() << " Run success! ";
  return RET_OK;
}

REG_KERNEL(kDSP, kNumberTypeFloat32, PrimitiveType_ExpFusion, DSPKernelCreator<ExpDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt16, PrimitiveType_ExpFusion, DSPKernelCreator<ExpDSPKernel>);
REG_KERNEL(kDSP, kNumberTypeInt32, PrimitiveType_ExpFusion, DSPKernelCreator<ExpDSPKernel>);
REG_KERNEL(kDSP, kNumberTypeComplex64, PrimitiveType_ExpFusion, DSPKernelCreator<ExpDSPKernel>);
REG_KERNEL(kDSP, kNumberTypeFloat16, PrimitiveType_ExpFusion, DSPKernelCreator<ExpDSPKernel>);
}  // namespace mindspore::kernel
