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

#include "src/litert/kernel/dsp/kernel/apply_momentum.h"

#include <cstdint>
#include <cstring>

#include "src/common/utils.h"
#include "src/litert/kernel/cpu/nnacl_c/nnacl_common.h"
#include "src/litert/kernel/cpu/nnacl_c/fp32_grad/optimizer.h"
#include "src/litert/kernel_registry.h"
#include "armc/include/operator.h"

using mindspore::kernel::KERNEL_ARCH::kDSP;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ApplyMomentum;

namespace mindspore::kernel {

ApplyMomentumDSPKernel::~ApplyMomentumDSPKernel() {
  if (allocator_ != nullptr) {
    if (float_params_buffer_ != nullptr) {
      allocator_->Free(float_params_buffer_);
      float_params_buffer_ = nullptr;
    }
    if (int_params_buffer_ != nullptr) {
      allocator_->Free(int_params_buffer_);
      int_params_buffer_ = nullptr;
    }
  }
  allocator_ = nullptr;
}

int ApplyMomentumDSPKernel::Prepare() {
  allocator_ = dsp_runtime_->GetAllocator();
  if (allocator_ == nullptr) {
    MS_LOG(ERROR) << "Get allocator failed!";
    return RET_ERROR;
  }
  return RET_OK;
}

int ApplyMomentumDSPKernel::CheckSpecs() {
  if (in_tensors_.size() != kApplyMomentumInputTensorSize) {
    MS_LOG(WARNING) << "Input size mismatch: expected " << kApplyMomentumInputTensorSize << ", got "
                    << in_tensors_.size();
    return RET_ERROR;
  }
  if (out_tensors_.size() != kApplyMomentumOutputTensorSize) {
    MS_LOG(WARNING) << "Output size mismatch: expected " << kApplyMomentumOutputTensorSize << ", got "
                    << out_tensors_.size();
    return RET_ERROR;
  }

  auto weight_shape = in_tensors_[kApplyMomentumWeightIdx]->shape();
  if (weight_shape != in_tensors_[kApplyMomentumAccumulateIdx]->shape() ||
      weight_shape != in_tensors_[kApplyMomentumGradientIdx]->shape()) {
    MS_LOG(WARNING) << "Weight, accumulate or gradient tensor shapes mismatch.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ApplyMomentumDSPKernel::Run() {
  auto *weight = in_tensors_[kApplyMomentumWeightIdx];
  int64_t elements_num = weight->ElementsNum();
  auto data_type = weight->data_type();
  auto *param = reinterpret_cast<ApplyMomentumParameter *>(op_parameter_);

  uint64_t weight_device_ptr = allocator_->GetDeviceMemPtr(weight->data());
  uint64_t accumulate_device_ptr = allocator_->GetDeviceMemPtr(in_tensors_[kApplyMomentumAccumulateIdx]->data());
  uint64_t grad_device_ptr = allocator_->GetDeviceMemPtr(in_tensors_[kApplyMomentumGradientIdx]->data());

  // Allocate and fill float params buffer
  size_t float_param_bytes = lite::DataTypeSize(data_type) * kApplyMomentumFloatParamSize;
  float_params_buffer_ = allocator_->Malloc(float_param_bytes);
  if (float_params_buffer_ == nullptr) {
    MS_LOG(ERROR) << "Malloc float params buffer failed!";
    return RET_ERROR;
  }

  const size_t scalar_indices[kApplyMomentumFloatParamSize] = {kApplyMomentumLrIdx, kApplyMomentumMomentumIdx};
  if (data_type == kNumberTypeFloat32) {
    float float_params[kApplyMomentumFloatParamSize] = {0.f};
    for (size_t i = 0; i < kApplyMomentumFloatParamSize; ++i) {
      const lite::Tensor *tensor = in_tensors_[scalar_indices[i]];
      if (tensor->data_type() != kNumberTypeFloat32) {
        MS_LOG(ERROR) << "Scalar tensor type mismatch: expected FP32.";
        return RET_ERROR;
      }
      float_params[i] = *(reinterpret_cast<const float *>(tensor->data()));
    }
    std::memcpy(float_params_buffer_, float_params, float_param_bytes);
  } else if (data_type == kNumberTypeFloat16) {
    uint16_t float16_params[kApplyMomentumFloatParamSize] = {0};
    for (size_t i = 0; i < kApplyMomentumFloatParamSize; ++i) {
      const lite::Tensor *tensor = in_tensors_[scalar_indices[i]];
      if (tensor->data_type() != kNumberTypeFloat16) {
        MS_LOG(ERROR) << "Scalar tensor type mismatch: expected FP16.";
        return RET_ERROR;
      }
      float16_params[i] = *(reinterpret_cast<const uint16_t *>(tensor->data()));
    }
    std::memcpy(float_params_buffer_, float16_params, float_param_bytes);
  } else {
    MS_LOG(ERROR) << "Unsupported data type for float params: " << data_type;
    return RET_ERROR;
  }

  // Allocate and fill int params buffer
  int_params_buffer_ = allocator_->Malloc(sizeof(int32_t) * kApplyMomentumIntParamSize);
  if (int_params_buffer_ == nullptr) {
    MS_LOG(ERROR) << "Malloc int params buffer failed!";
    return RET_ERROR;
  }

  auto *int_params = reinterpret_cast<int32_t *>(int_params_buffer_);
  int_params[0] = 0;
  int_params[1] = static_cast<int32_t>(elements_num);

  uint64_t float_params_device_ptr = allocator_->GetDeviceMemPtr(float_params_buffer_);
  uint64_t int_params_device_ptr = allocator_->GetDeviceMemPtr(int_params_buffer_);

  auto mem_type = GetMemType();
  int dtype = static_cast<int>(data_type);
  int mtype = static_cast<int>(mem_type);

  int ret = apply_momentum_func(weight_device_ptr, accumulate_device_ptr, grad_device_ptr, float_params_device_ptr,
                                int_params_device_ptr, param->use_nesterov_, core_mask_, dtype, mtype);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << this->name() << " Run failed!";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kDSP, kNumberTypeFloat32, PrimitiveType_ApplyMomentum, DSPKernelCreator<ApplyMomentumDSPKernel>)

#ifdef SUPPORT_FT04
REG_KERNEL(kDSP, kNumberTypeFloat16, PrimitiveType_ApplyMomentum, DSPKernelCreator<ApplyMomentumDSPKernel>)
#endif

// FT78 only supports Float32 for ApplyMomentum
}  // namespace mindspore::kernel
