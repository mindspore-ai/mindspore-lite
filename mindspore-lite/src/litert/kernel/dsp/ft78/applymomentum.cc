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

#include "src/litert/kernel/dsp/ft78/applymomentum.h"

#include <cstdint>
#include <cstring>
#include <limits>

#include "src/common/utils.h"
#include "src/litert/kernel/cpu/nnacl_c/nnacl_common.h"
#include "src/litert/kernel/cpu/nnacl_c/fp32_grad/optimizer.h"
#include "src/litert/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH::kDSP;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ApplyMomentum;

namespace mindspore::kernel {
int ApplyMomentumDSPKernel::Prepare() { return RET_OK; }

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

  auto data_type = in_tensors_[kApplyMomentumWeightIdx]->data_type();
  if (data_type != kNumberTypeFloat32) {
    MS_LOG(WARNING) << "Unsupported data type: " << static_cast<int>(data_type);
    return RET_ERROR;
  }

  auto check_scalar = [&](const lite::Tensor *tensor) -> bool {
    if (tensor == nullptr || tensor->ElementsNum() != 1) {
      return false;
    }
    auto tensor_type = tensor->data_type();
    return tensor_type == kNumberTypeFloat32;
  };

  if (!check_scalar(in_tensors_[kApplyMomentumLrIdx]) || !check_scalar(in_tensors_[kApplyMomentumMomentumIdx])) {
    MS_LOG(WARNING) << "Optimizer scalar tensors are invalid.";
    return RET_ERROR;
  }

  return RET_OK;
}

int ApplyMomentumDSPKernel::ApplyMomentumRunFp32() {
  kernel_name_ = "fp_applymomentum_s";
  core_mask_ = 0xf;
  return dsp_runtime_->RunKernel(kernel_name_, kernel_args_, core_mask_);
}

int ApplyMomentumDSPKernel::Run() {
  auto allocator = dsp_runtime_->GetAllocator();

  auto *weight = in_tensors_[kApplyMomentumWeightIdx];

  int64_t elements_num = weight->ElementsNum();

  auto *param = reinterpret_cast<ApplyMomentumParameter *>(op_parameter_);
  if (param == nullptr) {
    MS_LOG(ERROR) << "ApplyMomentum parameter is nullptr.";
    return RET_ERROR;
  }

  uint64_t weight_device_ptr = allocator->GetDeviceMemPtr(weight->data());
  uint64_t accumulate_device_ptr = allocator->GetDeviceMemPtr(in_tensors_[kApplyMomentumAccumulateIdx]->data());
  uint64_t grad_device_ptr = allocator->GetDeviceMemPtr(in_tensors_[kApplyMomentumGradientIdx]->data());

  size_t float_param_bytes = sizeof(float) * kApplyMomentumFloatParamSize;

  void *float_params_buffer = allocator->Malloc(float_param_bytes);
  if (float_params_buffer == nullptr) {
    MS_LOG(ERROR) << "Failed to allocate float parameter buffer.";
    return RET_ERROR;
  }
  auto free_float_buffer = [&]() {
    if (float_params_buffer != nullptr) {
      allocator->Free(float_params_buffer);
      float_params_buffer = nullptr;
    }
  };

  const size_t scalar_indices[kApplyMomentumFloatParamSize] = {kApplyMomentumLrIdx, kApplyMomentumMomentumIdx};
  float float_params[kApplyMomentumFloatParamSize] = {0.f};
  for (size_t i = 0; i < kApplyMomentumFloatParamSize; ++i) {
    const lite::Tensor *tensor = in_tensors_[scalar_indices[i]];
    if (tensor == nullptr || tensor->data() == nullptr) {
      free_float_buffer();
      MS_LOG(ERROR) << "Optimizer scalar tensor is invalid.";
      return RET_ERROR;
    }
    if (tensor->data_type() != kNumberTypeFloat32) {
      free_float_buffer();
      MS_LOG(ERROR) << "Scalar tensor type mismatch: expected FP32.";
      return RET_ERROR;
    }
    float_params[i] = *(reinterpret_cast<const float *>(tensor->data()));
  }
  std::memcpy(float_params_buffer, float_params, float_param_bytes);

  uint64_t float_params_device_ptr = allocator->GetDeviceMemPtr(float_params_buffer);

  void *int_params_buffer = allocator->Malloc(sizeof(int32_t) * kApplyMomentumIntParamSize);

  auto free_all_buffers = [&]() {
    if (float_params_buffer != nullptr) {
      allocator->Free(float_params_buffer);
      float_params_buffer = nullptr;
    }
    if (int_params_buffer != nullptr) {
      allocator->Free(int_params_buffer);
      int_params_buffer = nullptr;
    }
  };

  auto *int_params = reinterpret_cast<int32_t *>(int_params_buffer);
  int_params[0] = 0;
  int_params[1] = static_cast<int32_t>(elements_num);

  uint64_t int_params_device_ptr = allocator->GetDeviceMemPtr(int_params_buffer);
  if (int_params_device_ptr == 0) {
    free_all_buffers();
    MS_LOG(ERROR) << "Failed to obtain device pointer for int parameter buffer.";
    return RET_ERROR;
  }

  int use_nesterov = param->use_nesterov_ ? 1 : 0;
  SetKernelArg({weight_device_ptr, accumulate_device_ptr, grad_device_ptr, float_params_device_ptr,
                int_params_device_ptr, static_cast<uint64_t>(use_nesterov)});

  int ret = ApplyMomentumRunFp32();

  free_all_buffers();

  if (ret != RET_OK) {
    MS_LOG(ERROR) << this->name() << " Run failed! ";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kDSP, kNumberTypeFloat32, PrimitiveType_ApplyMomentum, DSPKernelCreator<ApplyMomentumDSPKernel>)
}  // namespace mindspore::kernel
