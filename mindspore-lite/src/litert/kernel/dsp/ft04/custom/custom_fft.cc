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

#include "src/litert/kernel/dsp/ft04/custom/custom_fft.h"
#include <vector>
#include "src/common/utils.h"

namespace mindspore::lite {
namespace {
const auto kComplex64 = DataType::kNumberTypeComplex64;
constexpr int kFFTMaxSize = 4 * 1024 * 1024;
}  // namespace

CustomFFTKernel::~CustomFFTKernel() {
  if (w_ptr_ != nullptr) {
    allocator_->Free(w_ptr_);
  }
}

int CustomFFTKernel::Prepare() {
  int ret = -1;
  core_mask_ = 0xf;
  allocator_ = dsp_runtime_->GetAllocator();
  // The w_ptr_ space size of FFT operator of DSP needs (n+48)*sizeof(std::complex<float>) bytes.
  w_ptr_ = allocator_->Malloc((length_ + 48) * sizeof(std::complex<float>));
  if (w_ptr_ == nullptr) {
    MS_LOG(ERROR) << "Malloc w ptr failed!";
    return kLiteError;
  }
  w_device_ptr_ = allocator_->GetDeviceMemPtr(w_ptr_);
  SetKernelArg({w_device_ptr_, static_cast<uint64_t>(length_), 0});
  kernel_name_ = "c64_getw_s";
  ret = dsp_runtime_->RunKernel(kernel_name_, kernel_args_, core_mask_);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Init w failed!";
    return kLiteError;
  }
  return kSuccess;
}

int CustomFFTKernel::CheckSpecs(const std::vector<MSTensor> &inputs, const std::vector<MSTensor> &outputs) {
  length_ = inputs.front().ElementNum();
  if (length_ > kFFTMaxSize) {
    MS_LOG(ERROR) << "FFT input size is too large!";
    return kLiteError;
  }
  if (!IsPowerOfTwo(length_)) {
    MS_LOG(ERROR) << "FFT input size is not power of 2!";
    return kLiteError;
  }
  auto shape_size = inputs.front().Shape().size();
  if (shape_size > 1) {
    MS_LOG(ERROR) << "FFT input shape size is not 1!";
    return kLiteError;
  }
  return kSuccess;
}

int CustomFFTKernel::Run() {
  int ret = -1;
  core_mask_ = 0xf;
  kernel_name_ = "c64_fft_nobitrev_s";
  auto tmp_ptr = allocator_->Malloc(2 * length_ * sizeof(std::complex<float>));
  if (tmp_ptr == nullptr) {
    MS_LOG(ERROR) << "Malloc tmp ptr failed!";
    return kLiteError;
  }
  auto tmp_device_ptr = allocator_->GetDeviceMemPtr(tmp_ptr);
  auto input_device_ptr = allocator_->GetDeviceMemPtr(in_tensors_.front()->data());
  auto output_device_ptr = allocator_->GetDeviceMemPtr(out_tensors_.front()->data());
  auto out_tmp_ptr = allocator_->Malloc(length_ * sizeof(std::complex<float>));
  if (out_tmp_ptr == nullptr) {
    MS_LOG(ERROR) << "Malloc out tmp ptr failed!";
    return kLiteError;
  }
  auto out_tmp_device_ptr = allocator_->GetDeviceMemPtr(out_tmp_ptr);
  SetKernelArg({input_device_ptr, w_device_ptr_, out_tmp_device_ptr, static_cast<uint64_t>(length_), tmp_device_ptr});
  ret = dsp_runtime_->RunKernel(kernel_name_, kernel_args_, core_mask_);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << this->name() << " Run failed! ";
    allocator_->Free(tmp_ptr);
    allocator_->Free(out_tmp_ptr);
    return kLiteError;
  }
  kernel_name_ = "c64_bitrev_s";
  SetKernelArg({out_tmp_device_ptr, static_cast<uint64_t>(length_), output_device_ptr, tmp_device_ptr});
  ret = dsp_runtime_->RunKernel(kernel_name_, kernel_args_, core_mask_);
  allocator_->Free(tmp_ptr);
  allocator_->Free(out_tmp_ptr);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << this->name() << " Run failed! ";
    return kLiteError;
  }
  MS_LOG(DEBUG) << this->name() << " Run success! ";
  return kSuccess;
}

REGISTER_CUSTOM_KERNEL(DSP, FTMatrix, kComplex64, Custom_FT_FFT, CustomKernelCreator<CustomFFTKernel>)
}  // namespace mindspore::lite
