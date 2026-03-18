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

#include "src/litert/kernel/dsp/kernel/custom/fft.h"
#include <vector>
#include <string>
#include "src/common/utils.h"
#include "armc/include/operator.h"

namespace mindspore::lite {

namespace {
constexpr size_t kMinDimension = 1;
constexpr size_t kMaxDimension = 2;
constexpr size_t kColIndex = 0;
constexpr size_t kRowIndex = 1;
constexpr size_t kMemoryMultiplierForTwiddle = 2;
constexpr size_t kMemoryMultiplierForTemp = 2;
constexpr int kFFTDirectionForward = 0;
const auto kComplex64 = DataType::kNumberTypeComplex64;
}  // namespace

FFTKernel::~FFTKernel() {
  if (allocator_ != nullptr) {
    if (w_ptr_ != nullptr) {
      allocator_->Free(w_ptr_);
    }
    if (temp_fft_ptr_ != nullptr) {
      allocator_->Free(temp_fft_ptr_);
    }
  }
  allocator_ = nullptr;
}

int FFTKernel::Prepare() {
  ParseAttrData();
  allocator_ = dsp_runtime_->GetAllocator();
  if (allocator_ == nullptr) {
    MS_LOG(ERROR) << "Get allocator failed!";
    return kLiteError;
  }
  auto data_size = inputs_.front().DataSize();
  size_t w_buffer_size = kMemoryMultiplierForTwiddle * data_size;
  w_ptr_ = allocator_->Malloc(w_buffer_size);
  if (w_ptr_ == nullptr) {
    MS_LOG(ERROR) << "Malloc w ptr failed!";
    return kLiteError;
  }
  w_device_ptr_ = allocator_->GetDeviceMemPtr(w_ptr_);
  if (w_device_ptr_ == 0) {
    MS_LOG(ERROR) << "Get w device ptr failed!";
    return kLiteError;
  }

  // Initialize w using twiddle_factor_func for FFT (direction_flag = 0)
  auto data_type = inputs_.front().DataType();
  auto mem_type = GetMemType();
  int dtype = static_cast<int>(data_type);
  int mtype = static_cast<int>(mem_type);
  int ret = twiddle_factor_func(w_device_ptr_, length_, kFFTDirectionForward, core_mask_, dtype, mtype);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Init w failed!";
    return kLiteError;
  }

  size_t temp_buffer_size = kMemoryMultiplierForTemp * data_size;
  temp_fft_ptr_ = allocator_->Malloc(temp_buffer_size);
  if (temp_fft_ptr_ == nullptr) {
    MS_LOG(ERROR) << "Malloc temp ptr failed!";
    return kLiteError;
  }

  return kSuccess;
}

int FFTKernel::CheckSpecs(const std::vector<MSTensor> &inputs, const std::vector<MSTensor> &outputs) {
  auto shape_size = inputs.front().Shape().size();
  if (shape_size == kMinDimension) {
    length_ = inputs.front().ElementNum();
  } else if (shape_size == kMaxDimension) {
    // only row calculation is supported
    length_ = inputs.front().Shape()[kRowIndex];
  } else {
    MS_LOG(ERROR) << "FFT input shape is not supported!";
    return kLiteError;
  }
  if (!IsPowerOfTwo(length_)) {
    MS_LOG(ERROR) << "FFT input size is not power of 2!";
    return kLiteError;
  }
  return kSuccess;
}

int FFTKernel::Run() {
  auto input_device_ptr = allocator_->GetDeviceMemPtr(in_tensors_.front()->data());
  if (input_device_ptr == 0) {
    MS_LOG(ERROR) << "Get input device ptr failed!";
    return kLiteError;
  }
  auto output_device_ptr = allocator_->GetDeviceMemPtr(out_tensors_.front()->data());
  if (output_device_ptr == 0) {
    MS_LOG(ERROR) << "Get output device ptr failed!";
    return kLiteError;
  }
  temp_fft_device_ptr_ = allocator_->GetDeviceMemPtr(temp_fft_ptr_);
  if (temp_fft_device_ptr_ == 0) {
    MS_LOG(ERROR) << "Get temp fft device ptr failed!";
    return kLiteError;
  }
  auto input_shape_size = in_tensors_.front()->shape().size();
  int batch_count = 1;
  if (input_shape_size == kMaxDimension) {
    batch_count = in_tensors_.front()->shape()[kColIndex];
  }

  auto data_type = inputs_.front().DataType();
  auto mem_type = GetMemType();
  int dtype = static_cast<int>(data_type);
  int mtype = static_cast<int>(mem_type);

  int ret = fft_func(input_device_ptr, w_device_ptr_, output_device_ptr, batch_count, length_, temp_fft_device_ptr_,
                     core_mask_, dtype, mtype);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << this->name() << " Run failed! ";
    return kLiteError;
  }
  return kSuccess;
}

REGISTER_CUSTOM_KERNEL(DSP, FTMatrix, kComplex64, Custom_FT_FFT, CustomKernelCreator<FFTKernel>)
}  // namespace mindspore::lite
