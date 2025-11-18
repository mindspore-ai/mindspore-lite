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

#include "src/litert/kernel/dsp/ft78/ragged_range.h"
#include <cstdint>
#include <string>
#include "src/litert/kernel_registry.h"
#include "schema/inner/model_generated.h"

using mindspore::kernel::KERNEL_ARCH::kDSP;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_RaggedRange;

namespace mindspore::kernel {
int RaggedRangeDSPKernel::CheckSpecs() {
  if (in_tensors_.size() != 3 || out_tensors_.size() != 2) {
    MS_LOG(WARNING) << "RaggedRange unexpected io sizes, in: " << in_tensors_.size()
                    << ", out: " << out_tensors_.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int RaggedRangeDSPKernel::Prepare() { return RET_OK; }

int RaggedRangeDSPKernel::CalcRows(int *rows, bool *starts_scalar, bool *limits_scalar, bool *deltas_scalar) {
  if (rows == nullptr || starts_scalar == nullptr || limits_scalar == nullptr || deltas_scalar == nullptr) {
    return RET_ERROR;
  }
  const auto &s0 = in_tensors_[0]->shape();
  const auto &s1 = in_tensors_[1]->shape();
  const auto &s2 = in_tensors_[2]->shape();
  *starts_scalar = s0.empty();
  *limits_scalar = s1.empty();
  *deltas_scalar = s2.empty();
  int non_scalar_rows = -1;
  if (!*starts_scalar) non_scalar_rows = s0[0];
  if (!*limits_scalar) {
    if (non_scalar_rows == -1) {
      non_scalar_rows = s1[0];
    } else if (non_scalar_rows != s1[0]) {
      return RET_ERROR;
    }
  }
  if (!*deltas_scalar) {
    if (non_scalar_rows == -1) {
      non_scalar_rows = s2[0];
    } else if (non_scalar_rows != s2[0]) {
      return RET_ERROR;
    }
  }
  *rows = (non_scalar_rows == -1) ? 1 : non_scalar_rows;
  return RET_OK;
}

int RaggedRangeDSPKernel::RunFp32() {
  kernel_name_ = "fp_raggedrange_s";
  return dsp_runtime_->RunKernel(kernel_name_, kernel_args_, core_mask_);
}

int RaggedRangeDSPKernel::RunFp64() {
  kernel_name_ = "dp_raggedrange_s";
  return dsp_runtime_->RunKernel(kernel_name_, kernel_args_, core_mask_);
}

int RaggedRangeDSPKernel::RunInt32() {
  kernel_name_ = "i32_raggedrange_s";
  return dsp_runtime_->RunKernel(kernel_name_, kernel_args_, core_mask_);
}

int RaggedRangeDSPKernel::RunInt16() {
  kernel_name_ = "i16_raggedrange_s";
  return dsp_runtime_->RunKernel(kernel_name_, kernel_args_, core_mask_);
}

int RaggedRangeDSPKernel::RunInt8() {
  kernel_name_ = "i8_raggedrange_s";
  return dsp_runtime_->RunKernel(kernel_name_, kernel_args_, core_mask_);
}

int RaggedRangeDSPKernel::Run() {
  int rows = 0;
  bool starts_scalar = false;
  bool limits_scalar = false;
  bool deltas_scalar = false;
  auto ret = CalcRows(&rows, &starts_scalar, &limits_scalar, &deltas_scalar);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "RaggedRange rows check failed.";
    return RET_ERROR;
  }

  auto allocator = dsp_runtime_->GetAllocator();
  uint64_t starts_dev = allocator->GetDeviceMemPtr(in_tensors_[0]->data());
  uint64_t limits_dev = allocator->GetDeviceMemPtr(in_tensors_[1]->data());
  uint64_t deltas_dev = allocator->GetDeviceMemPtr(in_tensors_[2]->data());
  uint64_t splits_dev = allocator->GetDeviceMemPtr(out_tensors_[0]->data());
  uint64_t values_dev = allocator->GetDeviceMemPtr(out_tensors_[1]->data());
  uint64_t rows_hex = 0;
  std::memcpy(&rows_hex, &rows, sizeof(int));

  SetKernelArg({starts_dev, limits_dev, deltas_dev, rows_hex, values_dev, splits_dev});

  auto out_dt = out_tensors_[1]->data_type();
  switch (out_dt) {
    case kNumberTypeFloat32:
      return RunFp32();
    case kNumberTypeFloat64:
      return RunFp64();
    case kNumberTypeInt32:
      return RunInt32();
    case kNumberTypeInt16:
      return RunInt16();
    case kNumberTypeInt8:
      return RunInt8();
    default:
      MS_LOG(ERROR) << "RaggedRange unsupported output dtype: " << static_cast<int>(out_dt);
      return RET_ERROR;
  }
}

REG_KERNEL(kDSP, kNumberTypeFloat32, PrimitiveType_RaggedRange, DSPKernelCreator<RaggedRangeDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeFloat64, PrimitiveType_RaggedRange, DSPKernelCreator<RaggedRangeDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt32, PrimitiveType_RaggedRange, DSPKernelCreator<RaggedRangeDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt16, PrimitiveType_RaggedRange, DSPKernelCreator<RaggedRangeDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt8, PrimitiveType_RaggedRange, DSPKernelCreator<RaggedRangeDSPKernel>)
}  // namespace mindspore::kernel
