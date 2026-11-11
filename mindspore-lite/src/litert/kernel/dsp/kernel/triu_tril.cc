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

#include "src/litert/kernel/dsp/kernel/triu_tril.h"
#include <climits>
#include "src/litert/kernel_registry.h"
#include "armc/include/operator.h"

using mindspore::kernel::KERNEL_ARCH::kDSP;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Tril;
using mindspore::schema::PrimitiveType_Triu;

namespace mindspore::kernel {

constexpr int kInIdx = 0;
constexpr int kKIdx = 1;
constexpr int kOutIdx = 0;
constexpr int kTriMinDim = 2;

static bool IsSupportedTriuTrilDtype(TypeId dtype) {
  if (dtype == kNumberTypeFloat32 || dtype == kNumberTypeInt16 || dtype == kNumberTypeInt32 ||
      dtype == kNumberTypeComplex64) {
    return true;
  }
#ifdef SUPPORT_FT04
  if (dtype == kNumberTypeFloat16) {
    return true;
  }
#endif
#ifdef SUPPORT_FT78
  if (dtype == kNumberTypeInt8 || dtype == kNumberTypeFloat64 || dtype == kNumberTypeComplex128) {
    return true;
  }
#endif
  return false;
}

int TriuTrilDSPKernelBase::CheckSpecs() {
  if (in_tensors_.size() != INPUT_TENSOR_SIZE_1 && in_tensors_.size() != INPUT_TENSOR_SIZE_2) {
    MS_LOG(ERROR) << OpName() << " kernel input size invalid: in=" << in_tensors_.size();
    return RET_ERROR;
  }
  if (out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(ERROR) << OpName() << " kernel output size invalid: out=" << out_tensors_.size();
    return RET_ERROR;
  }
  auto *input = in_tensors_[kInIdx];
  auto *output = out_tensors_[kOutIdx];
  if (static_cast<int>(input->shape().size()) < kTriMinDim) {
    MS_LOG(ERROR) << OpName() << " input must be at least 2D.";
    return RET_ERROR;
  }
  if (!IsSupportedTriuTrilDtype(input->data_type())) {
    MS_LOG(ERROR) << OpName() << " unsupported dtype: " << static_cast<int>(input->data_type());
    return RET_ERROR;
  }
  if (output->data_type() != input->data_type()) {
    MS_LOG(ERROR) << OpName() << " output dtype mismatch with input.";
    return RET_ERROR;
  }
  if (in_tensors_.size() == INPUT_TENSOR_SIZE_2) {
    auto *k_tensor = in_tensors_[kKIdx];
    const auto k_dtype = k_tensor->data_type();
    if (k_dtype != kNumberTypeInt32 && k_dtype != kNumberTypeInt64) {
      MS_LOG(ERROR) << OpName() << " k tensor must be int32/int64, got " << static_cast<int>(k_dtype);
      return RET_ERROR;
    }
    if (k_tensor->ElementsNum() != 1) {
      MS_LOG(ERROR) << OpName() << " k tensor must have 1 element, got " << k_tensor->ElementsNum();
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int TriuTrilDSPKernelBase::Prepare() { return RET_OK; }

int TriuTrilDSPKernelBase::Run() {
  auto *input = in_tensors_[kInIdx];
  auto *output = out_tensors_[kOutIdx];

  auto allocator = dsp_runtime_->GetAllocator();
  if (allocator == nullptr) {
    MS_LOG(ERROR) << OpName() << " allocator is null.";
    return RET_ERROR;
  }
  uint64_t in_ptr = allocator->GetDeviceMemPtr(input->data());
  uint64_t out_ptr = allocator->GetDeviceMemPtr(output->data());
  if (in_ptr == 0 || out_ptr == 0) {
    MS_LOG(ERROR) << OpName() << " device ptr is null.";
    return RET_ERROR;
  }

  // k comes from an int32/int64 host scalar tensor (default 0), not device memory.
  int k = 0;
  if (in_tensors_.size() == INPUT_TENSOR_SIZE_2) {
    auto *k_tensor = in_tensors_[kKIdx];
    if (k_tensor->data() == nullptr) {
      MS_LOG(ERROR) << OpName() << " k tensor data is null.";
      return RET_ERROR;
    }
    if (k_tensor->data_type() == kNumberTypeInt64) {
      k = static_cast<int>(*reinterpret_cast<const int64_t *>(k_tensor->data()));
    } else {
      k = *reinterpret_cast<const int32_t *>(k_tensor->data());
    }
  }

  const auto &shape = input->shape();
  const int height = shape[shape.size() - 2];
  const int width = shape[shape.size() - 1];
  int64_t out_elems = 1;
  for (size_t i = 0; i + kTriMinDim < shape.size(); ++i) {
    out_elems *= shape[i];
  }
  if (out_elems <= 0 || out_elems > INT_MAX) {
    MS_LOG(ERROR) << OpName() << " out_elems overflow: " << out_elems;
    return RET_ERROR;
  }

  const int dtype = static_cast<int>(input->data_type());
  const int mtype = static_cast<int>(GetMemType());
  int ret = Launch(in_ptr, out_ptr, k, height, width, out_elems, dtype, mtype);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << OpName() << " run failed, dtype=" << dtype;
    return RET_ERROR;
  }
  return RET_OK;
}

int TrilDSPKernel::Launch(uint64_t in_ptr, uint64_t out_ptr, int k, int height, int width, int64_t out_elems, int dtype,
                          int mtype) {
  return tril_func(in_ptr, out_ptr, k, height, width, out_elems, core_mask_, dtype, mtype);
}

int TriuDSPKernel::Launch(uint64_t in_ptr, uint64_t out_ptr, int k, int height, int width, int64_t out_elems, int dtype,
                          int mtype) {
  return triu_func(in_ptr, out_ptr, k, height, width, out_elems, core_mask_, dtype, mtype);
}

#define REG_TRIU_TRIL_KERNELS(prim_type, kernel_class)                            \
  REG_KERNEL(kDSP, kNumberTypeFloat32, prim_type, DSPKernelCreator<kernel_class>) \
  REG_KERNEL(kDSP, kNumberTypeInt16, prim_type, DSPKernelCreator<kernel_class>)   \
  REG_KERNEL(kDSP, kNumberTypeInt32, prim_type, DSPKernelCreator<kernel_class>)   \
  REG_KERNEL(kDSP, kNumberTypeComplex64, prim_type, DSPKernelCreator<kernel_class>)

REG_TRIU_TRIL_KERNELS(PrimitiveType_Tril, TrilDSPKernel)
REG_TRIU_TRIL_KERNELS(PrimitiveType_Triu, TriuDSPKernel)
#undef REG_TRIU_TRIL_KERNELS

#ifdef SUPPORT_FT04
// FT04 only: fp16.
REG_KERNEL(kDSP, kNumberTypeFloat16, PrimitiveType_Tril, DSPKernelCreator<TrilDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeFloat16, PrimitiveType_Triu, DSPKernelCreator<TriuDSPKernel>)
#endif

#ifdef SUPPORT_FT78
// FT78 only: int8, fp64 and complex128. No FT78 on-device test in this change.
REG_KERNEL(kDSP, kNumberTypeInt8, PrimitiveType_Tril, DSPKernelCreator<TrilDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeFloat64, PrimitiveType_Tril, DSPKernelCreator<TrilDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeComplex128, PrimitiveType_Tril, DSPKernelCreator<TrilDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt8, PrimitiveType_Triu, DSPKernelCreator<TriuDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeFloat64, PrimitiveType_Triu, DSPKernelCreator<TriuDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeComplex128, PrimitiveType_Triu, DSPKernelCreator<TriuDSPKernel>)
#endif
}  // namespace mindspore::kernel
