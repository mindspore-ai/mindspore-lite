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

#include "src/litert/kernel/dsp/kernel/gather_d.h"
#include "src/litert/kernel_registry.h"
#include "armc/include/operator.h"

using mindspore::kernel::KERNEL_ARCH::kDSP;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_GatherD;

namespace mindspore::kernel {

constexpr int kXIdx = 0;
constexpr int kAxisIdx = 1;
constexpr int kIndicesIdx = 2;
constexpr int kOutIdx = 0;

static bool IsSupportedGatherDDtype(TypeId dtype) {
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

int GatherDDSPKernel::CheckSpecs() {
  if (in_tensors_.size() != INPUT_TENSOR_SIZE_3) {
    MS_LOG(ERROR) << "gather_d kernel input size invalid: in=" << in_tensors_.size();
    return RET_ERROR;
  }
  if (out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(ERROR) << "gather_d kernel output size invalid: out=" << out_tensors_.size();
    return RET_ERROR;
  }
  auto *x = in_tensors_[kXIdx];
  auto *axis_tensor = in_tensors_[kAxisIdx];
  auto *indices = in_tensors_[kIndicesIdx];
  auto *output = out_tensors_[kOutIdx];
  if (!IsSupportedGatherDDtype(x->data_type())) {
    MS_LOG(ERROR) << "gather_d unsupported dtype: " << static_cast<int>(x->data_type());
    return RET_ERROR;
  }
  if (output->data_type() != x->data_type()) {
    MS_LOG(ERROR) << "gather_d output dtype mismatch with input.";
    return RET_ERROR;
  }
  if (indices->data_type() != kNumberTypeInt32) {
    MS_LOG(ERROR) << "gather_d indices must be int32, got " << static_cast<int>(indices->data_type());
    return RET_ERROR;
  }
  if (axis_tensor->data_type() != kNumberTypeInt32 || axis_tensor->ElementsNum() != 1) {
    MS_LOG(ERROR) << "gather_d axis must be an int32 scalar tensor.";
    return RET_ERROR;
  }
  if (x->shape().size() != indices->shape().size() || output->shape() != indices->shape()) {
    MS_LOG(ERROR) << "gather_d shape rank mismatch.";
    return RET_ERROR;
  }
  return RET_OK;
}

int GatherDDSPKernel::Prepare() { return RET_OK; }

int GatherDDSPKernel::Run() {
  auto *x = in_tensors_[kXIdx];
  auto *axis_tensor = in_tensors_[kAxisIdx];
  auto *indices = in_tensors_[kIndicesIdx];
  auto *output = out_tensors_[kOutIdx];

  auto allocator = dsp_runtime_->GetAllocator();
  if (allocator == nullptr) {
    MS_LOG(ERROR) << "GatherDDSPKernel allocator is null.";
    return RET_ERROR;
  }
  if (x->data() == nullptr || indices->data() == nullptr || output->data() == nullptr ||
      axis_tensor->data() == nullptr) {
    MS_LOG(ERROR) << "GatherDDSPKernel input/output data is null.";
    return RET_ERROR;
  }
  uint64_t x_ptr = allocator->GetDeviceMemPtr(x->data());
  uint64_t indices_ptr = allocator->GetDeviceMemPtr(indices->data());
  uint64_t out_ptr = allocator->GetDeviceMemPtr(output->data());
  if (x_ptr == 0 || indices_ptr == 0 || out_ptr == 0) {
    MS_LOG(ERROR) << "GatherDDSPKernel device ptr is null.";
    return RET_ERROR;
  }

  int axis = *reinterpret_cast<const int32_t *>(axis_tensor->data());
  const int ndim = static_cast<int>(x->shape().size());
  if (ndim <= 0) {
    MS_LOG(ERROR) << "GatherDDSPKernel invalid ndim=" << ndim;
    return RET_ERROR;
  }
  if (axis < -ndim || axis >= ndim) {
    MS_LOG(ERROR) << "GatherDDSPKernel invalid axis=" << axis << " ndim=" << ndim;
    return RET_ERROR;
  }
  if (axis < 0) {
    axis += ndim;
  }

  std::vector<int> x_shape = x->shape();
  std::vector<int> indices_shape = indices->shape();
  // Align with nnacl CheckIndexValue_int32_t: reject out-of-range indices before
  // the DSP kernel dereferences them (a device-side out-of-range read has no
  // host error signal). Negative indices are also rejected here and fall back
  // to CPU, where nnacl normalizes them; we must not rewrite them on the host
  // since the DSP kernel's own negative-index behavior is unknown.
  const int max_index = x_shape[axis];
  if (max_index <= 0) {
    MS_LOG(ERROR) << "GatherDDSPKernel invalid axis size=" << max_index;
    return RET_ERROR;
  }
  const auto *index_data = reinterpret_cast<const int32_t *>(indices->data());
  const auto index_count = static_cast<size_t>(indices->ElementsNum());
  for (size_t i = 0; i < index_count; ++i) {
    if (index_data[i] < 0 || index_data[i] >= max_index) {
      MS_LOG(ERROR) << "GatherDDSPKernel index out of range at " << i << ": " << index_data[i];
      return RET_ERROR;
    }
  }
  const int dtype = static_cast<int>(x->data_type());
  const int mtype = static_cast<int>(GetMemType());
  int ret = gather_d_func(x_ptr, out_ptr, indices_ptr, x_shape.data(), ndim, indices_shape.data(), ndim, axis,
                          core_mask_, dtype, mtype);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "GatherDDSPKernel run failed, dtype=" << dtype;
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kDSP, kNumberTypeFloat32, PrimitiveType_GatherD, DSPKernelCreator<GatherDDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt16, PrimitiveType_GatherD, DSPKernelCreator<GatherDDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt32, PrimitiveType_GatherD, DSPKernelCreator<GatherDDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeComplex64, PrimitiveType_GatherD, DSPKernelCreator<GatherDDSPKernel>)

#ifdef SUPPORT_FT04
// FT04 only: fp16.
REG_KERNEL(kDSP, kNumberTypeFloat16, PrimitiveType_GatherD, DSPKernelCreator<GatherDDSPKernel>)
#endif

#ifdef SUPPORT_FT78
// FT78 only: int8, fp64 and complex128. No FT78 on-device test in this change.
REG_KERNEL(kDSP, kNumberTypeInt8, PrimitiveType_GatherD, DSPKernelCreator<GatherDDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeFloat64, PrimitiveType_GatherD, DSPKernelCreator<GatherDDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeComplex128, PrimitiveType_GatherD, DSPKernelCreator<GatherDDSPKernel>)
#endif
}  // namespace mindspore::kernel
