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

#include "src/litert/kernel/dsp/kernel/fill.h"
#include <utility>
#include <climits>
#include "src/litert/kernel_registry.h"
#include "armc/include/operator.h"

using mindspore::kernel::KERNEL_ARCH::kDSP;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Fill;
using mindspore::schema::PrimitiveType_FillV2;

namespace mindspore::kernel {

constexpr int kOutIdx = 0;
// FillV2: in[0]=shape, in[1]=value; Fill: in[0]=value, in[1]=shape.
// See fill_infer.c / fillv2_infer.c for the source of truth.
constexpr int kFillV2ShapeIdx = 0;
constexpr int kFillV2ValueIdx = 1;
constexpr int kFillShapeIdx = 1;
constexpr int kFillValueIdx = 0;

static bool IsSupportedFillDtype(TypeId dtype) {
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

// Return (shape_idx, value_idx) matching the primitive's infer implementation.
static std::pair<int, int> GetIndices(schema::PrimitiveType prim) {
  if (prim == schema::PrimitiveType_FillV2) {
    return {kFillV2ShapeIdx, kFillV2ValueIdx};
  }
  return {kFillShapeIdx, kFillValueIdx};
}

int FillDSPKernel::CheckSpecs() {
  if (in_tensors_.size() != INPUT_TENSOR_SIZE_2) {
    MS_LOG(ERROR) << "fill kernel input size invalid: in=" << in_tensors_.size();
    return RET_ERROR;
  }
  if (out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(ERROR) << "fill kernel output size invalid: out=" << out_tensors_.size();
    return RET_ERROR;
  }
  const auto indices = GetIndices(type());
  auto *shape_tensor = in_tensors_[indices.first];
  auto *value_tensor = in_tensors_[indices.second];
  auto *output = out_tensors_[kOutIdx];
  if (!IsSupportedFillDtype(output->data_type())) {
    MS_LOG(ERROR) << "fill unsupported dtype: " << static_cast<int>(output->data_type());
    return RET_ERROR;
  }
  if (value_tensor->data_type() != output->data_type()) {
    MS_LOG(ERROR) << "fill value dtype mismatch with output.";
    return RET_ERROR;
  }
  if (value_tensor->ElementsNum() != 1) {
    MS_LOG(ERROR) << "fill value must be a scalar, elems=" << value_tensor->ElementsNum();
    return RET_ERROR;
  }
  const auto shape_dtype = shape_tensor->data_type();
  if (shape_dtype != kNumberTypeInt32 && shape_dtype != kNumberTypeInt64) {
    MS_LOG(ERROR) << "fill shape tensor must be int32/int64, got " << static_cast<int>(shape_dtype);
    return RET_ERROR;
  }
  if (static_cast<int>(shape_tensor->shape().size()) != 1) {
    MS_LOG(ERROR) << "fill shape tensor must be 1D.";
    return RET_ERROR;
  }
  return RET_OK;
}

int FillDSPKernel::Prepare() { return RET_OK; }

int FillDSPKernel::Run() {
  auto *value_tensor = in_tensors_[GetIndices(type()).second];
  auto *output = out_tensors_[kOutIdx];

  auto allocator = dsp_runtime_->GetAllocator();
  if (allocator == nullptr) {
    MS_LOG(ERROR) << "FillDSPKernel allocator is null.";
    return RET_ERROR;
  }
  if (value_tensor->data() == nullptr || output->data() == nullptr) {
    MS_LOG(ERROR) << "FillDSPKernel input/output data is null.";
    return RET_ERROR;
  }
  uint64_t value_ptr = allocator->GetDeviceMemPtr(value_tensor->data());
  uint64_t out_ptr = allocator->GetDeviceMemPtr(output->data());
  if (value_ptr == 0 || out_ptr == 0) {
    MS_LOG(ERROR) << "FillDSPKernel device ptr is null.";
    return RET_ERROR;
  }

  const auto elems = output->ElementsNum();
  if (elems <= 0 || elems > INT_MAX) {
    MS_LOG(ERROR) << "FillDSPKernel invalid length: " << elems;
    return RET_ERROR;
  }

  const int dtype = static_cast<int>(output->data_type());
  const int mtype = static_cast<int>(GetMemType());
  int ret = fill_func(out_ptr, value_ptr, static_cast<int>(elems), core_mask_, dtype, mtype);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "FillDSPKernel run failed, dtype=" << dtype;
    return RET_ERROR;
  }
  return RET_OK;
}

#define REG_FILL_KERNELS(prim_type)                                                \
  REG_KERNEL(kDSP, kNumberTypeFloat32, prim_type, DSPKernelCreator<FillDSPKernel>) \
  REG_KERNEL(kDSP, kNumberTypeInt16, prim_type, DSPKernelCreator<FillDSPKernel>)   \
  REG_KERNEL(kDSP, kNumberTypeInt32, prim_type, DSPKernelCreator<FillDSPKernel>)   \
  REG_KERNEL(kDSP, kNumberTypeComplex64, prim_type, DSPKernelCreator<FillDSPKernel>)

REG_FILL_KERNELS(PrimitiveType_Fill)
REG_FILL_KERNELS(PrimitiveType_FillV2)
#undef REG_FILL_KERNELS

#ifdef SUPPORT_FT04
// FT04 only: fp16.
REG_KERNEL(kDSP, kNumberTypeFloat16, PrimitiveType_Fill, DSPKernelCreator<FillDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeFloat16, PrimitiveType_FillV2, DSPKernelCreator<FillDSPKernel>)
#endif

#ifdef SUPPORT_FT78
// FT78 only: int8, fp64 and complex128. No FT78 on-device test in this change.
REG_KERNEL(kDSP, kNumberTypeInt8, PrimitiveType_Fill, DSPKernelCreator<FillDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeFloat64, PrimitiveType_Fill, DSPKernelCreator<FillDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeComplex128, PrimitiveType_Fill, DSPKernelCreator<FillDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt8, PrimitiveType_FillV2, DSPKernelCreator<FillDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeFloat64, PrimitiveType_FillV2, DSPKernelCreator<FillDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeComplex128, PrimitiveType_FillV2, DSPKernelCreator<FillDSPKernel>)
#endif
}  // namespace mindspore::kernel
