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

#include "src/litert/kernel/dsp/kernel/cast.h"
#include <algorithm>
#include <utility>
#include <vector>
#include "src/litert/kernel_registry.h"
#include "armc/include/operator.h"

using mindspore::kernel::KERNEL_ARCH::kDSP;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Cast;

namespace mindspore::kernel {

constexpr int kInIdx = 0;
constexpr int kOutIdx = 0;

// Supported (src, dst) pairs come from armc tests/cases/test_cast.c (CAST_MATRIX
// under TARGET_FT04 / TARGET_FT78). A combination outside this whitelist would
// build a non-existent DSP kernel name in cast_func, so reject it here to let
// the scheduler fall back to CPU.
bool IsSupportedCast(TypeId src, TypeId dst) {
  using Pair = std::pair<TypeId, TypeId>;
  const std::vector<Pair> common = {
    {kNumberTypeInt16, kNumberTypeInt32},       {kNumberTypeInt16, kNumberTypeFloat32},
    {kNumberTypeInt32, kNumberTypeInt16},       {kNumberTypeInt32, kNumberTypeFloat32},
    {kNumberTypeFloat32, kNumberTypeInt16},     {kNumberTypeFloat32, kNumberTypeInt32},
    {kNumberTypeFloat32, kNumberTypeComplex64}, {kNumberTypeComplex64, kNumberTypeFloat32}};
#ifdef SUPPORT_FT04
  const std::vector<Pair> ft04 = {{kNumberTypeInt16, kNumberTypeFloat16},   {kNumberTypeInt32, kNumberTypeFloat16},
                                  {kNumberTypeFloat16, kNumberTypeInt16},   {kNumberTypeFloat16, kNumberTypeInt32},
                                  {kNumberTypeFloat16, kNumberTypeFloat32}, {kNumberTypeFloat32, kNumberTypeFloat16}};
#endif
#ifdef SUPPORT_FT78
  const std::vector<Pair> ft78 = {
    {kNumberTypeInt8, kNumberTypeInt16},           {kNumberTypeInt8, kNumberTypeInt32},
    {kNumberTypeInt8, kNumberTypeFloat32},         {kNumberTypeInt8, kNumberTypeFloat64},
    {kNumberTypeInt16, kNumberTypeInt8},           {kNumberTypeInt16, kNumberTypeFloat64},
    {kNumberTypeInt32, kNumberTypeInt8},           {kNumberTypeInt32, kNumberTypeFloat64},
    {kNumberTypeFloat32, kNumberTypeInt8},         {kNumberTypeFloat32, kNumberTypeFloat64},
    {kNumberTypeFloat64, kNumberTypeInt8},         {kNumberTypeFloat64, kNumberTypeInt16},
    {kNumberTypeFloat64, kNumberTypeInt32},        {kNumberTypeFloat64, kNumberTypeFloat32},
    {kNumberTypeFloat64, kNumberTypeComplex128},   {kNumberTypeComplex64, kNumberTypeComplex128},
    {kNumberTypeComplex128, kNumberTypeComplex64}, {kNumberTypeComplex128, kNumberTypeFloat64}};
#endif
  const Pair want = {src, dst};
  const auto matches = [&want](const Pair &p) { return p == want; };
  if (std::any_of(common.begin(), common.end(), matches)) {
    return true;
  }
#ifdef SUPPORT_FT04
  if (std::any_of(ft04.begin(), ft04.end(), matches)) {
    return true;
  }
#endif
#ifdef SUPPORT_FT78
  if (std::any_of(ft78.begin(), ft78.end(), matches)) {
    return true;
  }
#endif
  return false;
}

int CastDSPKernel::CheckSpecs() {
  // Cast(x, dst_type) may carry dst_type either as an attribute (one input) or
  // as a tensor (two inputs). Source and destination types are taken from
  // in[0]/out[0] data_type, so both input forms are accepted.
  if (in_tensors_.size() != INPUT_TENSOR_SIZE_1 && in_tensors_.size() != INPUT_TENSOR_SIZE_2) {
    MS_LOG(ERROR) << "cast kernel input size invalid: in=" << in_tensors_.size();
    return RET_ERROR;
  }
  if (out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(ERROR) << "cast kernel output size invalid: out=" << out_tensors_.size();
    return RET_ERROR;
  }
  const auto *input = in_tensors_[kInIdx];
  const auto *output = out_tensors_[kOutIdx];
  const auto in_elems = input->ElementsNum();
  const auto out_elems = output->ElementsNum();
  if (in_elems <= 0 || in_elems != out_elems) {
    MS_LOG(ERROR) << "cast kernel element num invalid: in=" << in_elems << " out=" << out_elems;
    return RET_ERROR;
  }
  if (!IsSupportedCast(input->data_type(), output->data_type())) {
    MS_LOG(ERROR) << "cast kernel unsupported dtype pair: src=" << static_cast<int>(input->data_type())
                  << " dst=" << static_cast<int>(output->data_type());
    return RET_ERROR;
  }
  return RET_OK;
}

int CastDSPKernel::Prepare() { return RET_OK; }

int CastDSPKernel::Run() {
  auto *input = in_tensors_[kInIdx];
  auto *output = out_tensors_[kOutIdx];
  auto allocator = dsp_runtime_->GetAllocator();
  if (allocator == nullptr) {
    MS_LOG(ERROR) << "CastDSPKernel allocator is null.";
    return RET_ERROR;
  }
  uint64_t in_device_ptr = allocator->GetDeviceMemPtr(input->data());
  if (in_device_ptr == 0) {
    MS_LOG(ERROR) << "CastDSPKernel input device ptr is null.";
    return RET_ERROR;
  }
  uint64_t out_device_ptr = allocator->GetDeviceMemPtr(output->data());
  if (out_device_ptr == 0) {
    MS_LOG(ERROR) << "CastDSPKernel output device ptr is null.";
    return RET_ERROR;
  }
  uint64_t length = static_cast<uint64_t>(output->ElementsNum());
  int src_dtype = static_cast<int>(input->data_type());
  int dst_dtype = static_cast<int>(output->data_type());
  int mtype = static_cast<int>(GetMemType());
  int ret = cast_func(in_device_ptr, out_device_ptr, length, core_mask_, src_dtype, dst_dtype, mtype);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "CastDSPKernel run failed, src=" << src_dtype << " dst=" << dst_dtype;
    return RET_ERROR;
  }
  return RET_OK;
}

// Common (FT04 + FT78): int16/int32/fp32 mutual conversion and fp<->c64.
REG_KERNEL(kDSP, kNumberTypeInt16, PrimitiveType_Cast, DSPKernelCreator<CastDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt32, PrimitiveType_Cast, DSPKernelCreator<CastDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeFloat32, PrimitiveType_Cast, DSPKernelCreator<CastDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeComplex64, PrimitiveType_Cast, DSPKernelCreator<CastDSPKernel>)

#ifdef SUPPORT_FT04
// FT04 only: fp16 related conversions.
REG_KERNEL(kDSP, kNumberTypeFloat16, PrimitiveType_Cast, DSPKernelCreator<CastDSPKernel>)
#endif

#ifdef SUPPORT_FT78
// FT78 only: int8, fp64 and complex128 related conversions.
REG_KERNEL(kDSP, kNumberTypeInt8, PrimitiveType_Cast, DSPKernelCreator<CastDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeFloat64, PrimitiveType_Cast, DSPKernelCreator<CastDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeComplex128, PrimitiveType_Cast, DSPKernelCreator<CastDSPKernel>)
#endif
}  // namespace mindspore::kernel
