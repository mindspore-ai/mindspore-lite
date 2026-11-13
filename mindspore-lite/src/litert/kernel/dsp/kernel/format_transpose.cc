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

#include "src/litert/kernel/dsp/kernel/format_transpose.h"
#include <vector>
#include "src/litert/kernel_registry.h"
#include "armc/include/operator.h"

using mindspore::kernel::KERNEL_ARCH::kDSP;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_FormatTranspose;

namespace mindspore::kernel {

constexpr int kInIdx = 0;
constexpr int kOutIdx = 0;
constexpr int kShapeDim4D = 4;
constexpr int kNhwcBatchIdx = 0;
constexpr int kNhwcHeightIdx = 1;
constexpr int kNhwcWidthIdx = 2;
constexpr int kNhwcChannelIdx = 3;
constexpr int kNchwBatchIdx = 0;
constexpr int kNchwChannelIdx = 1;
constexpr int kNchwHeightIdx = 2;
constexpr int kNchwWidthIdx = 3;
constexpr int kFormatNchw = static_cast<int>(Format_NCHW);
constexpr int kFormatNhwc = static_cast<int>(Format_NHWC);

// The DSP kernel only permutes NHWC <-> NCHW; every other pair (including
// identical formats) is a no-op that leaves the output buffer untouched.
bool IsSupportedFormatPair(int src_format, int dst_format) {
  const bool nhwc_to_nchw = src_format == kFormatNhwc && dst_format == kFormatNchw;
  const bool nchw_to_nhwc = src_format == kFormatNchw && dst_format == kFormatNhwc;
  return nhwc_to_nchw || nchw_to_nhwc;
}

// The DSP library only builds kernels for these dtypes (see dsplib
// format_transpose.rst platform notes and armc test_format_transpose.c).
bool IsSupportedFormatTransposeDtype(TypeId dtype) {
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

// The input shape is expressed in src_format layout: NHWC is [N, H, W, C] and
// NCHW is [N, C, H, W]. The DSP kernel takes (batch, channel, plane) with
// plane = H * W, so derive those three values from the shape source layout.
bool DeriveTransposeDims(const std::vector<int> &shape, int src_format, int *batch, int *channel, int *plane) {
  if (shape.size() != kShapeDim4D) {
    return false;
  }
  if (src_format == kFormatNhwc) {
    *batch = shape[kNhwcBatchIdx];
    *plane = shape[kNhwcHeightIdx] * shape[kNhwcWidthIdx];
    *channel = shape[kNhwcChannelIdx];
  } else {
    *batch = shape[kNchwBatchIdx];
    *channel = shape[kNchwChannelIdx];
    *plane = shape[kNchwHeightIdx] * shape[kNchwWidthIdx];
  }
  return *batch > 0 && *channel > 0 && *plane > 0;
}

int FormatTransposeDSPKernel::CheckSpecs() {
  if (param_ == nullptr) {
    MS_LOG(ERROR) << "format_transpose kernel param is null.";
    return RET_ERROR;
  }
  if (in_tensors_.size() != INPUT_TENSOR_SIZE_1) {
    MS_LOG(ERROR) << "format_transpose kernel input size invalid: in=" << in_tensors_.size();
    return RET_ERROR;
  }
  if (out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(ERROR) << "format_transpose kernel output size invalid: out=" << out_tensors_.size();
    return RET_ERROR;
  }
  const auto *input = in_tensors_[kInIdx];
  const auto *output = out_tensors_[kOutIdx];
  if (!IsSupportedFormatTransposeDtype(input->data_type())) {
    MS_LOG(ERROR) << "format_transpose kernel unsupported dtype: " << static_cast<int>(input->data_type());
    return RET_ERROR;
  }
  if (output->data_type() != input->data_type()) {
    MS_LOG(ERROR) << "format_transpose kernel output dtype mismatch with input.";
    return RET_ERROR;
  }
  const int src_format = static_cast<int>(param_->src_format_);
  const int dst_format = static_cast<int>(param_->dst_format_);
  // Reject no-op format pairs here so the scheduler falls back to the CPU copy
  // path instead of dispatching to a DSP kernel that would not write output.
  if (!IsSupportedFormatPair(src_format, dst_format)) {
    MS_LOG(ERROR) << "format_transpose kernel unsupported format pair: src=" << src_format << " dst=" << dst_format;
    return RET_ERROR;
  }
  if (input->shape().size() != kShapeDim4D || output->shape().size() != kShapeDim4D) {
    MS_LOG(ERROR) << "format_transpose kernel requires 4D input and output.";
    return RET_ERROR;
  }
  const auto in_elems = input->ElementsNum();
  const auto out_elems = output->ElementsNum();
  if (in_elems <= 0 || in_elems != out_elems) {
    MS_LOG(ERROR) << "format_transpose kernel element num invalid: in=" << in_elems << " out=" << out_elems;
    return RET_ERROR;
  }
  return RET_OK;
}

int FormatTransposeDSPKernel::Prepare() { return RET_OK; }

int FormatTransposeDSPKernel::Run() {
  // DSPKernelCreator skips CheckSpecs when the output shape still contains -1,
  // so re-validate every invariant here before touching tensor indices.
  if (CheckSpecs() != RET_OK) {
    return RET_ERROR;
  }
  auto *input = in_tensors_[kInIdx];
  auto *output = out_tensors_[kOutIdx];
  auto allocator = dsp_runtime_->GetAllocator();
  if (allocator == nullptr) {
    MS_LOG(ERROR) << "FormatTransposeDSPKernel allocator is null.";
    return RET_ERROR;
  }
  uint64_t in_device_ptr = allocator->GetDeviceMemPtr(input->data());
  if (in_device_ptr == 0) {
    MS_LOG(ERROR) << "FormatTransposeDSPKernel input device ptr is null.";
    return RET_ERROR;
  }
  uint64_t out_device_ptr = allocator->GetDeviceMemPtr(output->data());
  if (out_device_ptr == 0) {
    MS_LOG(ERROR) << "FormatTransposeDSPKernel output device ptr is null.";
    return RET_ERROR;
  }

  const int src_format = static_cast<int>(param_->src_format_);
  const int dst_format = static_cast<int>(param_->dst_format_);
  int batch = 0;
  int channel = 0;
  int plane = 0;
  if (!DeriveTransposeDims(input->shape(), src_format, &batch, &channel, &plane)) {
    MS_LOG(ERROR) << "FormatTransposeDSPKernel invalid dims.";
    return RET_ERROR;
  }
  const int dtype = static_cast<int>(input->data_type());
  const int mtype = static_cast<int>(GetMemType());
  int ret = format_transpose_func(in_device_ptr, out_device_ptr, src_format, dst_format, batch, channel, plane,
                                  core_mask_, dtype, mtype);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "FormatTransposeDSPKernel run failed, dtype=" << dtype;
    return RET_ERROR;
  }
  return RET_OK;
}

// Common (FT04 + FT78): fp32, int16, int32 and complex64.
REG_KERNEL(kDSP, kNumberTypeFloat32, PrimitiveType_FormatTranspose, DSPKernelCreator<FormatTransposeDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt16, PrimitiveType_FormatTranspose, DSPKernelCreator<FormatTransposeDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt32, PrimitiveType_FormatTranspose, DSPKernelCreator<FormatTransposeDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeComplex64, PrimitiveType_FormatTranspose, DSPKernelCreator<FormatTransposeDSPKernel>)

#ifdef SUPPORT_FT04
// FT04 only: fp16.
REG_KERNEL(kDSP, kNumberTypeFloat16, PrimitiveType_FormatTranspose, DSPKernelCreator<FormatTransposeDSPKernel>)
#endif

#ifdef SUPPORT_FT78
// FT78 only: int8, fp64 and complex128. No FT78 on-device test in this change.
REG_KERNEL(kDSP, kNumberTypeInt8, PrimitiveType_FormatTranspose, DSPKernelCreator<FormatTransposeDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeFloat64, PrimitiveType_FormatTranspose, DSPKernelCreator<FormatTransposeDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeComplex128, PrimitiveType_FormatTranspose, DSPKernelCreator<FormatTransposeDSPKernel>)
#endif
}  // namespace mindspore::kernel
