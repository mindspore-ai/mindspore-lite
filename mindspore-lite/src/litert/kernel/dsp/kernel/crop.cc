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

#include "src/litert/kernel/dsp/kernel/crop.h"
#include <array>
#include <cstdint>
#include "src/litert/kernel/cpu/nnacl_c/crop_parameter.h"
#include "src/litert/kernel_registry.h"
#include "armc/include/operator.h"

using mindspore::kernel::KERNEL_ARCH::kDSP;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Crop;

namespace mindspore::kernel {

namespace {
constexpr size_t kDataIdx = 0;
constexpr size_t kShapeIdx = 1;
constexpr size_t kOutIdx = 0;
// CropPadOffset treats a single offset as a value that repeats over every dimension from axis on.
constexpr int kSingleOffset = 1;
}  // namespace

int64_t CropDSPKernel::ResolveAxis() const {
  const auto *param = reinterpret_cast<const CropParameter *>(op_parameter_);
  // crop_infer.c folds a negative axis into the rank the same way.
  return param->axis_ < 0 ? param->axis_ + static_cast<int64_t>(kRank) : param->axis_;
}

void CropDSPKernel::BuildFullOffset(int64_t axis) {
  // CropPadOffset in nnacl_c/base/crop_base.c expands the attribute offsets into one value pe
  // dimension: a lone offset repeats from axis on, a list maps one to one onto the trailing
  // dimensions. armc crop_func wants exactly the tail of that expansion, so Run passes it
  // full_offset_.data() + axis.
  const auto *param = reinterpret_cast<const CropParameter *>(op_parameter_);
  full_offset_.fill(0);
  for (size_t dim = static_cast<size_t>(axis); dim < full_offset_.size(); ++dim) {
    const size_t pos = dim - static_cast<size_t>(axis);
    if (param->offset_size_ == kSingleOffset) {
      full_offset_[dim] = static_cast<int>(param->offset_[0]);
    } else if (static_cast<int>(pos) < param->offset_size_) {
      full_offset_[dim] = static_cast<int>(param->offset_[pos]);
    }
  }
}

int CropDSPKernel::CheckGeometry() const {
  const auto *input = in_tensors_[kDataIdx];
  const auto *output = out_tensors_[kOutIdx];
  // The shapes and the offsets together decide which input elements the DSP reads, so everything
  // that would turn into an out of range read on the device has to be refused here.
  for (size_t dim = 0; dim < kRank; ++dim) {
    const auto in_dim = static_cast<int>(input->shape()[dim]);
    const auto out_dim = static_cast<int>(output->shape()[dim]);
    if (in_dim <= 0 || out_dim <= 0 || full_offset_[dim] < 0 || full_offset_[dim] > in_dim - out_dim) {
      MS_LOG(ERROR) << "Crop dimension " << dim << " reads " << out_dim << " elements at offset " << full_offset_[dim]
                    << " of an input that holds " << in_dim << ", which is outside the input buffer.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int CropDSPKernel::ResolveGeometry() {
  if (op_parameter_ == nullptr) {
    MS_LOG(ERROR) << "Crop kernel parameter is null.";
    return RET_ERROR;
  }
  const auto *param = reinterpret_cast<const CropParameter *>(op_parameter_);
  const auto axis = ResolveAxis();
  if (axis < 0 || axis >= static_cast<int64_t>(kRank)) {
    MS_LOG(ERROR) << "Crop axis " << param->axis_ << " is outside a 4D input.";
    return RET_ERROR;
  }
  if (param->offset_size_ < 0 || param->offset_size_ > static_cast<int>(kRank)) {
    MS_LOG(ERROR) << "Crop carries " << param->offset_size_ << " offsets, which does not fit a 4D input.";
    return RET_ERROR;
  }
  // crop_infer.c only accepts a list of offsets that reaches exactly down to the last dimension.
  if (param->offset_size_ > kSingleOffset && axis + param->offset_size_ != static_cast<int64_t>(kRank)) {
    MS_LOG(ERROR) << "Crop offsets from axis " << axis << " stop before the last dimension.";
    return RET_ERROR;
  }
  BuildFullOffset(axis);
  axis_ = static_cast<int>(axis);
  const auto *input = in_tensors_[kDataIdx];
  const auto *output = out_tensors_[kOutIdx];
  if (input->data_type() != output->data_type()) {
    MS_LOG(ERROR) << "Crop output dtype " << output->data_type() << " differs from the input dtype "
                  << input->data_type() << ".";
    return RET_ERROR;
  }
  if (input->shape().size() != kRank || output->shape().size() != kRank) {
    MS_LOG(ERROR) << "Crop needs a 4D input and output, got ranks " << input->shape().size() << " and "
                  << output->shape().size() << "; only NHWC cropping is implemented on the DSP.";
    return RET_ERROR;
  }
  return CheckGeometry();
}

int CropDSPKernel::Prepare() { return ResolveGeometry(); }

int CropDSPKernel::CheckSpecs() {
  // crop_infer.c takes the value tensor plus the tensor that carries the output shape.
  if (in_tensors_.size() != INPUT_TENSOR_SIZE_2) {
    MS_LOG(ERROR) << "Crop expects 2 inputs, the value and the output shape, got " << in_tensors_.size() << ".";
    return RET_ERROR;
  }
  if (out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(ERROR) << "Crop expects 1 output, got " << out_tensors_.size() << ".";
    return RET_ERROR;
  }
  if (in_tensors_[kDataIdx] == nullptr || in_tensors_[kShapeIdx] == nullptr || out_tensors_[kOutIdx] == nullptr) {
    MS_LOG(ERROR) << "Crop input, shape or output tensor is null.";
    return RET_ERROR;
  }
  return ResolveGeometry();
}

int CropDSPKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  auto allocator = dsp_runtime_->GetAllocator();
  if (allocator == nullptr) {
    MS_LOG(ERROR) << "Crop DSP allocator is null.";
    return RET_ERROR;
  }
  auto *input = in_tensors_[kDataIdx];
  auto *output = out_tensors_[kOutIdx];
  // ResolveGeometry settled the offsets before the first Run; a shorter shape that reached here
  // without it would let the loop below index past the end of a shape.
  if (input == nullptr || output == nullptr || input->shape().size() != kRank || output->shape().size() != kRank) {
    MS_LOG(ERROR) << "Crop runs on a missing tensor or a shape that is not 4D.";
    return RET_ERROR;
  }
  uint64_t in_device_ptr = allocator->GetDeviceMemPtr(input->data());
  if (in_device_ptr == 0) {
    MS_LOG(ERROR) << "Crop input device ptr is null.";
    return RET_ERROR;
  }
  uint64_t out_device_ptr = allocator->GetDeviceMemPtr(output->data());
  if (out_device_ptr == 0) {
    MS_LOG(ERROR) << "Crop output device ptr is null.";
    return RET_ERROR;
  }
  // The shapes are filled on the stack; only the offsets would have needed a buffer to hold them.
  std::array<int, kRank> in_shape;
  std::array<int, kRank> out_shape;
  for (size_t dim = 0; dim < kRank; ++dim) {
    in_shape[dim] = static_cast<int>(input->shape()[dim]);
    out_shape[dim] = static_cast<int>(output->shape()[dim]);
  }
  int ret = crop_func(in_device_ptr, out_device_ptr, in_shape.data(), out_shape.data(), full_offset_.data() + axis_,
                      axis_, static_cast<unsigned int>(core_mask_), static_cast<int>(input->data_type()),
                      static_cast<int>(GetMemType()));
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Crop run failed, ret: " << ret;
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kDSP, kNumberTypeFloat32, PrimitiveType_Crop, DSPKernelCreator<CropDSPKernel>)

#ifdef SUPPORT_FT04
REG_KERNEL(kDSP, kNumberTypeFloat16, PrimitiveType_Crop, DSPKernelCreator<CropDSPKernel>)
#endif
}  // namespace mindspore::kernel
