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

#include "src/litert/kernel/dsp/kernel/group_norm_fusion.h"
#include <cstdint>
#include "src/litert/kernel_registry.h"
#include "schema/inner/model_generated.h"
#include "armc/include/operator.h"

using mindspore::kernel::KERNEL_ARCH::kDSP;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_GroupNormFusion;

namespace mindspore::kernel {

namespace {
constexpr size_t kInputIdx = 0;
constexpr size_t kScaleIdx = 1;
constexpr size_t kOffsetIdx = 2;
constexpr size_t kOutputIdx = 0;
constexpr size_t kInputRank = 4;
constexpr int kInvalidChannel = -1;

// Validate 4D NHWC/NCHW input; return the format-aware channel count (>0) or kInvalidChannel.
int ValidateInputShape(const lite::Tensor *input) {
  if (input->shape().size() != kInputRank) {
    MS_LOG(ERROR) << "GroupNormFusion input must be 4D, got rank " << input->shape().size();
    return kInvalidChannel;
  }
  const auto format = input->format();
  if (format != mindspore::NHWC && format != mindspore::NCHW) {
    MS_LOG(ERROR) << "GroupNormFusion only supports NHWC/NCHW, got format " << static_cast<int>(format);
    return kInvalidChannel;
  }
  // DSP path uses NHWC where Channel() is the last dim; never hard-code shape indices.
  const int channel = input->Channel();
  if (input->Batch() < 1 || channel < 1 || input->Height() < 1 || input->Width() < 1) {
    MS_LOG(ERROR) << "GroupNormFusion invalid shape dims.";
    return kInvalidChannel;
  }
  return channel;
}
}  // namespace

GroupNormFusionDSPKernel::~GroupNormFusionDSPKernel() {
  if (allocator_ != nullptr) {
    if (mean_buffer_ != nullptr) {
      allocator_->Free(mean_buffer_);
      mean_buffer_ = nullptr;
    }
    if (variance_buffer_ != nullptr) {
      allocator_->Free(variance_buffer_);
      variance_buffer_ = nullptr;
    }
  }
  allocator_ = nullptr;
  stat_bytes_ = 0;
}

int GroupNormFusionDSPKernel::CheckSpecs() {
  if (in_tensors_.size() != INPUT_TENSOR_SIZE_3 || out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(ERROR) << "GroupNormFusion unexpected io sizes, in: " << in_tensors_.size()
                  << ", out: " << out_tensors_.size();
    return RET_ERROR;
  }
  auto dtype = in_tensors_[kInputIdx]->data_type();
  if (dtype != kNumberTypeFloat32 && dtype != kNumberTypeFloat16) {
    MS_LOG(ERROR) << "GroupNormFusion unsupported dtype: " << static_cast<int>(dtype);
    return RET_ERROR;
  }
  auto *input = in_tensors_[kInputIdx];
  const int channel = ValidateInputShape(input);
  if (channel < 1) {
    return RET_ERROR;
  }
  // The SDK indexes scale/offset by channel; validate dtype and length together.
  for (size_t i = kScaleIdx; i < in_tensors_.size(); ++i) {
    if (in_tensors_[i]->data_type() != dtype) {
      MS_LOG(ERROR) << "GroupNormFusion scale/offset dtype mismatch with input.";
      return RET_ERROR;
    }
    if (in_tensors_[i]->ElementsNum() != channel) {
      MS_LOG(ERROR) << "GroupNormFusion scale/offset elements " << in_tensors_[i]->ElementsNum() << " != channel "
                    << channel;
      return RET_ERROR;
    }
  }
  if (out_tensors_[kOutputIdx]->data_type() != dtype) {
    MS_LOG(ERROR) << "GroupNormFusion output dtype mismatch with input.";
    return RET_ERROR;
  }
  auto *param = reinterpret_cast<GroupNormParameter *>(op_parameter_);
  if (param == nullptr || param->num_groups_ < 1) {
    MS_LOG(ERROR) << "GroupNormFusion invalid num_groups.";
    return RET_ERROR;
  }
  if (channel % param->num_groups_ != 0) {
    MS_LOG(ERROR) << "GroupNormFusion channel " << channel << " not divisible by num_groups " << param->num_groups_;
    return RET_ERROR;
  }
  return RET_OK;
}

int GroupNormFusionDSPKernel::Prepare() {
  param_ = reinterpret_cast<GroupNormParameter *>(op_parameter_);
  if (param_ == nullptr) {
    MS_LOG(ERROR) << "GroupNormFusion param is null.";
    return RET_ERROR;
  }
  allocator_ = dsp_runtime_->GetAllocator();
  if (allocator_ == nullptr) {
    MS_LOG(ERROR) << "GroupNormFusion allocator is null.";
    return RET_ERROR;
  }
  // Use format-aware accessors: the DSP path carries 4D activations as NHWC,
  // where Channel() is the last dim. Hard-coding shape[1] would treat H as C.
  auto *input = in_tensors_[kInputIdx];
  param_->batch_ = input->Batch();
  param_->channel_ = input->Channel();
  param_->unit_ = input->Height() * input->Width();
  // Allocate stat buffers here (like CPU GroupNormPrepare), not in Run(): ReSize
  // re-runs Prepare on shape changes, and EnsureStatBuffers frees the old buffers
  // before reallocating. Run() then only consumes them.
  if (param_->num_groups_ < 1 || param_->batch_ < 1) {
    MS_LOG(ERROR) << "GroupNormFusion invalid num_groups/batch for stat buffers.";
    return RET_ERROR;
  }
  const int64_t stat_count = static_cast<int64_t>(param_->batch_) * param_->num_groups_;
  return EnsureStatBuffers(stat_count, input->data_type());
}

int GroupNormFusionDSPKernel::EnsureStatBuffers(int64_t count, TypeId dtype) {
  const size_t bytes = static_cast<size_t>(count) * lite::DataTypeSize(dtype);
  if (mean_buffer_ != nullptr && bytes <= stat_bytes_) {
    return RET_OK;
  }
  // Reallocate on growth (e.g. dynamic batch) or after a partial failure.
  if (mean_buffer_ != nullptr) {
    allocator_->Free(mean_buffer_);
    mean_buffer_ = nullptr;
  }
  if (variance_buffer_ != nullptr) {
    allocator_->Free(variance_buffer_);
    variance_buffer_ = nullptr;
  }
  stat_bytes_ = 0;
  mean_buffer_ = allocator_->Malloc(bytes);
  variance_buffer_ = allocator_->Malloc(bytes);
  if (mean_buffer_ == nullptr || variance_buffer_ == nullptr) {
    MS_LOG(ERROR) << "GroupNormFusion malloc mean/variance buffer failed.";
    if (mean_buffer_ != nullptr) {
      allocator_->Free(mean_buffer_);
      mean_buffer_ = nullptr;
    }
    if (variance_buffer_ != nullptr) {
      allocator_->Free(variance_buffer_);
      variance_buffer_ = nullptr;
    }
    return RET_ERROR;
  }
  stat_bytes_ = bytes;
  return RET_OK;
}

int GroupNormFusionDSPKernel::Run() {
  if (param_ == nullptr || allocator_ == nullptr) {
    MS_LOG(ERROR) << "GroupNormFusion not prepared.";
    return RET_ERROR;
  }
  auto *input = in_tensors_[kInputIdx];
  auto *scale = in_tensors_[kScaleIdx];
  auto *offset = in_tensors_[kOffsetIdx];
  auto *output = out_tensors_[kOutputIdx];
  const TypeId dtype = input->data_type();
  if (param_->batch_ < 1 || param_->num_groups_ < 1) {
    MS_LOG(ERROR) << "GroupNormFusion dims not prepared.";
    return RET_ERROR;
  }
  // Normally prepared in Prepare(); fall back here in case PreProcess skipped it.
  const int64_t stat_count = static_cast<int64_t>(param_->batch_) * param_->num_groups_;
  if (EnsureStatBuffers(stat_count, dtype) != RET_OK) {
    return RET_ERROR;
  }

  uint64_t input_dev = allocator_->GetDeviceMemPtr(input->data());
  uint64_t scale_dev = allocator_->GetDeviceMemPtr(scale->data());
  uint64_t offset_dev = allocator_->GetDeviceMemPtr(offset->data());
  uint64_t output_dev = allocator_->GetDeviceMemPtr(output->data());
  uint64_t mean_dev = allocator_->GetDeviceMemPtr(mean_buffer_);
  uint64_t variance_dev = allocator_->GetDeviceMemPtr(variance_buffer_);
  if (input_dev == 0 || scale_dev == 0 || offset_dev == 0 || output_dev == 0 || mean_dev == 0 || variance_dev == 0) {
    MS_LOG(ERROR) << "GroupNormFusion device ptr is null.";
    return RET_ERROR;
  }

  const int dtype_code = static_cast<int>(dtype);
  const int mtype = static_cast<int>(GetMemType());
  int ret = group_norm_fusion_func(input_dev, scale_dev, offset_dev, mean_dev, variance_dev, output_dev,
                                   param_->num_groups_, param_->channel_, param_->unit_, param_->batch_,
                                   param_->epsilon_, core_mask_, dtype_code, mtype);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << this->name() << " Run failed!";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kDSP, kNumberTypeFloat32, PrimitiveType_GroupNormFusion, DSPKernelCreator<GroupNormFusionDSPKernel>)

#ifdef SUPPORT_FT04
REG_KERNEL(kDSP, kNumberTypeFloat16, PrimitiveType_GroupNormFusion, DSPKernelCreator<GroupNormFusionDSPKernel>)
#endif
}  // namespace mindspore::kernel
