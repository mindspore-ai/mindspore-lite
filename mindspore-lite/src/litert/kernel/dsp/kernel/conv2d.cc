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

#include "src/litert/kernel/dsp/kernel/conv2d.h"
#include "src/litert/kernel_registry.h"
#include "armc/include/operator.h"

using mindspore::kernel::KERNEL_ARCH::kDSP;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Conv2DFusion;

namespace mindspore::kernel {

constexpr int kInputIdx = 0;
constexpr int kWeightIdx = 1;
constexpr int kBiasIdx = 2;
constexpr int kOutIdx = 0;
constexpr int kConvWeightDim = 4;
constexpr int kWeightChannelIdx = 3;

int Conv2dDSPKernel::CheckSpecs() {
  if (param_ == nullptr) {
    MS_LOG(ERROR) << "conv2d kernel parameter is null.";
    return RET_ERROR;
  }
  if (in_tensors_.size() != INPUT_TENSOR_SIZE_2 && in_tensors_.size() != INPUT_TENSOR_SIZE_3) {
    MS_LOG(ERROR) << "conv2d kernel input size invalid: in=" << in_tensors_.size();
    return RET_ERROR;
  }
  if (out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(ERROR) << "conv2d kernel output size invalid: out=" << out_tensors_.size();
    return RET_ERROR;
  }
  if (in_tensors_[kWeightIdx]->shape().size() != kConvWeightDim) {
    MS_LOG(ERROR) << "conv2d weight must be 4D.";
    return RET_ERROR;
  }
  // conv2d_func builds one kernel name from a single dtype, so all data tensors
  // must share the input dtype. Reject mixed-dtype cases here to let the
  // scheduler fall back to CPU instead of launching a wrong DSP kernel.
  const auto dtype = in_tensors_[kInputIdx]->data_type();
  if (dtype != kNumberTypeFloat32 && dtype != kNumberTypeFloat16) {
    MS_LOG(ERROR) << "conv2d unsupported input dtype: " << static_cast<int>(dtype);
    return RET_ERROR;
  }
  if (in_tensors_[kWeightIdx]->data_type() != dtype || out_tensors_[kOutIdx]->data_type() != dtype) {
    MS_LOG(ERROR) << "conv2d weight/output dtype mismatch with input.";
    return RET_ERROR;
  }
  if (in_tensors_.size() == INPUT_TENSOR_SIZE_3 && in_tensors_[kBiasIdx]->data_type() != dtype) {
    MS_LOG(ERROR) << "conv2d bias dtype mismatch with input.";
    return RET_ERROR;
  }
  // conv2d_func takes one pad per dimension and computes output size with
  // symmetric padding (pad*2). Asymmetric or runtime-derived (SAME) padding
  // would be computed incorrectly and may overflow the output buffer, so
  // reject these cases and let the scheduler fall back to CPU.
  if (param_->pad_u_ != param_->pad_d_ || param_->pad_l_ != param_->pad_r_) {
    MS_LOG(ERROR) << "conv2d only supports symmetric padding.";
    return RET_ERROR;
  }
  if (param_->pad_mode_ == Pad_same) {
    MS_LOG(ERROR) << "conv2d does not support Pad_same mode (padding derived at runtime).";
    return RET_ERROR;
  }
  // conv2d_func reads in_channel/group and expects weight[3] == in_channel/group.
  // Reject inconsistent group shapes here so the scheduler falls back to CPU
  // instead of launching a DSP kernel with wrong shapes.
  const int group = param_->group_;
  const int in_channel = in_tensors_[kInputIdx]->Channel();
  const int out_channel = out_tensors_[kOutIdx]->Channel();
  const auto &weight_shape = in_tensors_[kWeightIdx]->shape();
  if (group <= 0 || in_channel % group != 0 || out_channel % group != 0 ||
      weight_shape[kWeightChannelIdx] != in_channel / group) {
    MS_LOG(ERROR) << "conv2d invalid group=" << group << " in_channel=" << in_channel << " out_channel=" << out_channel;
    return RET_ERROR;
  }
  return RET_OK;
}

int Conv2dDSPKernel::Prepare() {
  param_ = reinterpret_cast<ConvParameter *>(op_parameter_);
  return RET_OK;
}

int Conv2dDSPKernel::Run() {
  if (param_ == nullptr) {
    MS_LOG(ERROR) << "conv2d kernel parameter is null.";
    return RET_ERROR;
  }
  auto *input = in_tensors_[kInputIdx];
  auto *weight = in_tensors_[kWeightIdx];
  auto *output = out_tensors_[kOutIdx];

  auto allocator = dsp_runtime_->GetAllocator();
  if (allocator == nullptr) {
    MS_LOG(ERROR) << "Conv2dDSPKernel allocator is null.";
    return RET_ERROR;
  }
  uint64_t in_ptr = allocator->GetDeviceMemPtr(input->data());
  uint64_t weight_ptr = allocator->GetDeviceMemPtr(weight->data());
  uint64_t out_ptr = allocator->GetDeviceMemPtr(output->data());
  if (in_ptr == 0 || weight_ptr == 0 || out_ptr == 0) {
    MS_LOG(ERROR) << "Conv2dDSPKernel device ptr is null.";
    return RET_ERROR;
  }
  // Bias is optional: a 3rd input tensor carries it, otherwise pass a null device pointer.
  uint64_t bias_ptr = 0;
  if (in_tensors_.size() == INPUT_TENSOR_SIZE_3) {
    bias_ptr = allocator->GetDeviceMemPtr(in_tensors_[kBiasIdx]->data());
    if (bias_ptr == 0) {
      MS_LOG(ERROR) << "Conv2dDSPKernel bias device ptr is null.";
      return RET_ERROR;
    }
  }

  const int batch = input->Batch();
  const int input_h = input->Height();
  const int input_w = input->Width();
  const int in_channel = input->Channel();
  const int out_channel = output->Channel();
  const int dtype = static_cast<int>(input->data_type());
  const int mtype = static_cast<int>(GetMemType());

  int ret =
    conv2d_func(in_ptr, weight_ptr, bias_ptr, out_ptr, batch, input_h, input_w, in_channel, out_channel,
                param_->kernel_h_, param_->kernel_w_, param_->stride_h_, param_->stride_w_, param_->pad_u_,
                param_->pad_l_, param_->dilation_h_, param_->dilation_w_, param_->group_, core_mask_, dtype, mtype);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Conv2dDSPKernel run failed, dtype=" << dtype;
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kDSP, kNumberTypeFloat32, PrimitiveType_Conv2DFusion, DSPKernelCreator<Conv2dDSPKernel>)

#ifdef SUPPORT_FT04
REG_KERNEL(kDSP, kNumberTypeFloat16, PrimitiveType_Conv2DFusion, DSPKernelCreator<Conv2dDSPKernel>)
#endif
}  // namespace mindspore::kernel
