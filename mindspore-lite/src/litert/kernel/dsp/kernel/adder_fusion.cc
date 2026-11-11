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

#include "src/litert/kernel/dsp/kernel/adder_fusion.h"
#include "src/litert/kernel_registry.h"
#include "src/litert/kernel/cpu/nnacl_c/conv_parameter.h"
#include "armc/include/operator.h"

using mindspore::kernel::KERNEL_ARCH::kDSP;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_AdderFusion;

namespace mindspore::kernel {

namespace {
constexpr int kInputIdx = 0;
constexpr int kWeightIdx = 1;
constexpr int kBiasIdx = 2;
constexpr int kOutIdx = 0;
constexpr int kTensorRank4 = 4;
constexpr int kWeightOutIdx = 0;
constexpr int kWeightChannelIdx = 3;
}  // namespace

int AdderFusionDSPKernel::Prepare() { return RET_OK; }

int AdderFusionDSPKernel::CheckSpecs() {
  // adder_func always takes a bias address, so a bias-less AdderFusion graph cannot be
  // served by this kernel: reject it explicitly here instead of dereferencing a missing
  // tensor, and let the scheduler fall back to the CPU kernel.
  if (in_tensors_.size() != INPUT_TENSOR_SIZE_3) {
    MS_LOG(ERROR) << "AdderFusion DSP kernel requires 3 inputs (input, weight, bias), got " << in_tensors_.size()
                  << "; AdderFusion without bias is not supported by the DSP kernel.";
    return RET_ERROR;
  }
  if (out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(ERROR) << "AdderFusion expects 1 output, got " << out_tensors_.size();
    return RET_ERROR;
  }
  if (in_tensors_[kInputIdx] == nullptr || in_tensors_[kWeightIdx] == nullptr || in_tensors_[kBiasIdx] == nullptr ||
      out_tensors_[kOutIdx] == nullptr) {
    MS_LOG(ERROR) << "AdderFusion got null tensor.";
    return RET_ERROR;
  }
  if (in_tensors_[kInputIdx]->shape().size() != kTensorRank4 ||
      in_tensors_[kWeightIdx]->shape().size() != kTensorRank4) {
    MS_LOG(ERROR) << "AdderFusion expects rank-4 input and weight tensors.";
    return RET_ERROR;
  }
  if (op_parameter_ == nullptr) {
    MS_LOG(ERROR) << "AdderFusion kernel parameter is null.";
    return RET_ERROR;
  }
  auto *param = reinterpret_cast<ConvParameter *>(op_parameter_);
  // adder_func takes one pad per dimension and computes the output with symmetric
  // padding, so asymmetric padding would silently produce a wrong result. Reject it
  // here to let the scheduler fall back to CPU.
  if (param->pad_u_ != param->pad_d_ || param->pad_l_ != param->pad_r_) {
    MS_LOG(ERROR) << "AdderFusion only supports symmetric padding.";
    return RET_ERROR;
  }
  // adder_func reads in_channel/group and expects weight[3] == in_channel / group;
  // reject inconsistent group shapes here instead of launching a DSP kernel with
  // wrong shapes (same contract as conv2d.cc).
  const int group = param->group_;
  const int in_channel = in_tensors_[kInputIdx]->Channel();
  const int out_channel = out_tensors_[kOutIdx]->Channel();
  const auto &weight_shape = in_tensors_[kWeightIdx]->shape();
  if (group <= 0 || in_channel % group != 0 || out_channel % group != 0 ||
      weight_shape[kWeightChannelIdx] != in_channel / group || weight_shape[kWeightOutIdx] != out_channel) {
    MS_LOG(ERROR) << "AdderFusion invalid group=" << group << " in_channel=" << in_channel
                  << " out_channel=" << out_channel << " weight_channel=" << weight_shape[kWeightChannelIdx];
    return RET_ERROR;
  }
  return RET_OK;
}

int AdderFusionDSPKernel::Run() {
  auto *input = in_tensors_[kInputIdx];
  auto *weight = in_tensors_[kWeightIdx];
  auto *bias = in_tensors_[kBiasIdx];
  auto *output = out_tensors_[kOutIdx];
  auto *param = reinterpret_cast<ConvParameter *>(op_parameter_);
  if (param == nullptr) {
    MS_LOG(ERROR) << "AdderFusionDSPKernel parameter is null.";
    return RET_ERROR;
  }

  const int batch = input->Batch();
  const int input_h = input->Height();
  const int input_w = input->Width();
  const int in_channel = input->Channel();
  const int out_channel = weight->Batch();
  const int kernel_h = weight->Height();
  const int kernel_w = weight->Width();
  const int group = param->group_;

  auto data_type = input->data_type();
  auto mem_type = GetMemType();
  auto allocator = dsp_runtime_->GetAllocator();
  if (allocator == nullptr) {
    MS_LOG(ERROR) << "AdderFusionDSPKernel failed to get DSP allocator.";
    return RET_ERROR;
  }
  uint64_t input_device_ptr = allocator->GetDeviceMemPtr(input->data());
  if (input_device_ptr == 0) {
    MS_LOG(ERROR) << "AdderFusionDSPKernel input device ptr is null.";
    return RET_ERROR;
  }
  uint64_t weight_device_ptr = allocator->GetDeviceMemPtr(weight->data());
  if (weight_device_ptr == 0) {
    MS_LOG(ERROR) << "AdderFusionDSPKernel weight device ptr is null.";
    return RET_ERROR;
  }
  uint64_t bias_device_ptr = allocator->GetDeviceMemPtr(bias->data());
  if (bias_device_ptr == 0) {
    MS_LOG(ERROR) << "AdderFusionDSPKernel bias device ptr is null.";
    return RET_ERROR;
  }
  uint64_t output_device_ptr = allocator->GetDeviceMemPtr(output->data());
  if (output_device_ptr == 0) {
    MS_LOG(ERROR) << "AdderFusionDSPKernel output device ptr is null.";
    return RET_ERROR;
  }

  int ret = adder_func(input_device_ptr, weight_device_ptr, bias_device_ptr, output_device_ptr, batch, input_h, input_w,
                       in_channel, out_channel, kernel_h, kernel_w, param->stride_h_, param->stride_w_, param->pad_u_,
                       param->pad_l_, param->dilation_h_, param->dilation_w_, group, core_mask_,
                       static_cast<int>(data_type), static_cast<int>(mem_type));
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "AdderFusionDSPKernel run failed.";
    return RET_ERROR;
  }

  return RET_OK;
}

REG_KERNEL(kDSP, kNumberTypeFloat32, PrimitiveType_AdderFusion, DSPKernelCreator<AdderFusionDSPKernel>)

#ifdef SUPPORT_FT04
REG_KERNEL(kDSP, kNumberTypeFloat16, PrimitiveType_AdderFusion, DSPKernelCreator<AdderFusionDSPKernel>)
#endif
}  // namespace mindspore::kernel
