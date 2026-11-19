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

#include "src/litert/kernel/dsp/kernel/conv2d_backprop_filter.h"
#include <algorithm>
#include <array>
#include "src/litert/kernel_registry.h"
#include "src/litert/kernel/cpu/nnacl_c/conv_parameter.h"
#include "armc/include/operator.h"

using mindspore::kernel::KERNEL_ARCH::kDSP;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Conv2DBackpropFilterFusion;

namespace mindspore::kernel {

namespace {
constexpr size_t kDyIdx = 0;
constexpr size_t kXIdx = 1;
constexpr size_t kWeightShapeIdx = 2;
constexpr size_t kOutIdx = 0;
// A convolution works on [batch, height, width, channel], and the weight the gradient produces is
// [out channel, kernel height, kernel width, in channel].
constexpr int kConvRank = 4;
constexpr size_t kBatchDim = 0;
constexpr size_t kHeightDim = 1;
constexpr size_t kWidthDim = 2;
constexpr size_t kChannelDim = 3;
// conv2d_backprop_filter_func has no bias to accumulate here, Lite computes that as its own gradient.
constexpr uint64_t kNoBiasData = 0;
}  // namespace

const lite::Tensor *Conv2DBackpropFilterDSPKernel::Dy() const {
  return in_tensors_.size() > kDyIdx ? in_tensors_[kDyIdx] : nullptr;
}

const lite::Tensor *Conv2DBackpropFilterDSPKernel::X() const {
  return in_tensors_.size() > kXIdx ? in_tensors_[kXIdx] : nullptr;
}

const lite::Tensor *Conv2DBackpropFilterDSPKernel::WeightShape() const {
  return in_tensors_.size() > kWeightShapeIdx ? in_tensors_[kWeightShapeIdx] : nullptr;
}

const lite::Tensor *Conv2DBackpropFilterDSPKernel::WeightGrad() const {
  return out_tensors_.size() > kOutIdx ? out_tensors_[kOutIdx] : nullptr;
}

int Conv2DBackpropFilterDSPKernel::Prepare() { return ResolveGeometry(); }

int Conv2DBackpropFilterDSPKernel::CheckSpecs() { return ResolveGeometry(); }

int Conv2DBackpropFilterDSPKernel::CheckTensors() const {
  if (op_parameter_ == nullptr) {
    MS_LOG(ERROR) << "Conv2DBackpropFilter kernel parameter is null.";
    return RET_ERROR;
  }
  if (in_tensors_.size() < INPUT_TENSOR_SIZE_3) {
    MS_LOG(ERROR) << "Conv2DBackpropFilter needs the gradient, the input and the weight shape, got "
                  << in_tensors_.size() << " inputs.";
    return RET_ERROR;
  }
  if (out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(ERROR) << "Conv2DBackpropFilter expects 1 output, got " << out_tensors_.size() << ".";
    return RET_ERROR;
  }
  for (size_t i = 0; i < INPUT_TENSOR_SIZE_3; ++i) {
    if (in_tensors_[i] == nullptr) {
      MS_LOG(ERROR) << "Conv2DBackpropFilter input " << i << " is null.";
      return RET_ERROR;
    }
  }
  auto *weight_grad = WeightGrad();
  if (weight_grad == nullptr) {
    MS_LOG(ERROR) << "Conv2DBackpropFilter output tensor is null.";
    return RET_ERROR;
  }
  if (static_cast<int>(Dy()->shape().size()) != kConvRank || static_cast<int>(X()->shape().size()) != kConvRank) {
    MS_LOG(ERROR) << "Conv2DBackpropFilter expects NHWC inputs, the gradient has rank " << Dy()->shape().size()
                  << " and the input has rank " << X()->shape().size() << ".";
    return RET_ERROR;
  }
  // The DSP reads the gradient and the activation as one dtype and writes the weight in that same
  // dtype, so a mismatch would be read as garbage rather than as a conversion.
  const auto data_type = Dy()->data_type();
  if (X()->data_type() != data_type || weight_grad->data_type() != data_type) {
    MS_LOG(ERROR) << "Conv2DBackpropFilter needs one dtype for all three: gradient " << data_type << ", input "
                  << X()->data_type() << ", weight " << weight_grad->data_type() << ".";
    return RET_ERROR;
  }
  // The third input only carries the weight shape; the output already holds it after inference.
  if (WeightShape()->data_type() != kNumberTypeInt32 || WeightShape()->ElementsNum() != kConvRank) {
    MS_LOG(ERROR) << "Conv2DBackpropFilter weight shape input must hold " << kConvRank << " integers, got "
                  << WeightShape()->ElementsNum() << " of dtype " << WeightShape()->data_type() << ".";
    return RET_ERROR;
  }
  return RET_OK;
}

int Conv2DBackpropFilterDSPKernel::ResolveGeometry() {
  if (CheckTensors() != RET_OK) {
    return RET_ERROR;
  }
  auto *param = reinterpret_cast<ConvParameter *>(op_parameter_);
  const auto &dy_shape = Dy()->shape();
  const auto &x_shape = X()->shape();
  batch_ = dy_shape[kBatchDim];
  if (batch_ != x_shape[kBatchDim]) {
    MS_LOG(ERROR) << "Conv2DBackpropFilter batch differs: gradient " << batch_ << ", input " << x_shape[kBatchDim]
                  << ".";
    return RET_ERROR;
  }
  input_h_ = x_shape[kHeightDim];
  input_w_ = x_shape[kWidthDim];
  input_channel_ = x_shape[kChannelDim];
  output_h_ = dy_shape[kHeightDim];
  output_w_ = dy_shape[kWidthDim];
  output_channel_ = dy_shape[kChannelDim];
  const std::array<int, kConvRank> extents = {input_h_, input_w_, output_h_, output_w_};
  const auto bad_extent = std::find_if(extents.begin(), extents.end(), [](int extent) { return extent <= 0; });
  if (bad_extent != extents.end()) {
    MS_LOG(ERROR) << "Conv2DBackpropFilter has a spatial extent of " << *bad_extent << ".";
    return RET_ERROR;
  }
  // conv2d_backprop_filter_func rejects a group that does not divide both channel counts.
  const int group = param->group_;
  if (group <= 0 || input_channel_ % group != 0 || output_channel_ % group != 0) {
    MS_LOG(ERROR) << "Conv2DBackpropFilter group " << group << " does not divide the channels " << input_channel_
                  << " and " << output_channel_ << ".";
    return RET_ERROR;
  }
  if (input_channel_ <= 0 || output_channel_ <= 0 || batch_ <= 0) {
    MS_LOG(ERROR) << "Conv2DBackpropFilter has a channel or batch extent of " << input_channel_ << ", "
                  << output_channel_ << ", " << batch_ << ".";
    return RET_ERROR;
  }
  const std::array<int, 6> positives = {param->kernel_h_, param->kernel_w_,   param->stride_h_,
                                        param->stride_w_, param->dilation_h_, param->dilation_w_};
  const auto bad_setting = std::find_if(positives.begin(), positives.end(), [](int value) { return value <= 0; });
  if (bad_setting != positives.end()) {
    MS_LOG(ERROR) << "Conv2DBackpropFilter has a kernel, stride or dilation setting of " << *bad_setting << ".";
    return RET_ERROR;
  }
  // The pads handed to the DSP are these fields as they stand. Under Pad_same the padding a forward
  // conv really uses is derived at runtime and the fields can still say 0, which would make the
  // gradient accumulate at positions the forward pass cropped. conv2d.cc declines the same shape of
  // case, so both kernels make the scheduler see the same contract.
  if (param->pad_mode_ == Pad_same) {
    MS_LOG(ERROR) << "Conv2DBackpropFilter cannot use Pad_same padding, it is derived at runtime.";
    return RET_ERROR;
  }
  const std::array<int, 4> pads = {param->pad_u_, param->pad_d_, param->pad_l_, param->pad_r_};
  const auto bad_pad = std::find_if(pads.begin(), pads.end(), [](int pad) { return pad < 0; });
  if (bad_pad != pads.end()) {
    MS_LOG(ERROR) << "Conv2DBackpropFilter pads with " << *bad_pad << ".";
    return RET_ERROR;
  }
  return CheckWeightShape();
}

int Conv2DBackpropFilterDSPKernel::CheckWeightShape() const {
  auto *param = reinterpret_cast<ConvParameter *>(op_parameter_);
  const auto &weight_shape = WeightGrad()->shape();
  if (static_cast<int>(weight_shape.size()) != kConvRank) {
    MS_LOG(ERROR) << "Conv2DBackpropFilter weight output has rank " << weight_shape.size() << ", not " << kConvRank
                  << ".";
    return RET_ERROR;
  }
  // One filter per output channel and kernel position, over the channels of its own group.
  const std::array<int, kConvRank> expected = {output_channel_, param->kernel_h_, param->kernel_w_,
                                               input_channel_ / param->group_};
  for (int dim = 0; dim < kConvRank; ++dim) {
    if (weight_shape[dim] != expected[dim]) {
      MS_LOG(ERROR) << "Conv2DBackpropFilter weight dimension " << dim << " is " << weight_shape[dim] << " but the "
                    << "gradient joins to " << expected[dim] << ".";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int Conv2DBackpropFilterDSPKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  auto allocator = dsp_runtime_->GetAllocator();
  if (allocator == nullptr) {
    MS_LOG(ERROR) << "Conv2DBackpropFilter DSP allocator is null.";
    return RET_ERROR;
  }
  // A graph whose weight shape was unknown at creation never reached CheckSpecs, and a resized graph
  // may skip Prepare, so the geometry is settled again before any pointer goes over.
  if (ResolveGeometry() != RET_OK) {
    return RET_ERROR;
  }
  auto *param = reinterpret_cast<ConvParameter *>(op_parameter_);
  const uint64_t dy_device_ptr = allocator->GetDeviceMemPtr(Dy()->data());
  if (dy_device_ptr == 0) {
    MS_LOG(ERROR) << "Conv2DBackpropFilter gradient device ptr is null.";
    return RET_ERROR;
  }
  const uint64_t x_device_ptr = allocator->GetDeviceMemPtr(X()->data());
  if (x_device_ptr == 0) {
    MS_LOG(ERROR) << "Conv2DBackpropFilter input device ptr is null.";
    return RET_ERROR;
  }
  const uint64_t dw_device_ptr = allocator->GetDeviceMemPtr(WeightGrad()->data());
  if (dw_device_ptr == 0) {
    MS_LOG(ERROR) << "Conv2DBackpropFilter weight device ptr is null.";
    return RET_ERROR;
  }
  int ret = conv2d_backprop_filter_func(
    dy_device_ptr, x_device_ptr, dw_device_ptr, kNoBiasData, batch_, input_h_, input_w_, input_channel_, output_h_,
    output_w_, output_channel_, param->kernel_h_, param->kernel_w_, param->stride_h_, param->stride_w_, param->pad_u_,
    param->pad_d_, param->pad_l_, param->pad_r_, param->dilation_h_, param->dilation_w_, param->group_,
    static_cast<unsigned int>(core_mask_), static_cast<int>(Dy()->data_type()), static_cast<int>(GetMemType()));
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Conv2DBackpropFilter run failed, ret: " << ret;
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kDSP, kNumberTypeFloat32, PrimitiveType_Conv2DBackpropFilterFusion,
           DSPKernelCreator<Conv2DBackpropFilterDSPKernel>)

#ifdef SUPPORT_FT04
REG_KERNEL(kDSP, kNumberTypeFloat16, PrimitiveType_Conv2DBackpropFilterFusion,
           DSPKernelCreator<Conv2DBackpropFilterDSPKernel>)
#endif
}  // namespace mindspore::kernel
