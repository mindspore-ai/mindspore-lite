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

#include "src/litert/delegate/pnna/op/deconv2d_pnna.h"
#include <algorithm>
#include <vector>
#include "src/litert/delegate/delegate_utils.h"

namespace mindspore {
namespace lite {

int PNNADeConv2d::UpdateDeConv2dPad(uint32_t input_size, uint32_t output_size, uint32_t filter_height_or_width,
                                    schema::PadMode pad_mode, uint32_t *pad_top_or_left, uint32_t *pad_bottom_or_right,
                                    uint32_t stride_height_or_width) {
  if (pad_top_or_left == nullptr || pad_bottom_or_right == nullptr) {
    MS_LOG(ERROR) << "Invalid pad parameters.";
    return RET_ERROR;
  }
  auto pad_size = (filter_height_or_width + stride_height_or_width * (input_size - 1) - output_size) / 2;
  pad_size = pad_size > 0 ? pad_size : 0;
  *pad_top_or_left = pad_size;
  *pad_bottom_or_right = pad_size;
  return RET_OK;
}

int PNNADeConv2d::SetDeConv2dPadAndStride() {
  auto conv = op_primitive_->value_as_Conv2dTransposeFusion();
  if (conv->pad_list() != nullptr && conv->pad_list()->size() == DIMENSION_4D) {
    pad_height_top_ = static_cast<uint32_t>(*(conv->pad_list()->begin() + PAD_UP));
    pad_height_bottom_ = static_cast<uint32_t>(*(conv->pad_list()->begin() + PAD_DOWN));
    pad_width_left_ = static_cast<uint32_t>(*(conv->pad_list()->begin() + PAD_LEFT));
    pad_width_right_ = static_cast<uint32_t>(*(conv->pad_list()->begin() + PAD_RIGHT));
  }
  MS_CHECK_TRUE_RET(conv->stride() != nullptr && conv->stride()->size() == DIMENSION_2D, RET_ERROR);
  stride_height_ = static_cast<uint32_t>(*(conv->stride()->begin()));
  stride_width_ = static_cast<uint32_t>(*(conv->stride()->begin() + 1));
  MS_CHECK_TRUE_RET(conv->dilation() != nullptr && conv->dilation()->size() == DIMENSION_2D, RET_ERROR);
  dilation_height_ = static_cast<uint32_t>(*(conv->dilation()->begin()));
  dilation_width_ = static_cast<uint32_t>(*(conv->dilation()->begin() + 1));
  return RET_OK;
}

int PNNADeConv2d::InitParams() {
  auto conv = op_primitive_->value_as_Conv2dTransposeFusion();
  MS_CHECK_TRUE_RET(conv != nullptr, RET_ERROR);
  act_type_ = conv->activation_type();
  MS_CHECK_TRUE_RET(act_type_ == schema::ActivationType_NO_ACTIVATION, RET_ERROR);
  auto weigh_format = in_tensors_[kWeightIndex].format();
  if (weigh_format == NHWC) {
    oc_count_ = static_cast<uint32_t>(in_tensors_[kWeightIndex].Shape()[Index3]);
  } else {
    oc_count_ = static_cast<uint32_t>(in_tensors_[kWeightIndex].Shape()[Index1]);
  }
  auto out_paddings = conv->output_paddings();
  std::vector<uint32_t> out_paddings_;
  if (out_paddings != nullptr) {
    (void)std::transform(out_paddings->begin(), out_paddings->end(), std::back_inserter(out_paddings_),
                         [](uint32_t x) { return x; });
    if (out_paddings_.size() == 1) {
      output_padding_height_ = out_paddings_[Index0];
      output_padding_width_ = out_paddings_[Index0];
    } else if (out_paddings_.size() == 2) {
      output_padding_height_ = out_paddings_[Index0];
      output_padding_width_ = out_paddings_[Index1];
    }
  }
  std::vector<uint32_t> kernel_size_;
  auto kernel_size = conv->kernel_size();
  if (kernel_size != nullptr) {
    (void)std::transform(kernel_size->begin(), kernel_size->end(), std::back_inserter(kernel_size_),
                         [](uint32_t x) { return x; });
    if (kernel_size_.size() == 1) {
      kernel_size_height_ = kernel_size_[Index0];
      kernel_size_width_ = kernel_size_[Index0];
    } else if (kernel_size_.size() == 2) {
      kernel_size_height_ = kernel_size_[Index0];
      kernel_size_width_ = kernel_size_[Index1];
    }
  }
  group_ = static_cast<uint32_t>(conv->group());
  in_channel_ = static_cast<uint32_t>(conv->in_channel());
  out_channel_ = static_cast<uint32_t>(conv->out_channel());
  is_dw_conv_ = (group_ != 1 && group_ == in_channel_ && group_ == out_channel_) ||
                (group_ == 1 && in_channel_ == 1 && out_channel_ == 1);
  is_group_conv_ = group_ != 1 && !is_dw_conv_;
  pad_mode_ = conv->pad_mode();
  switch (pad_mode_) {
    case schema::PadMode::PadMode_PAD:
      pad_type_ = pnna::PadType::AUTO;
      break;
    case schema::PadMode::PadMode_SAME:
      pad_type_ = pnna::PadType::SAME;
      break;
    case schema::PadMode::PadMode_VALID:
      pad_type_ = pnna::PadType::VALID;
      break;
    default:
      pad_type_ = pnna::PadType::AUTO;
      break;
  }
  auto ret = SetDeConv2dPadAndStride();
  MS_CHECK_TRUE_RET(ret == RET_OK, RET_ERROR);
  return RET_OK;
}

int PNNADeConv2d::AddOpToPNNAModel(PNNASubGraph *graph) {
  MS_CHECK_TRUE_RET(graph != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(graph->graph() != nullptr, RET_ERROR);
  uint32_t output_channel_size;
  uint32_t filter_height;
  uint32_t filter_width;
  uint32_t filter_channel_size;
  auto ret = GetConvFilterDims(in_tensors_[kWeightIndex], &output_channel_size, &filter_channel_size, &filter_height,
                               &filter_width, is_dw_conv_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "GetDeConvFilterDims failed.";
    return RET_ERROR;
  }
  if (pad_mode_ != schema::PadMode::PadMode_PAD) {
    // update pad size.
    ret = UpdateDeConv2dPad(in_tensors_[kInputIndex].Shape()[kNCHW_H], out_tensors_[OUTPUT_INDEX].Shape()[kNCHW_W],
                            filter_height, pad_mode_, &pad_height_top_, &pad_height_bottom_, stride_height_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Update pad_height_top_ and pad_height_bottom_ failed.";
      return RET_ERROR;
    }
    ret = UpdateDeConv2dPad(in_tensors_[kInputIndex].Shape()[kNCHW_W], out_tensors_[OUTPUT_INDEX].Shape()[kNCHW_H],
                            filter_width, pad_mode_, &pad_width_left_, &pad_width_right_, stride_width_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Update pad_width_left_ and pad_width_right_ failed.";
      return RET_ERROR;
    }
  }
  std::vector<std::shared_ptr<pnna::Tensor>> input_tensors;
  auto input_tensor = graph->GetMappedTensor(&in_tensors_[kInputIndex]);
  if (!input_tensor) {
    input_tensor = graph->ConvertOperand(&in_tensors_[kInputIndex]);
  }
  input_tensors.emplace_back(input_tensor);
  // handle constant input.
  ret = HandleConstantInputs(graph, &in_tensors_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "handle constant inputs failed.";
    return RET_ERROR;
  }
  auto kernel_tensor = graph->ConvertOperand(&in_tensors_[kWeightIndex]);
  input_tensors.emplace_back(kernel_tensor);
  if (in_tensors_.size() >= kInputSize2) {
    auto bias_tensor = graph->GetMappedTensor(&in_tensors_[kBiasIndex]);
    if (!bias_tensor) {
      bias_tensor = graph->ConvertOperand(&in_tensors_[kBiasIndex]);
    }
    input_tensors.emplace_back(bias_tensor);
  }
  auto output_tensor = graph->ConvertOperand(&out_tensors_[OUTPUT_INDEX]);
  if (is_dw_conv_) {
    auto deconv2d_op = graph->graph()->CreateOperation<pnna::ops::DeConv2d>(
      oc_count_, pad_type_, std::array<uint32_t, 2>({kernel_size_height_, kernel_size_width_}),
      std::array<uint32_t, 2>({stride_height_, stride_width_}),
      std::array<uint32_t, 2>({dilation_height_, dilation_width_}),
      std::array<uint32_t, 4>({pad_width_left_, pad_width_right_, pad_height_top_, pad_height_bottom_}), group_);
    (*deconv2d_op).BindInputs(input_tensors).BindOutput(output_tensor);
  } else {
    auto deconv2d_op = graph->graph()->CreateOperation<pnna::ops::DeConv2d>(
      oc_count_, pad_type_, std::array<uint32_t, 2>({kernel_size_height_, kernel_size_width_}),
      std::array<uint32_t, 2>({stride_height_, stride_width_}),
      std::array<uint32_t, 2>({output_padding_height_, output_padding_height_}),
      std::array<uint32_t, 4>({pad_width_left_, pad_width_right_, pad_height_top_, pad_height_bottom_}));
    (*deconv2d_op).BindInputs(input_tensors).BindOutput(output_tensor);
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
