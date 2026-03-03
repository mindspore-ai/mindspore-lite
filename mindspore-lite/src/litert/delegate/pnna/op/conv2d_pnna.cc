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

#include "src/litert/delegate/pnna/op/conv2d_pnna.h"
#include "src/litert/delegate/delegate_utils.h"
#include "src/common/common.h"
#include "src/litert/cxx_api/tensor/tensor_impl.h"
#include "nnacl_c/int8/pack_int8.h"
#include "src/litert/delegate/pnna/pnna_utils.h"

namespace mindspore {
namespace lite {
void UpdateConv2DPadAndDilation(uint32_t input_size, uint32_t filter_height_or_width, schema::PadMode pad_mode,
                                uint32_t *pad_top_or_left, uint32_t *pad_bottom_or_right,
                                uint32_t stride_height_or_width, uint32_t *dilation_height_or_width) {
  if (pad_mode == schema::PadMode::PadMode_SAME) {
    auto output_size = (input_size + stride_height_or_width - 1) / stride_height_or_width;
    int32_t pad_size = (output_size - 1) * stride_height_or_width + filter_height_or_width - input_size;
    pad_size = pad_size < 0 ? 0 : pad_size;
    *pad_top_or_left = pad_size / 2;
    *pad_bottom_or_right = pad_size - *pad_top_or_left;
    *dilation_height_or_width = 1;
  } else if (pad_mode == schema::PadMode::PadMode_VALID) {
    *pad_top_or_left = 0;
    *pad_bottom_or_right = 0;
  }
}

int PNNAConv2d::InitParams() {
  auto conv = op_primitive_->value_as_Conv2DFusion();
  MS_ASSERT(conv != nullptr);
  group_ = static_cast<uint32_t>(conv->group());
  in_channel_ = static_cast<uint32_t>(conv->in_channel());
  out_channel_ = static_cast<uint32_t>(conv->out_channel());
  is_dw_conv_ = group_ != 1 && group_ == in_channel_ && group_ == out_channel_;
  is_group_conv_ = group_ != 1 && !is_dw_conv_;

  pad_mode_ = conv->pad_mode();
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
  act_type_ = conv->activation_type();
  return RET_OK;
}

int PNNAConv2d::AddOpToPNNAModel(PNNASubGraph *graph) {
  MS_CHECK_TRUE_RET(graph != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(graph->graph() != nullptr, RET_ERROR);

  int ret = HandleConstantInputs(graph, &in_tensors_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "handle constant inputs failed.";
    return RET_ERROR;
  }

  /* (1) For NCHW, [C_out, 1, filter_height, filter_width] when */
  /* depthwise_mode=TRUE, otherwise is [C_out, C_in, filter_height, */
  /* filter_width] */
  /* (2) For NHWC, [1, filter_height, filter_width, C_out] when */
  /* depthwise_mode=TRUE, otherwise is [C_out, filter_height, filter_width,*/
  /* C_in] */
  /* (3) For HWCN, [filter_height, filter_width, C_in, C_out] when */
  /* depthwise_mode=TRUE, otherwise is [filter_height, filter_width, */
  /* C_in, C_out] */
  uint32_t output_channel_size;
  uint32_t filter_height;
  uint32_t filter_width;
  uint32_t filter_channel_size;

  ret = GetConvFilterDims(in_tensors_[kWeightIndex], &output_channel_size, &filter_channel_size, &filter_height,
                          &filter_width, is_dw_conv_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "GetDeConvFilterDims failed.";
    return RET_ERROR;
  }
  if (pad_mode_ != schema::PadMode::PadMode_PAD) {
    UpdateConv2DPadAndDilation(in_tensors_[kInputIndex].Shape()[kNCHW_H], filter_height, pad_mode_, &pad_height_top_,
                               &pad_height_bottom_, stride_height_, &dilation_height_);
    UpdateConv2DPadAndDilation(in_tensors_[kInputIndex].Shape()[kNCHW_W], filter_width, pad_mode_, &pad_width_left_,
                               &pad_width_right_, stride_width_, &dilation_width_);
  }

  std::vector<std::shared_ptr<pnna::Tensor>> input_tensors;
  auto input_tensor = graph->GetMappedTensor(&in_tensors_[kInputIndex]);
  if (!input_tensor) {
    input_tensor = graph->ConvertOperand(&in_tensors_[kInputIndex]);
  }
  input_tensors.emplace_back(input_tensor);
  int32_t multiplier = 0;
  auto filter_tensor = graph->ConvertOperand(&in_tensors_[kWeightIndex]);
  if (is_dw_conv_) {
    multiplier = output_channel_size / group_;
    std::vector<int64_t> filter_dimensions(in_tensors_[1].Shape().data(),
                                           in_tensors_[1].Shape().data() + in_tensors_[1].Shape().size());
    MS_CHECK_GT(in_tensors_[1].Shape().size(), 2, RET_ERROR);
    // Oc,1,H,W -> 1,Oc,H,W
    filter_dimensions[0] = filter_dimensions[1];
    filter_dimensions[1] = output_channel_size;
    auto filter_spec = filter_tensor->GetSpec();
    auto quant_param = filter_tensor->GetQuantization();
    if (quant_param.Type() == pnna::QuantType::SYMMETRIC_PER_CHANNEL) {
      quant_param.SetChannelDim(2);
    }
    filter_spec.SetShape(ConvertToPnnaShapeType(filter_dimensions));
    filter_spec.SetQuantization(quant_param);
    filter_tensor = graph->graph()->CreateTensor(filter_spec, in_tensors_[kWeightIndex].Data().get());
  }
  input_tensors.emplace_back(filter_tensor);
  if (in_tensors_.size() >= kInputSize2) {
    auto bias_tensor = graph->GetMappedTensor(&in_tensors_[kBiasIndex]);
    if (!bias_tensor) {
      bias_tensor = graph->ConvertOperand(&in_tensors_[kBiasIndex]);
    }
    input_tensors.emplace_back(bias_tensor);
  }
  auto output_tensor = graph->ConvertOperand(&out_tensors_[kOutputIndex]);

  MS_LOG(DEBUG) << " pad_width_left_: " << pad_width_left_ << " pad_width_right_: " << pad_width_right_
                << " pad_height_top_: " << pad_height_top_ << " pad_height_bottom_: " << pad_height_bottom_;
  MS_LOG(DEBUG) << " stride_width_: " << stride_width_ << " stride_height_: " << stride_height_;
  MS_LOG(DEBUG) << " dilation_width_: " << dilation_width_ << " dilation_height_: " << dilation_height_;
  MS_LOG(DEBUG) << " multiplier: " << multiplier;

  auto conv2d_op = graph->graph()->CreateOperation<pnna::ops::Conv2d>(
    std::array<uint32_t, 4>({pad_width_left_, pad_width_right_, pad_height_top_, pad_height_bottom_}),
    std::array<uint32_t, 2>({stride_width_, stride_height_}),
    std::array<uint32_t, 2>({dilation_width_, dilation_height_}), multiplier);
  (*conv2d_op).BindInputs(input_tensors).BindOutput(output_tensor);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
