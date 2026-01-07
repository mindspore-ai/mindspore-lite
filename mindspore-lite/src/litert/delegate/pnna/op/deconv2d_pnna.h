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

#ifndef MINDSPORE_LITE_SRC_LITERT_DELEGATE_PNNA_OP_DECONV2D_PNNA_H_
#define MINDSPORE_LITE_SRC_LITERT_DELEGATE_PNNA_OP_DECONV2D_PNNA_H_

#include <string>
#include <vector>
#include "src/litert/delegate/pnna/op/pnna_op.h"

namespace mindspore {
namespace lite {
class PNNADeConv2d : public PNNAOp {
 public:
  PNNADeConv2d(const std::string &name, const schema::Primitive *primitive,
               const std::vector<mindspore::MSTensor> &in_tensors, const std::vector<mindspore::MSTensor> &out_tensors,
               schema::QuantType quant_type)
      : PNNAOp(name, primitive, in_tensors, out_tensors, quant_type) {}

  ~PNNADeConv2d() override {}
  bool IsSupport() override { return true; };
  int InitParams() override;
  int AddOpToPNNAModel(PNNASubGraph *graph) override;
  int SetDeConv2dPadAndStride();
  int GetDeConv2dFilterDims(mindspore::MSTensor filter_tensor, uint32_t *c_out, uint32_t *c_in, uint32_t *h,
                            uint32_t *w, bool is_depthwise_mode);
  int UpdateDeConv2dPad(uint32_t input_size, uint32_t output_size, uint32_t filter_height_or_width,
                        schema::PadMode pad_mode, uint32_t *pad_top_or_left, uint32_t *pad_bottom_or_right,
                        uint32_t stride_height_or_width);

 private:
  schema::PadMode pad_mode_;
  uint32_t group_;
  uint32_t in_channel_;
  uint32_t out_channel_;
  uint32_t oc_count_;
  bool is_dw_conv_ = false;
  bool is_group_conv_ = false;
  uint32_t pad_height_top_;
  uint32_t pad_height_bottom_;
  uint32_t pad_width_left_;
  uint32_t pad_width_right_;
  uint32_t stride_height_;
  uint32_t stride_width_;
  uint32_t dilation_height_;
  uint32_t dilation_width_;
  uint32_t output_padding_height_;
  uint32_t output_padding_width_;
  uint16_t kernel_size_height_;
  uint16_t kernel_size_width_;
  pnna::PadType pad_type_;
  schema::ActivationType act_type_ = schema::ActivationType_NO_ACTIVATION;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_LITERT_DELEGATE_PNNA_OP_DECONV2D_PNNA_H_
