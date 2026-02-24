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
#ifndef MINDSPORE_LITE_SRC_LITERT_DELEGATE_PNNA_OP_CONV2D_PNNA_H_
#define MINDSPORE_LITE_SRC_LITERT_DELEGATE_PNNA_OP_CONV2D_PNNA_H_

#include "src/litert/delegate/pnna/op/pnna_op.h"

namespace mindspore {
namespace lite {
class PNNAConv2d : public PNNAOp {
 public:
  PNNAConv2d(const std::string &name, const schema::Primitive *primitive,
             const std::vector<mindspore::MSTensor> &in_tensors, const std::vector<mindspore::MSTensor> &out_tensors,
             schema::QuantType quant_type)
      : PNNAOp(name, primitive, in_tensors, out_tensors, quant_type) {}

  ~PNNAConv2d() override {}

  bool IsSupport() override { return true; };
  int InitParams() override;
  int AddOpToPNNAModel(PNNASubGraph *graph) override;

 private:
  schema::PadMode pad_mode_;
  uint32_t group_ = 0;
  uint32_t in_channel_ = 0;
  uint32_t out_channel_ = 0;
  bool is_dw_conv_ = false;
  bool is_group_conv_ = false;
  uint32_t pad_height_top_ = 0;
  uint32_t pad_height_bottom_ = 0;
  uint32_t pad_width_left_ = 0;
  uint32_t pad_width_right_ = 0;
  uint32_t stride_height_ = 0;
  uint32_t stride_width_ = 0;
  uint32_t dilation_height_ = 1;
  uint32_t dilation_width_ = 1;
  schema::ActivationType act_type_ = schema::ActivationType_NO_ACTIVATION;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_LITERT_DELEGATE_PNNA_OP_CONV2D_PNNA_H_
