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

#include "src/litert/delegate/pnna/op/pad_pnna.h"
#include <algorithm>

namespace mindspore {
namespace lite {
namespace {
constexpr size_t kFrontSizeIndex = 0;
constexpr size_t kBackSizeIndex = 1;
}  // namespace

bool PNNAPad::IsSupport() { return true; }

int PNNAPad::InitParams() {
  auto pad = op_primitive_->value_as_PadFusion();
  MS_CHECK_TRUE_RET(pad != nullptr, RET_ERROR);
  const_val_ = static_cast<int32_t>(pad->constant_value());
  auto paddings = pad->paddings();
  MS_CHECK_TRUE_RET(paddings != nullptr, RET_ERROR);
  auto fb_paddings = paddings->data();
  auto padding_size = fb_paddings->size();
  front_size_.reserve(padding_size);
  back_size_.reserve(padding_size);
  for (size_t i = 0; i < padding_size; ++i) {
    auto padding_addr = (fb_paddings->begin() + i)->data();
    auto padding_value = std::vector<uint32_t>(padding_addr->begin(), padding_addr->end());
    MS_CHECK_TRUE_RET(padding_value.size() == 2, RET_ERROR);
    front_size_.push_back(padding_value[kFrontSizeIndex]);
    back_size_.push_back(padding_value[kBackSizeIndex]);
  }
  std::reverse(front_size_.begin(), front_size_.end());
  std::reverse(back_size_.begin(), back_size_.end());
  auto padding_mode = pad->padding_mode();
  switch (padding_mode) {
    case schema::PaddingMode::PaddingMode_CONSTANT:
      pad_mode_ = pnna::ops::Pad::pad_mode_type::PAD_MODE_CONSTANT;
      break;
    case schema::PaddingMode::PaddingMode_REFLECT:
      pad_mode_ = pnna::ops::Pad::pad_mode_type::PAD_MODE_REFLECT;
      break;
    case schema::PaddingMode::PaddingMode_SYMMETRIC:
      pad_mode_ = pnna::ops::Pad::pad_mode_type::PAD_MODE_SYMMETRIC;
      break;
    case schema::PaddingMode::PaddingMode_MODE_RESERVED:
      pad_mode_ = pnna::ops::Pad::pad_mode_type::PAD_MODE_EDGE;
      break;
    default:
      pad_mode_ = pnna::ops::Pad::pad_mode_type::PAD_MODE_CONSTANT;
      break;
  }
  return RET_OK;
}

int PNNAPad::AddOpToPNNAModel(PNNASubGraph *graph) {
  MS_CHECK_TRUE_RET(graph != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(graph->graph() != nullptr, RET_ERROR);
  auto input_tensor = graph->GetMappedTensor(&in_tensors_[Index0]);
  if (!input_tensor) {
    input_tensor = graph->ConvertOperand(&in_tensors_[Index0]);
  }
  auto output_tensor = graph->ConvertOperand(&out_tensors_[kOutputIndex]);
  auto pad_op = graph->graph()->CreateOperation<pnna::ops::Pad>(front_size_, back_size_, const_val_, pad_mode_);
  pad_op->BindInputs({input_tensor});
  pad_op->BindOutputs({output_tensor});
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
