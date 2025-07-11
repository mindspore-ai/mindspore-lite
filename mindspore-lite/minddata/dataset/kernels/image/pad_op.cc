/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
#include "mindspore-lite/minddata/dataset/kernels/image/pad_op.h"

#include "mindspore-lite/minddata/dataset/include/dataset/constants.h"
#include "mindspore-lite/minddata/dataset/kernels/image/image_utils.h"
#include "mindspore-lite/minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
const BorderType PadOp::kDefBorderType = BorderType::kConstant;
const uint8_t PadOp::kDefFillR = 0;
const uint8_t PadOp::kDefFillG = 0;
const uint8_t PadOp::kDefFillB = 0;

PadOp::PadOp(int32_t pad_top, int32_t pad_bottom, int32_t pad_left, int32_t pad_right, BorderType padding_mode,
             uint8_t fill_r, uint8_t fill_g, uint8_t fill_b)
    : pad_top_(pad_top),
      pad_bottom_(pad_bottom),
      pad_left_(pad_left),
      pad_right_(pad_right),
      boarder_type_(padding_mode),
      fill_r_(fill_r),
      fill_g_(fill_g),
      fill_b_(fill_b) {}

Status PadOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  return Pad(input, output, pad_top_, pad_bottom_, pad_left_, pad_right_, boarder_type_, fill_r_, fill_g_, fill_b_);
}

Status PadOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  TensorShape out({-1, -1, 3});  // we don't know what is output image size, but we know it should be 3 channels
  CHECK_FAIL_RETURN_UNEXPECTED(!inputs.empty(), "Pad: inputs cannot be empty.");
  if (inputs[0].Rank() == 1) {
    outputs.emplace_back(out);
  }
  CHECK_FAIL_RETURN_UNEXPECTED(
    !outputs.empty(),
    "Pad: invalid input shape, expected 1D input, but got input dimension is:" + std::to_string(inputs[0].Rank()));
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
