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

#include "mindspore-lite/minddata/dataset/kernels/image/decode_op.h"

#ifndef ENABLE_ANDROID
#include "mindspore-lite/minddata/dataset/kernels/image/image_utils.h"
#else
#include "mindspore-lite/minddata/dataset/kernels/image/lite_image_utils.h"
#endif
#include "mindspore-lite/minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
const bool DecodeOp::kDefRgbFormat = true;

DecodeOp::DecodeOp(bool rgb) : is_rgb_format_(rgb) {
  if (is_rgb_format_) {  // RGB color mode
    MS_LOG(DEBUG) << "Decode color mode is RGB.";
  } else {
    MS_LOG(DEBUG) << "Decode color mode is BGR.";
  }
}

Status DecodeOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  // check the input tensor shape
  if (input->Rank() != 1) {
    RETURN_STATUS_UNEXPECTED("Decode: invalid input shape, only support 1D input, got rank: " +
                             std::to_string(input->Rank()));
  }
  if (is_rgb_format_) {  // RGB color mode
    return Decode(input, output);
  } else {  // BGR color mode
    RETURN_STATUS_UNEXPECTED(
      "Decode: only support Decoded into RGB image, check input parameter 'rgb' first, its value should be 'True'.");
  }
}

Status DecodeOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  TensorShape out({-1, -1, 3});  // we don't know what is output image size, but we know it should be 3 channels
  if (inputs[0].Rank() == 1) {
    (void)outputs.emplace_back(out);
  }
  CHECK_FAIL_RETURN_UNEXPECTED(
    !outputs.empty(),
    "Decode: invalid input shape, expected 1D input, but got input dimension is:" + std::to_string(inputs[0].Rank()));
  return Status::OK();
}

Status DecodeOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  CHECK_FAIL_RETURN_UNEXPECTED(!inputs.empty(), "Decode: inputs cannot be empty.");
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  outputs[0] = DataType(DataType::DE_UINT8);
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
