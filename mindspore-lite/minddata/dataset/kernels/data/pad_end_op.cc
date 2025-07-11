/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "mindspore-lite/minddata/dataset/kernels/data/pad_end_op.h"

#include "mindspore-lite/minddata/dataset/core/tensor.h"
#include "mindspore-lite/minddata/dataset/kernels/data/data_utils.h"
#include "mindspore-lite/minddata/dataset/kernels/tensor_op.h"

namespace mindspore {
namespace dataset {
Status PadEndOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  return PadEnd(input, output, output_shape_.AsVector(), pad_val_);
}

Status PadEndOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  for (auto i = 0; i < inputs.size(); ++i) {
    (void)outputs.emplace_back(TensorShape(output_shape_.AsVector()));
  }
  CHECK_FAIL_RETURN_UNEXPECTED(!outputs.empty(), "PadEnd: invalid input shape.");
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
