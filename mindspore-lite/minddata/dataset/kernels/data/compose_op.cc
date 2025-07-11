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
#include "mindspore-lite/minddata/dataset/kernels/data/compose_op.h"

#include "mindspore-lite/minddata/dataset/core/tensor.h"
#include "mindspore-lite/minddata/dataset/kernels/tensor_op.h"
#include "mindspore-lite/minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
Status ComposeOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  std::vector<TensorShape> in_shapes = inputs;
  for (auto &op : ops_) {
    RETURN_IF_NOT_OK(op->OutputShape(in_shapes, outputs));
    in_shapes = std::move(outputs);  // outputs become empty after move
  }
  outputs = std::move(in_shapes);
  return Status::OK();
}

Status ComposeOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  std::vector<DataType> in_types = inputs;
  for (auto &op : ops_) {
    RETURN_IF_NOT_OK(op->OutputType(in_types, outputs));
    in_types = std::move(outputs);  // outputs become empty after move
  }
  outputs = std::move(in_types);
  return Status::OK();
}

Status ComposeOp::Compute(const TensorRow &inputs, TensorRow *outputs) {
  IO_CHECK_VECTOR(inputs, outputs);
  CHECK_FAIL_RETURN_UNEXPECTED(!ops_.empty(), "Compose: transform list should not be empty.");
  TensorRow in_rows = inputs;
  for (auto &op : ops_) {
    RETURN_IF_NOT_OK(op->Compute(in_rows, outputs));
    in_rows = std::move(*outputs);  // after move, *outputs become empty
  }
  (*outputs) = std::move(in_rows);
  return Status::OK();
}

ComposeOp::ComposeOp(const std::vector<std::shared_ptr<TensorOp>> &ops) : ops_(ops) {}
}  // namespace dataset
}  // namespace mindspore
