/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IR_TENSOR_OPERATION_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IR_TENSOR_OPERATION_H_

#include <memory>
#include <string>

#include "mindspore-lite/minddata/dataset/kernels/tensor_op.h"
#include "mindspore-lite/minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
// Abstract class to represent a dataset in the data pipeline.
class TensorOperation : public std::enable_shared_from_this<TensorOperation> {
 public:
  /// \brief Constructor
  TensorOperation() : random_op_(false) {}

  /// \brief Constructor
  explicit TensorOperation(bool random) : random_op_(random) {}

  /// \brief Destructor
  virtual ~TensorOperation() = default;

  /// \brief Pure virtual function to convert a TensorOperation class into a runtime TensorOp object.
  /// \return shared pointer to the newly created TensorOp.
  virtual std::shared_ptr<TensorOp> Build() = 0;

  virtual Status ValidateParams() { return Status::OK(); }

  virtual std::string Name() const = 0;

  /// \brief Check whether the operation is deterministic.
  /// \return true if this op is a random op (returns non-deterministic result e.g. RandomCrop)
  bool IsRandomOp() const { return random_op_; }

  virtual Status to_json(nlohmann::json *out_json) { return Status::OK(); }

  virtual MapTargetDevice Type() { return MapTargetDevice::kCpu; }

 protected:
  bool random_op_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IR_TENSOR_OPERATION_H_
