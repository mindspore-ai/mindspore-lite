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
#include "mindspore-lite/minddata/dataset/kernels/data/random_apply_op.h"

#include "mindspore-lite/minddata/dataset/core/tensor.h"
#include "mindspore-lite/minddata/dataset/kernels/tensor_op.h"
#include "mindspore-lite/minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
RandomApplyOp::RandomApplyOp(const std::vector<std::shared_ptr<TensorOp>> &ops, double prob)
    : prob_(prob), rand_double_(0.0, 1.0) {
  compose_ = std::make_unique<ComposeOp>(ops);
}

uint32_t RandomApplyOp::NumOutput() {
  if (compose_->NumOutput() != NumInput()) {
    MS_LOG(WARNING) << "NumOutput!=NumInput (randomApply would randomly affect number of outputs).";
    return 0;
  }
  return compose_->NumOutput();
}

Status RandomApplyOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(compose_->OutputShape(inputs, outputs));
  // randomApply either runs all ops or do nothing. If the two methods don't give the same result. return unknown shape.
  if (inputs != outputs) {  // when RandomApply is not applied, input should be the same as output
    outputs.clear();
    outputs.resize(NumOutput(), TensorShape::CreateUnknownRankShape());
  }
  return Status::OK();
}

Status RandomApplyOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(compose_->OutputType(inputs, outputs));
  if (inputs != outputs) {  // when RandomApply is not applied, input should be the same as output
    outputs.clear();
    outputs.resize(NumOutput(), DataType(DataType::DE_UNKNOWN));
  }
  return Status::OK();
}

Status RandomApplyOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  if (rand_double_(random_generator_) <= prob_) {
    RETURN_IF_NOT_OK(compose_->Compute(input, output));
  } else {
    *output = input;  // copy over the tensors
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
