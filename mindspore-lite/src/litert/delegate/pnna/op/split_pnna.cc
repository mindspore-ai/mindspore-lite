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

#include "src/litert/delegate/pnna/op/split_pnna.h"

namespace mindspore {
namespace lite {
bool PNNASplit::IsSupport() {
  if (!size_splits_.empty() &&
      std::any_of(size_splits_.begin(), size_splits_.end(), [&](int x) { return x != size_splits_.front(); })) {
    return false;
  }
  auto shape = in_tensors_.front().Shape();
  if (std::find(shape.begin(), shape.end(), static_cast<int64_t>(-1)) != shape.end()) {
    return false;
  }
  int axis = axis_ < 0 ? axis_ + static_cast<int>(shape.size()) : axis_;
  MS_CHECK_TRUE_RET(axis < static_cast<int>(shape.size()), false);
  if (size_splits_.empty()) {
    if (shape[axis] % split_num_ != 0) {
      MS_LOG(ERROR) << "Default split size is not usable.";
      return false;
    }
    int split_size = shape.at(axis) / split_num_;
    for (int i = 0; i < split_num_; i++) {
      size_splits_.push_back(split_size);
    }
  }
  if (size_splits_.front() <= 0 || shape.at(axis) % size_splits_.front() != 0) {
    return false;
  }
  split_num_ = shape.at(axis) / size_splits_.front();
  return true;
}

int PNNASplit::InitParams() {
  auto split = op_primitive_->value_as_Split();
  MS_CHECK_TRUE_RET(split != nullptr, RET_ERROR);
  axis_ = split->axis();
  split_num_ = split->output_num();
  auto split_sizes_vector = split->size_splits();
  if (split_sizes_vector != nullptr && static_cast<int>(split_sizes_vector->size()) <= split_num_) {
    (void)std::transform(split_sizes_vector->begin(), split_sizes_vector->end(), std::back_inserter(size_splits_),
                         [](int x) { return x; });
  }
  return RET_OK;
}

int PNNASplit::AddOpToPNNAModel(PNNASubGraph *graph) {
  MS_CHECK_TRUE_RET(graph != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(graph->graph() != nullptr, RET_ERROR);

  auto input_tensor = graph->GetMappedTensor(&in_tensors_[0]);
  if (!input_tensor) {
    input_tensor = graph->ConvertOperand(&in_tensors_[0]);
  }
  std::vector<std::shared_ptr<pnna::Tensor>> output_tensors(split_num_);
  // WHCN
  uint32_t mapped_axis = ConvertToPnnaAxis(axis_, in_tensors_.front().Shape().size());
  std::vector<uint32_t> mapped_split;
  for (int i = 0; i < split_num_; i++) {
    mapped_split.push_back(size_splits_[i]);
    output_tensors[i] = graph->ConvertOperand(&out_tensors_[i]);
  }
  auto split_op = graph->graph()->CreateOperation<pnna::ops::Split>(mapped_axis, mapped_split);
  split_op->BindInputs({input_tensor});
  split_op->BindOutputs(output_tensors);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
