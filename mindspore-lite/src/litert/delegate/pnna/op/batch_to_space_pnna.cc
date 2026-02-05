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

#include "src/litert/delegate/pnna/op/batch_to_space_pnna.h"
#include <algorithm>
#include <utility>
#include <vector>

namespace mindspore {
namespace lite {

namespace {
constexpr size_t kCropSize1D = 2;
constexpr size_t kCropSize2D = 4;
constexpr size_t kFirstDimensionStart = 0;
constexpr size_t kFirstDimensionEnd = 1;
constexpr size_t kSecondDimensionStart = 2;
constexpr size_t kSecondDimensionEnd = 3;
}  // namespace

bool PNNABatchToSpace::IsSupport() {
  auto input = in_tensors_.front();
  return input.Shape().size() == DIMENSION_4D;
}

int PNNABatchToSpace::InitParams() {
  auto batch2space = op_primitive_->value_as_BatchToSpace();
  MS_CHECK_TRUE_RET(batch2space != nullptr, RET_ERROR);
  auto block_size = batch2space->block_size();
  MS_CHECK_TRUE_RET(block_size != nullptr, RET_ERROR);
  (void)std::transform(block_size->begin(), block_size->end(), std::back_inserter(block_size_),
                       [](int x) { return x; });
  auto crops = batch2space->crops();
  MS_CHECK_TRUE_RET(crops != nullptr, RET_ERROR);
  auto crop_data = crops->data();
  MS_CHECK_TRUE_RET(crop_data != nullptr, RET_ERROR);
  auto crop_begin = crop_data->begin();
  auto crop = crop_begin->data();
  MS_CHECK_TRUE_RET(crop != nullptr, RET_ERROR);
  (void)std::transform(crop->begin(), crop->end(), std::back_inserter(crop_), [](int x) { return x; });
  return RET_OK;
}

std::vector<int> PNNABatchToSpace::ConvertToPnnaCrop(const std::vector<int> &crop) {
  if (crop.empty()) {
    MS_LOG(ERROR) << "Unsupported empty crop size";
    return {};
  }
  std::vector<int> result(crop.size());
  if (crop.size() == kCropSize1D) {
    result[kFirstDimensionStart] = crop[kFirstDimensionEnd];
    result[kFirstDimensionEnd] = crop[kFirstDimensionStart];

  } else if (crop.size() == kCropSize2D) {
    result[kFirstDimensionStart] = crop[kSecondDimensionStart];
    result[kFirstDimensionEnd] = crop[kSecondDimensionEnd];
    result[kSecondDimensionStart] = crop[kFirstDimensionStart];
    result[kSecondDimensionEnd] = crop[kFirstDimensionEnd];
  } else {
    MS_LOG(ERROR) << "Unsupported crop size: " << crop.size();
    return {};
  }
  return result;
}

std::vector<int> PNNABatchToSpace::ConvertToPnnaBlockSize(const std::vector<int> &block_size) {
  if (block_size.empty()) {
    MS_LOG(ERROR) << "Unsupported empty block size";
    return {};
  }
  std::vector<int> result(block_size.size());
  for (size_t i = 0; i < block_size.size(); ++i) {
    result[i] = block_size[block_size.size() - 1 - i];
  }
  return result;
}

int PNNABatchToSpace::AddOpToPNNAModel(PNNASubGraph *graph) {
  MS_CHECK_TRUE_RET(graph != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(graph->graph() != nullptr, RET_ERROR);
  // Convert to pnna tensors and operators
  auto input_tensor = graph->GetMappedTensor(&in_tensors_[Index0]);
  if (!input_tensor) {
    input_tensor = graph->ConvertOperand(&in_tensors_[Index0]);
  }
  auto output_tensor = graph->ConvertOperand(&out_tensors_[kOutputIndex]);
  auto block_size = ConvertToPnnaBlockSize(block_size_);
  MS_CHECK_TRUE_RET(!block_size.empty(), RET_ERROR);
  auto crop = ConvertToPnnaCrop(crop_);
  MS_CHECK_TRUE_RET(!crop.empty(), RET_ERROR);
  auto batch2space_op = graph->graph()->CreateOperation<pnna::ops::Batch2Space>(block_size, crop);
  batch2space_op->BindInputs({input_tensor});
  batch2space_op->BindOutputs({output_tensor});
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
