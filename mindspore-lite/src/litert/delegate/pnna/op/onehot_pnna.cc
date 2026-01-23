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

#include "src/litert/delegate/pnna/op/onehot_pnna.h"
#include <vector>
#include "src/litert/delegate/pnna/pnna_utils.h"
#include "src/litert/kernel/cpu/nnacl_c/int8/quantize.h"

namespace mindspore {

namespace {
constexpr int32_t kOneHotInputSize = 3;
constexpr int32_t kDepthTensorIndex = 1;
constexpr int32_t kOffOnTensorIndex = 2;
constexpr int32_t kOffValueIndex = 0;
constexpr int32_t kOnValueIndex = 1;
}  // namespace

namespace lite {
bool PNNAOneHot::IsSupport() {
  if (in_tensors_.size() != kOneHotInputSize) {
    return false;
  }
  for (size_t i = kDepthTensorIndex; i < in_tensors_.size(); ++i) {
    if (!in_tensors_[i].IsConst()) {
      return false;
    }
  }
  return true;
}

int PNNAOneHot::InitParams() {
  auto onehot = op_primitive_->value_as_OneHot();
  MS_CHECK_TRUE_RET(onehot != nullptr, RET_ERROR);
  axis_ = onehot->axis();

  auto depth_tensor = in_tensors_.at(kDepthTensorIndex);
  MS_CHECK_TRUE_RET(depth_tensor.IsConst(), RET_ERROR);
  MS_CHECK_TRUE_RET(
    depth_tensor.DataType() == DataType::kNumberTypeInt64 || depth_tensor.DataType() == DataType::kNumberTypeInt32,
    RET_ERROR);

  auto depth_data = depth_tensor.MutableData();
  MS_CHECK_TRUE_RET(depth_data != nullptr, RET_ERROR);
  if (depth_tensor.DataType() == DataType::kNumberTypeInt64) {
    auto depth_data_int64 = reinterpret_cast<const int64_t *>(depth_data);
    depth_ = static_cast<int32_t>(*depth_data_int64);
  } else {
    auto depth_data_int32 = reinterpret_cast<const int32_t *>(depth_data);
    depth_ = *depth_data_int32;
  }

  auto off_on_tensor = in_tensors_.at(kOffOnTensorIndex);
  MS_CHECK_TRUE_RET(off_on_tensor.IsConst(), RET_ERROR);

  auto off_on_data = off_on_tensor.MutableData();
  MS_CHECK_TRUE_RET(off_on_data != nullptr, RET_ERROR);

  if (off_on_tensor.DataType() == DataType::kNumberTypeFloat32) {
    auto off_on_data_float = reinterpret_cast<const float *>(off_on_data);
    off_value_ = off_on_data_float[kOffValueIndex];
    on_value_ = off_on_data_float[kOnValueIndex];
  } else if (off_on_tensor.DataType() == DataType::kNumberTypeInt32) {
    auto off_on_data_int32 = reinterpret_cast<const int32_t *>(off_on_data);
    off_value_ = static_cast<float>(off_on_data_int32[kOffValueIndex]);
    on_value_ = static_cast<float>(off_on_data_int32[kOnValueIndex]);
  } else if (off_on_tensor.DataType() == DataType::kNumberTypeInt64) {
    auto off_on_data_int64 = reinterpret_cast<const int64_t *>(off_on_data);
    off_value_ = static_cast<float>(off_on_data_int64[kOffValueIndex]);
    on_value_ = static_cast<float>(off_on_data_int64[kOnValueIndex]);
  } else if (off_on_tensor.DataType() == DataType::kNumberTypeInt8) {
    auto off_on_data_int8 = reinterpret_cast<const int8_t *>(off_on_data);
    std::vector<int8_t> off_on_data_vec(off_on_tensor.ElementNum());
    std::memcpy(off_on_data_vec.data(), off_on_data_int8, sizeof(int8_t) * off_on_tensor.ElementNum());
    auto off_on_data_quant = off_on_tensor.QuantParams();
    MS_CHECK_TRUE_RET(!off_on_data_quant.empty(), RET_ERROR);
    auto off_on_data_scale = off_on_data_quant.front().scale;
    auto off_on_data_zero_point = off_on_data_quant.front().zero_point;
    std::vector<float> off_on_data_dequant(off_on_tensor.ElementNum());
    Dequantize(off_on_data_vec.data(), off_on_data_vec.size(), off_on_data_scale, off_on_data_zero_point,
               off_on_data_dequant.data());
    off_value_ = off_on_data_dequant[kOffValueIndex];
    on_value_ = off_on_data_dequant[kOnValueIndex];
  } else {
    return RET_ERROR;
  }
  return RET_OK;
}

int PNNAOneHot::AddOpToPNNAModel(PNNASubGraph *graph) {
  MS_CHECK_TRUE_RET(graph != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(graph->graph() != nullptr, RET_ERROR);
  auto input_tensor = graph->GetMappedTensor(&in_tensors_[kInputIndex]);
  if (!input_tensor) {
    input_tensor = graph->ConvertOperand(&in_tensors_[kInputIndex]);
  }
  auto output_tensor = graph->ConvertOperand(&out_tensors_[kOutputIndex]);
  auto onehot = graph->graph()->CreateOperation<pnna::ops::OneHot>(
    depth_, on_value_, off_value_, ConvertToPnnaAxis(axis_, out_tensors_.front().Shape().size()));
  (*onehot).BindInput(input_tensor).BindOutput(output_tensor);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
