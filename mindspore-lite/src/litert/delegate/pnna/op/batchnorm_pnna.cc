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

#include "src/litert/delegate/pnna/op/batchnorm_pnna.h"
#include <vector>
#include "src/litert/kernel/cpu/nnacl_c/int8/quantize.h"

namespace mindspore {
namespace lite {
namespace {
constexpr size_t kScaleIndex = 1;
constexpr size_t kOffsetIndex = 2;
constexpr size_t kMeanIndex = 3;
constexpr size_t kVarianceIndex = 4;
}  // namespace

bool PNNABatchnorm::IsSupport() {
  auto input = in_tensors_.front();
  if (!in_tensors_[kScaleIndex].IsConst() && !in_tensors_[kOffsetIndex].IsConst() &&
      !in_tensors_[kMeanIndex].IsConst() && !in_tensors_[kVarianceIndex].IsConst()) {
    return false;
  }
  return input.Shape().size() == DIMENSION_4D;
}

int PNNABatchnorm::InitParams() {
  auto batchnorm = op_primitive_->value_as_FusedBatchNorm();
  MS_CHECK_TRUE_RET(batchnorm != nullptr, RET_ERROR);
  epsilon_ = batchnorm->epsilon();
  return RET_OK;
}

MSTensor PNNABatchnorm::DequantizeTensor(const MSTensor &tensor) {
  auto quant = tensor.QuantParams();
  auto scales = static_cast<float>(quant[0].scale);
  auto zp = quant[0].zero_point;
  auto shape = tensor.Shape();
  auto data = reinterpret_cast<const int8_t *>(tensor.Data().get());
  std::vector<float> vec(tensor.ElementNum());
  Dequantize(data, tensor.ElementNum(), scales, zp, vec.data());
  auto dequant_tensor = MSTensor(tensor.Name() + "_int8tofp32", DataType::kNumberTypeFloat32, shape, vec.data(),
                                 tensor.ElementNum() * sizeof(float));
  MS_CHECK_TRUE_RET(dequant_tensor != nullptr, MSTensor());
  dequant_tensor.SetFormat(tensor.format());
  return dequant_tensor;
}

int PNNABatchnorm::AddOpToPNNAModel(PNNASubGraph *graph) {
  MS_CHECK_TRUE_RET(graph != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(graph->graph() != nullptr, RET_ERROR);
  auto input_tensor = graph->GetMappedTensor(&in_tensors_[kInputIndex]);
  if (!input_tensor) {
    input_tensor = graph->ConvertOperand(&in_tensors_[kInputIndex]);
  }
  if (in_tensors_[kInputIndex].DataType() == DataType::kNumberTypeInt8) {
    in_tensors_.at(kScaleIndex) = DequantizeTensor(in_tensors_[kScaleIndex]);
    in_tensors_.at(kOffsetIndex) = DequantizeTensor(in_tensors_[kOffsetIndex]);
    in_tensors_.at(kMeanIndex) = DequantizeTensor(in_tensors_[kMeanIndex]);
    in_tensors_.at(kVarianceIndex) = DequantizeTensor(in_tensors_[kVarianceIndex]);
  }
  auto scale = graph->ConvertOperand(&in_tensors_[kScaleIndex]);
  auto offset = graph->ConvertOperand(&in_tensors_[kOffsetIndex]);
  auto mean = graph->ConvertOperand(&in_tensors_[kMeanIndex]);
  auto variance = graph->ConvertOperand(&in_tensors_[kVarianceIndex]);
  auto output_tensor = graph->ConvertOperand(&out_tensors_[kOutputIndex]);
  auto batch_norm_op = graph->graph()->CreateOperation<pnna::ops::BatchNorm>(epsilon_);
  batch_norm_op->BindInputs({input_tensor, mean, variance, scale, offset});
  batch_norm_op->BindOutputs({output_tensor});
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
