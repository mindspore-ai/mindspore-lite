/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "src/litert/delegate/pnna/op/pooling_pnna.h"

namespace mindspore {
namespace lite {
bool PNNAPooling::IsSupport() {
  if (type_ != schema::PrimitiveType_AvgPoolFusion && type_ != schema::PrimitiveType_MaxPoolFusion) {
    MS_LOG(ERROR) << "Unsupported pooling operation type " << static_cast<int>(type_) << " is found.";
    return false;
  }
  return true;
}

int PNNAPooling::SetPoolingParams(const flatbuffers::Vector<int64_t> *pads, const flatbuffers::Vector<int64_t> *strides,
                                  const flatbuffers::Vector<int64_t> *kernel_size, bool is_global) {
  MS_CHECK_TRUE_RET(strides != nullptr && kernel_size != nullptr, RET_ERROR);
  if (pad_mode_ == schema::PadMode_PAD) {
    MS_CHECK_TRUE_RET(pads != nullptr, RET_ERROR);
    MS_CHECK_TRUE_RET(pads->size() == DIMENSION_4D, RET_ERROR);
    pad_list_.at(0) = (static_cast<uint32_t>(*(pads->begin() + PAD_LEFT)));
    pad_list_.at(1) = (static_cast<uint32_t>(*(pads->begin() + PAD_RIGHT)));
    pad_list_.at(2) = (static_cast<uint32_t>(*(pads->begin() + PAD_UP)));
    pad_list_.at(3) = (static_cast<uint32_t>(*(pads->begin() + PAD_DOWN)));
  }
  MS_CHECK_TRUE_RET(strides->size() == DIMENSION_2D, RET_ERROR);
  strides_.at(0) = (static_cast<uint32_t>(*(strides->begin() + 1)));
  strides_.at(1) = (static_cast<uint32_t>(*(strides->begin())));
  if (is_global) {
    MS_CHECK_TRUE_RET(in_tensors_.at(0).Shape().size() == DIMENSION_4D, RET_ERROR);
    kernel_size_.at(0) = in_tensors_.at(0).Shape().at(2);
    kernel_size_.at(1) = in_tensors_.at(0).Shape().at(1);
  } else if (kernel_size != nullptr && kernel_size->size() == DIMENSION_2D) {
    kernel_size_.at(0) = static_cast<uint32_t>(*(kernel_size->begin() + 1));
    kernel_size_.at(1) = static_cast<uint32_t>(*(kernel_size->begin()));
  }
  return RET_OK;
}

int PNNAPooling::InitParams() {
  bool is_global = false;
  const flatbuffers::Vector<int64_t> *pads = nullptr;
  const flatbuffers::Vector<int64_t> *strides = nullptr;
  const flatbuffers::Vector<int64_t> *kernel_size = nullptr;
  if (type_ == schema::PrimitiveType_AvgPoolFusion) {
    auto pool = op_primitive_->value_as_AvgPoolFusion();
    MS_CHECK_TRUE_RET(pool != nullptr, RET_ERROR);
    act_type_ = pool->activation_type();
    pads = pool->pad();
    strides = pool->strides();
    kernel_size = pool->kernel_size();
    is_global = pool->global();
    ceil_mode_ = pool->round_mode();
    pad_mode_ = pool->pad_mode();
  } else if (type_ == schema::PrimitiveType_MaxPoolFusion) {
    auto pool = op_primitive_->value_as_MaxPoolFusion();
    MS_CHECK_TRUE_RET(pool != nullptr, RET_ERROR);
    act_type_ = pool->activation_type();
    pads = pool->pad();
    strides = pool->strides();
    kernel_size = pool->kernel_size();
    is_global = pool->global();
    ceil_mode_ = pool->round_mode();
    pad_mode_ = pool->pad_mode();
  }
  return SetPoolingParams(pads, strides, kernel_size, is_global);
}

int PNNAPooling::AddOpToPNNAModel(PNNASubGraph *graph) {
  MS_CHECK_TRUE_RET(graph->graph() != nullptr, RET_ERROR);
  pnna::PoolType pool_type;
  if (type_ == schema::PrimitiveType_AvgPoolFusion) {
    pool_type = pnna::PoolType::AVG;
  } else if (type_ == schema::PrimitiveType_MaxPoolFusion) {
    pool_type = pnna::PoolType::MAX;
  }
  auto input_tensor = graph->GetMappedTensor(&in_tensors_[0]);
  if (!input_tensor) {
    input_tensor = graph->ConvertOperand(&in_tensors_[0]);
  }
  auto output_tensor = graph->ConvertOperand(&out_tensors_[0]);
  pnna::RoundType round_type = ceil_mode_ == schema::RoundMode_CEIL ? pnna::RoundType::CEILING : pnna::RoundType::FLOOR;
  if (pad_mode_ == schema::PadMode_SAME) {
    auto pad_type = pnna::PadType::SAME;
    auto pool2d =
      graph->graph()->CreateOperation<pnna::ops::Pool2d>(pool_type, pad_type, kernel_size_, strides_, round_type);
    (*pool2d).BindInputs({input_tensor}).BindOutputs({output_tensor});
  } else if (pad_mode_ == schema::PadMode_VALID) {
    auto pad_type = pnna::PadType::VALID;
    auto pool2d =
      graph->graph()->CreateOperation<pnna::ops::Pool2d>(pool_type, pad_type, kernel_size_, strides_, round_type);
    (*pool2d).BindInputs({input_tensor}).BindOutputs({output_tensor});
  } else {
    auto pool2d =
      graph->graph()->CreateOperation<pnna::ops::Pool2d>(pool_type, pad_list_, kernel_size_, strides_, round_type);
    (*pool2d).BindInputs({input_tensor}).BindOutputs({output_tensor});
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
