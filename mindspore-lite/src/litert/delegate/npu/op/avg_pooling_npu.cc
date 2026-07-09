/**
 * Copyright 2020-2026 Huawei Technologies Co., Ltd
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

#include "src/litert/delegate/npu/op/avg_pooling_npu.h"
#include "src/litert/delegate/npu/npu_converter_utils.h"
#include "src/litert/delegate/npu/npu_manager.h"

namespace mindspore::lite {
constexpr int MAX_HW_SIZE = 65534;

int AvgPoolingNPUOp::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                               const std::vector<mindspore::MSTensor> &out_tensors) {
  auto pooling_prim = primitive->value_as_AvgPoolFusion();
  if (pooling_prim == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }
  auto stride_h = static_cast<int>(*(pooling_prim->strides()->begin()));
  auto stride_w = static_cast<int>(*(pooling_prim->strides()->begin() + 1));
  if (pooling_prim->pad() != nullptr) {
    CHECK_LESS_RETURN(pooling_prim->pad()->size(), DIMENSION_4D);
    auto pad_u = static_cast<int>(*(pooling_prim->pad()->begin() + PAD_UP));
    auto pad_d = static_cast<int>(*(pooling_prim->pad()->begin() + PAD_DOWN));
    auto pad_l = static_cast<int>(*(pooling_prim->pad()->begin() + PAD_LEFT));
    auto pad_r = static_cast<int>(*(pooling_prim->pad()->begin() + PAD_RIGHT));
    if (pad_u < 0 || pad_d < 0 || pad_l < 0 || pad_r < 0 || pad_u > stride_h || pad_l > stride_w) {
      MS_LOG(WARNING) << "Npu pooling does not support pad < 0 or pad > stride.";
      return RET_NOT_SUPPORT;
    }
  }
  CHECK_LESS_RETURN(in_tensors.size(), 1);
  auto input_shape = in_tensors.front().Shape();
  auto height = input_shape.at(NHWC_H);
  auto width = input_shape.at(NHWC_W);
  if (!NPUManager::CheckDDKVerGreatEqual("100.330.011.032") && height * width > MAX_HW_SIZE) {
    MS_LOG(WARNING) << "The pooling size of " << name_ << " exceeds the max size that NPU support.";
    return RET_NOT_SUPPORT;
  }
  return RET_OK;
}

// SetPoolingParam(), Init(), SetNPUInputs(), GetNPUOp(), ~AvgPoolingNPUOp() are implemented by PoolingNPUOp base class
}  // namespace mindspore::lite
