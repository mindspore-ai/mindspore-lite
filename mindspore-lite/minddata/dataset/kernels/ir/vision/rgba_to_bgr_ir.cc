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
#include "mindspore-lite/minddata/dataset/kernels/ir/vision/rgba_to_bgr_ir.h"

#ifndef ENABLE_ANDROID
#include "mindspore-lite/minddata/dataset/kernels/image/rgba_to_bgr_op.h"
#endif

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// RgbaToBgrOperation.
RgbaToBgrOperation::RgbaToBgrOperation() = default;

RgbaToBgrOperation::~RgbaToBgrOperation() = default;

std::string RgbaToBgrOperation::Name() const { return kRgbaToBgrOperation; }

Status RgbaToBgrOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> RgbaToBgrOperation::Build() {
  std::shared_ptr<RgbaToBgrOp> tensor_op = std::make_shared<RgbaToBgrOp>();
  return tensor_op;
}

Status RgbaToBgrOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  *operation = std::make_shared<vision::RgbaToBgrOperation>();
  return Status::OK();
}
#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
