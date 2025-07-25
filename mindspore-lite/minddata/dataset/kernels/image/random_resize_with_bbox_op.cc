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

#include "mindspore-lite/minddata/dataset/kernels/image/random_resize_with_bbox_op.h"

#include "mindspore-lite/minddata/dataset/kernels/image/resize_with_bbox_op.h"
#include "mindspore-lite/minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
Status RandomResizeWithBBoxOp::Compute(const TensorRow &input, TensorRow *output) {
  // Randomly selects from the following four interpolation methods
  // 0-bilinear, 1-nearest_neighbor, 2-bicubic, 3-area
  IO_CHECK_VECTOR(input, output);
  auto interpolation = static_cast<InterpolationMode>(distribution_(random_generator_));
  std::shared_ptr<TensorOp> resize_with_bbox_op = std::make_shared<ResizeWithBBoxOp>(size1_, size2_, interpolation);
  RETURN_IF_NOT_OK(resize_with_bbox_op->Compute(input, output));
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
