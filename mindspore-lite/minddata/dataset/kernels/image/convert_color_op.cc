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

#include "mindspore-lite/minddata/dataset/kernels/image/convert_color_op.h"

#include "mindspore-lite/minddata/dataset/core/cv_tensor.h"
#include "mindspore-lite/minddata/dataset/kernels/image/image_utils.h"
#include "mindspore-lite/minddata/dataset/util/random.h"
#include "mindspore-lite/minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
ConvertColorOp::ConvertColorOp(ConvertMode convert_mode) : convert_mode_(convert_mode) {}

Status ConvertColorOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  return ConvertColor(input, output, convert_mode_);
}
}  // namespace dataset
}  // namespace mindspore
