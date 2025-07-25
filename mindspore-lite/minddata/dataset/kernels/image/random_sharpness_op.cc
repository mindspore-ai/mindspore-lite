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

#include "mindspore-lite/minddata/dataset/kernels/image/random_sharpness_op.h"

#include "mindspore-lite/minddata/dataset/kernels/image/image_utils.h"

namespace mindspore {
namespace dataset {
/// constructor
RandomSharpnessOp::RandomSharpnessOp(float start_degree, float end_degree)
    : start_degree_(start_degree), end_degree_(end_degree) {}

/// main function call for random sharpness : Generate the random degrees
Status RandomSharpnessOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  float random_double = distribution_(random_generator_);
  /// get the degree sharpness range
  /// the way this op works (uniform distribution)
  /// assumption here is that mDegreesEnd > mDegreeStart so we always get positive number
  float degree_range = (end_degree_ - start_degree_) / 2;
  float mid = (end_degree_ + start_degree_) / 2;
  float alpha = mid + random_double * degree_range;
  return AdjustSharpness(input, output, alpha);
}
}  // namespace dataset
}  // namespace mindspore
