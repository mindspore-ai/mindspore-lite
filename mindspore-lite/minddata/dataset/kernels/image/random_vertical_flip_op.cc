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

#include "mindspore-lite/minddata/dataset/kernels/image/random_vertical_flip_op.h"

#include "mindspore-lite/minddata/dataset/kernels/image/image_utils.h"
#include "mindspore-lite/minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
Status RandomVerticalFlipOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  const auto output_count = input.size();
  output->resize(output_count);

  for (const auto &image : input) {
    RETURN_IF_NOT_OK(ValidateImageDtype("RandomVerticalFlip", image->type()));
    RETURN_IF_NOT_OK(ValidateImageRank("RandomVerticalFlip", image->Rank()));
  }

  if (distribution_(random_generator_)) {
    for (size_t i = 0; i < input.size(); i++) {
      RETURN_IF_NOT_OK(VerticalFlip(input[i], &(*output)[i]));
    }
    return Status::OK();
  }
  *output = input;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
