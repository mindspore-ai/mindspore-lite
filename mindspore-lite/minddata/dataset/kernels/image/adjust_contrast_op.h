/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_ADJUST_CONTRAST_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_ADJUST_CONTRAST_OP_H_

#include <memory>
#include <string>

#include "mindspore-lite/minddata/dataset/core/cv_tensor.h"
#include "mindspore-lite/minddata/dataset/core/tensor.h"
#include "mindspore-lite/minddata/dataset/kernels/tensor_op.h"
#include "mindspore-lite/minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class AdjustContrastOp : public TensorOp {
 public:
  explicit AdjustContrastOp(float contrast_factor) : contrast_factor_(contrast_factor) {}

  ~AdjustContrastOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kAdjustContrastOp; }

 private:
  float contrast_factor_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_CONTRAST_OP_H_
