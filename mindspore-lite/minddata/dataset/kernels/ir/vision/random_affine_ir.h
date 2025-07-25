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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IR_VISION_RANDOM_AFFINE_IR_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IR_VISION_RANDOM_AFFINE_IR_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "include/api/status.h"
#include "mindspore-lite/minddata/dataset/include/dataset/constants.h"
#include "mindspore-lite/minddata/dataset/include/dataset/transforms.h"
#include "mindspore-lite/minddata/dataset/kernels/ir/tensor_operation.h"

namespace mindspore {
namespace dataset {
namespace vision {
constexpr char kRandomAffineOperation[] = "RandomAffine";

class RandomAffineOperation : public TensorOperation {
 public:
  RandomAffineOperation(const std::vector<float_t> &degrees, const std::vector<float_t> &translate_range,
                        const std::vector<float_t> &scale_range, const std::vector<float_t> &shear_ranges,
                        InterpolationMode interpolation, const std::vector<uint8_t> &fill_value);

  ~RandomAffineOperation() override;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override;

  Status to_json(nlohmann::json *out_json) override;

  static Status from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation);

 private:
  std::vector<float_t> degrees_;          // min_degree, max_degree
  std::vector<float_t> translate_range_;  // maximum x translation percentage, maximum y translation percentage
  std::vector<float_t> scale_range_;      // min_scale, max_scale
  std::vector<float_t> shear_ranges_;     // min_x_shear, max_x_shear, min_y_shear, max_y_shear
  InterpolationMode interpolation_;
  std::vector<uint8_t> fill_value_;
};
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IR_VISION_RANDOM_AFFINE_IR_H_
