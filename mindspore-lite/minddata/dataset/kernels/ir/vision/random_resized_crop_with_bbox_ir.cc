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

#include "mindspore-lite/minddata/dataset/kernels/ir/vision/random_resized_crop_with_bbox_ir.h"

#include <algorithm>

#ifndef ENABLE_ANDROID
#include "mindspore-lite/minddata/dataset/kernels/image/random_crop_and_resize_with_bbox_op.h"
#endif
#include "mindspore-lite/minddata/dataset/kernels/ir/validators.h"
#include "mindspore-lite/minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// RandomResizedCropWithBBoxOperation
RandomResizedCropWithBBoxOperation::RandomResizedCropWithBBoxOperation(const std::vector<int32_t> &size,
                                                                       const std::vector<float> &scale,
                                                                       const std::vector<float> &ratio,
                                                                       InterpolationMode interpolation,
                                                                       int32_t max_attempts)
    : size_(size), scale_(scale), ratio_(ratio), interpolation_(interpolation), max_attempts_(max_attempts) {}

RandomResizedCropWithBBoxOperation::~RandomResizedCropWithBBoxOperation() = default;

std::string RandomResizedCropWithBBoxOperation::Name() const { return kRandomResizedCropWithBBoxOperation; }

Status RandomResizedCropWithBBoxOperation::ValidateParams() {
  // size
  RETURN_IF_NOT_OK(ValidateVectorSize("RandomResizedCropWithBBox", size_));
  // scale
  RETURN_IF_NOT_OK(ValidateVectorScale("RandomResizedCropWithBBox", scale_));
  // ratio
  RETURN_IF_NOT_OK(ValidateVectorRatio("RandomResizedCropWithBBox", ratio_));
  // max_attempts
  if (max_attempts_ < 1) {
    std::string err_msg = "RandomResizedCropWithBBox: max_attempts must be greater than or equal to 1, got: " +
                          std::to_string(max_attempts_);
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  // interpolation
  if (interpolation_ != InterpolationMode::kLinear && interpolation_ != InterpolationMode::kNearestNeighbour &&
      interpolation_ != InterpolationMode::kCubic && interpolation_ != InterpolationMode::kArea &&
      interpolation_ != InterpolationMode::kCubicPil) {
    std::string err_msg = "RandomResizedCropWithBBox: Invalid InterpolationMode, check input value of enum.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomResizedCropWithBBoxOperation::Build() {
  constexpr size_t dimension_zero = 0;
  constexpr size_t dimension_one = 1;
  constexpr size_t size_two = 2;

  int32_t height = size_[dimension_zero];
  int32_t width = size_[dimension_zero];
  // User specified the width value.
  if (size_.size() == size_two) {
    width = size_[dimension_one];
  }
  std::shared_ptr<RandomCropAndResizeWithBBoxOp> tensor_op = std::make_shared<RandomCropAndResizeWithBBoxOp>(
    height, width, scale_[dimension_zero], scale_[dimension_one], ratio_[dimension_zero], ratio_[dimension_one],
    interpolation_, max_attempts_);
  return tensor_op;
}

Status RandomResizedCropWithBBoxOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["size"] = size_;
  args["scale"] = scale_;
  args["ratio"] = ratio_;
  args["interpolation"] = interpolation_;
  args["max_attempts"] = max_attempts_;
  *out_json = args;
  return Status::OK();
}

Status RandomResizedCropWithBBoxOperation::from_json(nlohmann::json op_params,
                                                     std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "size", kRandomResizedCropWithBBoxOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "scale", kRandomResizedCropWithBBoxOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "ratio", kRandomResizedCropWithBBoxOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "interpolation", kRandomResizedCropWithBBoxOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "max_attempts", kRandomResizedCropWithBBoxOperation));
  std::vector<int32_t> size = op_params["size"];
  std::vector<float> scale = op_params["scale"];
  std::vector<float> ratio = op_params["ratio"];
  auto interpolation = static_cast<InterpolationMode>(op_params["interpolation"]);
  int32_t max_attempts = op_params["max_attempts"];
  *operation =
    std::make_shared<vision::RandomResizedCropWithBBoxOperation>(size, scale, ratio, interpolation, max_attempts);
  return Status::OK();
}
#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
