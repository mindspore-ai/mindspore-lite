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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IR_VISION_AUTO_CONTRAST_IR_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IR_VISION_AUTO_CONTRAST_IR_H_

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
constexpr char kAutoContrastOperation[] = "AutoContrast";

class AutoContrastOperation : public TensorOperation {
 public:
  AutoContrastOperation(float cutoff, const std::vector<uint32_t> &ignore, const std::string &device_target = "CPU");

  ~AutoContrastOperation() override;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override;

  Status to_json(nlohmann::json *out_json) override;

  static Status from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation);

  MapTargetDevice Type() override;

 private:
  float cutoff_;
  std::vector<uint32_t> ignore_;
  std::string device_target_;  // CPU, Ascend
};
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IR_VISION_AUTO_CONTRAST_IR_H_
