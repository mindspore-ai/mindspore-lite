/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_ASCEND_RESOURCE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_ASCEND_RESOURCE_H_

#include <memory>
#include <string>
#include "mindspore-lite/minddata/dataset/core/device_resource.h"
#include "mindspore-lite/minddata/dataset/core/device_tensor.h"
#include "mindspore-lite/minddata/dataset/core/tensor.h"

namespace mindspore {
namespace dataset {

class AscendResource : public DeviceResource {
 public:
  AscendResource() = default;
  ~AscendResource() = default;

  Status InitResource(uint32_t device_id) override;

  Status FinalizeResource() override;

  Status Sink(const mindspore::MSTensor &host_input, std::shared_ptr<DeviceTensor> *device_input) override;

  Status Pop(const std::shared_ptr<DeviceTensor> &device_output, std::shared_ptr<Tensor> *host_output) override;

  std::shared_ptr<void> GetInstance() override;

  Status DeviceDataRelease() override;

  void *GetContext() override;

  void *GetStream() override;

 private:
  std::shared_ptr<void> processor_;
};

}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_ASCEND_RESOURCE_H_
