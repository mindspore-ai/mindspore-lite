/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_DEVICE_QUEUE_TRACING_H
#define MINDSPORE_DEVICE_QUEUE_TRACING_H

#include <string>
#include <vector>
#include "mindspore-lite/minddata/dataset/engine/perf/profiling.h"
#include "mindspore-lite/minddata/dataset/util/path.h"

namespace mindspore {
namespace dataset {
class DeviceQueueTracing : public Tracing {
 public:
  // Constructor
  DeviceQueueTracing() = default;

  // Destructor
  ~DeviceQueueTracing() override = default;

  std::string Name() const override { return kDeviceQueueTracingName; };

 protected:
  Path GetFileName(const std::string &dir_path, const std::string &rank_id) override;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_DEVICE_QUEUE_TRACING_H
