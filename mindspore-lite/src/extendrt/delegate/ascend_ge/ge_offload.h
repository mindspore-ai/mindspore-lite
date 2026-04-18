/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_GE_OFFLOAD_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_GE_OFFLOAD_H_

#include "include/api/status.h"
#include "extendrt/delegate/ascend_ge/ge_session_manager.h"

namespace mindspore {
struct OffLoadMemoryInfo {
  size_t const_memory_size = 0;
  size_t feature_memory_size = 0;
  void *device_const_memory = nullptr;  // shared memory, is shared_const_memory_
  void *host_const_memory = nullptr;
  void *device_feature_memory = nullptr;
};

class GeOffLoad {
 public:
  GeOffLoad() {}
  ~GeOffLoad() = default;

  static GeOffLoad &GetInstance();

  Status CalculateMemorySize(const std::shared_ptr<ge::Session> &ge_session, uint32_t graph_id);

  Status InitConstAndFeatureMemory(const std::shared_ptr<ge::Session> &ge_session, uint32_t graph_id);

  Status UpdateSharedDeviceMemory(uint32_t graph_id);

 private:
  // graph_id <=> memory info
  std::map<uint32_t, std::shared_ptr<OffLoadMemoryInfo>> all_offload_memory_info_;
  void *shared_const_memory_ = nullptr;
  size_t shared_const_memory_size_ = 0;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_GE_OFFLOAD_H_
