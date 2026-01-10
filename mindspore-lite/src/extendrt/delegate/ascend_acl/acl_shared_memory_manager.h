/**
 * Copyright 2022-2026 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_ACL_ACL_SHARED_MEMORY_MANAGER_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_ACL_ACL_SHARED_MEMORY_MANAGER_H_

#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <memory>
#include <utility>
#include <thread>
#include "include/api/status.h"
#include "extendrt/delegate/ascend_acl/acl_model_options.h"

namespace mindspore {
struct AclMemoryInfo {
  void *memory_address = nullptr;
  size_t memory_size = 0;
};

struct SharedMemoryInfo {
  int32_t device_id;
  std::thread::id thread_id;
  std::string model_path;
  AclMemoryInfo memory_info;
  bool allocated;
};
class AclSharedMemoryManager {
 public:
  AclSharedMemoryManager() {}
  ~AclSharedMemoryManager();

  AclSharedMemoryManager(const AclSharedMemoryManager &) = delete;
  AclSharedMemoryManager &operator=(const AclSharedMemoryManager &) = delete;

  static AclSharedMemoryManager &GetInstance();

  Status PrepareMutiModelShare(const void *om_data, size_t om_data_size,
                               const std::shared_ptr<AclModelOptions> &options);
  std::pair<void *, void *> ShareWorkspaceProcess(const size_t &work_size, const size_t &weight_size,
                                                  const std::shared_ptr<AclModelOptions> &options);
  std::pair<void *, void *> ShareWeightspaceProcess(const size_t &work_size,
                                                    const std::shared_ptr<AclModelOptions> &options);
  std::pair<void *, void *> ShareWorkspaceAndWeightspaceProcess(const size_t &work_size,
                                                                const std::shared_ptr<AclModelOptions> &options);

  void ReleaseDeviceMem(int32_t device_id, std::string model_path);
  void Lock(int32_t device_id);
  void Unlock(int32_t device_id);

 private:
  Status UpdateWeightSpace(std::string model_path, size_t weight_size, int32_t device_id);
  Status UpdateWorkSpace(size_t work_size, int32_t device_id);
  Status GetModelWorkMem(void **work_ptr, int32_t device_id);
  Status GetModelWeightMem(void **weight_ptr, std::string model_path, int32_t device_id);

 private:
  std::mutex acl_shared_memory_prepare_mutex_;
  std::mutex acl_shared_memory_execute_mutex_;
  std::map<int32_t, std::mutex> shared_lock_for_device_id_map_;
  // device_id: <memory_info, is_initialize>
  std::map<int32_t, std::pair<AclMemoryInfo, bool>> shared_work_memory_info_map_;
  // device_id: {model_path: shared_memory_info}
  std::map<int32_t, std::map<std::string, SharedMemoryInfo>> shared_weight_memory_info_map_;
  AclMemoryInfo shared_weight_mem_info_ = {nullptr, 0};
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_ACL_ACL_SHARED_MEMORY_MANAGER_H_
