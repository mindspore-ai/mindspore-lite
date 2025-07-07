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

#ifndef delegate_MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ASCEND_SRC_ACL_MEM_MANAGER_H_
#define delegate_MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ASCEND_SRC_ACL_MEM_MANAGER_H_

#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <memory>
#include <utility>
#include <thread>
#include "include/errorcode.h"

namespace mindspore {
using mindspore::lite::STATUS;

struct AclModelMemInfo {
  void *mem_addr = nullptr;
  size_t mem_size = 0;
};

struct MemShareInfo {
  int32_t device_id;
  std::thread::id thread_id;
  std::string model_path;
  AclModelMemInfo mem_info;
  bool allocated;
};
class AclMemManager {
 public:
  AclMemManager() {}
  ~AclMemManager();

  AclMemManager(const AclMemManager &) = delete;
  AclMemManager &operator=(const AclMemManager &) = delete;

  static AclMemManager &GetInstance() {
    static AclMemManager instance;
    return instance;
  }
  STATUS UpdateWorkspace(size_t work_size, size_t weight_size, int32_t device_id);
  STATUS UpdateWorkspace(size_t work_size, int32_t device_id);
  STATUS UpdateWeightspace(std::string model_path, size_t weight_size, int32_t device_id);
  STATUS GetModelWorkMem(AclModelMemInfo *acl_work_mem_info, int32_t device_id);
  STATUS GetModelWorkMem(void **work_ptr, int32_t device_id);
  STATUS GetModelWeightMem(AclModelMemInfo *acl_weight_mem_info);
  STATUS GetModelWeightMem(void **weight_ptr, std::string model_path, int32_t device_id);
  void ReleaseDeviceMem(int32_t device_id, std::string model_path);
  void Lock() { return acl_execute_mutex_.lock(); }
  void Unlock() { return acl_execute_mutex_.unlock(); }
  void Lock(int32_t device_id);
  void Unlock(int32_t device_id);

 private:
  std::mutex acl_mem_alloc_mutex_;
  std::mutex acl_execute_mutex_;
  std::map<int32_t, std::mutex> device_lock_map_;
  std::map<int32_t, std::pair<AclModelMemInfo, bool>> work_mem_info_map_;
  std::map<int32_t, std::map<std::string, MemShareInfo>> weight_mem_info_map_;
  AclModelMemInfo weight_mem_info_ = {nullptr, 0};
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ASCEND_SRC_ACL_MEM_MANAGER_H_
