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

#include "extendrt/delegate/ascend_acl/acl_shared_memory_manager.h"
#include <utility>
#include <memory>
#include <algorithm>
#include <map>
#include <string>
#include "src/common/log_adapter.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_base_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_mdl_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "include/api/status.h"

namespace mindspore {
AclSharedMemoryManager &AclSharedMemoryManager::GetInstance() {
  static AclSharedMemoryManager instance;
  return instance;
}

Status AclSharedMemoryManager::UpdateWeightSpace(std::string model_path, size_t weight_size, int32_t device_id) {
  MS_LOG(DEBUG) << "model path: " << model_path << ", weight size: " << weight_size << ", device id: " << device_id;
  if (shared_weight_memory_info_map_.find(device_id) == shared_weight_memory_info_map_.end()) {
    MS_LOG(DEBUG) << "find device id in weight memory map.";
    AclMemoryInfo new_weight_mem = {nullptr, weight_size};
    SharedMemoryInfo mem_share_info;
    mem_share_info.device_id = device_id;
    mem_share_info.model_path = "";
    mem_share_info.memory_info = new_weight_mem;
    mem_share_info.allocated = false;
    std::map<std::string, SharedMemoryInfo> inner_map;
    inner_map.insert(std::make_pair(model_path, mem_share_info));
    shared_weight_memory_info_map_.insert(std::make_pair(device_id, inner_map));
  } else if (shared_weight_memory_info_map_.at(device_id).find(model_path) ==
             shared_weight_memory_info_map_.at(device_id).end()) {
    MS_LOG(DEBUG) << "find model path id in weight memory info map.";
    AclMemoryInfo new_weight_mem = {nullptr, weight_size};
    SharedMemoryInfo mem_share_info;
    mem_share_info.device_id = device_id;
    mem_share_info.model_path = "";
    mem_share_info.memory_info = new_weight_mem;
    mem_share_info.allocated = false;
    shared_weight_memory_info_map_.at(device_id).insert(std::make_pair(model_path, mem_share_info));
  }
  return kSuccess;
}

Status AclSharedMemoryManager::UpdateWorkSpace(size_t work_size, int32_t device_id) {
  auto it = shared_work_memory_info_map_.find(device_id);
  if (it == shared_work_memory_info_map_.end()) {
    AclMemoryInfo new_work_mem = {nullptr, 0};
    shared_work_memory_info_map_.insert(std::make_pair(device_id, std::make_pair(new_work_mem, false)));
  } else if (it->second.second == true) {
    MS_LOG(ERROR) << "Device " << device_id << " has alloc memory!";
    return kLiteError;
  }
  MS_LOG(DEBUG) << "Get device success.";
  it = shared_work_memory_info_map_.find(device_id);
  if (it == shared_work_memory_info_map_.end()) {
    MS_LOG(ERROR) << "Get mem failed!";
    return kLiteError;
  }
  MS_LOG(DEBUG) << "Begin record work size.";
  if (work_size > it->second.first.memory_size) {
    it->second.first.memory_size = work_size;
    MS_LOG(DEBUG) << "Update work_size = " << it->second.first.memory_size << " successful.";
  }
  return kSuccess;
}

Status AclSharedMemoryManager::PrepareMutiModelShare(const void *om_data, size_t om_data_size,
                                                     const std::shared_ptr<AclModelOptions> &options) {
  size_t work_size = 0;
  size_t weight_size = 0;
  auto acl_ret = CALL_ASCEND_API(aclmdlQuerySizeFromMem, om_data, om_data_size, &work_size, &weight_size);
  if (acl_ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Call aclmdlQuerySizeFromMem failed, ret = " << acl_ret;
    return Status(kLiteAclInitFailed, "Call aclmdlQuerySizeFromMem failed.");
  }
  MS_LOG(INFO) << "work_size: " << work_size << " weight_size: " << weight_size;
  auto ret = UpdateWorkSpace(work_size, options->device_id);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "update workspace failed, ret = " << ret;
    return Status(kLiteError, "update workspace failed.");
  }
  ret = UpdateWeightSpace(options->model_path, weight_size, options->device_id);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "update weightspace failed, ret = " << ret;
    return Status(kLiteError, "update weightspace failed.");
  }
  return kSuccess;
}

std::pair<void *, void *> AclSharedMemoryManager::ShareWorkspaceProcess(
  const size_t &work_size, const size_t &weight_size, const std::shared_ptr<AclModelOptions> &options) {
  MS_LOG(INFO) << "Share work space.";
  void *work_ptr = nullptr;
  void *weight_ptr = nullptr;
  if (work_size == 0) {
    MS_LOG(WARNING) << "Dynamic input model not support share workspace.";
    work_ptr = nullptr;
  } else {
    auto ret = GetModelWorkMem(&work_ptr, options->device_id);
    MS_CHECK_TRUE_MSG(ret == kSuccess, std::make_pair(weight_ptr, work_ptr), "Get work mem failed!");
  }
  auto acl_ret = CALL_ASCEND_API(aclrtMalloc, &(weight_ptr), weight_size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (acl_ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Call aclrtMalloc failed, err_code = " << acl_ret;
    return std::make_pair(weight_ptr, work_ptr);
  }
  return std::make_pair(weight_ptr, work_ptr);
}

std::pair<void *, void *> AclSharedMemoryManager::ShareWeightspaceProcess(
  const size_t &work_size, const std::shared_ptr<AclModelOptions> &options) {
  MS_LOG(INFO) << "Share weight space.";
  void *work_ptr = nullptr;
  void *weight_ptr = nullptr;
  auto ret = GetModelWeightMem(&weight_ptr, options->model_path, options->device_id);
  MS_CHECK_TRUE_MSG(ret == kSuccess, std::make_pair(weight_ptr, work_ptr), "Get weight mem failed!");
  if (work_size == 0) {
    work_ptr = nullptr;
  } else {
    auto acl_ret = CALL_ASCEND_API(aclrtMalloc, &(work_ptr), work_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (acl_ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Call aclrtMalloc failed, err_code = " << acl_ret;
      return std::make_pair(weight_ptr, work_ptr);
    }
  }
  return std::make_pair(weight_ptr, work_ptr);
}

std::pair<void *, void *> AclSharedMemoryManager::ShareWorkspaceAndWeightspaceProcess(
  const size_t &work_size, const std::shared_ptr<AclModelOptions> &options) {
  MS_LOG(INFO) << "Share workspace and weight space.";

  void *work_ptr = nullptr;
  void *weight_ptr = nullptr;

  auto model_path = options->model_path;
  auto ret = GetModelWeightMem(&weight_ptr, model_path, options->device_id);
  MS_CHECK_TRUE_MSG(ret == kSuccess, std::make_pair(weight_ptr, work_ptr), "Get weight mem failed!");
  if (work_size == 0) {
    work_ptr = nullptr;
    MS_LOG(WARNING) << "Dynamic input model not support share workspace.";
  } else {
    ret = GetModelWorkMem(&work_ptr, options->device_id);
    MS_CHECK_TRUE_MSG(ret == kSuccess, std::make_pair(weight_ptr, work_ptr), "Get work mem failed!");
  }
  return std::make_pair(weight_ptr, work_ptr);
}

Status AclSharedMemoryManager::GetModelWorkMem(void **work_ptr, int32_t device_id) {
  MS_CHECK_TRUE_MSG(work_ptr != nullptr, kLiteError, "work_ptr is nullptr!");
  std::unique_lock<std::mutex> acl_mtx(acl_shared_memory_prepare_mutex_);

  auto it = shared_work_memory_info_map_.find(device_id);
  if (it == shared_work_memory_info_map_.end()) {
    MS_LOG(ERROR) << "Get work mem failed!";
    return kLiteError;
  }
  it->second.second = true;
  MS_LOG(DEBUG) << "Get device id success.";
  if (it->second.first.memory_address == nullptr) {
    if (it->second.first.memory_size == 0) {
      return kLiteError;
    }
    MS_LOG(DEBUG) << "Begin alloc mem addr.";
    auto acl_ret = CALL_ASCEND_API(aclrtMalloc, &(it->second.first.memory_address), it->second.first.memory_size,
                                   ACL_MEM_MALLOC_HUGE_FIRST);
    if (acl_ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Call aclrtMalloc failed, err_code = " << acl_ret;
      return kLiteError;
    }
    MS_LOG(DEBUG) << "Malloc work mem success, max work size is " << it->second.first.memory_size;
  }
  *work_ptr = it->second.first.memory_address;
  return kSuccess;
}

Status AclSharedMemoryManager::GetModelWeightMem(void **weight_ptr, std::string model_path, int32_t device_id) {
  MS_CHECK_TRUE_MSG(weight_ptr != nullptr, kLiteError, "weight_ptr is nullptr!");
  std::unique_lock<std::mutex> acl_mtx(acl_shared_memory_prepare_mutex_);
  if (shared_weight_memory_info_map_.find(device_id) == shared_weight_memory_info_map_.end()) {
    MS_LOG(ERROR) << "Can't get weight mem of device " << device_id << "!";
    return kLiteError;
  }
  if (shared_weight_memory_info_map_.at(device_id).find(model_path) ==
      shared_weight_memory_info_map_.at(device_id).end()) {
    MS_LOG(ERROR) << "Can't get weight mem of device " << device_id << " of model path " << model_path << "!";
    return kLiteError;
  }
  auto &share_mem_info = shared_weight_memory_info_map_.at(device_id).at(model_path);

  if (share_mem_info.memory_info.memory_address == nullptr) {
    if (share_mem_info.memory_info.memory_size == 0) {
      MS_LOG(ERROR) << "Weight size if 0!";
      return kLiteError;
    }
    auto acl_ret = CALL_ASCEND_API(aclrtMalloc, &(share_mem_info.memory_info.memory_address),
                                   share_mem_info.memory_info.memory_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (acl_ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Call aclrtMalloc failed, err_code : " << acl_ret << "!";
      return kLiteError;
    }
    MS_LOG(DEBUG) << "Malloc weight size is " << share_mem_info.memory_info.memory_size << "!";
  }
  *weight_ptr = share_mem_info.memory_info.memory_address;
  return kSuccess;
}

void AclSharedMemoryManager::Lock(int32_t device_id) {
  acl_shared_memory_execute_mutex_.lock();
  if (shared_lock_for_device_id_map_.find(device_id) == shared_lock_for_device_id_map_.end()) {
    shared_lock_for_device_id_map_.emplace(std::piecewise_construct, std::forward_as_tuple(device_id),
                                           std::forward_as_tuple());
  }
  acl_shared_memory_execute_mutex_.unlock();
  return shared_lock_for_device_id_map_.at(device_id).lock();
}

void AclSharedMemoryManager::Unlock(int32_t device_id) {
  acl_shared_memory_execute_mutex_.lock();
  if (shared_lock_for_device_id_map_.find(device_id) == shared_lock_for_device_id_map_.end()) {
    shared_lock_for_device_id_map_.emplace(std::piecewise_construct, std::forward_as_tuple(device_id),
                                           std::forward_as_tuple());
  }
  acl_shared_memory_execute_mutex_.unlock();
  return shared_lock_for_device_id_map_.at(device_id).unlock();
}

void AclSharedMemoryManager::ReleaseDeviceMem(int32_t device_id, std::string model_path) {
  MS_LOG(DEBUG) << "start ReleaseDeviceMem";
  for (auto &device_id_iter : shared_work_memory_info_map_) {
    if (device_id_iter.first != device_id) {
      continue;
    }
    if (device_id_iter.second.first.memory_address != nullptr) {
      (void)CALL_ASCEND_API(aclrtFree, device_id_iter.second.first.memory_address);
      device_id_iter.second.first.memory_address = nullptr;
    }
  }
  for (auto &device_id_iter : shared_weight_memory_info_map_) {
    if (device_id_iter.first != device_id) {
      continue;
    }
    for (auto &model_path_iter : device_id_iter.second) {
      if (model_path_iter.first != model_path) {
        continue;
      }
      if (model_path_iter.second.memory_info.memory_address != nullptr) {
        (void)CALL_ASCEND_API(aclrtFree, model_path_iter.second.memory_info.memory_address);
        model_path_iter.second.memory_info.memory_address = nullptr;
      }
    }
  }
  MS_LOG(DEBUG) << "ReleaseDeviceMem end";
}

AclSharedMemoryManager::~AclSharedMemoryManager() {
  MS_LOG(DEBUG) << "delete AclSharedMemoryManager";
  for (auto &memory_info_pair : shared_work_memory_info_map_) {
    if (memory_info_pair.second.first.memory_address != nullptr) {
      (void)CALL_ASCEND_API(aclrtFree, memory_info_pair.second.first.memory_address);
      memory_info_pair.second.first.memory_address = nullptr;
      memory_info_pair.second.first.memory_size = 0;
    }
  }
  if (shared_weight_mem_info_.memory_address != nullptr) {
    (void)CALL_ASCEND_API(aclrtFree, shared_weight_mem_info_.memory_address);
    shared_weight_mem_info_.memory_address = nullptr;
    shared_weight_mem_info_.memory_size = 0;
  }
  for (auto &device_id_iter : shared_weight_memory_info_map_) {
    for (auto &model_path_iter : device_id_iter.second) {
      if (model_path_iter.second.memory_info.memory_address != nullptr) {
        (void)CALL_ASCEND_API(aclrtFree, model_path_iter.second.memory_info.memory_address);
        model_path_iter.second.memory_info.memory_address = nullptr;
        model_path_iter.second.memory_info.memory_size = 0;
      }
    }
  }
  MS_LOG(DEBUG) << "delete AclSharedMemoryManager end";
}
}  // namespace mindspore
