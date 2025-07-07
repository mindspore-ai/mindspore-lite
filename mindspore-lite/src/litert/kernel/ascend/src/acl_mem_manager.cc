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

#include "src/litert/kernel/ascend/src/acl_mem_manager.h"
#include <utility>
#include <memory>
#include <algorithm>
#include <map>
#include <string>
#include "src/common/log_adapter.h"
#include "plugin/res_manager/ascend/symbol_interface/acl_rt_symbol.h"
#include "plugin/res_manager/ascend/symbol_interface/symbol_utils.h"

namespace mindspore::kernel {
namespace acl {
STATUS AclMemManager::UpdateWorkspace(size_t work_size, size_t weight_size, int32_t device_id) {
  auto it = work_mem_info_map_.find(device_id);
  if (it == work_mem_info_map_.end()) {
    AclModelMemInfo new_work_mem = {nullptr, 0};
    work_mem_info_map_.insert(std::make_pair(device_id, std::make_pair(new_work_mem, false)));
  } else if (it->second.second == true) {
    MS_LOG(ERROR) << "Device " << device_id << " has alloc memory!";
    return lite::RET_ERROR;
  }

  it = work_mem_info_map_.find(device_id);
  if (it == work_mem_info_map_.end()) {
    MS_LOG(ERROR) << "Get mem failed!";
    return lite::RET_ERROR;
  }

  if (work_size > it->second.first.mem_size) {
    it->second.first.mem_size = work_size;
    MS_LOG(DEBUG) << "Update work_size = " << it->second.first.mem_size << " successful.";
  }

  if (weight_size > weight_mem_info_.mem_size) {
    weight_mem_info_.mem_size = weight_size;
    MS_LOG(DEBUG) << "Update weight_size = " << weight_size << " successful.";
  }
  return lite::RET_OK;
}

STATUS AclMemManager::UpdateWorkspace(size_t work_size, int32_t device_id) {
  auto it = work_mem_info_map_.find(device_id);
  if (it == work_mem_info_map_.end()) {
    AclModelMemInfo new_work_mem = {nullptr, 0};
    work_mem_info_map_.insert(std::make_pair(device_id, std::make_pair(new_work_mem, false)));
  } else if (it->second.second == true) {
    MS_LOG(ERROR) << "Device " << device_id << " has alloc memory!";
    return lite::RET_ERROR;
  }
  MS_LOG(DEBUG) << "Get device success.";
  it = work_mem_info_map_.find(device_id);
  if (it == work_mem_info_map_.end()) {
    MS_LOG(ERROR) << "Get mem failed!";
    return lite::RET_ERROR;
  }
  MS_LOG(DEBUG) << "Begin record work size.";
  if (work_size > it->second.first.mem_size) {
    it->second.first.mem_size = work_size;
    MS_LOG(DEBUG) << "Update work_size = " << it->second.first.mem_size << " successful.";
  }
  return lite::RET_OK;
}

STATUS AclMemManager::UpdateWeightspace(std::string model_path, size_t weight_size, int32_t device_id) {
  if (weight_mem_info_map_.find(device_id) == weight_mem_info_map_.end()) {
    AclModelMemInfo new_weight_mem = {nullptr, weight_size};
    MemShareInfo mem_share_info;
    mem_share_info.device_id = device_id;
    mem_share_info.model_path = "";
    mem_share_info.mem_info = new_weight_mem;
    mem_share_info.allocated = false;
    std::map<std::string, MemShareInfo> inner_map;
    inner_map.insert(std::make_pair(model_path, mem_share_info));
    weight_mem_info_map_.insert(std::make_pair(device_id, inner_map));
  } else if (weight_mem_info_map_.at(device_id).find(model_path) == weight_mem_info_map_.at(device_id).end()) {
    AclModelMemInfo new_weight_mem = {nullptr, weight_size};
    MemShareInfo mem_share_info;
    mem_share_info.device_id = device_id;
    mem_share_info.model_path = "";
    mem_share_info.mem_info = new_weight_mem;
    mem_share_info.allocated = false;
    weight_mem_info_map_.at(device_id).insert(std::make_pair(model_path, mem_share_info));
  }
  return lite::RET_OK;
}

STATUS AclMemManager::GetModelWorkMem(AclModelMemInfo *acl_work_mem_info, int32_t device_id) {
  std::unique_lock<std::mutex> acl_mtx(acl_mem_alloc_mutex_);

  auto it = work_mem_info_map_.find(device_id);
  if (it == work_mem_info_map_.end()) {
    MS_LOG(ERROR) << "Get work mem failed!";
    return lite::RET_ERROR;
  }
  it->second.second = true;

  if (it->second.first.mem_addr == nullptr) {
    if (it->second.first.mem_size == 0) {
      return lite::RET_ERROR;
    }
    auto acl_ret =
      CALL_ASCEND_API(aclrtMalloc, &(it->second.first.mem_addr), it->second.first.mem_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (acl_ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Call aclrtMalloc failed, err_code = " << acl_ret;
      return lite::RET_ERROR;
    }
    MS_LOG(DEBUG) << "Malloc max work size is " << it->second.first.mem_size;
  }
  *acl_work_mem_info = it->second.first;
  return lite::RET_OK;
}

STATUS AclMemManager::GetModelWorkMem(void **work_ptr, int32_t device_id) {
  MS_CHECK_TRUE_MSG(work_ptr != nullptr, lite::RET_NULL_PTR, "work_ptr is nullptr!");
  std::unique_lock<std::mutex> acl_mtx(acl_mem_alloc_mutex_);

  auto it = work_mem_info_map_.find(device_id);
  if (it == work_mem_info_map_.end()) {
    MS_LOG(ERROR) << "Get work mem failed!";
    return lite::RET_ERROR;
  }
  it->second.second = true;
  MS_LOG(DEBUG) << "Get device id success.";
  if (it->second.first.mem_addr == nullptr) {
    if (it->second.first.mem_size == 0) {
      return lite::RET_ERROR;
    }
    MS_LOG(DEBUG) << "Begin alloc mem addr.";
    auto acl_ret =
      CALL_ASCEND_API(aclrtMalloc, &(it->second.first.mem_addr), it->second.first.mem_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (acl_ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Call aclrtMalloc failed, err_code = " << acl_ret;
      return lite::RET_ERROR;
    }
    MS_LOG(DEBUG) << "Malloc work mem success, max work size is " << it->second.first.mem_size;
  }
  *work_ptr = it->second.first.mem_addr;
  return lite::RET_OK;
}

STATUS AclMemManager::GetModelWeightMem(AclModelMemInfo *acl_weight_mem_info) {
  std::unique_lock<std::mutex> acl_mtx(acl_mem_alloc_mutex_);
  if (weight_mem_info_.mem_addr == nullptr) {
    if (weight_mem_info_.mem_size == 0) {
      return lite::RET_ERROR;
    }
    auto acl_ret =
      CALL_ASCEND_API(aclrtMalloc, &weight_mem_info_.mem_addr, weight_mem_info_.mem_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (acl_ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Call aclrtMalloc failed, err_code = " << acl_ret;
      return lite::RET_ERROR;
    }
    MS_LOG(DEBUG) << "Malloc max weight size is " << weight_mem_info_.mem_size;
  }
  *acl_weight_mem_info = weight_mem_info_;
  return lite::RET_OK;
}

STATUS AclMemManager::GetModelWeightMem(void **weight_ptr, std::string model_path, int32_t device_id) {
  MS_CHECK_TRUE_MSG(weight_ptr != nullptr, lite::RET_NULL_PTR, "weight_ptr is nullptr!");
  std::unique_lock<std::mutex> acl_mtx(acl_mem_alloc_mutex_);
  if (weight_mem_info_map_.find(device_id) == weight_mem_info_map_.end()) {
    MS_LOG(ERROR) << "Can't get weight mem of device " << device_id << "!";
    return lite::RET_ERROR;
  }
  if (weight_mem_info_map_.at(device_id).find(model_path) == weight_mem_info_map_.at(device_id).end()) {
    MS_LOG(ERROR) << "Can't get weight mem of device " << device_id << " of model path " << model_path << "!";
    return lite::RET_ERROR;
  }
  auto &share_mem_info = weight_mem_info_map_.at(device_id).at(model_path);

  if (share_mem_info.mem_info.mem_addr == nullptr) {
    if (share_mem_info.mem_info.mem_size == 0) {
      MS_LOG(ERROR) << "Weight size if 0!";
      return lite::RET_ERROR;
    }
    auto acl_ret = CALL_ASCEND_API(aclrtMalloc, &(share_mem_info.mem_info.mem_addr), share_mem_info.mem_info.mem_size,
                                   ACL_MEM_MALLOC_HUGE_FIRST);
    if (acl_ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Call aclrtMalloc failed, err_code : " << acl_ret << "!";
      return lite::RET_ERROR;
    }
    MS_LOG(DEBUG) << "Malloc weight size is " << share_mem_info.mem_info.mem_size << "!";
  }
  *weight_ptr = share_mem_info.mem_info.mem_addr;
  return lite::RET_OK;
}

void AclMemManager::Lock(int32_t device_id) {
  acl_execute_mutex_.lock();
  if (device_lock_map_.find(device_id) == device_lock_map_.end()) {
    device_lock_map_.emplace(std::piecewise_construct, std::forward_as_tuple(device_id), std::forward_as_tuple());
  }
  acl_execute_mutex_.unlock();
  return device_lock_map_.at(device_id).lock();
}

void AclMemManager::Unlock(int32_t device_id) {
  acl_execute_mutex_.lock();
  if (device_lock_map_.find(device_id) == device_lock_map_.end()) {
    device_lock_map_.emplace(std::piecewise_construct, std::forward_as_tuple(device_id), std::forward_as_tuple());
  }
  acl_execute_mutex_.unlock();
  return device_lock_map_.at(device_id).unlock();
}

void AclMemManager::ReleaseDeviceMem(int32_t device_id, std::string model_path) {
  for (auto &device_id_iter : work_mem_info_map_) {
    if (device_id_iter.first != device_id) {
      continue;
    }
    if (device_id_iter.second.first.mem_addr != nullptr) {
      (void)CALL_ASCEND_API(aclrtFree, device_id_iter.second.first.mem_addr);
      device_id_iter.second.first.mem_addr = nullptr;
    }
  }
  for (auto &device_id_iter : weight_mem_info_map_) {
    if (device_id_iter.first != device_id) {
      continue;
    }
    for (auto &model_path_iter : device_id_iter.second) {
      if (model_path_iter.first != model_path) {
        continue;
      }
      if (model_path_iter.second.mem_info.mem_addr != nullptr) {
        (void)CALL_ASCEND_API(aclrtFree, model_path_iter.second.mem_info.mem_addr);
        model_path_iter.second.mem_info.mem_addr = nullptr;
      }
    }
  }
}

AclMemManager::~AclMemManager() {
  for (auto &mem_info_pair : work_mem_info_map_) {
    if (mem_info_pair.second.first.mem_addr != nullptr) {
      (void)CALL_ASCEND_API(aclrtFree, mem_info_pair.second.first.mem_addr);
      mem_info_pair.second.first.mem_addr = nullptr;
      mem_info_pair.second.first.mem_size = 0;
    }
  }
  if (weight_mem_info_.mem_addr != nullptr) {
    (void)CALL_ASCEND_API(aclrtFree, weight_mem_info_.mem_addr);
    weight_mem_info_.mem_addr = nullptr;
    weight_mem_info_.mem_size = 0;
  }
  for (auto &device_id_iter : weight_mem_info_map_) {
    for (auto &model_path_iter : device_id_iter.second) {
      if (model_path_iter.second.mem_info.mem_addr != nullptr) {
        (void)CALL_ASCEND_API(aclrtFree, model_path_iter.second.mem_info.mem_addr);
        model_path_iter.second.mem_info.mem_addr = nullptr;
        model_path_iter.second.mem_info.mem_size = 0;
      }
    }
  }
}
}  // namespace acl
}  // namespace mindspore::kernel
