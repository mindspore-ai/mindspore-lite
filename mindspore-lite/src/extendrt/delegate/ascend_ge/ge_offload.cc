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

#include "extendrt/delegate/ascend_ge/ge_offload.h"
#include "src/common/common.h"
#include "src/common/log_util.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_base_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_mdl_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"
namespace mindspore {
GeOffLoad &GeOffLoad::GetInstance() {
  static GeOffLoad instance;
  return instance;
}

Status GeOffLoad::CalculateMemorySize(const std::shared_ptr<ge::Session> &ge_session, uint32_t graph_id) {
  MS_LOG(INFO) << "calculate memory size for , graph id " << graph_id;
  ge::Status ret = ge_session->CompileGraph(graph_id);
  if (ret != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "Call GE CompileGraph Failed: " << ge::GEGetErrorMsg();
    return kLiteError;
  }
  MS_LOG(INFO) << "graph id is: " << graph_id;

  auto summary = ge_session->GetCompiledGraphSummary(graph_id);
  MS_CHECK_TRUE_MSG(summary != nullptr, kLiteError, "ge session summary is nullptr.");

  auto memory_info = std::make_shared<OffLoadMemoryInfo>();
  MS_CHECK_TRUE_MSG(memory_info != nullptr, kLiteError, "memory info is nullptr.");

  ret = summary->GetConstMemorySize(memory_info->const_memory_size);
  if (ret != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "GetConstMemorySize failed!";
    return kLiteError;
  }
  MS_LOG(INFO) << "model const memory size:" << memory_info->const_memory_size;

  ret = summary->GetFeatureMemorySize(memory_info->feature_memory_size);
  if (ret != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "GetFeatureMemorySize failed!";
    return kLiteError;
  }
  auto offload_info_item = all_offload_memory_info_.find(graph_id);
  if (offload_info_item != all_offload_memory_info_.end()) {
    MS_LOG(WARNING) << "this graph ID(" << graph_id
                    << ") has already calculated the memory size, there is no need to repeat the calculation";
    return kLiteError;
  }

  all_offload_memory_info_[graph_id] = memory_info;
  // update shared memory size
  if (memory_info->const_memory_size > shared_const_memory_size_) {
    shared_const_memory_size_ = memory_info->const_memory_size;
  }
  return kSuccess;
}

Status GeOffLoad::InitConstAndFeatureMemory(const std::shared_ptr<ge::Session> &ge_session, uint32_t graph_id) {
  MS_LOG(INFO) << "compile graph with offload, graph id " << graph_id;
  auto offload_info_item = all_offload_memory_info_.find(graph_id);
  if (offload_info_item == all_offload_memory_info_.end()) {
    MS_LOG(WARNING) << "this graph ID(" << graph_id << ") has bot calculated the memory size";
    return kLiteError;
  }

  ge::Status ret = ge_session->CompileGraph(graph_id);
  if (ret != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "Call GE CompileGraph Failed: " << ge::GEGetErrorMsg();
    return kLiteError;
  }
  MS_LOG(INFO) << "graph id is: " << graph_id;

  auto summary = ge_session->GetCompiledGraphSummary(graph_id);
  MS_CHECK_TRUE_MSG(summary != nullptr, kLiteError, "ge session summary is nullptr.");

  MS_LOG(INFO) << "const mem size:" << all_offload_memory_info_[graph_id]->const_memory_size;

  auto acl_ret = CALL_ASCEND_API(aclrtMalloc, &all_offload_memory_info_[graph_id]->device_const_memory,
                                 all_offload_memory_info_[graph_id]->const_memory_size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (acl_ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Call aclrtMalloc failed, err_code = " << acl_ret;
    return kLiteError;
  }
  ge_session->SetGraphConstMemoryBase(graph_id, all_offload_memory_info_[graph_id]->device_const_memory,
                                      all_offload_memory_info_[graph_id]->const_memory_size);

  acl_ret = CALL_ASCEND_API(aclrtMalloc, &all_offload_memory_info_[graph_id]->device_feature_memory,
                            all_offload_memory_info_[graph_id]->feature_memory_size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (acl_ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Call aclrtMalloc failed, err_code = " << acl_ret;
    return kLiteError;
  }
  ge_session->UpdateGraphFeatureMemoryBase(graph_id, all_offload_memory_info_[graph_id]->device_feature_memory,
                                           all_offload_memory_info_[graph_id]->feature_memory_size);

  acl_ret = CALL_ASCEND_API(aclrtMallocHost, &all_offload_memory_info_[graph_id]->host_const_memory,
                            all_offload_memory_info_[graph_id]->const_memory_size);
  if (acl_ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Call aclrtMallocHost failed, err_code = " << acl_ret;
    return kLiteError;
  }

  acl_ret = CALL_ASCEND_API(aclrtMemcpy, all_offload_memory_info_[graph_id]->host_const_memory,
                            all_offload_memory_info_[graph_id]->const_memory_size,
                            all_offload_memory_info_[graph_id]->device_const_memory,
                            all_offload_memory_info_[graph_id]->const_memory_size, ACL_MEMCPY_DEVICE_TO_HOST);
  if (acl_ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "copy from device to host failed!, err_code = " << acl_ret;
    return kLiteError;
  }
  return kSuccess;
}

Status GeOffLoad::UpdateSharedDeviceMemory(uint32_t graph_id) {
  auto acl_ret = CALL_ASCEND_API(aclrtMemcpy, all_offload_memory_info_[graph_id]->device_const_memory,
                                 all_offload_memory_info_[graph_id]->const_memory_size,
                                 all_offload_memory_info_[graph_id]->host_const_memory,
                                 all_offload_memory_info_[graph_id]->const_memory_size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (acl_ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "copy from host to device failed!, err_code = " << acl_ret;
    return kLiteError;
  }
  return kSuccess;
}

}  // namespace mindspore
