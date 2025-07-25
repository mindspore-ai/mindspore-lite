/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "cxx_api/graph/acl/acl_env_guard.h"
#include "src/common/log_adapter.h"
#include "utils/ms_utils.h"
#include "plugin/res_manager/ascend/symbol_interface/acl_symbol.h"
#include "plugin/res_manager/ascend/symbol_interface/symbol_utils.h"

namespace mindspore {
std::shared_ptr<AclEnvGuard> AclEnvGuard::global_acl_env_ = nullptr;
std::mutex AclEnvGuard::global_acl_env_mutex_;

AclInitAdapter &AclInitAdapter::GetInstance() {
  static AclInitAdapter instance = {};
  return instance;
}

aclError AclInitAdapter::AclInit(const char *config_file) {
  std::lock_guard<std::mutex> lock(flag_mutex_);
  if (init_flag_) {
    return ACL_SUCCESS;
  }

  init_flag_ = true;
  return CALL_ASCEND_API(aclInit, config_file);
}

aclError AclInitAdapter::AclFinalize() {
  std::lock_guard<std::mutex> lock(flag_mutex_);
  if (!init_flag_) {
    MS_LOG(INFO) << "Acl had been finalized.";
    return ACL_SUCCESS;
  }

  MS_LOG(INFO) << "Begin to aclFinalize.";
  init_flag_ = false;
  return CALL_ASCEND_API(aclFinalize);
}

aclError AclInitAdapter::ForceFinalize() {
  std::lock_guard<std::mutex> lock(flag_mutex_);
  MS_LOG(INFO) << "Begin to force aclFinalize.";
  init_flag_ = false;
  return CALL_ASCEND_API(aclFinalize);
}

AclEnvGuard::AclEnvGuard() : errno_(AclInitAdapter::GetInstance().AclInit(nullptr)) {
  if (errno_ != ACL_SUCCESS && errno_ != ACL_ERROR_REPEAT_INITIALIZE) {
    MS_LOG(ERROR) << "Execute aclInit failed.";
    return;
  }
  MS_LOG(INFO) << "Execute aclInit success.";
}

AclEnvGuard::~AclEnvGuard() {
  TRY_AND_CATCH_WITH_EXCEPTION(errno_ = AclInitAdapter::GetInstance().AclFinalize(),
                               "AclInitAdapter GetInstance failed");
  if (errno_ != ACL_SUCCESS && errno_ != ACL_ERROR_REPEAT_FINALIZE) {
    MS_LOG(ERROR) << "Execute AclFinalize failed.";
  }
  MS_LOG(INFO) << "Execute AclFinalize success.";
}

std::shared_ptr<AclEnvGuard> AclEnvGuard::GetAclEnv() {
  std::lock_guard<std::mutex> lock(global_acl_env_mutex_);
  std::shared_ptr<AclEnvGuard> acl_env = global_acl_env_;
  if (acl_env != nullptr) {
    MS_LOG(INFO) << "Acl has been initialized, skip.";
  } else {
    acl_env = std::make_shared<AclEnvGuard>();
    aclError ret = acl_env->GetErrno();
    if (ret != ACL_SUCCESS && ret != ACL_ERROR_REPEAT_INITIALIZE) {
      MS_LOG(ERROR) << "Execute aclInit failed.";
      return nullptr;
    }
    global_acl_env_ = acl_env;
    MS_LOG(INFO) << "Execute aclInit success.";
  }
  return acl_env;
}
}  // namespace mindspore
