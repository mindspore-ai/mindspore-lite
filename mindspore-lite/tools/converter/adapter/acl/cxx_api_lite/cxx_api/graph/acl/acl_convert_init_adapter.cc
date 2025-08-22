/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "cxx_api/graph/acl/acl_convert_init_adapter.h"
#include <map>
#include <string>
#include "src/common/log_adapter.h"
#include "utils/ms_utils.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace mindspore {
AclConvertInitAdapter &AclConvertInitAdapter::GetInstance() {
  static AclConvertInitAdapter instance = {};
  return instance;
}

aclError AclConvertInitAdapter::AclInit(const char *config_file) {
  std::lock_guard<std::mutex> lock(flag_mutex_);
  if (init_flag_) {
    return ACL_SUCCESS;
  }

  init_flag_ = true;
  return CALL_ASCEND_API(aclInit, config_file);
}

aclError AclConvertInitAdapter::AclFinalize() {
  std::lock_guard<std::mutex> lock(flag_mutex_);
  if (!init_flag_) {
    MS_LOG(INFO) << "Acl had been finalized.";
    return ACL_SUCCESS;
  }

  MS_LOG(INFO) << "Begin to aclFinalize.";
  init_flag_ = false;
  return CALL_ASCEND_API(aclFinalize);
}

aclError AclConvertInitAdapter::ForceFinalize() {
  std::lock_guard<std::mutex> lock(flag_mutex_);
  MS_LOG(INFO) << "Begin to force aclFinalize.";
  init_flag_ = false;
  return CALL_ASCEND_API(aclFinalize);
}

ge::graphStatus AclConvertInitAdapter::AclBuildInit(const std::map<std::string, std::string> &init_options) {
  std::lock_guard<std::mutex> lock(build_flag_mutex_);
  if (!init_build_flag_) {
    init_build_flag_ = true;
    return ge::aclgrphBuildInitialize(init_options);
  } else {
    return ge::GRAPH_SUCCESS;
  }
}
}  // namespace mindspore
