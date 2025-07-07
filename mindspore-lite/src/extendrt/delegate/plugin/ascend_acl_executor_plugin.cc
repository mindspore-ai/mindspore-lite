/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#include "extendrt/delegate/plugin/ascend_acl_executor_plugin.h"
#include <string>
#include "src/common/log_adapter.h"
#if !defined(_WIN32)
#include "extendrt/cxx_api/dlutils.h"
#endif

namespace mindspore::lite {
namespace {
constexpr auto kAscendAclPluginSoName = "libascend_acl_plugin.so";
constexpr auto kFunCreateAscendAclPluginImpl = "CreateAscendAclExecutorPluginImpl";
}  // namespace
AscendAclExecutorPlugin::AscendAclExecutorPlugin() = default;
AscendAclExecutorPlugin::~AscendAclExecutorPlugin() {
#if !defined(_WIN32)
  MS_LOG(DEBUG) << "~AscendAclExecutorPlugin() begin.";
  acl_plugin_impl_ = nullptr;
  DLSoClose(handle_);
  MS_LOG(DEBUG) << "~AscendAclExecutorPlugin() end.";
#endif
}

AscendAclExecutorPlugin &AscendAclExecutorPlugin::GetInstance() {
  static AscendAclExecutorPlugin instance;
  return instance;
}

bool AscendAclExecutorPlugin::Register() {
#if !defined(_WIN32)
  if (is_registered_) {
    return true;
  }
  auto ret =
    DLSoPath({"libmindspore-lite.so", "_c_lite", "tools/converter/lib"}, kAscendAclPluginSoName, &plugin_path_);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Get real path of " << kAscendAclPluginSoName << " failed.";
    return false;
  }
  MS_LOG(INFO) << "Find ascend acl plugin so success, path = " << plugin_path_;
  void *function = nullptr;
  ret = DLSoOpen(plugin_path_, kFunCreateAscendAclPluginImpl, &handle_, &function);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "DLSoOpen failed, so path: " << plugin_path_ << ", err: " << ret.ToString();
    return false;
  }
  auto create_plugin_impl_func = reinterpret_cast<AscendAclExecutorPluginImplBase *(*)(void)>(function);
  if (create_plugin_impl_func == nullptr) {
    MS_LOG(ERROR) << "Cast " << kFunCreateAscendAclPluginImpl << " failed.";
    return false;
  }
  acl_plugin_impl_ = std::shared_ptr<AscendAclExecutorPluginImplBase>(create_plugin_impl_func());
  if (acl_plugin_impl_ == nullptr) {
    MS_LOG(ERROR) << "Create Ascend acl plugin implement failed.";
    return false;
  }
  is_registered_ = true;
  MS_LOG(INFO) << "Register Ascend acl plugin success.";
#endif
  return true;
}
}  // namespace mindspore::lite
