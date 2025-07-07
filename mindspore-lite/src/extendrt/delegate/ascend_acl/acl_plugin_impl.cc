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

#include <memory>
#include "extendrt/delegate/ascend_acl/acl_plugin_impl.h"

namespace mindspore {
std::shared_ptr<AclGraphExecutor> AscendAclExecutorPluginImpl::InitAclGraphExecutor(
  const std::shared_ptr<mindspore::Context> &context, const ConfigInfos &config_infos) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "Parameter context cannot be nullptr";
    return nullptr;
  }
  auto acl_graph_executor = std::make_shared<mindspore::AclGraphExecutor>(context, config_infos);
  if (acl_graph_executor == nullptr) {
    MS_LOG(ERROR) << "Failed to create GeGraphExecutor";
    return nullptr;
  }
  if (!acl_graph_executor->Init()) {
    MS_LOG(ERROR) << "Failed to init ge graph executor";
    return nullptr;
  }
  return acl_graph_executor;
}

AscendAclExecutorPluginImpl *CreateAscendAclExecutorPluginImpl() { return new AscendAclExecutorPluginImpl(); }
}  // namespace mindspore
