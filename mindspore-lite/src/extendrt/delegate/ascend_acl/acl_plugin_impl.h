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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_ASCEND_ACL_ACL_PLUGIN_IMPL_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_ASCEND_ACL_ACL_PLUGIN_IMPL_H_

#include <memory>
#include "include/api/status.h"
#include "src/common/log_adapter.h"
#include "extendrt/delegate/plugin/ascend_acl_executor_plugin.h"
// #include "extendrt/delegate/ascend_ge/ge_device_context.h"
#include "extendrt/delegate/ascend_acl/acl_graph_executor.h"

namespace mindspore {
class AscendAclExecutorPluginImpl : public lite::AscendAclExecutorPluginImplBase {
 public:
  AscendAclExecutorPluginImpl() = default;
  ~AscendAclExecutorPluginImpl() = default;
//   Status AdaptGraph(FuncGraphPtr graph) const override;
//   bool AoeTuning(const FuncGraphPtr &graph, const std::shared_ptr<mindspore::Context> &context,
//                  const ConfigInfos &config_infos) override;

//   bool OfflineBuildGraph(const FuncGraphPtr &graph, const std::shared_ptr<mindspore::Context> &context,
//                          const ConfigInfos &config_infos) override;

 private:
  std::shared_ptr<AclGraphExecutor> InitAclGraphExecutor(const std::shared_ptr<mindspore::Context> &context,
                                                       const ConfigInfos &config_infos);
};

extern "C" MS_API AscendAclExecutorPluginImpl *CreateAscendAclExecutorPluginImpl();
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_ASCEND_ACL_ACL_PLUGIN_IMPL_H_
