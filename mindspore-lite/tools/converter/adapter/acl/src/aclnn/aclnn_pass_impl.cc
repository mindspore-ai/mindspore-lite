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

#include "tools/converter/adapter/acl/src/aclnn/aclnn_pass_impl.h"
#include "tools/converter/optimizer_manager.h"
#include "common/common.h"
#include "cxx_api/graph/acl/acl_convert_init_adapter.h"
#include "tools/common/custom_ascend_utils.h"

namespace mindspore {
namespace opt {

AclnnPassImpl::AclnnPassImpl(const std::shared_ptr<ConverterPara> &param, bool is_subgraph)
    : AclPassImpl(param), is_subgraph_(is_subgraph) {
  shape_infer_ = std::make_unique<AclnnShapeInfer>(param, user_options_cfg_);
  graph_builder_ = std::make_unique<AclnnGraphBuilder>(param, user_options_cfg_);
}
AclnnPassImpl::~AclnnPassImpl() {
  if (!is_subgraph_) {
    AclConvertInitAdapter::GetInstance().AclBuildFinalize();
  }
}

bool AclnnPassImpl::Run(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_MSG(shape_infer_ != nullptr, false, "shape_infer_ is nullptr.");
  MS_CHECK_TRUE_MSG(graph_builder_ != nullptr, false, "graph_builder_ is nullptr.");
  auto ret = shape_infer_->InferShape(func_graph);
  MS_CHECK_TRUE_MSG(ret == lite::RET_OK, false, "Infer shape failed.");

  ret = graph_builder_->RunGraphSplitPass(func_graph);
  MS_CHECK_TRUE_MSG(ret == lite::RET_OK, false, "RunGraphSplitPass failed.");

  ret = graph_builder_->BuildMixedAclnnGraph(func_graph);
  MS_CHECK_TRUE_MSG(ret == lite::RET_OK, false, "BuildMixedAclnnGraph failed.");

  ret = graph_builder_->CleanCNodeAttr(func_graph);
  MS_CHECK_TRUE_MSG(ret == lite::RET_OK, false, "CleanCNodeAttr failed.");

  return true;
}

STATUS AclnnPassImpl::BuildGraph(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_MSG(func_graph != nullptr, lite::RET_NULL_PTR, "func_graph is nullptr.");
  Buffer om_data;
  if (ConvertGraphToOm(func_graph, &om_data) != lite::RET_OK) {
    MS_LOG(ERROR) << "Convert graph  to om failed.";
    return lite::RET_ERROR;
  }
  if (!CustomAscendUtils::CreateCustomFuncGraph(func_graph, om_data, "ACL_om_data", {}, {})) {
    MS_LOG(ERROR) << "Create custom func graph failed";
    return lite::RET_ERROR;
  }
  MS_LOG(DEBUG) << "Build graph success.";
  return lite::RET_OK;
}

}  // namespace opt
}  // namespace mindspore
