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

#include "tools/converter/adapter/acl/src/aclnn/aclnn_graph_builder.h"
#include "tools/common/string_util.h"
#include "tools/converter/optimizer_manager.h"
#include "common/common.h"
#include "common/utils.h"
#include "mindspore/core/include/ir/graph_utils.h"
#include "mindspore/ccsrc/include/utils/anfalgo.h"
#include "tools/common/custom_ascend_utils.h"

namespace mindspore {
namespace opt {

AclnnGraphBuilder::AclnnGraphBuilder(const std::shared_ptr<ConverterPara> &param,
                                     const lite::acl::AclModelOptionCfg &user_options_cfg)
    : param_(param),
      user_options_cfg_(user_options_cfg),
      subgraph_processor_(std::make_unique<AclnnSubgraphProcessor>(param, user_options_cfg)) {}

STATUS AclnnGraphBuilder::RunGraphSplitPass(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_MSG(param_ != nullptr, lite::RET_NULL_PTR, "param_ is nullptr!");
  param_->aclModelOptionCfgParam = user_options_cfg_;
  MS_CHECK_TRUE_MSG(lite::RunOptimizerPass(func_graph, {"MixedAclnnPass"}), false, "graph split pass failed!");
  return lite::RET_OK;
}

STATUS AclnnGraphBuilder::BuildMixedAclnnGraph(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_MSG(func_graph != nullptr, lite::RET_NULL_PTR, "func_graph is nullptr!");
  auto nodes = TopoSort(func_graph->get_return());

  for (auto &node : nodes) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_CHECK_TRUE_MSG(cnode != nullptr, lite::RET_NULL_PTR, "cast to cnode failed.");
    std::string kernel_name = common::AnfAlgo::GetCNodeName(cnode);
    if (kernel_name != lite::kNameCustomAclnnSubgraph) {
      continue;
    }

    auto ret = subgraph_processor_->ProcessCustomAclnnSubgraph(cnode, func_graph);
    if (ret != lite::RET_OK) {
      MS_LOG(ERROR) << "ProcessCustomAclnnSubgraph failed for: " << cnode->fullname_with_scope();
      return ret;
    }
  }
  return lite::RET_OK;
}

STATUS AclnnGraphBuilder::CleanCNodeAttr(const FuncGraphPtr &graph) {
  MS_CHECK_TRUE_MSG(graph != nullptr, lite::RET_NULL_PTR, "graph is nullptr");
  auto nodes = TopoSort(graph->get_return());
  for (auto &node : nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_CHECK_TRUE_MSG(cnode != nullptr, lite::RET_NULL_PTR, "cnode is nullptr");
    cnode->EraseAttr(lite::kNameGraphAttr);
    cnode->EraseAttr(lite::kNameShapeAttr);
    cnode->EraseAttr(lite::kNameShapeValueAttr);
    cnode->EraseAttr(lite::kNameCNodeValueAttr);
  }
  return lite::RET_OK;
}

}  // namespace opt
}  // namespace mindspore
