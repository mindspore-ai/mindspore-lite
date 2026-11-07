/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#define USE_DEPRECATED_API
#include "tools/optimizer/common/node_pass_extends.h"
#include <unordered_set>
#include <deque>
#include <algorithm>
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "ir/manager.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/optimizer/common/fusion_trace_utils.h"

namespace mindspore {
namespace opt {
bool NodePass::Run(const FuncGraphPtr &func_graph) {
  MS_LOG(ERROR) << "stub func";
  return false;
}

AnfNodePtr LiteNodePass::FinishFusionProcess(const FuncGraphPtr &func_graph, const AnfNodePtr &root,
                                             const AnfNodePtr &ret,
                                             const std::vector<AnfNodePtr> &matched_parts) const {
  if (func_graph == nullptr || ret == nullptr || trace_policy_ == TracePolicy::kManual) {
    return ret;
  }
  auto survivor = ret->cast<CNodePtr>();
  if (survivor == nullptr) {
    return ret;
  }
  // The returned node is the replacement the caller replaces `root` with, so it absorbs the
  // matched root's origin identity.
  if (ret != root) {
    TraceNewFusionNode(survivor, {root});
  }
  if (trace_policy_ != TracePolicy::kAutoRegion) {
    return ret;
  }
  // Absorb-by-disappearance scoped to THIS match: equiv-listed CNode parts that lost their last
  // user during Process were consumed by the fusion, so their identity moves to the survivor.
  // Fusions that fan identity out to several survivors must stay kManual and trace explicitly.
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    return ret;
  }
  const auto &users = manager->node_users();
  for (const auto &part : matched_parts) {
    auto cnode = dyn_cast<CNode>(part);
    if (cnode == nullptr || cnode == survivor || cnode == root) {
      continue;
    }
    // Losing the last user either empties the entry or erases it from the map.
    auto it = users.find(cnode);
    if (it == users.end() || it->second.empty()) {
      TraceAbsorbNames(survivor, cnode);
    }
  }
  return ret;
}

bool LiteNodePass::Run(const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    MS_LOG(ERROR) << "func_graph is nullptr.";
    return false;
  }
  FuncGraphManagerPtr manager = func_graph->manager();
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr.";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return false;
  }
  manager->AddFuncGraph(func_graph);

  std::unordered_set<AnfNodePtr> seen_node;
  std::deque<AnfNodePtr> to_process{func_graph->output()};
  bool changes = false;
  while (!to_process.empty()) {
    AnfNodePtr node = to_process.front();
    to_process.pop_front();
    if (seen_node.count(node) > 0 || !manager->all_nodes().contains(node)) {
      continue;
    }
    (void)seen_node.insert(node);
    AnfNodePtr new_node = Run(func_graph, node);
    bool change = (new_node != nullptr);
    if (new_node != nullptr && new_node != node) {
      (void)manager->Replace(node, new_node);
      (void)seen_node.erase(node);
    } else if (new_node == nullptr) {
      new_node = node;
    }
    MS_CHECK_TRUE_MSG(new_node != nullptr, false, "new_node is nullptr!");
    if (IsValueNode<FuncGraph>(new_node)) {
      auto const_func_graph = GetValueNode<FuncGraphPtr>(new_node);
      if (const_func_graph == nullptr) {
        lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
        return false;
      }
      to_process.push_back(const_func_graph->output());
    } else if (new_node->isa<CNode>()) {
      if (IsGraphKernel(new_node)) {
        to_process.push_back(new_node);
      }
      auto cnode = new_node->cast<CNodePtr>();
      if (cnode == nullptr) {
        lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
        return false;
      }
      auto inputs = cnode->inputs();
      (void)to_process.insert(to_process.end(), inputs.begin(), inputs.end());
    }
    changes = changes || change;
    if (changes) {
      MS_LOG(DEBUG) << "pass " << this->name() << "changed node:" << new_node->fullname_with_scope();
    }
  }
  return changes;
}
}  // namespace opt
}  // namespace mindspore
