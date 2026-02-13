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

#include "tools/optimizer/fusion/mixed_aclnn_pass.h"
#include <unordered_map>
#include <unordered_set>
#include <deque>
#include <utility>
#include <set>
#include <algorithm>
#include <map>
#include "mindspore/core/include/ir/graph_utils.h"
#include "tools/common/string_util.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "infer/custom.h"
#include "infer/tuple_get_item.h"
#include "common/common.h"

namespace mindspore::opt {
namespace {
constexpr size_t kTargetNodeSize = 2;
constexpr auto kUniqueName = "uniq_name";
}  // namespace

STATUS SubgraphInfoBuilder::GetSuccessors(const FuncGraphPtr &graph, const AnfNodePtr &predecessor,
                                          std::vector<Edge> *successor) {
  MS_CHECK_TRUE_MSG(graph != nullptr, lite::RET_NULL_PTR, "graph is nullptr.");
  MS_CHECK_TRUE_MSG(predecessor != nullptr, lite::RET_NULL_PTR, "predecessor is nullptr.");
  MS_CHECK_TRUE_MSG(successor != nullptr, lite::RET_NULL_PTR, "successor is nullptr.");
  auto manager = graph->manager();
  MS_CHECK_TRUE_MSG(manager != nullptr, lite::RET_NULL_PTR, "manager is nullptr.");

  for (auto &node : manager->node_users()[predecessor]) {
    MS_CHECK_TRUE_MSG(node.first != nullptr, lite::RET_NULL_PTR, "node is nullptr");
    MS_CHECK_TRUE_MSG(node.first->isa<CNode>(), lite::RET_ERROR, "node should be a cnode");
    auto cnode = node.first->cast<CNodePtr>();
    MS_CHECK_TRUE_MSG(cnode != nullptr, lite::RET_NULL_PTR, "cnode is nullptr.");
    successor->push_back({predecessor, cnode, node.second});
  }
  return lite::RET_OK;
}

STATUS SubgraphInfoBuilder::ProcessInputs(const CNodePtr &old_cnode, const size_t &input_id,
                                          const FuncGraphPtr &subgraph,
                                          const std::unordered_map<AnfNodePtr, AnfNodePtr> &node_map,
                                          std::vector<AnfNodePtr> *new_inputs) {
  MS_CHECK_TRUE_MSG(old_cnode != nullptr, lite::RET_NULL_PTR, "old_cnode is nullptr");
  MS_CHECK_TRUE_MSG(subgraph != nullptr, lite::RET_NULL_PTR, "subgraph is nullptr");
  MS_CHECK_TRUE_MSG(new_inputs != nullptr, lite::RET_NULL_PTR, "new_inputs is nullptr");
  auto old_input = old_cnode->input(input_id);
  MS_CHECK_TRUE_MSG(old_input != nullptr, lite::RET_NULL_PTR, "old_input is nullptr");
  AnfNodePtr new_input = nullptr;
  auto it = node_map.find(old_input);
  if (it != node_map.end()) {  // processed, skip
    new_input = it->second;
  } else if (old_input->isa<CNode>()) {  // require value from another subgraph
    MS_LOG(INFO) << old_input->fullname_with_scope() << " will be a input in subgraph";
    auto param = subgraph->add_parameter();
    MS_CHECK_TRUE_MSG(param != nullptr, lite::RET_NULL_PTR, "param is nullptr");
    auto abstract = old_input->abstract();
    MS_CHECK_TRUE_MSG(abstract != nullptr, lite::RET_NULL_PTR, "abstract is nullptr");
    auto abstract_clone = abstract->Clone();
    MS_CHECK_TRUE_MSG(abstract_clone != nullptr, lite::RET_NULL_PTR, "abstract_clone is nullptr");
    param->set_abstract(abstract_clone);
    param->set_name(old_input->fullname_with_scope());

    new_input = param;
  } else if (old_input->isa<Parameter>()) {
    auto old_param = old_input->cast<ParameterPtr>();
    MS_CHECK_TRUE_MSG(old_param != nullptr, lite::RET_NULL_PTR, "cast ParameterPtr failed.");

    auto param = subgraph->add_parameter();
    MS_CHECK_TRUE_MSG(param != nullptr, lite::RET_NULL_PTR, "param is nullptr");
    auto abstract = old_param->abstract();
    MS_CHECK_TRUE_MSG(abstract != nullptr, lite::RET_NULL_PTR, "abstract is nullptr");
    auto abstract_clone = abstract->Clone();
    MS_CHECK_TRUE_MSG(abstract_clone != nullptr, lite::RET_NULL_PTR, "abstract_clone is nullptr");

    auto default_param = old_param->default_param();
    if (default_param) {
      param->set_default_param(default_param);
    }
    param->set_abstract(abstract_clone);
    param->set_name(old_input->fullname_with_scope());

    new_input = param;
  } else if (old_input->isa<ValueNode>()) {  // value node
    new_input = std::make_shared<ValueNode>(GetValueNode<ValuePtr>(old_input));
  }
  if (new_input == nullptr) {
    MS_LOG(ERROR) << "process input for " << old_cnode->fullname_with_scope() << ":" << input_id << "failed.";
    return lite::RET_ERROR;
  }
  new_inputs->push_back(new_input);
  return lite::RET_OK;
}

STATUS SubgraphInfoBuilder::BFSShapeValueCNode(const CNodePtr &cnode, std::vector<CNodePtr> *out) {
  MS_CHECK_TRUE_MSG(cnode != nullptr, lite::RET_NULL_PTR, "cnode is nullptr");
  MS_CHECK_TRUE_MSG(out != nullptr, lite::RET_NULL_PTR, "out is nullptr");
  out->clear();
  if (cnode->GetAttr(lite::kNameShapeValueAttr) == nullptr) {
    return lite::RET_OK;
  }
  out->push_back(cnode);
  size_t work_index = 0;
  do {
    auto &work_node = (*out)[work_index];
    auto inputs = work_node->inputs();
    for (size_t i = 1; i < inputs.size(); i++) {
      const auto &input_node = inputs[i];
      if (!input_node->isa<CNode>()) {
        continue;
      }
      auto input_cnode = input_node->cast<CNodePtr>();
      MS_CHECK_TRUE_MSG(input_cnode != nullptr, lite::RET_NULL_PTR, "input_cnode is nullptr");
      if (input_cnode->GetAttr(lite::kNameShapeValueAttr)) {
        out->push_back(input_cnode);
      }
    }
  } while (++work_index < out->size());
  return lite::RET_OK;
}

STATUS SubgraphInfoBuilder::CloneCNode(const CNodePtr &old_cnode, const FuncGraphPtr &graph,
                                       std::unordered_map<AnfNodePtr, AnfNodePtr> *old_to_new_map,
                                       std::unordered_map<AnfNodePtr, AnfNodePtr> *new_to_old_map) {
  MS_CHECK_TRUE_MSG(old_cnode != nullptr, lite::RET_NULL_PTR, "old_cnode is nullptr");
  MS_CHECK_TRUE_MSG(graph != nullptr, lite::RET_NULL_PTR, "graph is nullptr");
  MS_CHECK_TRUE_MSG(old_to_new_map != nullptr, lite::RET_NULL_PTR, "old_to_new_map is nullptr");
  MS_CHECK_TRUE_MSG(new_to_old_map != nullptr, lite::RET_NULL_PTR, "new_to_old_map is nullptr");
  auto prim_c = GetValueNode<PrimitivePtr>(old_cnode->input(0));
  std::vector<AnfNodePtr> new_inputs = {std::make_shared<ValueNode>(prim_c)};

  for (size_t j = 1; j < old_cnode->size(); j++) {
    auto ret = ProcessInputs(old_cnode, j, graph, *old_to_new_map, &new_inputs);
    if (ret != lite::RET_OK) {
      MS_LOG(ERROR) << "ProcessInputs for " << old_cnode->fullname_with_scope() << "failed.";
      return ret;
    }

    new_to_old_map->insert_or_assign(new_inputs[j], old_cnode->input(j));
    old_to_new_map->insert_or_assign(old_cnode->input(j), new_inputs[j]);
  }

  auto new_cnode = graph->NewCNode(new_inputs);
  MS_CHECK_TRUE_MSG(new_cnode != nullptr, lite::RET_NULL_PTR, "new_cnode is nullptr");
  new_cnode->set_fullname_with_scope(old_cnode->fullname_with_scope());

  auto abstract = old_cnode->abstract();
  MS_CHECK_TRUE_MSG(abstract != nullptr, lite::RET_NULL_PTR, "abstract is nullptr");
  auto abstract_clone = abstract->Clone();
  MS_CHECK_TRUE_MSG(abstract_clone != nullptr, lite::RET_NULL_PTR, "abstract is nullptr");
  new_cnode->set_abstract(abstract_clone);
  old_to_new_map->insert_or_assign(old_cnode, new_cnode);
  new_to_old_map->insert_or_assign(new_cnode, old_cnode);
  return lite::RET_OK;
}

STATUS SubgraphInfoBuilder::BuildSubGraphInfo(const FuncGraphPtr &original_graph,
                                              const std::unordered_map<AnfNodePtr, size_t> &node_graph_id_map,
                                              const std::vector<AnfNodePtr> &subgraph_node, size_t graph_id,
                                              SubgraphInfo *graph_info,
                                              std::map<int64_t, std::unordered_set<CNodePtr>> *extra_outputs) {
  MS_CHECK_TRUE_MSG(original_graph != nullptr, lite::RET_NULL_PTR, "original_graph is nullptr.");
  MS_CHECK_TRUE_MSG(graph_info != nullptr, lite::RET_NULL_PTR, "graph_info is nullptr.");
  MS_CHECK_TRUE_MSG(extra_outputs != nullptr, lite::RET_NULL_PTR, "extra_outputs is nullptr.");
  auto graph = std::make_shared<FuncGraph>();

  std::unordered_map<AnfNodePtr, AnfNodePtr> old_to_new_map;
  std::unordered_map<AnfNodePtr, AnfNodePtr> new_to_old_map;

  for (auto &node : subgraph_node) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }

    auto old_cnode = node->cast<CNodePtr>();
    MS_CHECK_TRUE_MSG(old_cnode != nullptr, lite::RET_NULL_PTR, "cast to cnode failed!");
    auto ret = CloneCNode(old_cnode, graph, &old_to_new_map, &new_to_old_map);
    if (ret != lite::RET_OK) {
      MS_LOG(ERROR) << "Clone CNode: " << old_cnode->fullname_with_scope() << " failed.";
      return ret;
    }
  }

  auto ret = HandleShapeValueInputs(original_graph, graph, &old_to_new_map, &new_to_old_map, graph_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "HandleShapeValueInputs failed.";
    return ret;
  }

  ret = ProcessExtraOutputs(node_graph_id_map, graph, new_to_old_map, extra_outputs);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ProcessExtraOutputs failed.";
    return ret;
  }

  (*graph_info) = SubgraphInfo{graph, {}, old_to_new_map, new_to_old_map};
  return RET_OK;
}

STATUS SubgraphInfoBuilder::BuildSubGraphOutputs(const FuncGraphPtr &original_graph,
                                                 const std::unordered_map<AnfNodePtr, size_t> &node_graph_id_map,
                                                 const std::vector<AnfNodePtr> &subgraph_node,
                                                 const std::unordered_set<CNodePtr> &required_outputs,
                                                 SubgraphInfo *graph_info) {
  MS_CHECK_TRUE_MSG(original_graph != nullptr, lite::RET_NULL_PTR, "original_graph is nullptr.");
  MS_CHECK_TRUE_MSG(graph_info != nullptr, lite::RET_NULL_PTR, "subgraph is nullptr");

  std::vector<Edge> graph_outputs;
  std::unordered_set<AnfNodePtr> outputs_set;
  for (auto &node : subgraph_node) {
    MS_CHECK_TRUE_MSG(node != nullptr, lite::RET_NULL_PTR, "node is nullptr");
    if (!node->isa<CNode>()) {
      continue;
    }
    auto old_cnode = node->cast<CNodePtr>();
    MS_CHECK_TRUE_MSG(old_cnode != nullptr, lite::RET_NULL_PTR, "cast to cnode failed.");

    std::vector<Edge> outputs;
    auto ret = GetSuccessors(original_graph, old_cnode, &outputs);
    MS_CHECK_TRUE_MSG(ret == lite::RET_OK, ret, "GetSuccessors failed.");

    for (auto &o : outputs) {
      MS_CHECK_TRUE_MSG(o.source != nullptr, lite::RET_NULL_PTR, "output source is nullptr");
      MS_CHECK_TRUE_MSG(o.target != nullptr, lite::RET_NULL_PTR, "output target is nullptr");
      auto it = node_graph_id_map.find(o.target);
      if (it == node_graph_id_map.end()) {
        MS_LOG(ERROR) << "get graph id for node:" << o.target->fullname_with_scope() << " failed.";
        return lite::RET_ERROR;
      }

      if (graph_info->old_to_new_map.find(o.target) == graph_info->old_to_new_map.end() &&
          !outputs_set.count(o.source)) {
        graph_outputs.push_back(o);
        outputs_set.insert(o.source);
      }
    }
  }

  std::unordered_set<AnfNodePtr> return_nodes;

  std::transform(graph_outputs.begin(), graph_outputs.end(), std::inserter(return_nodes, return_nodes.begin()),
                 [&](const Edge &o) { return graph_info->old_to_new_map[o.source]; });

  std::transform(required_outputs.begin(), required_outputs.end(), std::inserter(return_nodes, return_nodes.begin()),
                 [&](const auto &node) { return graph_info->old_to_new_map[node]; });

  std::vector<AnfNodePtr> new_graph_return_vec(return_nodes.begin(), return_nodes.end());
  auto ret = BuildReturnNode(graph_info->graph, new_graph_return_vec);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "build return node failed!";
    return lite::RET_ERROR;
  }

  std::vector<AnfNodePtr> old_graph_return_vec;
  std::transform(new_graph_return_vec.begin(), new_graph_return_vec.end(), std::back_inserter(old_graph_return_vec),
                 [&](const auto &node) { return graph_info->new_to_old_map[node]; });
  graph_info->outputs = std::move(old_graph_return_vec);
  auto final_inputs = graph_info->graph->get_inputs();
  for (size_t i = 0; i < final_inputs.size(); i++) {
    auto input_param = final_inputs[i]->cast<ParameterPtr>();
    input_param->set_name(std::to_string(i));
  }
  return lite::RET_OK;
}

STATUS SubgraphInfoBuilder::SplitGraph(const FuncGraphPtr &original_graph, const std::set<std::string> &split_op_names,
                                       std::vector<std::vector<AnfNodePtr>> *out_subgraphs,
                                       std::unordered_map<AnfNodePtr, size_t> *out_node_to_graph_id) {
  MS_CHECK_TRUE_MSG(original_graph != nullptr, lite::RET_NULL_PTR, "original_graph is nullptr.");
  MS_CHECK_TRUE_MSG(out_subgraphs != nullptr, lite::RET_NULL_PTR, "out_subgraphs is nullptr.");
  MS_CHECK_TRUE_MSG(out_node_to_graph_id != nullptr, lite::RET_NULL_PTR, "out_node_to_graph_id is nullptr.");
  std::unordered_map<AnfNodePtr, size_t> node_to_graph_id;
  std::vector<std::vector<AnfNodePtr>> subgraph_nodes(1);
  auto all_nodes = TopoSort(original_graph->get_return());

  for (auto &node : all_nodes) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto node_name = cnode->fullname_with_scope();

    MS_LOG(INFO) << "MixedAclnnPass check node: " << node_name;
    if (node_name == kReturnOpName || node_name == "return tuple") {
      node_to_graph_id.insert({node, 0});  // 0 means original_graph
      break;
    }

    if (split_op_names.find(node_name) != split_op_names.end()) {
      node_to_graph_id.insert({cnode, 0});
      if (!subgraph_nodes.back().empty()) {
        subgraph_nodes.emplace_back();
      }
      continue;
    }

    subgraph_nodes.back().push_back(cnode);
    node_to_graph_id.insert({cnode, subgraph_nodes.size()});
  }
  if (subgraph_nodes.back().empty()) {
    subgraph_nodes.pop_back();
  }
  *out_subgraphs = std::move(subgraph_nodes);
  *out_node_to_graph_id = std::move(node_to_graph_id);
  return RET_OK;
}

STATUS SubgraphInfoBuilder::BuildAllSubgraphInfo(const FuncGraphPtr &original_graph,
                                                 const std::unordered_map<AnfNodePtr, size_t> &node_to_graph_id,
                                                 const std::vector<std::vector<AnfNodePtr>> &subgraph_nodes,
                                                 std::vector<SubgraphInfo> *out) {
  MS_CHECK_TRUE_MSG(original_graph != nullptr, lite::RET_NULL_PTR, "original_graph is nullptr.");
  MS_CHECK_TRUE_MSG(out != nullptr, lite::RET_NULL_PTR, "out is nullptr.");
  std::vector<SubgraphInfo> subgraphs;
  std::map<int64_t, std::unordered_set<CNodePtr>> extra_outputs;
  for (size_t i = 0; i < subgraph_nodes.size(); i++) {
    if (subgraph_nodes[i].empty()) {
      MS_LOG(ERROR) << "subgraph_nodes[" << i << "] is empty.";
      return lite::RET_ERROR;
    }
    SubgraphInfo subgraph;
    auto ret = BuildSubGraphInfo(original_graph, node_to_graph_id, subgraph_nodes[i], i + 1, &subgraph, &extra_outputs);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Build subgraph failed, graph_id: " << i;
      return ret;
    }

    subgraphs.push_back(subgraph);
  }

  for (size_t i = 0; i < subgraph_nodes.size(); i++) {
    auto ret =
      BuildSubGraphOutputs(original_graph, node_to_graph_id, subgraph_nodes[i], extra_outputs[i + 1], &subgraphs[i]);
    if (ret != lite::RET_OK) {
      MS_LOG(ERROR) << "Build subgraph outputs failed, graph_id: " << i;
      return ret;
    }
  }
  *out = std::move(subgraphs);
  return RET_OK;
}

STATUS SubgraphInfoBuilder::HandleShapeValueInputs(const FuncGraphPtr &original_graph, const FuncGraphPtr &subgraph,
                                                   std::unordered_map<AnfNodePtr, AnfNodePtr> *old_to_new_map,
                                                   std::unordered_map<AnfNodePtr, AnfNodePtr> *new_to_old_map,
                                                   size_t graph_id) {
  MS_CHECK_TRUE_MSG(original_graph != nullptr, lite::RET_NULL_PTR, "original_graph is nullptr.");
  MS_CHECK_TRUE_MSG(subgraph != nullptr, lite::RET_NULL_PTR, "subgraph is nullptr.");
  MS_CHECK_TRUE_MSG(old_to_new_map != nullptr, lite::RET_NULL_PTR, "old_to_new_map is nullptr.");
  MS_CHECK_TRUE_MSG(new_to_old_map != nullptr, lite::RET_NULL_PTR, "new_to_old_map is nullptr.");

  auto inputs = subgraph->get_inputs();
  for (auto &input : inputs) {
    MS_CHECK_TRUE_MSG(input != nullptr, lite::RET_NULL_PTR, "input is nullptr");
    auto old_it = new_to_old_map->find(input);
    if (old_it == new_to_old_map->end()) {
      MS_LOG(ERROR) << "cannot find origin node for " << input->fullname_with_scope();
      return lite::RET_ERROR;
    }
    if (!old_it->second->isa<CNode>()) {
      continue;
    }
    auto input_cnode = old_it->second->cast<CNodePtr>();
    if (!input_cnode->GetAttr(lite::kNameShapeValueAttr)) {
      continue;
    }
    MS_LOG(INFO) << input_cnode->fullname_with_scope()
                 << " produce a shape value. it cannot be a input for graph:" << graph_id;

    std::vector<CNodePtr> extra_cnodes;
    auto ret = BFSShapeValueCNode(input_cnode, &extra_cnodes);
    if (ret != lite::RET_OK) {
      MS_LOG(ERROR) << "search shape value depend of node:" << input_cnode->fullname_with_scope() << " failed.";
      return ret;
    }

    old_to_new_map->erase(input_cnode);
    for (auto it = extra_cnodes.rbegin(); it < extra_cnodes.rend(); it++) {
      const auto &old_cnode = *it;
      if (old_to_new_map->find(old_cnode) != old_to_new_map->end()) {
        continue;
      }
      ret = CloneCNode(old_cnode, subgraph, old_to_new_map, new_to_old_map);
      if (ret != lite::RET_OK) {
        MS_LOG(ERROR) << "Clone CNode: " << old_cnode->fullname_with_scope() << " failed.";
        return ret;
      }
    }

    auto manager = original_graph->manager();
    MS_CHECK_TRUE_MSG(manager != nullptr, lite::RET_NULL_PTR, "manager is nullptr");
    auto node_users = manager->node_users()[input_cnode];
    for (auto &user : node_users) {
      auto new_it = old_to_new_map->find(user.first);
      if (new_it != old_to_new_map->end()) {
        if (!new_it->second->isa<CNode>()) {
          MS_LOG(ERROR) << new_it->second->fullname_with_scope() << " is not a CNode. It cannot be user of"
                        << input->fullname_with_scope();
          return lite::RET_ERROR;
        }

        auto new_node = (*old_to_new_map)[input_cnode];
        auto user_cnode = new_it->second->cast<CNodePtr>();
        user_cnode->set_input(user.second, new_node);
      }
    }
    subgraph->DropNode(input);
  }
  return RET_OK;
}

STATUS SubgraphInfoBuilder::ProcessExtraOutputs(const std::unordered_map<AnfNodePtr, size_t> &node_graph_id_map,
                                                const FuncGraphPtr &subgraph,
                                                const std::unordered_map<AnfNodePtr, AnfNodePtr> &new_to_old_map,
                                                std::map<int64_t, std::unordered_set<CNodePtr>> *extra_outputs) {
  MS_CHECK_TRUE_MSG(subgraph != nullptr, lite::RET_NULL_PTR, "subgraph is nullptr.");
  MS_CHECK_TRUE_MSG(extra_outputs != nullptr, lite::RET_NULL_PTR, "extra_outputs is nullptr.");

  auto inputs = subgraph->get_inputs();
  for (auto &input : inputs) {
    MS_CHECK_TRUE_MSG(input != nullptr, lite::RET_NULL_PTR, "input is nullptr");
    auto old_it = new_to_old_map.find(input);
    if (old_it == new_to_old_map.end()) {
      MS_LOG(ERROR) << "cannot find origin node for " << input->fullname_with_scope();
      return lite::RET_ERROR;
    }
    if (!old_it->second->isa<CNode>()) {
      continue;
    }
    auto input_cnode = old_it->second->cast<CNodePtr>();
    auto it = node_graph_id_map.find(input_cnode);
    if (it == node_graph_id_map.end()) {
      MS_LOG(ERROR) << "graph id of " << input_cnode->fullname_with_scope() << " cannot be found.";
      return lite::RET_ERROR;
    }

    (*extra_outputs)[it->second].insert(input_cnode);
  }
  return RET_OK;
}

STATUS SubgraphInfoBuilder::Build(const FuncGraphPtr &original_graph, const std::set<std::string> &split_op_names,
                                  std::vector<SubgraphInfo> *out_subgraphs,
                                  std::unordered_map<AnfNodePtr, size_t> *out_node_to_graph_id) {
  MS_CHECK_TRUE_MSG(original_graph != nullptr, lite::RET_NULL_PTR, "original_grap his nullptr.");
  MS_CHECK_TRUE_MSG(out_subgraphs != nullptr, lite::RET_NULL_PTR, "out_subgraphs is nullptr.");
  MS_CHECK_TRUE_MSG(out_node_to_graph_id != nullptr, lite::RET_NULL_PTR, "out_node_to_graph_id is nullptr.");
  std::vector<std::vector<AnfNodePtr>> subgraph_nodes;
  auto ret = SplitGraph(original_graph, split_op_names, &subgraph_nodes, out_node_to_graph_id);
  MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "SplitGraph failed.");

  ret = BuildAllSubgraphInfo(original_graph, *out_node_to_graph_id, subgraph_nodes, out_subgraphs);
  MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "BuildAllSubgraphInfo failed.");
  return lite::RET_OK;
}

STATUS MixedAclnnPass::TransposeValueVec(const std::vector<std::vector<ValuePtr>> &matrix,
                                         std::vector<std::vector<ValuePtr>> *ret) {
  MS_CHECK_TRUE_MSG(ret != nullptr, lite::RET_NULL_PTR, "ret is nullptr");
  if (matrix.empty()) {
    *ret = {};
    return lite::RET_OK;
  }

  const size_t rows = matrix.size();
  const size_t cols = matrix[0].size();

  if (!std::all_of(matrix.begin(), matrix.end(), [cols](const auto &row) { return row.size() == cols; })) {
    MS_LOG(ERROR) << "Matrix rows have different sizes";
    return lite::RET_ERROR;
  }

  std::vector<std::vector<ValuePtr>> result;
  result.reserve(cols);

  for (size_t j = 0; j < cols; ++j) {
    std::vector<ValuePtr> column;
    column.reserve(rows);
    for (size_t i = 0; i < rows; ++i) {
      column.push_back(matrix[i][j]);
    }

    result.push_back(std::move(column));
  }
  *ret = std::move(result);
  return lite::RET_OK;
}

STATUS MixedAclnnPass::CollectSubgraphInputs(const SubgraphInfo &subgraph,
                                             const std::unordered_set<AnfNodePtr> &begin_nodes,
                                             const std::unordered_map<AnfNodePtr, size_t> &node_to_graph_id,
                                             const std::vector<SubgraphInfo> &subgraphs,
                                             const std::unordered_map<size_t, std::vector<AnfNodePtr>> &custom_outputs,
                                             std::vector<AnfNodePtr> *new_inputs) {
  auto graph_inputs = subgraph.graph->get_inputs();
  for (auto &new_node : graph_inputs) {
    auto it_old_node = subgraph.new_to_old_map.find(new_node);
    if (it_old_node == subgraph.new_to_old_map.end()) {
      MS_LOG(ERROR) << "cannot find original node for " << new_node->fullname_with_scope();
      return lite::RET_ERROR;
    }
    auto old_node = it_old_node->second;
    auto find = begin_nodes.find(old_node);
    if (find != begin_nodes.end()) {
      new_inputs->push_back(*find);
      continue;
    }
    auto it_graph_id = node_to_graph_id.find(old_node);
    if (it_graph_id == node_to_graph_id.end()) {
      MS_LOG(ERROR) << "cannot find graph id for " << old_node->fullname_with_scope();
      return lite::RET_ERROR;
    }
    auto graph_id = it_graph_id->second;
    if (graph_id == 0) {
      new_inputs->push_back(old_node);
      continue;
    }
    auto graph_index = graph_id - 1;
    auto it = std::find(subgraphs[graph_index].outputs.begin(), subgraphs[graph_index].outputs.end(), old_node);
    if (it == subgraphs[graph_index].outputs.end()) {
      MS_LOG(ERROR) << "collect custom inputs failed.";
      return lite::RET_ERROR;
    }
    auto index = std::distance(subgraphs[graph_index].outputs.begin(), it);
    if (!custom_outputs.count(graph_id) || custom_outputs.at(graph_id).size() < static_cast<size_t>(index + 1)) {
      MS_LOG(ERROR) << "collect custom inputs failed. cannot get previous output.";
      return lite::RET_ERROR;
    }
    new_inputs->push_back(custom_outputs.at(graph_id)[index]);
  }
  return lite::RET_OK;
}

STATUS MixedAclnnPass::ReplaceWithSingleOutput(const FuncGraphPtr &original_graph, const CNodePtr &output,
                                               const AnfNodePtr &new_custom_node, const FuncGraphManagerPtr &manager) {
  MS_CHECK_TRUE_MSG(output != nullptr, lite::RET_NULL_PTR, "output is nullptr");
  auto abstract = output->abstract();
  MS_CHECK_TRUE_MSG(abstract != nullptr, lite::RET_NULL_PTR, "abstract is nullptr");
  auto abstract_clone = abstract->Clone();
  MS_CHECK_TRUE_MSG(abstract_clone != nullptr, lite::RET_NULL_PTR, "abstract_clone is nullptr");
  auto new_node = new_custom_node->cast<CNodePtr>();
  new_node->set_abstract(abstract_clone);
  new_node->AddAttr(lite::kNameShapeAttr, output->GetAttr(lite::kNameShapeAttr));
  if (!manager->Replace(output, new_custom_node)) {
    MS_LOG(ERROR) << "Replace custom node failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS MixedAclnnPass::ReplaceWithMultipleOutputs(const FuncGraphPtr &original_graph, const SubgraphInfo &subgraph,
                                                  const AnfNodePtr &new_custom_node, const FuncGraphManagerPtr &manager,
                                                  AnfNodePtrList *outputs) {
  AbstractBasePtrList abstract_list;
  std::vector<ValuePtr> dyn_shapes;
  for (size_t j = 0; j < subgraph.outputs.size(); j++) {
    auto output = subgraph.outputs[j];
    MS_CHECK_TRUE_MSG(output != nullptr, lite::RET_NULL_PTR, "output is nullptr");
    auto cnode = output->cast<CNodePtr>();
    MS_CHECK_TRUE_MSG(cnode != nullptr, lite::RET_NULL_PTR, "cnode is nullptr");
    auto abstract = cnode->abstract();
    MS_CHECK_TRUE_MSG(abstract != nullptr, lite::RET_NULL_PTR, "abstract is nullptr");
    auto abstract_clone = abstract->Clone();
    MS_CHECK_TRUE_MSG(abstract_clone != nullptr, lite::RET_NULL_PTR, "abstract_clone is nullptr");
    abstract_list.emplace_back(abstract_clone);

    mindspore::ops::TupleGetItem tuple_get_item_prim;
    auto tuple_get_item_prim_c = tuple_get_item_prim.GetPrim();
    MS_CHECK_TRUE_MSG(tuple_get_item_prim_c != nullptr, lite::RET_NULL_PTR,
                      "create tuple_get_item_prim_c return nullptr");
    auto tuple_get_item_prim_value = NewValueNode(tuple_get_item_prim_c);
    MS_CHECK_TRUE_MSG(tuple_get_item_prim_value != nullptr, lite::RET_NULL_PTR, "create ValueNode return nullptr");
    auto get_item_value = NewValueNode(MakeValue<int32_t>(j));
    MS_CHECK_TRUE_MSG(get_item_value != nullptr, lite::RET_NULL_PTR, "create ValueNode return nullptr");
    std::vector<AnfNodePtr> inputs{tuple_get_item_prim_value, new_custom_node, get_item_value};
    CNodePtr get_item_cnode = original_graph->NewCNode(inputs);
    MS_CHECK_TRUE_MSG(get_item_cnode != nullptr, lite::RET_NULL_PTR, "get_item_cnode is nullptr.");
    auto new_node = new_custom_node->cast<CNodePtr>();
    get_item_cnode->set_fullname_with_scope(new_node->fullname_with_scope() + "_getitem_" + std::to_string(j));
    get_item_cnode->set_abstract(abstract_clone);
    if (!manager->Replace(output, get_item_cnode)) {
      MS_LOG(ERROR) << "Replace custom node failed.";
      return RET_ERROR;
    }
    outputs->push_back(get_item_cnode);
    get_item_cnode->AddAttr(lite::kNameShapeAttr, cnode->GetAttr(lite::kNameShapeAttr));
    dyn_shapes.push_back(cnode->GetAttr(lite::kNameShapeAttr));
  }
  auto new_abstract_list = std::make_shared<abstract::AbstractTuple>(abstract_list);
  CHECK_NULL_RETURN(new_abstract_list);
  auto new_node = new_custom_node->cast<CNodePtr>();
  new_node->set_abstract(new_abstract_list);

  std::vector<std::vector<ValuePtr>> final_shapes;
  for (auto &output_shape : dyn_shapes) {
    MS_CHECK_TRUE_MSG(output_shape->isa<ValueSequence>(), lite::RET_ERROR, "output_shape should be ValueSequence");
    final_shapes.push_back(GetValue<std::vector<ValuePtr>>(output_shape));
  }

  std::vector<std::vector<ValuePtr>> shapes_attr;
  auto ret = TransposeValueVec(final_shapes, &shapes_attr);
  MS_CHECK_TRUE_MSG(ret == lite::RET_OK, ret, "transpose final_shapes failed.");
  new_node->AddAttr(lite::kNameShapeAttr, MakeValue(shapes_attr));
  return RET_OK;
}

STATUS MixedAclnnPass::CreateCustomAclnnSubgraph(const FuncGraphPtr &original_graph,
                                                 const std::unordered_set<AnfNodePtr> &begin_nodes,
                                                 std::vector<SubgraphInfo> &subgraphs,
                                                 std::unordered_map<AnfNodePtr, size_t> &node_to_graph_id) {
  MS_CHECK_TRUE_MSG(original_graph != nullptr, lite::RET_NULL_PTR, "original_graph is nullptr.");
  auto manager = Manage(original_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr.";
    return RET_ERROR;
  }
  std::unordered_map<size_t, std::vector<AnfNodePtr>> custom_outputs;

  for (size_t i = 0; i < subgraphs.size(); i++) {
    auto subgraph = subgraphs[i];
    auto cur_graph_id = i + 1;
    if (subgraph.outputs.empty()) {
      MS_LOG(ERROR) << "subgraph " << cur_graph_id << " outputs is empty";
      return RET_ERROR;
    }
    mindspore::ops::Custom prim;
    prim.AddAttr(kUniqueName, api::MakeValue<std::string>(lite::kNameCustomAclnnSubgraph));
    auto prim_c = prim.GetPrim();

    std::vector<AnfNodePtr> new_inputs;
    auto ret = CollectSubgraphInputs(subgraph, begin_nodes, node_to_graph_id, subgraphs, custom_outputs, &new_inputs);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "collect subgraph inputs for graph: " << cur_graph_id << " failed.";
      return ret;
    }

    auto new_custom_node = original_graph->NewCNode(prim_c, new_inputs);
    new_custom_node->AddAttr(lite::kNameGraphAttr, MakeValue(subgraph.graph));
    new_custom_node->set_fullname_with_scope("custom_" + std::to_string(cur_graph_id));
    AnfNodePtrList outputs;
    if (subgraph.outputs.size() == 1) {
      outputs.push_back(new_custom_node);
      auto output = subgraph.outputs.front()->cast<CNodePtr>();
      ret = ReplaceWithSingleOutput(original_graph, output, new_custom_node, manager);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "replace failed. graph: " << cur_graph_id;
        return ret;
      }
    } else {
      ret = ReplaceWithMultipleOutputs(original_graph, subgraph, new_custom_node, manager, &outputs);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "replace failed. graph: " << cur_graph_id;
        return ret;
      }
    }
    custom_outputs.insert({cur_graph_id, outputs});
  }

  return RET_OK;
}

bool MixedAclnnPass::Run(const FuncGraphPtr &original_graph) {
  MS_CHECK_TRUE_MSG(original_graph != nullptr, false, "original_graph is nullptr!");
  MS_CHECK_TRUE_MSG(param_ != nullptr, false, "param_ is nullptr!");
  auto split_nodes_vec = lite::SplitStringToVector(param_->aclModelOptionCfgParam.aclnn_nodes, ",");
  std::set<std::string> split_op_names(split_nodes_vec.begin(), split_nodes_vec.end());

  SubgraphInfoBuilder builder;
  std::vector<SubgraphInfo> subgraphs;
  std::unordered_map<AnfNodePtr, size_t> node_to_graph_id;
  auto ret = builder.Build(original_graph, split_op_names, &subgraphs, &node_to_graph_id);
  MS_CHECK_TRUE_MSG(ret == RET_OK, false, "Build subgraph info failed.");

  auto model_inputs = original_graph->get_inputs();
  std::unordered_set<AnfNodePtr> begin_nodes(model_inputs.begin(), model_inputs.end());
  ret = CreateCustomAclnnSubgraph(original_graph, begin_nodes, subgraphs, node_to_graph_id);
  MS_CHECK_TRUE_MSG(ret == RET_OK, false, "CreateCustomNodesAndReplace failed.");
  return true;
}

}  // namespace mindspore::opt
