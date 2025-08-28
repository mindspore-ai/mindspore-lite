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
#include "tools/optimizer/fusion/graph_split_pass.h"
#include <unordered_map>
#include <unordered_set>
#include <deque>
#include <utility>
#include <set>
#include <algorithm>
#include <map>
#include "infer/return.h"
#include "tools/converter/export_model.h"
#include "infer/make_tuple.h"
#include "mindspore/core/include/ir/graph_utils.h"

namespace mindspore::opt {
namespace {
constexpr size_t kTargetNodeSize = 2;
}

STATUS BuildReturnNode(const FuncGraphPtr &anf_graph, const std::vector<AnfNodePtr> &return_inputs) {
  MS_CHECK_TRUE_RET(anf_graph != nullptr, lite::RET_NULL_PTR);
  auto return_prim = std::make_shared<ops::Return>();
  if (return_prim == nullptr) {
    MS_LOG(ERROR) << "new return failed!";
    return lite::RET_NULL_PTR;
  }
  if (return_inputs.empty()) {
    MS_LOG(ERROR) << "return input is empty";
    return lite::RET_ERROR;
  }
  auto final_return = return_inputs;
  AbstractBasePtr abstract = nullptr;
  if (return_inputs.size() == 1) {
    anf_graph->set_output(return_inputs.front(), false);
    abstract = return_inputs.front()->abstract();
  } else if (return_inputs.size() > 1) {
    auto make_tuple_prim_ptr = std::make_shared<ops::MakeTuple>();
    if (make_tuple_prim_ptr == nullptr) {
      MS_LOG(DEBUG) << "new maketyple failed";
      return lite::RET_NULL_PTR;
    }
    AbstractBasePtrList elem;
    std::transform(return_inputs.begin(), return_inputs.end(), std::back_inserter(elem),
                   [](auto &node) { return node->abstract(); });
    auto make_tuple_prim_c = make_tuple_prim_ptr->GetPrim();
    MS_CHECK_TRUE_MSG(make_tuple_prim_c != nullptr, lite::RET_NULL_PTR, "make_tuple_prim_c is nullptr!");
    auto make_tuple_cnode = anf_graph->NewCNode(make_tuple_prim_c, return_inputs);
    if (make_tuple_cnode == nullptr) {
      MS_LOG(ERROR) << "new cnode failed!";
      return lite::RET_NULL_PTR;
    }
    make_tuple_cnode->set_fullname_with_scope("return tuple");
    make_tuple_cnode->set_abstract(std::make_shared<abstract::AbstractTuple>(elem));
    abstract = make_tuple_cnode->abstract();
    final_return = {make_tuple_cnode};
  } else {
    MS_LOG(ERROR) << "Return inputs is 0!";
    return lite::RET_ERROR;
  }
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "Input node abstract is null, node:" << final_return.front()->fullname_with_scope();
    return lite::RET_ERROR;
  }
  auto return_prim_c = return_prim->GetPrim();
  CHECK_NULL_RETURN(return_prim_c);
  auto return_cnode = anf_graph->NewCNode(return_prim_c, final_return);
  if (return_cnode == nullptr) {
    MS_LOG(ERROR) << "new cnode error";
    return lite::RET_ERROR;
  }
  return_cnode->set_fullname_with_scope("Return");
  return_cnode->set_abstract(abstract);
  anf_graph->set_return(return_cnode);
  return lite::RET_OK;
}

bool IsWeight(const AnfNodePtr &node) {
  return (utils::isa<ParameterPtr>(node) && node->cast<ParameterPtr>() != nullptr &&
          node->cast<ParameterPtr>()->has_default());
}

STATUS BuildInputOutputMap(const std::vector<std::vector<std::string>> &subgraph_output_names,
                           const std::vector<std::vector<std::string>> &subgraph_input_names,
                           const std::vector<std::string> &main_graph_input_names,
                           const std::vector<std::string> &main_graph_output_names,
                           std::vector<std::vector<int>> *subgraph_input_to_main_graph_map,
                           std::vector<std::vector<int>> *subgraph_output_to_main_graph_map,
                           std::vector<std::vector<std::vector<int>>> *subgraph_output_to_subgraph_input_map) {
  std::map<std::string, std::vector<int>> subgraph_input_name_to_index_map;
  for (size_t i = 0; i < subgraph_input_names.size(); i++) {
    auto single_subgraph_input_names = subgraph_input_names[i];
    subgraph_input_to_main_graph_map->push_back(std::vector<int>(single_subgraph_input_names.size(), -1));
    for (size_t j = 0; j < single_subgraph_input_names.size(); j++) {
      subgraph_input_name_to_index_map[single_subgraph_input_names[j]] = {int32_t(i), int32_t(j)};
      auto input_name_iter =
        std::find(main_graph_input_names.begin(), main_graph_input_names.end(), single_subgraph_input_names[j]);
      if (input_name_iter != main_graph_input_names.end()) {
        int main_input_index = input_name_iter - main_graph_input_names.begin();
        subgraph_input_to_main_graph_map->at(i)[j] = main_input_index;
      }
    }
  }
  for (size_t i = 0; i < subgraph_output_names.size(); i++) {
    auto single_subgraph_output_names = subgraph_output_names[i];
    subgraph_output_to_subgraph_input_map->push_back(
      std::vector<std::vector<int>>(single_subgraph_output_names.size(), std::vector<int>(2, -1)));
    for (size_t j = 0; j < single_subgraph_output_names.size(); j++) {
      auto output_name = single_subgraph_output_names[j];
      auto input_output_iter = subgraph_input_name_to_index_map.find(output_name);
      if (input_output_iter != subgraph_input_name_to_index_map.end()) {
        subgraph_output_to_subgraph_input_map->at(i)[j] = input_output_iter->second;
      }
    }
  }
  for (size_t i = 0; i < subgraph_output_names.size(); i++) {
    auto single_subgraph_output_names = subgraph_output_names[i];
    subgraph_output_to_main_graph_map->push_back(std::vector<int>(single_subgraph_output_names.size(), -1));
    for (size_t j = 0; j < single_subgraph_output_names.size(); j++) {
      auto output_name_iter =
        std::find(main_graph_output_names.begin(), main_graph_output_names.end(), single_subgraph_output_names[j]);
      if (output_name_iter != main_graph_output_names.end()) {
        int main_output_index = output_name_iter - main_graph_output_names.begin();
        subgraph_output_to_main_graph_map->at(i)[j] = main_output_index;
      }
    }
  }
  return lite::RET_OK;
}

bool IsParentNode(const AnfNodePtr &input_node, const AnfNodePtr &output_node) {
  std::deque<AnfNodePtr> queue;
  queue.push_back(output_node);
  std::unordered_set<AnfNodePtr> visited;
  visited.insert(output_node);
  while (!queue.empty()) {
    auto node = queue.front();
    queue.pop_front();
    if (node == input_node) {
      return true;
    }
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_CHECK_TRUE_MSG(cnode != nullptr, false, "cnode is nullptr!");
    for (size_t j = 1; j < cnode->size(); ++j) {
      auto inp_node = cnode->input(j);
      if (visited.find(inp_node) == visited.end()) {
        queue.push_back(inp_node);
      }
      visited.insert(inp_node);
    }
  }
  return false;
}

STATUS GetSubgraphStopNodes(const std::vector<std::vector<AnfNodePtr>> &boundaries, const size_t &current_index,
                            const std::vector<AnfNodePtr> &output_nodes, std::vector<AnfNodePtr> *stop_nodes) {
  for (int i = current_index - 1; i >= 0; i--) {
    auto current_stop_nodes = boundaries[i];
    for (size_t j = 0; j < output_nodes.size(); j++) {
      for (size_t k = 0; k < current_stop_nodes.size(); k++) {
        if (IsParentNode(current_stop_nodes[k], output_nodes[j])) {
          stop_nodes->insert(stop_nodes->end(), current_stop_nodes.begin(), current_stop_nodes.end());
          return lite::RET_OK;
        }
      }
    }
  }
  MS_LOG(ERROR) << "Can not found prenode for current output nodes";
  return lite::RET_ERROR;
}

STATUS GetSuccessor(const FuncGraphPtr &original_graph, const AnfNodePtr &predecessor,
                    std::vector<AnfNodePtr> *successor) {
  auto all_nodes = TopoSort(original_graph->get_return());
  for (auto node : all_nodes) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_CHECK_TRUE_MSG(cnode != nullptr, lite::RET_ERROR, "cnode is nullptr!");
    auto inputs = cnode->inputs();
    auto it = std::find(cnode->inputs().begin(), cnode->inputs().end(), predecessor);
    if (it != cnode->inputs().end()) {
      successor->push_back(node);
    }
  }
  return lite::RET_OK;
}

STATUS ExtendSubgraphInputOutput(
  const std::vector<std::vector<std::string>> &subgraph_input_name,
  const std::vector<std::vector<std::string>> &subgraph_output_name,
  const std::vector<std::pair<std::vector<std::string>, std::vector<std::string>>> &subgraph_input_output,
  const std::vector<std::string> &main_graph_input_names,
  std::vector<std::vector<std::vector<std::string>>> *extended_subgraph_input_output,
  std::vector<std::vector<std::vector<std::string>>> *infer_path_root_nodes) {
  size_t min_graph_id = INT32_MAX;
  size_t max_graph_id = 0;

  for (auto input_output_pair : subgraph_input_output) {
    auto input_names = input_output_pair.first;
    std::set<std::string> extra_inputs;
    for (auto name : input_names) {
      for (size_t subgraph_id = 0; subgraph_id < subgraph_input_name.size(); subgraph_id++) {
        auto curr_subgraph_input = subgraph_input_name[subgraph_id];
        bool exists = std::any_of(curr_subgraph_input.begin(), curr_subgraph_input.end(),
                                  [&name](const auto &curr_input) { return curr_input == name; });
        if (exists) {
          min_graph_id = std::min(min_graph_id, subgraph_id);
          max_graph_id = std::max(max_graph_id, subgraph_id);
          extra_inputs.insert(curr_subgraph_input.begin(), curr_subgraph_input.end());
        }
      }
    }
    std::vector<std::string> collected_output_nodes;
    for (size_t graph_id = min_graph_id; graph_id < max_graph_id; graph_id++) {
      auto curr_subgraph_output_names = subgraph_output_name[graph_id];
      collected_output_nodes.insert(collected_output_nodes.end(), curr_subgraph_output_names.begin(),
                                    curr_subgraph_output_names.end());
    }
    for (auto key : collected_output_nodes) {
      if (std::find(input_names.begin(), input_names.end(), key) == input_names.end()) {
        extra_inputs.erase(key);
      }
    }

    std::vector<std::string> extended_input(extra_inputs.begin(), extra_inputs.end());
    for (auto key : main_graph_input_names) {
      extra_inputs.erase(key);
    }
    std::vector<std::string> extra_inputs_vec(extra_inputs.begin(), extra_inputs.end());
    if (!extra_inputs_vec.empty()) {
      extended_subgraph_input_output->push_back({{}, extra_inputs_vec});
      infer_path_root_nodes->push_back({{}, extra_inputs_vec});
    }
    extended_subgraph_input_output->push_back({extended_input, input_output_pair.second});
    infer_path_root_nodes->push_back({input_names, input_output_pair.second});
  }
  return lite::RET_OK;
}

STATUS GetInferPath(const std::map<std::string, size_t> &subgraph_output_map,
                    const std::vector<std::vector<std::string>> &subgraph_input_name,
                    const std::vector<std::vector<std::string>> &target_node, std::vector<int32_t> *infer_path) {
  if (target_node.size() != kTargetNodeSize) {
    MS_LOG(ERROR) << "Must set subgraph input and output!";
    return lite::RET_ERROR;
  }
  auto target_input = target_node[0];
  auto target_output = target_node[1];
  int32_t min_index = INT32_MAX;
  int32_t max_index = 0;
  for (auto node : target_output) {
    if (subgraph_output_map.find(node) == subgraph_output_map.end()) {
      MS_LOG(ERROR) << "node " << node << " not in output subgraph map!";
      return kLiteError;
    }
    max_index = std::max(static_cast<int32_t>(subgraph_output_map.at(node)), max_index);
  }
  for (size_t subgraph_id = 0; subgraph_id < subgraph_input_name.size(); subgraph_id++) {
    auto single_graph_inputs = subgraph_input_name[subgraph_id];
    for (auto name : single_graph_inputs) {
      if (std::find(target_input.begin(), target_input.end(), name) != target_input.end()) {
        min_index = std::min(min_index, static_cast<int32_t>(subgraph_id));
      }
    }
  }
  if (min_index == INT32_MAX && target_input.size() != 0) {
    MS_LOG(ERROR) << "Can not find the input you set";
    return kLiteError;
  } else if (min_index == INT32_MAX && target_input.size() == 0) {
    min_index = 0;
  }
  if (max_index < min_index) {
    MS_LOG(ERROR) << "max_index should larger or equal to min_index! min_index:" << min_index
                  << ", max_index:" << max_index;
    return lite::RET_ERROR;
  }
  infer_path->push_back(min_index);
  infer_path->push_back(max_index);
  return lite::RET_OK;
}

STATUS CollectSubgraphNodes(const FuncGraphPtr &original_graph, const std::vector<AnfNodePtr> &output_nodes,
                            const std::unordered_set<AnfNodePtr> &stop_nodes, const size_t &boundaries_id,
                            std::unordered_set<AnfNodePtr> *nodes_in_subgraph,
                            std::map<AnfNodePtr, size_t> *node_graph_id_map) {
  MS_CHECK_TRUE_MSG(nodes_in_subgraph != nullptr, lite::RET_ERROR, "nodes_in_subgraph is nullptr!");
  MS_CHECK_TRUE_MSG(node_graph_id_map != nullptr, lite::RET_ERROR, "node_graph_id_map is nullptr!");
  std::deque<AnfNodePtr> queue(output_nodes.begin(), output_nodes.end());
  std::unordered_set<AnfNodePtr> visited(output_nodes.begin(), output_nodes.end());

  while (!queue.empty()) {
    auto current_node = queue.front();
    queue.pop_front();
    if (stop_nodes.count(current_node)) {
      continue;
    }
    if (!utils::isa<CNodePtr>(current_node)) {
      continue;
    }
    if (nodes_in_subgraph->find(current_node) != nodes_in_subgraph->end()) {
      continue;
    }
    if (node_graph_id_map->find(current_node) == node_graph_id_map->end()) {
      ((*node_graph_id_map)[current_node]) = boundaries_id - 1;
    } else {
      continue;
    }
    nodes_in_subgraph->insert(current_node);
    auto abstract = current_node->abstract();
    size_t output_num = 1;
    if (abstract->isa<abstract::AbstractTuple>()) {
      auto abstract_tuple = abstract->cast<abstract::AbstractTuplePtr>();
      output_num = abstract_tuple->elements().size();
    }
    if (output_num > 1) {
      std::vector<AnfNodePtr> successors;
      GetSuccessor(original_graph, current_node, &successors);
      for (auto successor : successors) {
        nodes_in_subgraph->insert(successor);
        (*node_graph_id_map)[successor] = boundaries_id - 1;
      }
    }

    auto cnode = current_node->cast<CNodePtr>();
    MS_CHECK_TRUE_MSG(cnode != nullptr, lite::RET_ERROR, "cnode is nullptr!");
    for (size_t j = 1; j < cnode->size(); j++) {
      auto inp_node = cnode->input(j);
      if (visited.find(inp_node) == visited.end()) {
        queue.push_back(inp_node);
      }
      visited.insert(inp_node);
    }
  }
  return RET_OK;
}

STATUS ProcessNodeInputs(const std::map<AnfNodePtr, size_t> &node_graph_id_map, const CNodePtr &old_cnode,
                         const size_t &boundaries_id, const size_t &input_id,
                         const std::unordered_set<AnfNodePtr> &begin_nodes, const FuncGraphPtr &subgraph,
                         std::unordered_map<AnfNodePtr, AnfNodePtr> *node_map, std::vector<AnfNodePtr> *new_inputs,
                         std::vector<std::vector<std::string>> *subgraph_input_names,
                         std::vector<std::vector<AnfNodePtr>> *subgraph_output_vec,
                         std::vector<std::unordered_map<AnfNodePtr, AnfNodePtr>> *node_map_vec) {
  auto old_input = old_cnode->input(input_id);
  if (node_map->count(old_input)) {
    new_inputs->push_back((*node_map)[old_input]);
  } else if (begin_nodes.find(old_input) != begin_nodes.end()) {
    auto param = subgraph->add_parameter();
    MS_CHECK_TRUE_MSG(param != nullptr, lite::RET_ERROR, "param is nullptr!");
    auto abstract = old_input->abstract();
    MS_CHECK_TRUE_MSG(abstract != nullptr, lite::RET_ERROR, "abstract is nullptr!");
    auto abstract_clone = abstract->Clone();
    MS_CHECK_TRUE_MSG(abstract_clone != nullptr, lite::RET_ERROR, "abstract_clone is nullptr!");
    param->set_abstract(abstract_clone);
    param->set_name(old_input->fullname_with_scope());
    new_inputs->push_back(param);
    (*node_map)[old_input] = param;
    subgraph_input_names->at(boundaries_id - 1).push_back(old_input->fullname_with_scope());
  } else if (node_graph_id_map.find(old_input) != node_graph_id_map.end() &&
             node_graph_id_map.at(old_input) != boundaries_id - 1) {
    auto param = subgraph->add_parameter();
    MS_CHECK_TRUE_MSG(param != nullptr, lite::RET_ERROR, "param is nullptr!");
    auto abstract = old_input->abstract();
    MS_CHECK_TRUE_MSG(abstract != nullptr, lite::RET_ERROR, "abstract is nullptr!");
    auto abstract_clone = abstract->Clone();
    MS_CHECK_TRUE_MSG(abstract_clone != nullptr, lite::RET_ERROR, "abstract_clone is nullptr!");
    param->set_abstract(abstract_clone);
    param->set_name(old_input->fullname_with_scope());
    new_inputs->push_back(param);
    (*node_map)[old_input] = param;
    subgraph_input_names->at(boundaries_id - 1).push_back(old_input->fullname_with_scope());
    if (std::find(subgraph_output_vec->at(node_graph_id_map.at(old_input)).begin(),
                  subgraph_output_vec->at(node_graph_id_map.at(old_input)).end(),
                  node_map_vec->at(node_graph_id_map.at(old_input))[old_input]) ==
        subgraph_output_vec->at(node_graph_id_map.at(old_input)).end()) {
      subgraph_output_vec->at(node_graph_id_map.at(old_input))
        .push_back(node_map_vec->at(node_graph_id_map.at(old_input))[old_input]);
    }
  } else if (utils::isa<ValueNodePtr>(old_input)) {
    new_inputs->push_back(old_input);
  }
  return RET_OK;
}

STATUS BuildSubGraph(const FuncGraphPtr &original_graph, const std::unordered_set<AnfNodePtr> &nodes_in_subgraph,
                     const std::map<AnfNodePtr, size_t> &node_graph_id_map,
                     const std::unordered_set<AnfNodePtr> &begin_nodes, const size_t &boundaries_id,
                     const std::vector<AnfNodePtr> &output_nodes,
                     std::vector<std::vector<AnfNodePtr>> *subgraph_output_vec,
                     std::vector<std::unordered_map<AnfNodePtr, AnfNodePtr>> *node_map_vec,
                     std::vector<FuncGraphPtr> *subgraphs,
                     std::vector<std::vector<std::string>> *subgraph_input_names) {
  MS_CHECK_TRUE_MSG(original_graph != nullptr, lite::RET_ERROR, "original_graph is nullptr!");
  MS_CHECK_TRUE_MSG(subgraph_output_vec != nullptr, lite::RET_ERROR, "subgraph_output_vec is nullptr!");
  MS_CHECK_TRUE_MSG(node_map_vec != nullptr, lite::RET_ERROR, "node_map_vec is nullptr!");
  MS_CHECK_TRUE_MSG(subgraphs != nullptr, lite::RET_ERROR, "subgraphs is nullptr!");
  auto subgraph = std::make_shared<FuncGraph>();
  std::unordered_map<AnfNodePtr, AnfNodePtr> node_map;
  std::unordered_set<AnfNodePtr> all_required_inputs;
  std::unordered_set<AnfNodePtr> tensor_produced_internally;
  for (const auto &node : nodes_in_subgraph) {
    tensor_produced_internally.insert(node);
    auto cnode = node->cast<CNodePtr>();
    for (size_t j = 1; j < cnode->size(); j++) {
      all_required_inputs.insert(cnode->input(j));
    }
  }
  std::vector<AnfNodePtr> required_initializers;
  std::copy_if(all_required_inputs.begin(), all_required_inputs.end(), std::back_inserter(required_initializers),
               [](const auto &input) { return IsWeight(input); });
  for (const auto &old_input_node : required_initializers) {
    if (!utils::isa<ParameterPtr>(old_input_node)) {
      continue;
    }
    auto old_param = old_input_node->cast<ParameterPtr>();
    MS_CHECK_TRUE_MSG(old_param != nullptr, lite::RET_ERROR, "old_input_node cast to parameterptr failed!");
    auto param = subgraph->add_parameter();
    MS_CHECK_TRUE_MSG(param != nullptr, lite::RET_ERROR, "param is nullptr!");
    auto default_param = old_param->default_param();
    auto abstract = old_param->abstract()->Clone();
    MS_CHECK_TRUE_MSG(abstract != nullptr, lite::RET_ERROR, "abstract is nullptr!");
    auto abstract_clone = abstract->Clone();
    MS_CHECK_TRUE_MSG(abstract_clone != nullptr, lite::RET_ERROR, "abstract is nullptr!");
    auto name = old_param->fullname_with_scope();
    param->set_default_param(default_param);
    param->set_abstract(abstract_clone);
    param->set_name(name);
    node_map[old_param] = param;
  }
  auto ordered_cnodes = TopoSort(original_graph->get_return());
  for (const auto &node : ordered_cnodes) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto old_cnode = node->cast<CNodePtr>();
    MS_CHECK_TRUE_MSG(old_cnode != nullptr, lite::RET_ERROR, "old cnode cast to cnode failed!");
    if (nodes_in_subgraph.find(old_cnode) != nodes_in_subgraph.end()) {
      std::vector<AnfNodePtr> new_inputs;
      new_inputs.push_back(old_cnode->input(0));
      for (size_t j = 1; j < old_cnode->size(); j++) {
        if (ProcessNodeInputs(node_graph_id_map, old_cnode, boundaries_id, j, begin_nodes, subgraph, &node_map,
                              &new_inputs, subgraph_input_names, subgraph_output_vec, node_map_vec) != lite::RET_OK) {
          MS_LOG(ERROR) << "ProcessNodeInputs failed!";
          return lite::RET_ERROR;
        }
      }
      auto new_cnode = subgraph->NewCNode(new_inputs);
      new_cnode->set_fullname_with_scope(old_cnode->fullname_with_scope());
      auto abstract = old_cnode->abstract();
      MS_CHECK_TRUE_MSG(abstract != nullptr, lite::RET_ERROR, "abstract is nullptr!");
      auto abstract_clone = abstract->Clone();
      MS_CHECK_TRUE_MSG(abstract_clone != nullptr, lite::RET_ERROR, "abstract_clone is nullptr!");
      new_cnode->set_abstract(abstract_clone);
      node_map[old_cnode] = new_cnode;
    }
  }
  std::vector<AnfNodePtr> final_outputs;
  for (const auto &old_output_node : output_nodes) {
    if (node_map.count(old_output_node)) {
      final_outputs.push_back(node_map[old_output_node]);
    }
  }
  node_map_vec->push_back(node_map);
  subgraph_output_vec->push_back(final_outputs);
  static auto sub_manager = Manage(subgraph);
  subgraph->set_manager(sub_manager);
  subgraph->set_attr("graph_name", MakeValue("subgraph_" + std::to_string(boundaries_id - 1)));
  subgraphs->push_back(subgraph);
  return RET_OK;
}

STATUS RecordModelInfos(
  const FuncGraphPtr &original_graph, const std::vector<std::vector<std::string>> &subgraph_output_names,
  const std::vector<std::vector<std::string>> &subgraph_input_names,
  const std::vector<std::string> &main_graph_input_names, const std::vector<std::string> &main_graph_output_names,
  const std::vector<std::pair<std::vector<std::string>, std::vector<std::string>>> &subgraph_input_output) {
  std::vector<std::vector<int>> subgraph_input_to_main_graph_map;
  std::vector<std::vector<int>> subgraph_output_to_main_graph_map;
  std::vector<std::vector<std::vector<int>>> subgraph_output_to_subgraph_input_map;
  if (BuildInputOutputMap(subgraph_output_names, subgraph_input_names, main_graph_input_names, main_graph_output_names,
                          &subgraph_input_to_main_graph_map, &subgraph_output_to_main_graph_map,
                          &subgraph_output_to_subgraph_input_map) != lite::RET_OK) {
    MS_LOG(ERROR) << "build input output map failed!";
    return RET_ERROR;
  }
  std::map<std::string, size_t> subgraph_output_map;
  for (size_t subgraph_id = 0; subgraph_id < subgraph_output_names.size(); subgraph_id++) {
    auto current_subgraph_output_names = subgraph_output_names[subgraph_id];
    for (auto name : current_subgraph_output_names) {
      subgraph_output_map[name] = subgraph_id;
    }
  }
  std::vector<std::vector<int32_t>> subgraph_infer_paths;
  std::vector<std::vector<std::vector<std::string>>> extended_subgraph_input_output;
  std::vector<std::vector<std::vector<std::string>>> infer_path_root_nodes;
  if (ExtendSubgraphInputOutput(subgraph_input_names, subgraph_output_names, subgraph_input_output,
                                main_graph_input_names, &extended_subgraph_input_output,
                                &infer_path_root_nodes) != lite::RET_OK) {
    MS_LOG(ERROR) << "ExtendSubgraphInputOutput failed!";
    return RET_ERROR;
  }
  for (auto input_output_pair : infer_path_root_nodes) {
    std::vector<int32_t> infer_path;
    if (GetInferPath(subgraph_output_map, subgraph_input_names, input_output_pair, &infer_path) != lite::RET_OK) {
      MS_LOG(ERROR) << "GetInferPath failed!";
      return RET_ERROR;
    }
    subgraph_infer_paths.push_back(infer_path);
  }
  original_graph->set_attr("input_map", MakeValue<std::vector<std::vector<int>>>(subgraph_input_to_main_graph_map));
  original_graph->set_attr("output_map", MakeValue<std::vector<std::vector<int>>>(subgraph_output_to_main_graph_map));
  original_graph->set_attr("output_to_input_map", MakeValue<std::vector<std::vector<std::vector<int>>>>(
                                                    subgraph_output_to_subgraph_input_map));
  original_graph->set_attr("subgraph_output_names",
                           MakeValue<std::vector<std::vector<std::string>>>(subgraph_output_names));
  original_graph->set_attr("subgraph_input_names",
                           MakeValue<std::vector<std::vector<std::string>>>(subgraph_input_names));
  original_graph->set_attr("graph_input_names", MakeValue<std::vector<std::string>>(main_graph_input_names));
  original_graph->set_attr("subgraph_infer_path", MakeValue<std::vector<std::vector<int>>>(subgraph_infer_paths));
  original_graph->set_attr(
    "extended_subgraph_input_output",
    MakeValue<std::vector<std::vector<std::vector<std::string>>>>(extended_subgraph_input_output));
  return RET_OK;
}

STATUS BuildBoundaries(const FuncGraphPtr &original_graph, const std::vector<std::string> &split_op_names,
                       std::vector<std::string> *main_graph_output_names,
                       std::vector<std::vector<AnfNodePtr>> *boundaries,
                       std::vector<std::string> *main_graph_input_names) {
  auto model_inputs = original_graph->get_inputs();
  std::set<std::string> graph_input_name;
  for (auto input : model_inputs) {
    graph_input_name.insert(input->fullname_with_scope());
  }
  auto return_node = original_graph->get_return();
  std::vector<AnfNodePtr> graph_outputs;
  for (size_t i = 1; i < return_node->inputs().size(); i++) {
    graph_outputs.push_back(return_node->inputs()[i]);
    main_graph_output_names->push_back(return_node->inputs()[i]->fullname_with_scope());
  }
  std::vector<std::string> split_op_names_without_input;
  for (auto s : split_op_names) {
    if (graph_input_name.find(s) == graph_input_name.end() &&
        std::find(main_graph_output_names->begin(), main_graph_output_names->end(), s) ==
          main_graph_output_names->end()) {
      split_op_names_without_input.push_back(s);
    }
  }
  std::vector<std::string> sorted_split_op_names = {};
  std::unordered_map<std::string, CNodePtr> node_by_name;
  auto all_nodes = TopoSort(original_graph->get_return());
  for (const auto &node : all_nodes) {
    if (utils::isa<CNodePtr>(node)) {
      auto cnode = node->cast<CNodePtr>();
      MS_CHECK_TRUE_MSG(cnode != nullptr, lite::RET_ERROR, "cnode is nullptr");
      if (!cnode->fullname_with_scope().empty()) {
        node_by_name[cnode->fullname_with_scope()] = cnode;
        if (std::find(split_op_names_without_input.begin(), split_op_names_without_input.end(),
                      cnode->fullname_with_scope()) != split_op_names_without_input.end()) {
          sorted_split_op_names.push_back(cnode->fullname_with_scope());
        }
      }
    }
  }
  for (auto name : split_op_names_without_input) {
    if (node_by_name.find(name) == node_by_name.end()) {
      MS_LOG(ERROR) << "The node you set not in graph, node name:" << name;
      return lite::RET_ERROR;
    }
  }
  if (sorted_split_op_names.size() != split_op_names_without_input.size()) {
    MS_LOG(ERROR) << "Size of sorted_split_op_names should equal to split_op_names_without_input!";
    return lite::RET_ERROR;
  }
  boundaries->push_back(model_inputs);
  std::transform(model_inputs.begin(), model_inputs.end(), std::back_inserter(*main_graph_input_names),
                 [](auto &input) { return input->fullname_with_scope(); });
  for (const auto &op_name : sorted_split_op_names) {
    if (node_by_name.find(op_name) == node_by_name.end()) {
      MS_LOG(ERROR) << "Node:" << op_name << " not found in the graph!";
      return lite::RET_ERROR;
    }
    auto split_node = node_by_name[op_name];
    auto abstract = split_node->abstract();
    MS_CHECK_TRUE_MSG(abstract != nullptr, lite::RET_ERROR, "abstract is nullptr!");
    size_t output_num = 1;
    if (abstract->isa<abstract::AbstractTuple>()) {
      auto abstract_tuple = abstract->cast<abstract::AbstractTuplePtr>();
      MS_CHECK_TRUE_MSG(abstract_tuple != nullptr, lite::RET_ERROR, "abstract_tuple is nullptr!");
      output_num = abstract_tuple->elements().size();
    }
    std::vector<AnfNodePtr> split_node_vec;
    if (output_num > 1) {
      if (GetSuccessor(original_graph, split_node, &split_node_vec) != lite::RET_OK) {
        MS_LOG(ERROR) << "GetSuccessor failed!";
        return lite::RET_ERROR;
      }
    } else {
      split_node_vec.push_back(split_node);
    }
    boundaries->push_back(split_node_vec);
  }
  boundaries->push_back(graph_outputs);
  return lite::RET_OK;
}

bool GraphSplitPass::Run(const FuncGraphPtr &original_graph) {
  MS_CHECK_TRUE_MSG(original_graph != nullptr, false, "original_graph is nullptr!");
  MS_CHECK_TRUE_MSG(param_ != nullptr, false, "param_ is nullptr!");
  if (param_->splitGraphCfg.split_node_names.empty()) {
    return true;
  }
  auto split_op_names = param_->splitGraphCfg.split_node_names;
  auto subgraph_input_output = param_->splitGraphCfg.subgraph_input_output;
  std::vector<std::vector<std::string>> subgraph_output_names;
  std::vector<std::vector<std::string>> subgraph_input_names;
  std::vector<std::string> main_graph_input_names;
  std::vector<std::string> main_graph_output_names;
  auto model_inputs = original_graph->get_inputs();
  std::map<AnfNodePtr, size_t> node_graph_id_map;
  std::vector<std::vector<AnfNodePtr>> subgraph_output_vec;
  std::vector<std::unordered_map<AnfNodePtr, AnfNodePtr>> node_map_vec;
  std::vector<std::vector<AnfNodePtr>> boundaries;
  if (BuildBoundaries(original_graph, split_op_names, &main_graph_output_names, &boundaries, &main_graph_input_names) !=
      RET_OK) {
    MS_LOG(ERROR) << "BuildBoundaries failed!";
    return false;
  }
  std::vector<FuncGraphPtr> subgraphs;
  for (size_t i = 1; i < boundaries.size(); i++) {
    const auto &output_nodes = boundaries[i];
    std::vector<AnfNodePtr> stop_nodes_vec;
    if (GetSubgraphStopNodes(boundaries, i, output_nodes, &stop_nodes_vec) != lite::RET_OK) {
      MS_LOG(ERROR) << "Get subgraph stop nodes failed!";
      return false;
    }
    subgraph_input_names.push_back({});
    subgraph_output_names.push_back({});
    std::unordered_set<AnfNodePtr> stop_nodes(stop_nodes_vec.begin(), stop_nodes_vec.end());
    std::unordered_set<AnfNodePtr> begin_nodes(model_inputs.begin(), model_inputs.end());
    begin_nodes.insert(stop_nodes_vec.begin(), stop_nodes_vec.end());

    // 反向追溯收集节点
    std::unordered_set<AnfNodePtr> nodes_in_subgraph;

    if (CollectSubgraphNodes(original_graph, output_nodes, stop_nodes, i, &nodes_in_subgraph, &node_graph_id_map) !=
        RET_OK) {
      MS_LOG(ERROR) << "CollectSubgraphNodes failed!";
      return false;
    }

    if (BuildSubGraph(original_graph, nodes_in_subgraph, node_graph_id_map, begin_nodes, i, output_nodes,
                      &subgraph_output_vec, &node_map_vec, &subgraphs, &subgraph_input_names) != RET_OK) {
      MS_LOG(ERROR) << "BuildSubGraph failed!";
      return false;
    }
  }
  for (size_t i = 0; i < subgraphs.size(); i++) {
    for (auto subgraph_output : subgraph_output_vec[i]) {
      subgraph_output_names[i].push_back(subgraph_output->fullname_with_scope());
    }
    if (BuildReturnNode(subgraphs[i], subgraph_output_vec[i]) != lite::RET_OK) {
      MS_LOG(ERROR) << "build return node failed!";
      return false;
    }
  }
  if (RecordModelInfos(original_graph, subgraph_output_names, subgraph_input_names, main_graph_input_names,
                       main_graph_output_names, subgraph_input_output) != RET_OK) {
    MS_LOG(ERROR) << "RecordModelInfos failed!";
    return false;
  }
  original_graph->set_attr("subgraphs", MakeValue<std::vector<FuncGraphPtr>>(subgraphs));
  return true;
}
}  // namespace mindspore::opt
