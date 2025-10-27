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

#include "tools/optimizer/graph/add_variable_node_pass.h"
#include <memory>
#include <vector>
#include <set>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <regex>
#include <string>
#include "src/common/log_adapter.h"
#include "src/common/log_util.h"
#include "src/common/common.h"
#include "tools/common/tensor_util.h"
#include "tools/common/func_graph_utils.h"
#include "tools/optimizer/fusion/matmul_allreduce_fusion.h"
#include "op_def/auto_generate/gen_lite_ops.h"
#include "tools/common/parse_config_utils.h"
#include "op_def/conv_pool_ops.h"
#include "include/errorcode.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_b.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_c.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"
#include "mindspore/core/include/ir/graph_utils.h"
#include "mindspore/ops/infer/custom.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kInputSize3 = 3;
constexpr size_t kInputIndex2 = 2;
constexpr size_t kConstantMatmulWeightShapeSize = 2;
constexpr size_t kConstantConvWeightShapeSize = 4;
constexpr size_t kWeightInitLen = 1;
constexpr size_t kIndex0 = 0;
constexpr size_t kIndex1 = 1;
constexpr size_t kIndex2 = 2;
constexpr size_t kIndex3 = 3;
constexpr size_t kIndex4 = 4;
constexpr float kInitZero = 0.0;
constexpr float kInitOne = 1.0;
constexpr size_t kInitBatchSize = 1;
constexpr size_t kMaxConfigLen = 1e6;
constexpr uint16_t kFloatOne = 15360;

bool MatchPattern(const std::string &input) {
  std::regex pattern(R"(^([^:;]+):(\d+(?:,\d+)*);([^:;]+)$)");
  return std::regex_match(input, pattern);
}
}  // namespace

lite::STATUS InsertVariableNodePass::ParseInsertNode(std::string file_path, std::set<std::string> *variable_nodes,
                                                     std::vector<std::string> *node_name_list) {
  MS_CHECK_TRUE_RET(variable_nodes != nullptr, lite::RET_NULL_PTR);
  MS_CHECK_TRUE_RET(node_name_list != nullptr, lite::RET_NULL_PTR);
  std::ifstream file;
  auto ret = lite::ReadFileToIfstream(file_path, &file);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "read file to ifstream failed!";
    return ret;
  }
  size_t config_len = 0;
  std::string line;
  while (std::getline(file, line)) {
    if (!MatchPattern(line)) {
      MS_LOG(ERROR) << "Format of config error, it should be 'weight_name:num1,num2,num3;node_name', input config:"
                    << line;
      return RET_ERROR;
    }
    config_len++;
    if (config_len >= kMaxConfigLen) {
      MS_LOG(ERROR) << "Support max config len is " << kMaxConfigLen << ", current len:" << config_len << "!";
      return RET_ERROR;
    }
    auto pos_colon = line.find(':');
    if (pos_colon == std::string::npos) {
      MS_LOG(ERROR) << "Parse variable weight file error!";
      file.close();
      return RET_ERROR;
    }
    auto variable_para_name = line.substr(0, pos_colon);
    (*node_name_list).push_back(variable_para_name);
    variable_nodes->insert(variable_para_name);
  }
  file.close();
  return RET_OK;
}

void InsertVariableNodePass::InitWeightParam(std::string *variable_weights_file, int32_t *max_weight_batch) {
  if (param_->config_infos.find(lite::kAscendContextSection) != param_->config_infos.end()) {
    auto ascend_context = param_->config_infos.at(lite::kAscendContextSection);
    if (ascend_context.find(lite::kVariableWeightsFile) != ascend_context.end()) {
      *variable_weights_file = ascend_context.at(lite::kVariableWeightsFile);
    }
    if (ascend_context.find(lite::kMaxWeightBatch) != ascend_context.end()) {
      *max_weight_batch = std::stoi(ascend_context.at(lite::kMaxWeightBatch));
    }
  }
}

lite::STATUS InsertVariableNodePass::RecordParameterVariableName(
  const FuncGraphPtr &func_graph, const ParameterPtr &para_node, const string &search_key,
  std::unordered_map<std::string, std::string> *node_name_map,
  std::unordered_map<std::string, AbstractBasePtr> *node_abstract_map) {
  MS_CHECK_TRUE_RET(node_name_map != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(para_node != nullptr, RET_ERROR);
  (*node_name_map)[search_key] = para_node->fullname_with_scope() + "_const";
  (*node_abstract_map)[search_key] = para_node->abstract();
  return RET_OK;
}

FuncGraphPtr InsertVariableNodePass::CreateUpdateGraph(const std::vector<std::string> &const_names,
                                                       const std::vector<AbstractBasePtr> &abstarcts) {
  MS_CHECK_TRUE_MSG(const_names.size() == abstarcts.size(), nullptr,
                    "size of const_names must equal to size of abstracts's size!");
  auto func_graph = std::make_shared<FuncGraph>();
  std::vector<AnfNodePtr> graph_outputs = {};
  for (size_t i = 0; i < const_names.size(); i++) {
    auto param = func_graph->add_parameter();
    MS_CHECK_TRUE_MSG(param != nullptr, nullptr, "param is nullptr!");
    auto name = const_names[i];
    auto abstract = abstarcts[i];
    MS_CHECK_TRUE_MSG(abstract != nullptr, nullptr, "abstract is nullptr!");
    param->set_abstract(abstract->Clone());  // node name and abstract of weight
    param->set_name(name + "_data");
    auto variable_prim = std::make_unique<ops::Custom>();
    MS_CHECK_TRUE_MSG(variable_prim != nullptr, nullptr, "variable_prim is nullptr!");
    variable_prim->set_type("Variable");
    std::vector<std::string> variable_input_names = {"x"};
    std::vector<std::string> variable_output_names = {"y"};
    variable_prim->AddAttr("input_names", api::MakeValue(variable_input_names));
    variable_prim->AddAttr("output_names", api::MakeValue(variable_output_names));
    variable_prim->AddAttr(kAttrRegOpName, api::MakeValue("Variable"));
    auto variable_prim_c = variable_prim->GetPrim();
    MS_CHECK_TRUE_MSG(variable_prim_c != nullptr, nullptr, "variable_prim_c is nullptr!");
    auto variable_cnode = func_graph->NewCNode(variable_prim_c, {});
    MS_CHECK_TRUE_MSG(variable_cnode != nullptr, nullptr, "variable_cnode is nullptr");
    variable_cnode->set_fullname_with_scope(name + "_var");
    variable_cnode->set_abstract(abstract->Clone());
    auto assign_prim = std::make_unique<ops::Custom>();
    MS_CHECK_TRUE_MSG(assign_prim != nullptr, nullptr, "assign_prim is nullptr!");
    assign_prim->set_type("Assign");
    std::vector<std::string> assign_input_names = {"input0", "input1"};
    std::vector<std::string> assign_output_names = {"output0"};
    assign_prim->AddAttr("input_names", api::MakeValue(assign_input_names));
    assign_prim->AddAttr("output_names", api::MakeValue(assign_output_names));
    assign_prim->AddAttr(kAttrRegOpName, api::MakeValue("Assign"));
    auto assign_prim_c = assign_prim->GetPrim();
    MS_CHECK_TRUE_MSG(assign_prim_c != nullptr, nullptr, "assign_prim_c is nullptr!");
    std::vector<AnfNodePtr> assign_inputs = {variable_cnode, param};
    auto assign_cnode = func_graph->NewCNode(assign_prim_c, assign_inputs);
    MS_CHECK_TRUE_MSG(assign_cnode != nullptr, nullptr, "assign_cnode is nullptr!");
    assign_cnode->set_fullname_with_scope(name + "_assign");
    assign_cnode->set_abstract(abstract->Clone());
    graph_outputs.push_back(assign_cnode);
  }
  // update graph should not has output
  // single output graph insert identity node as graph output, then delete it in IdentityOptimization function to make
  // sure update graph has no output node.
  if (graph_outputs.size() == 1) {
    auto prim = std::make_unique<ops::Identity>();
    MS_CHECK_TRUE_MSG(prim != nullptr, nullptr, "prim is nullptr!");
    auto prim_c = prim->GetPrim();
    MS_CHECK_TRUE_MSG(prim_c != nullptr, nullptr, "prim_c is nullptr!");
    auto identity_cnode = func_graph->NewCNode(prim_c, graph_outputs);
    MS_CHECK_TRUE_MSG(identity_cnode != nullptr, nullptr, "identity_cnode is nullptr");
    identity_cnode->set_abstract(graph_outputs[0]->abstract()->Clone());
    identity_cnode->set_fullname_with_scope(graph_outputs[0]->fullname_with_scope() + "_identity");
    graph_outputs[0] = identity_cnode;
  }
  auto ret = BuildReturnNode(func_graph, graph_outputs);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "BuildReturnNode failed!";
    return nullptr;
  }
  func_graph->set_attr("is_update_graph", MakeValue<bool>(true));
  return func_graph;
}

lite::STATUS InsertVariableNodePass::BuildVariableNode(FuncGraphPtr func_graph) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, RET_ERROR);
  std::string variable_weights_file = "";
  int32_t max_weight_batch = 1;
  InitWeightParam(&variable_weights_file, &max_weight_batch);
  MS_CHECK_TRUE_RET(variable_weights_file != "", RET_OK);
  std::set<std::string> variable_nodes;
  std::unordered_map<std::string, std::string> node_name_map;
  std::unordered_map<std::string, AbstractBasePtr> node_abstract_map;
  std::vector<std::string> node_name_list;
  auto ret = ParseInsertNode(variable_weights_file, &variable_nodes, &node_name_list);
  MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "ParseInsertNode failed!");
  uint32_t matched_num = 0;
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    MS_CHECK_TRUE_RET(node != nullptr, false);
    auto node_name = node->fullname_with_scope();
    if (utils::isa<ParameterPtr>(node)) {
      if (variable_nodes.find(node_name) == variable_nodes.end()) {
        continue;
      }
      auto parameter = node->cast<ParameterPtr>();
      if (parameter == nullptr || !parameter->has_default()) {
        continue;
      }
      ret = RecordParameterVariableName(func_graph, parameter, node_name, &node_name_map, &node_abstract_map);
      MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "Record parameter variable name failed!");
    } else {
      continue;
    }
    matched_num++;
  }
  if (matched_num != variable_nodes.size()) {
    MS_LOG(ERROR) << "matched num:" << matched_num << " != all node num:" << variable_nodes.size() << "!";
    return RET_ERROR;
  }
  for (auto s : node_name_list) {
    if (node_name_map.find(s) == node_name_map.end()) {
      continue;
    }
    param_->variable_node_names.push_back(node_name_map[s]);
    param_->variable_node_abstracts.push_back(node_abstract_map[s]);
  }
  auto update_graph = CreateUpdateGraph(param_->variable_node_names, param_->variable_node_abstracts);
  if (update_graph == nullptr) {
    MS_LOG(ERROR) << "update graph is nullptr!";
    return RET_ERROR;
  }
  param_->update_graph = update_graph;
  return RET_OK;
}

bool InsertVariableNodePass::Run(const FuncGraphPtr &graph) {
  if (BuildVariableNode(graph) != RET_OK) {
    MS_LOG(ERROR) << "build variable node failed!";
    return false;
  }
  if (param_->variable_node_names.size() > 0) {
    graph->set_attr(lite::kBundleModel, MakeValue("True"));
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
