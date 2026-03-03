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

#include "tools/converter/adapter/acl/src/aclnn/aclnn_subgraph_processor.h"
#include "tools/converter/adapter/acl/src/aclnn/aclnn_pass_impl.h"
#include "tools/converter/adapter/acl/src/aclnn/aclnn_utils.h"
#include "tools/common/string_util.h"
#include "tools/converter/optimizer_manager.h"
#include "tools/converter/ms_depend/utils.h"
#include "common/common.h"
#include "common/utils.h"
#include "mindspore/core/include/ir/graph_utils.h"
#include "mindspore/ccsrc/include/utils/anfalgo.h"
#include "cxx_api/graph/acl/acl_convert_init_adapter.h"
#include "utils/utils.h"

namespace mindspore {
namespace opt {

AclnnSubgraphProcessor::AclnnSubgraphProcessor(const std::shared_ptr<ConverterPara> &param,
                                               const lite::acl::AclModelOptionCfg &user_options_cfg)
    : param_(param), user_options_cfg_(user_options_cfg) {}

STATUS AclnnSubgraphProcessor::ValidateSubgraphInputs(const CNodePtr &cnode, const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_MSG(cnode != nullptr, lite::RET_NULL_PTR, "cnode is nullptr.");
  MS_CHECK_TRUE_MSG(func_graph != nullptr, lite::RET_NULL_PTR, "func_graph is nullptr.");
  if (cnode->inputs().size() != func_graph->get_inputs().size() + 1) {
    MS_LOG(ERROR) << "subgraph inputs size != cnode inputs. node: " << cnode->fullname_with_scope();
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

STATUS AclnnSubgraphProcessor::CollectSubgraphInputShapes(const CNodePtr &cnode,
                                                          const std::vector<std::vector<int64_t>> &global_dim_groups,
                                                          OrderShapes *shapes,
                                                          std::vector<std::vector<int64_t>> *dim_groups) {
  MS_CHECK_TRUE_MSG(cnode != nullptr, lite::RET_NULL_PTR, "cnode is nullptr.");
  MS_CHECK_TRUE_MSG(shapes != nullptr, lite::RET_NULL_PTR, "shapes is nullptr.");
  MS_CHECK_TRUE_MSG(dim_groups != nullptr, lite::RET_NULL_PTR, "dim_groups is nullptr.");

  *dim_groups = std::vector<std::vector<int64_t>>(global_dim_groups.size(), std::vector<int64_t>());

  for (size_t i = 1; i < cnode->inputs().size(); i++) {
    auto node = cnode->input(i);
    MS_CHECK_TRUE_MSG(node != nullptr, lite::RET_NULL_PTR, "node is nullptr.");
    std::vector<int64_t> shape;
    std::vector<std::vector<int64_t>> groups;

    if (node->isa<Parameter>()) {
      auto parameter = node->cast<ParameterPtr>();
      MS_CHECK_TRUE_MSG(parameter != nullptr, lite::RET_NULL_PTR, "parameter is nullptr.");
      if (!parameter->has_default()) {
        auto ret = GetGlobalInputDynShape(parameter->name(), &shape, &groups);
        MS_CHECK_TRUE_MSG(ret == lite::RET_OK, ret, "get global input shape failed.");
      } else {
        MS_LOG(ERROR) << "parameter with default should inline. name: " << parameter->name();
        return lite::RET_ERROR;
      }
    } else if (node->isa<CNode>()) {
      auto input_cnode = node->cast<CNodePtr>();
      MS_CHECK_TRUE_MSG(input_cnode != nullptr, lite::RET_NULL_PTR, "input_cnode is nullptr.");
      auto ret = GetCNodeInputDynShape(input_cnode, &shape, &groups);
      MS_CHECK_TRUE_MSG(ret == lite::RET_OK, ret, "get cnode input shape failed.");
    }
    MS_CHECK_TRUE_MSG(groups.size() == global_dim_groups.size(), lite::RET_ERROR,
                      "input groups num != global groups num");
    shapes->emplace_back(std::to_string(i - 1),
                         shape);  // input name in subgraph is sequence start at 0.
    for (size_t j = 0; j < groups.size(); j++) {
      (*dim_groups)[j].insert((*dim_groups)[j].end(), groups[j].begin(), groups[j].end());
    }
  }
  return lite::RET_OK;
}

STATUS AclnnSubgraphProcessor::PreparePass(const FuncGraphPtr &func_graph, const OrderShapes &shapes,
                                           const std::vector<std::vector<int64_t>> &dim_groups,
                                           std::unique_ptr<AclPassImpl> &acl_pass) {
  MS_CHECK_TRUE_MSG(func_graph != nullptr, lite::RET_NULL_PTR, "func_graph is nullptr.");

  std::unordered_set<std::string> dim_groups_str;
  std::transform(dim_groups.begin(), dim_groups.end(), std::inserter(dim_groups_str, dim_groups_str.begin()),
                 [](auto &group) { return lite::Join(",", group); });

  std::vector<std::string> shapes_str;
  std::map<std::string, std::vector<int64_t>> shape_map;
  auto param = std::make_shared<ConverterPara>(*param_);
  param->aclModelOptionCfgParam = user_options_cfg_;
  if (dim_groups_str.size() == 1) {
    OrderShapes static_shape;
    auto ret = AclnnUtils::ApplyDims(shapes, dim_groups.front(), &static_shape);
    MS_CHECK_TRUE_MSG(ret == lite::RET_OK, ret, "apply dims failed.");
    std::transform(static_shape.begin(), static_shape.end(), std::back_inserter(shapes_str),
                   [](auto &s) { return lite::Join(":", s.first, lite::Join(",", s.second)); });
    shape_map = {static_shape.begin(), static_shape.end()};
    param->aclModelOptionCfgParam.build_options_map.erase(lite::kDynamicDimsSearchKey);
  } else {
    std::transform(shapes.begin(), shapes.end(), std::back_inserter(shapes_str),
                   [](auto &s) { return lite::Join(":", s.first, lite::Join(",", s.second)); });
    shape_map = {shapes.begin(), shapes.end()};
    param->aclModelOptionCfgParam.build_options_map[lite::kDynamicDimsSearchKey] = lite::Join(";", dim_groups_str);
  }
  param->aclModelOptionCfgParam.build_options_map[lite::kInputShapeKey] = lite::Join(";", shapes_str);

  auto ret = AclnnUtils::ApplyInputShape(func_graph, shape_map);
  MS_CHECK_TRUE_MSG(ret == lite::RET_OK, ret, "Apply input shape on graph failed.");
  acl_pass = std::make_unique<AclnnPassImpl>(param, true);
  return lite::RET_OK;
}

STATUS AclnnSubgraphProcessor::RunAclPassForSubgraph(const CNodePtr &cnode, const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_MSG(cnode != nullptr, lite::RET_NULL_PTR, "cnode is nullptr.");
  MS_CHECK_TRUE_MSG(func_graph != nullptr, lite::RET_NULL_PTR, "func_graph is nullptr.");
  auto ret = ValidateSubgraphInputs(cnode, func_graph);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "ValidateSubgraphInputs failed.";
    return ret;
  }

  std::vector<std::vector<int64_t>> global_dim_groups;
  ret =
    AclnnUtils::ParseDynamicDims(user_options_cfg_.build_options_map[lite::kDynamicDimsSearchKey], &global_dim_groups);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "ParseDynamicDims failed.";
    return ret;
  }

  OrderShapes shapes;
  std::vector<std::vector<int64_t>> dim_groups;
  ret = CollectSubgraphInputShapes(cnode, global_dim_groups, &shapes, &dim_groups);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "CollectSubgraphInputShapes failed.";
    return ret;
  }
  std::unique_ptr<AclPassImpl> acl_pass;
  ret = PreparePass(func_graph, shapes, dim_groups, acl_pass);
  if (ret != lite::RET_OK || acl_pass == nullptr) {
    MS_LOG(ERROR) << "PrepareBuildOptionsAndExecute failed.";
    return ret;
  }
  if (!acl_pass->AclPassImpl::Run(func_graph)) {
    MS_LOG(ERROR) << "Run acl pass failed.";
    return lite::RET_ERROR;
  }

  return lite::RET_OK;
}

STATUS AclnnSubgraphProcessor::GetGlobalInputDynShape(const std::string &input_name, std::vector<int64_t> *shapes,
                                                      std::vector<std::vector<int64_t>> *dim_groups) {
  MS_CHECK_TRUE_MSG(shapes != nullptr, lite::RET_NULL_PTR, "shape is nullptr");
  MS_CHECK_TRUE_MSG(dim_groups != nullptr, lite::RET_NULL_PTR, "dim_groups is nullptr");
  auto &build_options = user_options_cfg_.build_options_map;

  OrderShapes global_shapes;
  auto ret = AclnnUtils::ParseInputShape(build_options[lite::kInputShapeKey], &global_shapes);
  MS_CHECK_TRUE_MSG(ret == lite::RET_OK, ret, "Parse input_shape failed.");

  std::vector<std::vector<int64_t>> global_dim_groups;
  ret = AclnnUtils::ParseDynamicDims(build_options[lite::kDynamicDimsSearchKey], &global_dim_groups);
  MS_CHECK_TRUE_MSG(ret == lite::RET_OK, ret, "Parse ge.dynamicDims failed.");

  auto it = std::find_if(global_shapes.begin(), global_shapes.end(),
                         [&input_name](auto item) { return item.first == input_name; });

  if (it == global_shapes.end()) {
    MS_LOG(ERROR) << "cannot find input_shape for [" << input_name << "] in config.";
    return lite::RET_ERROR;
  }

  auto dim_index = AclnnUtils::GetDynDimPosInGlobalGroup(global_shapes, std::distance(global_shapes.begin(), it));
  *shapes = it->second;
  dim_groups->clear();

  for (auto &group : global_dim_groups) {
    std::vector<int64_t> dims;
    std::transform(dim_index.begin(), dim_index.end(), std::back_inserter(dims),
                   [&](const auto &index) { return group[index]; });
    dim_groups->push_back(dims);
  }
  return lite::RET_OK;
}

STATUS AclnnSubgraphProcessor::GetCNodeInputDynShape(const CNodePtr &cnode, std::vector<int64_t> *shape,
                                                     std::vector<std::vector<int64_t>> *dim_groups) {
  MS_CHECK_TRUE_MSG(cnode != nullptr, lite::RET_NULL_PTR, "cnode is nullptr.");
  MS_CHECK_TRUE_MSG(shape != nullptr, lite::RET_NULL_PTR, "shape is nullptr.");
  MS_CHECK_TRUE_MSG(dim_groups != nullptr, lite::RET_NULL_PTR, "dim_groups is nullptr");
  std::vector<ValuePtr> shapes_value;
  if (common::AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimTupleGetItem)) {
    MS_CHECK_TRUE_MSG(cnode->size() == lite::kTupleGetItemInputSize, lite::RET_ERROR,
                      "TupleGetItem's input size is incorrect.");
    auto get_item_input_node = cnode->input(1);
    MS_CHECK_TRUE_MSG(get_item_input_node != nullptr, lite::RET_NULL_PTR, "get_item_input_node is nullptr");
    auto get_item_input_cnode = get_item_input_node->cast<CNodePtr>();
    MS_CHECK_TRUE_MSG(get_item_input_cnode != nullptr, lite::RET_NULL_PTR, "get_item_input_cnode is nullptr");
    auto idx = common::AnfAlgo::GetTupleGetItemOutIndex(cnode);
    if (!utils::isa<abstract::AbstractTuplePtr>(get_item_input_cnode->abstract())) {
      MS_LOG(ERROR) << "TupleGetItem's abstract is not AbstractTuple";
      return lite::RET_ERROR;
    }
    auto shapes_attr = get_item_input_cnode->GetAttr(lite::kNameShapeAttr);
    MS_CHECK_TRUE_MSG(shapes_attr != nullptr, lite::RET_NULL_PTR, "shapes_attr is nullptr");
    if (!shapes_attr->isa<ValueSequence>()) {
      MS_LOG(ERROR) << "attr: " << lite::kNameShapeAttr << " should be a ValueSequence";
      return lite::RET_ERROR;
    }
    auto get_item_input_cnode_shape = GetValue<std::vector<ValuePtr>>(shapes_attr);
    for (auto &shape_tuple : get_item_input_cnode_shape) {
      if (!shape_tuple->isa<ValueSequence>()) {
        MS_LOG(ERROR) << "attr: " << lite::kNameShapeAttr << " for get_item_input should be a ValueSequence";
        return lite::RET_ERROR;
      }
      auto shape_list = GetValue<std::vector<ValuePtr>>(shape_tuple);
      MS_CHECK_TRUE_MSG(idx < shape_list.size(), lite::RET_ERROR, "idx should be less than shape_list size.");
      shapes_value.push_back(shape_list[idx]);
    }
  } else {
    auto shapes_attr = cnode->GetAttr(lite::kNameShapeAttr);
    MS_CHECK_TRUE_MSG(shapes_attr != nullptr, lite::RET_NULL_PTR, "shapes_attr is nullptr");
    if (!shapes_attr->isa<ValueSequence>()) {
      MS_LOG(ERROR) << "attr: " << lite::kNameShapeAttr << " should be a ValueSequence";
      return lite::RET_ERROR;
    }

    shapes_value = GetValue<std::vector<ValuePtr>>(shapes_attr);
  }

  std::vector<ShapeVector> static_shapes;
  for (size_t i = 0; i < shapes_value.size(); i++) {
    if (shapes_value[i] == nullptr) {
      MS_LOG(ERROR) << "shapes_value[" << i << "] is nullptr.";
      return lite::RET_ERROR;
    }
    if (!shapes_value[i]->isa<tensor::MetaTensor>()) {
      MS_LOG(ERROR) << "input from cnode should be a tensor.";
      return lite::RET_ERROR;
    }

    auto meta_tensor = shapes_value[i]->cast<tensor::MetaTensorPtr>();
    if (meta_tensor == nullptr) {
      MS_LOG(ERROR) << "shapes_value[" << i << "] cast to MetaTensor failed.";
      return lite::RET_ERROR;
    }
    static_shapes.push_back(meta_tensor->shape());
  }
  MS_CHECK_TRUE_MSG(!static_shapes.empty(), lite::RET_ERROR, "static_shapes is empty.");

  bool same_shape_len = std::all_of(static_shapes.begin(), static_shapes.end(),
                                    [&](const ShapeVector &t) { return t.size() == static_shapes.front().size(); });

  MS_CHECK_TRUE_MSG(same_shape_len, lite::RET_ERROR, "length of dims in static shapes is not equal.");

  if (static_shapes.size() == 1) {
    *shape = static_shapes.front();
  } else {
    *shape = std::vector<int64_t>(static_shapes.front().size(), -1);
    dim_groups->clear();
    std::copy(static_shapes.begin(), static_shapes.end(), std::back_inserter(*dim_groups));
  }
  return lite::RET_OK;
}

STATUS AclnnSubgraphProcessor::ExtractSubgraphOM(const CNodePtr &custom_node, const CNodePtr &cnode,
                                                 const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_MSG(custom_node != nullptr, lite::RET_NULL_PTR, "custom_node is nullptr.");
  MS_CHECK_TRUE_MSG(cnode != nullptr, lite::RET_NULL_PTR, "cnode is nullptr.");
  MS_CHECK_TRUE_MSG(func_graph != nullptr, lite::RET_NULL_PTR, "func_graph is nullptr.");

  auto om_node = custom_node->inputs().back();
  MS_CHECK_TRUE_MSG(om_node->isa<Parameter>(), lite::RET_ERROR, "om node should be a Parameter");
  auto om_parameter = om_node->cast<ParameterPtr>();
  MS_CHECK_TRUE_MSG(om_parameter != nullptr, lite::RET_NULL_PTR, "om_parameter is nullptr.");
  auto om_value = om_parameter->default_param();
  MS_CHECK_TRUE_MSG(om_value != nullptr, lite::RET_NULL_PTR, "om_value is nullptr.");

  auto prim_node = custom_node->inputs().front();
  MS_CHECK_TRUE_MSG(prim_node->isa<ValueNode>(), lite::RET_ERROR, "prim node should be a ValueNode");
  auto prim_value_node = prim_node->cast<ValueNodePtr>();
  MS_CHECK_TRUE_MSG(prim_value_node != nullptr, lite::RET_NULL_PTR, "prim_value_node is nullptr.");
  auto prim_value = prim_value_node->value();
  MS_CHECK_TRUE_MSG(prim_value != nullptr, lite::RET_NULL_PTR, "prim_value is nullptr.");

  cnode->set_input(0, std::make_shared<ValueNode>(prim_value));
  auto inputs = cnode->inputs();

  auto param = func_graph->add_parameter();
  MS_CHECK_TRUE_MSG(param != nullptr, lite::RET_NULL_PTR, "param is nullptr.");
  auto abstract = om_parameter->abstract();
  MS_CHECK_TRUE_MSG(abstract != nullptr, lite::RET_NULL_PTR, "abstract is nullptr");
  auto abstract_clone = abstract->Clone();
  MS_CHECK_TRUE_MSG(abstract_clone != nullptr, lite::RET_NULL_PTR, "abstract_clone is nullptr.");
  param->set_default_param(om_value);
  param->set_abstract(abstract_clone);
  param->set_name("ACL_om_data_" + cnode->fullname_with_scope());

  inputs.push_back(param);
  cnode->set_inputs(inputs);
  return lite::RET_OK;
}

STATUS AclnnSubgraphProcessor::FindCustomAscend(const FuncGraphPtr &graph, std::vector<CNodePtr> *out) {
  MS_CHECK_TRUE_MSG(graph != nullptr, lite::RET_NULL_PTR, "graph is nullptr");
  MS_CHECK_TRUE_MSG(out != nullptr, lite::RET_NULL_PTR, "out is nullptr");
  auto nodes = TopoSort(graph->get_return());
  for (auto &node : nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_CHECK_TRUE_MSG(cnode != nullptr, lite::RET_NULL_PTR, "cnode is nullptr");
    std::string kernel_name = common::AnfAlgo::GetCNodeName(cnode);
    if (kernel_name == lite::kNameCustomAscend) {
      out->push_back(cnode);
    }
  }
  return lite::RET_OK;
}

STATUS AclnnSubgraphProcessor::ProcessCustomAclnnSubgraph(const CNodePtr &cnode, const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_MSG(cnode != nullptr, lite::RET_NULL_PTR, "cnode is nullptr.");
  MS_CHECK_TRUE_MSG(func_graph != nullptr, lite::RET_NULL_PTR, "func_graph is nullptr.");

  auto graph_value = cnode->GetAttr(lite::kNameGraphAttr);
  MS_CHECK_TRUE_MSG(graph_value != nullptr, lite::RET_NULL_PTR, "get attr graph failed.");
  auto graph = GetValue<FuncGraphPtr>(graph_value);
  MS_CHECK_TRUE_MSG(graph != nullptr, lite::RET_NULL_PTR, "get value as FuncGraph failed.");

  auto ret = RunAclPassForSubgraph(cnode, graph);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "RunAclPass on graph: " << cnode->fullname_with_scope() << " failed.";
    return ret;
  }

  std::vector<CNodePtr> custom_nodes;
  ret = FindCustomAscend(graph, &custom_nodes);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "FindCustomAscend on graph: " << cnode->fullname_with_scope() << " failed.";
    return ret;
  }
  if (custom_nodes.size() != 1) {
    MS_LOG(ERROR) << "size of CustomAscend should be 1. graph: " << cnode->fullname_with_scope();
    return lite::RET_ERROR;
  }

  auto custom_node = custom_nodes.front();
  ret = ExtractSubgraphOM(custom_node, cnode, func_graph);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "ExtractSubgraphOM failed for: " << cnode->fullname_with_scope();
    return ret;
  }

  return lite::RET_OK;
}

}  // namespace opt
}  // namespace mindspore
