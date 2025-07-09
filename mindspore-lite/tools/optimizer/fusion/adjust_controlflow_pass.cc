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

#define USE_DEPRECATED_API
#include "tools/optimizer/fusion/adjust_controlflow_pass.h"
#include <memory>
#include <vector>
#include <string>
#include <map>
#include <functional>
#include "tools/converter/ops/ops_def.h"
#include "include/errorcode.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "mindspore/ops/op_def/op_name.h"
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "common/log_util.h"
#include "mindspore/ops/op_def/lite_ops.h"
#include "ops_utils/op_constants.h"
#include "tools/converter/export_model.h"

namespace mindspore {
namespace opt {
int32_t AdjustControlflowPass::AdjustBranchs(const FuncGraphPtr &branch, const FuncGraphPtr &func_graph) {
  auto node_list = TopoSort(branch->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    if (!opt::CheckPrimitiveType(node, prim::kPrimIf)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (AdjustControlflow(cnode, func_graph) != lite::RET_OK) {
      MS_LOG(ERROR) << "This node run AdjustControlflow failed! Node_name is: " << cnode->fullname_with_scope() << "!";
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}

int32_t AdjustControlflowPass::AdjustControlflow(const CNodePtr &cnode, const FuncGraphPtr &func_graph) {
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "cnode is nullptr!";
    return lite::RET_ERROR;
  }
  if (cnode->size() < ops::kSize3) {
    MS_LOG(ERROR) << "If node size should larger than 3! current size:" << cnode->size();
    return lite::RET_ERROR;
  }
  auto value_node = cnode->input(0)->cast<ValueNodePtr>();
  if (value_node == nullptr) {
    MS_LOG(ERROR) << "Value node[" << cnode->fullname_with_scope() << "] is nullptr!";
    return lite::RET_ERROR;
  }
  auto then_value_node = cnode->input(kIndex1)->cast<ValueNodePtr>();
  if (then_value_node == nullptr) {
    MS_LOG(ERROR) << "Then branch of If node is nullptr!";
    return lite::RET_ERROR;
  }
  auto else_value_node = cnode->input(kIndex2)->cast<ValueNodePtr>();
  if (else_value_node == nullptr) {
    MS_LOG(ERROR) << "Else branch of If node is nullptr!";
    return lite::RET_ERROR;
  }

  auto branch_then = GetValue<FuncGraphPtr>(then_value_node->value());
  if (branch_then == nullptr) {
    MS_LOG(ERROR) << "Value of then branch is null!";
    return lite::RET_ERROR;
  }

  auto branch_else = GetValue<FuncGraphPtr>(else_value_node->value());
  if (branch_else == nullptr) {
    MS_LOG(ERROR) << "Value of else branch is null!";
    return lite::RET_ERROR;
  }
  auto new_param = std::make_shared<ConverterPara>();
  new_param->fmk_type = converter::kFmkTypeMs;
  new_param->save_type = kMindIR;
  std::map<FuncGraphPtr, FuncGraphPtr> cloned_func_graph;
  auto mirror_graph_then = lite::CloneFuncGraph(branch_then, new_param, &cloned_func_graph);
  MS_CHECK_TRUE_MSG(mirror_graph_then != nullptr, lite::RET_NULL_PTR, "mirror_graph_then create failed!");

  auto mirror_graph_else = lite::CloneFuncGraph(branch_else, new_param, &cloned_func_graph);
  MS_CHECK_TRUE_MSG(mirror_graph_else != nullptr, lite::RET_NULL_PTR, "mirror_graph_else create failed!");

  static auto manager_then = Manage(mirror_graph_then);
  MS_CHECK_TRUE_MSG(manager_then != nullptr, lite::RET_NULL_PTR, "manager_then create failed!");
  mirror_graph_then->set_manager(manager_then);

  static auto manager_else = Manage(mirror_graph_else);
  MS_CHECK_TRUE_MSG(manager_else != nullptr, lite::RET_NULL_PTR, "manager_else create failed!");
  mirror_graph_else->set_manager(manager_else);

  if (AdjustBranchs(mirror_graph_then, func_graph) != lite::RET_OK) {
    MS_LOG(ERROR) << "Adjust then_value_node failed!";
    return lite::RET_ERROR;
  }
  if (AdjustBranchs(mirror_graph_else, func_graph) != lite::RET_OK) {
    MS_LOG(ERROR) << "Adjust else_value_node failed!";
    return lite::RET_ERROR;
  }
  auto mirror_value_then = NewValueNode(mirror_graph_then);
  MS_CHECK_TRUE_MSG(mirror_value_then != nullptr, lite::RET_NULL_PTR, "mirror_value_then create failed!");
  auto mirror_value_else = NewValueNode(mirror_graph_else);
  MS_CHECK_TRUE_MSG(mirror_value_else != nullptr, lite::RET_NULL_PTR, "mirror_value_else create failed!");

  auto if_inputs = cnode->inputs();
  if_inputs[kIndex2] = mirror_value_else;
  if_inputs.insert(if_inputs.begin() + kIndex2, {mirror_value_then});
  cnode->set_inputs(if_inputs);
  return lite::RET_OK;
}

bool AdjustControlflowPass::Run(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, false);
  auto node_list = TopoSort(func_graph->get_return());
  auto manager = Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "Manager is nullptr!";
    return false;
  }
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    if (!opt::CheckPrimitiveType(node, prim::kPrimIf)) {
      continue;
    }
    MS_LOG(INFO) << "begin process if node";
    auto if_node = node->cast<CNodePtr>();
    MS_CHECK_TRUE_RET(if_node != nullptr, false);
    if (AdjustControlflow(if_node, func_graph) != lite::RET_OK) {
      MS_LOG(ERROR) << "This node run AdjustControlflow failed! Node_name is: " << if_node->fullname_with_scope()
                    << "!";
      return false;
    }
    MS_LOG(INFO) << "This node run AdjustControlflowPass success : " << if_node->fullname_with_scope();
  }
  MS_LOG(INFO) << "AdjustControlflowPass end.";
  return true;
}
}  // namespace opt
}  // namespace mindspore
