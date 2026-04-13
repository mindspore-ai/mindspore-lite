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

#include "tools/optimizer/fusion/adjust_reducesum_pass.h"
#include <memory>
#include <vector>
#include <string>
#include "ops_utils/op_utils.h"
#include "tools/common/tensor_util.h"
#include "mindspore/ops/op_def/lite_ops.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "mindspore/ops/op_def/auto_generate/gen_lite_ops.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_c.h"
#include "mindspore/core/include/ir/graph_utils.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_r.h"

namespace mindspore {
namespace opt {
namespace {
static constexpr size_t kIndex1 = 1;
static constexpr size_t kIndex2 = 2;
static constexpr size_t kIndex3 = 3;
static constexpr int kNumberFlatten = -1;

CNodePtr CreateReshapeCNode(const FuncGraphPtr &func_graph, const AnfNodePtr &cnode,
                            const std::vector<int32_t> &shape) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
  MS_CHECK_TRUE_RET(cnode != nullptr, nullptr);
  MS_CHECK_TRUE_RET(cnode->abstract() != nullptr, nullptr);
  auto shape_parm_node =
    opt::BuildIntVecParameterNode(func_graph, shape, cnode->fullname_with_scope() + "_input_shape_perm");
  MS_CHECK_TRUE_MSG(shape_parm_node != nullptr, nullptr, "create shape_parm_node return nullptr!");
  std::vector<AnfNodePtr> op_inputs = {cnode, shape_parm_node};
  auto reshape_prim = std::make_shared<ops::Reshape>();
  MS_CHECK_TRUE_MSG(reshape_prim != nullptr, nullptr, "create reshape_prim return nullptr!");
  auto reshape_prim_c = reshape_prim->GetPrim();
  MS_CHECK_TRUE_MSG(reshape_prim_c != nullptr, nullptr, "create prim_c return nullptr!");
  auto reshape_node = func_graph->NewCNode(reshape_prim_c, op_inputs);
  MS_CHECK_TRUE_MSG(reshape_node != nullptr, nullptr, "create reshape_node return nullptr!");
  reshape_node->set_fullname_with_scope(cnode->fullname_with_scope() + "_reshape");
  reshape_node->set_abstract(cnode->abstract()->Clone());
  return reshape_node;
}

bool IsAxesEmpty(AnfNodePtr axes_input) {
  MS_CHECK_TRUE_RET(axes_input != nullptr, true);
  if (utils::isa<ValueNodePtr>(axes_input) && axes_input->cast<ValueNodePtr>() != nullptr &&
      axes_input->cast<ValueNodePtr>()->value() != nullptr) {
    auto value_ptr = axes_input->cast<ValueNodePtr>()->value();
    if (value_ptr->type() != nullptr &&
        (value_ptr->type()->type_id() == kNumberTypeInt64 || value_ptr->type()->type_id() == kNumberTypeInt32)) {
      return false;
    } else if (value_ptr->type() != nullptr && value_ptr->type()->type_id() == kObjectTypeTuple) {
      auto type_tuple = value_ptr->type()->cast<TuplePtr>();
      MS_CHECK_TRUE_MSG(type_tuple != nullptr, true, "type_tuple is nullptr!");
      return type_tuple->elements().empty();
    }
  }
  return true;
}

Status FillAxesForNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode, const std::vector<int32_t> &axes_values) {
  auto axes_value_node = opt::BuildIntVecValueNode(func_graph, axes_values);
  MS_CHECK_TRUE_MSG(axes_value_node != nullptr, kLiteError, "Create axes value node failed!");
  auto inputs = cnode->inputs();
  if (inputs.size() >= kIndex3) {
    inputs[kIndex2] = axes_value_node->cast<AnfNodePtr>();
  } else {
    inputs.push_back(axes_value_node->cast<AnfNodePtr>());
  }
  cnode->set_inputs(inputs);
  return kSuccess;
}

Status AdjustReduceSum(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_CHECK_TRUE_MSG(cnode->inputs().size() > kIndex1, kLiteError, "input size should large than 1!");
  auto value_node = cnode->input(0)->cast<ValueNodePtr>();
  if (value_node == nullptr) {
    MS_LOG(ERROR) << "Value node[" << cnode->fullname_with_scope() << "] is nullptr!";
    return kLiteError;
  }
  auto src_prim = GetValueNode<PrimitivePtr>(value_node);
  if (src_prim == nullptr) {
    MS_LOG(ERROR) << "Value node[" << cnode->fullname_with_scope() << "] cast to primitive failed!";
    return kLiteError;
  }
  auto skip_mode_ptr = src_prim->GetAttr("skip_mode");
  if (skip_mode_ptr != nullptr) {
    auto skip_mode = GetValue<bool>(skip_mode_ptr);
    if (skip_mode == true) {
      return kSuccess;
    }
  }
  if (cnode->inputs().size() >= kIndex3) {
    auto axes_input = cnode->input(kIndex2);
    if (!IsAxesEmpty(axes_input)) {
      return kSuccess;
    }
  }

  auto keep_dims_ptr = src_prim->GetAttr("keep_dims");
  bool keep_dims = false;
  if (keep_dims_ptr != nullptr) {
    keep_dims = GetValue<bool>(keep_dims_ptr);
  }

  if (keep_dims) {
    // For keep_dims=true, follow MindSpore ReduceAxisUpdate::BuildAxis strategy:
    // fill axis with [0, 1, ..., rank-1] to preserve output dimensions.
    auto input_abstract = cnode->input(kIndex1)->abstract();
    MS_CHECK_TRUE_MSG(input_abstract != nullptr, kLiteError, "input abstract is nullptr!");
    auto input_shape = input_abstract->GetShape();
    MS_CHECK_TRUE_MSG(input_shape != nullptr, kLiteError, "input shape is nullptr!");
    auto shape_vec = input_shape->GetShapeVector();
    if (shape_vec.empty()) {
      MS_LOG(INFO) << "Input shape is empty for keep_dims=true, skip: " << cnode->fullname_with_scope();
      return kSuccess;
    }
    std::vector<int32_t> full_axis;
    for (size_t i = 0; i < shape_vec.size(); i++) {
      full_axis.push_back(static_cast<int32_t>(i));
    }
    MS_LOG(INFO) << "Fill full axis list for keep_dims=true ReduceSum node: " << cnode->fullname_with_scope()
                 << ", axis size: " << full_axis.size();
    return FillAxesForNode(func_graph, cnode, full_axis);
  }

  // For keep_dims=false, flatten input to 1D and reduce on axis 0.
  auto reshape_node = CreateReshapeCNode(func_graph, cnode->input(kIndex1), {kNumberFlatten});
  MS_CHECK_TRUE_MSG(reshape_node != nullptr, kLiteNullptr, "reshape node is nullptr!");
  auto graph_manager = func_graph->manager();
  MS_CHECK_TRUE_MSG(graph_manager != nullptr, kLiteNullptr, "graph_manager is nullptr!");
  if (!graph_manager->Replace(cnode->input(1), reshape_node)) {
    MS_LOG(ERROR) << "Failed to replace input of reducesum node by reshape node, cnode: "
                  << cnode->fullname_with_scope() << ", input size: " << cnode->size();
    return kLiteError;
  }
  return FillAxesForNode(func_graph, cnode, {0});
}
}  // namespace
bool AdjustReduceSumPass::Run(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, false);
  MS_LOG(INFO) << "AdjustReduceSumPass start.";
  auto fmk_type_ptr = func_graph->get_attr("fmk");
  if (fmk_type_ptr != nullptr && (GetValue<int>(fmk_type_ptr) != static_cast<int>(converter::kFmkTypeMs) &&
                                  GetValue<int>(fmk_type_ptr) != static_cast<int>(converter::kFmkTypeOnnx))) {
    return true;
  }
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
    if (!opt::CheckPrimitiveType(node, prim::kPrimReduceSum)) {
      continue;
    }
    auto reducesum_node = node->cast<CNodePtr>();
    MS_CHECK_TRUE_RET(reducesum_node != nullptr, false);
    if (AdjustReduceSum(func_graph, reducesum_node) != kSuccess) {
      MS_LOG(ERROR) << "This node run AdjustReduceSum failed! Node_name is: " << reducesum_node->fullname_with_scope()
                    << "!";
      return false;
    }
    MS_LOG(INFO) << "This node run AdjustReduceSum success : " << reducesum_node->fullname_with_scope();
  }
  MS_LOG(INFO) << "AdjustReduceSum end.";
  return true;
}
}  // namespace opt
}  // namespace mindspore
