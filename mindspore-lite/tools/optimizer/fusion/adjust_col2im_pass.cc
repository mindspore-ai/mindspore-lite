/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "tools/optimizer/fusion/adjust_col2im_pass.h"
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include "infer/resize.h"
#include "ops_utils/op_utils.h"
#include "tools/common/tensor_util.h"
#include "mindspore/ops/op_def/lite_ops.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "mindspore/ops/op_def/auto_generate/gen_lite_ops.h"
#include "infer/range_v2.h"
#include "mindspore/ops/op_def/image_ops.h"
#include "nnacl/op_base.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_c.h"

namespace mindspore {
namespace opt {
namespace {
STATUS AddConstInputToAttr(const CNodePtr &cnode, const size_t input_index, const std::string &arg_name,
                           const std::string &arg_handler, const PrimitivePtr &primitive) {
  if (input_index >= cnode->size() - 1) {
    MS_LOG(ERROR) << "The index of args in op_def `" << input_index << "` should less than the inputs size minus one `"
                  << cnode->size() - 1 << "`!";
    return RET_ERROR;
  }
  auto input_node = cnode->inputs()[input_index + 1];

  ValuePtr value = nullptr;
  if (input_node->isa<ValueNode>()) {
    auto value_node = input_node->cast<ValueNodePtr>();
    value = value_node->value();
  } else if (input_node->isa<Parameter>()) {
    auto parameter_node = input_node->cast<ParameterPtr>();
    CHECK_NULL_RETURN(parameter_node->abstract());
    value = parameter_node->abstract()->BuildValue();
  }
  if (value == nullptr) {
    MS_LOG(ERROR) << cnode->ToString() << " is not Value!";
    return lite::RET_ERROR;
  }
  if (value->isa<ValueAny>()) {
    MS_LOG(ERROR) << cnode->ToString() << " is ValueAny!";
    return lite::RET_ERROR;
  }

  if (!value->isa<tensor::Tensor>()) {
    primitive->AddAttr(arg_name, value);
    return RET_OK;
  }
  auto value_vector = CheckAndConvertUtils::CheckTensorIntValue(arg_name, value, primitive->name());
  auto tensor = value->cast<tensor::TensorPtr>();
  auto tensor_shape = tensor->shape_c();
  MS_LOG(DEBUG) << cnode->ToString() << " 's input[" << input_index << "] is tensor.";
  if (tensor_shape.empty()) {
    primitive->AddAttr(arg_name, MakeValue(value_vector[0]));
  } else {
    primitive->AddAttr(arg_name, MakeValue(value_vector));
  }
  return lite::RET_OK;
}

STATUS AdjustCol2im(const CNodePtr &cnode) {
  auto value_node = cnode->input(0)->cast<ValueNodePtr>();
  if (value_node == nullptr) {
    MS_LOG(ERROR) << "Value node[" << cnode->fullname_with_scope() << "] is nullptr!";
    return lite::RET_ERROR;
  }
  auto src_prim = GetValueNode<PrimitivePtr>(value_node);
  if (src_prim == nullptr) {
    MS_LOG(ERROR) << "Value node[" << cnode->fullname_with_scope() << "] cast to primitive failed!";
    return lite::RET_ERROR;
  }
  if (AddConstInputToAttr(cnode, Index2, "kernel_size", "", src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "AddConstInputToAttr failed!";
    return lite::RET_ERROR;
  }

  auto input = cnode->input(Index1);
  CHECK_NULL_RETURN(input);
  auto func_graph = cnode->func_graph();
  CHECK_NULL_RETURN(func_graph);
  auto shape_node = opt::GenShapeNode(func_graph, input, cnode->fullname_with_scope() + "_shape");
  CHECK_NULL_RETURN(shape_node);
  AnfNodePtr gather_node0 =
    opt::GenGatherNode(func_graph, shape_node, {0}, cnode->fullname_with_scope() + "_gather0", {0});
  CHECK_NULL_RETURN(gather_node0);
  auto gather_node1 = opt::GenGatherNode(func_graph, shape_node, {1}, cnode->fullname_with_scope() + "_gather1", {0});
  CHECK_NULL_RETURN(gather_node1);
  AnfNodePtr gather_node2 =
    opt::GenGatherNode(func_graph, shape_node, {Index2}, cnode->fullname_with_scope() + "_gather2", {0});
  CHECK_NULL_RETURN(gather_node2);
  if (cnode->abstract() != nullptr) {
    gather_node0->set_abstract(cnode->abstract()->Clone());
    gather_node1->set_abstract(cnode->abstract()->Clone());
    gather_node2->set_abstract(cnode->abstract()->Clone());
  }
  int32_t kernel_mul = 1;
  if (src_prim->HasAttr("kernel_size")) {
    auto kernel_size = GetValue<std::vector<int64_t>>(src_prim->GetAttr("kernel_size"));
    kernel_mul = std::accumulate(kernel_size.begin(), kernel_size.end(), 1, std::multiplies<int32_t>());
  } else {
    MS_LOG(ERROR) << "Col2im must have attr kernel_size!";
    return lite::RET_ERROR;
  }
  std::vector<int> div_input2 = {kernel_mul};
  AnfNodePtr div_node = opt::GenDivNode(func_graph, gather_node1, div_input2, cnode->fullname_with_scope() + "_div");
  CHECK_NULL_RETURN(div_node);
  AnfNodePtr input_kernel_node =
    opt::BuildIntVecParameterNode(func_graph, div_input2, cnode->fullname_with_scope() + "_kernel_mul");
  if (input_kernel_node == nullptr) {
    MS_LOG(ERROR) << "Make input1 node failed!";
    return lite::RET_ERROR;
  }
  std::vector<AnfNodePtr> concat_input = {gather_node0, div_node, input_kernel_node, gather_node2};
  auto concat_node = opt::GenConcatNode(func_graph, concat_input, cnode->fullname_with_scope() + "_concat", 0);

  auto reshape_node = opt::GenReshapeNode(func_graph, input, concat_node, cnode->fullname_with_scope() + "_reshape");
  auto manager = Manage(func_graph);
  CHECK_NULL_RETURN(manager);
  if (!manager->Replace(input, reshape_node)) {
    MS_LOG(ERROR) << "Replace node failed!";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}
}  // namespace
bool AdjustCol2imPass::Run(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, false);
  MS_LOG(INFO) << "AdjustCol2imPass start.";
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
    if (!opt::CheckPrimitiveType(node, prim::kPrimCol2Im)) {
      continue;
    }
    auto col2im_node = node->cast<CNodePtr>();
    MS_CHECK_TRUE_RET(col2im_node != nullptr, false);
    if (AdjustCol2im(col2im_node) != lite::RET_OK) {
      MS_LOG(ERROR) << "This node run AdjustCol2im failed! Node_name is: " << col2im_node->fullname_with_scope() << "!";
      return false;
    }
    MS_LOG(INFO) << "This node run AdjustCol2im success : " << col2im_node->fullname_with_scope();
  }
  MS_LOG(INFO) << "AdjustCol2im end.";
  return true;
}
}  // namespace opt
}  // namespace mindspore
