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
#include "tools/converter/import/convert_extend_ops/utils.h"
#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/converter/import/convert_extend_ops/convert_extend_ops_pass.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_b.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_c.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_r.h"

namespace mindspore::opt {
TypeId GetSingleNodeOutputTypeId(const mindspore::AnfNodePtr &node) {
  TypePtr type = node->Type();
  if (node->isa<CNode>()) {
    auto input0 = node->cast<CNodePtr>()->input(0);
    auto input0_value_node = dyn_cast_ptr<ValueNode>(input0);
    if (input0_value_node != nullptr && input0_value_node->value()->isa<Primitive>()) {
      const auto &abs = node->abstract();
      MS_CHECK_TRUE_MSG(abs != nullptr, kTypeUnknown, "get abstract from CNode failed.");
      type = abs->BuildType();
    }
  }

  MS_CHECK_TRUE_MSG(type != nullptr, kTypeUnknown, "type is nullptr.");
  if (type->isa<TensorType>()) {
    const auto &tensor_type = type->cast<TensorTypePtr>();
    MS_CHECK_TRUE_MSG(tensor_type != nullptr, kTypeUnknown, "cast TensorType failed.");
    const auto &element = tensor_type->element();
    return element->type_id();
  } else {
    return type->type_id();
  }
}

AnfNodePtr GetCastNode(const FuncGraphPtr &func_graph, const mindspore::AnfNodePtr &node, const TypeId &dst_type_id) {
  auto cast_prim = std::make_shared<Primitive>(prim::kPrimCast->name());
  MS_CHECK_TRUE_MSG(cast_prim != nullptr, nullptr, "create Cast Primitive failed.");
  cast_prim->AddAttr("input_names", MakeValue(std::vector<std::string>{"x", "dst_type"}));
  cast_prim->AddAttr("output_names", MakeValue(std::vector<std::string>{"output"}));
  cast_prim->AddAttr("primitive_function", MakeValue<bool>(true));

  auto dst_type = std::make_shared<Int64Imm>(static_cast<int64_t>(dst_type_id));
  std::vector<AnfNodePtr> cast_inputs = {NewValueNode(cast_prim), node, NewValueNode(dst_type)};
  auto cast_shape = node->Shape();
  auto cast_node = func_graph->NewCNode(cast_inputs);
  MS_CHECK_TRUE_MSG(cast_node != nullptr, nullptr, "create Cast CNode failed.");
  auto cast_abs = std::make_shared<mindspore::abstract::AbstractTensor>(TypeIdToType(dst_type_id), cast_shape);
  cast_node->set_abstract(cast_abs);
  cast_node->set_scope(node->scope());

  return cast_node;
}

AnfNodePtr GetReshapeNode(const FuncGraphPtr &func_graph, const mindspore::AnfNodePtr &node,
                          const ShapeVector &dst_shape) {
  auto reshape_prim = std::make_shared<Primitive>(prim::kPrimReshape->name());
  MS_CHECK_TRUE_MSG(reshape_prim != nullptr, nullptr, "create Reshape Primitive failed.");
  reshape_prim->AddAttr("primitive_function", MakeValue<bool>(true));

  auto dst_shape_value = MakeValue(dst_shape);
  MS_CHECK_TRUE_MSG(dst_shape_value != nullptr, nullptr, "create dst_shape_value failed.");
  auto dst_shape_node = NewValueNode(dst_shape_value);
  MS_CHECK_TRUE_MSG(dst_shape_node != nullptr, nullptr, "create dst_shape_node failed.");
  dst_shape_node->set_abstract(dst_shape_value->ToAbstract());

  std::vector<AnfNodePtr> reshape_inputs = {NewValueNode(reshape_prim), node, dst_shape_node};
  auto reshape_node = func_graph->NewCNode(reshape_inputs);
  MS_CHECK_TRUE_MSG(reshape_node != nullptr, nullptr, "create Reshape CNode failed.");
  auto type_id = GetSingleNodeOutputTypeId(node);
  auto reshape_abs = std::make_shared<mindspore::abstract::AbstractTensor>(TypeIdToType(type_id), dst_shape);
  reshape_node->set_abstract(reshape_abs);
  reshape_node->set_scope(node->scope());

  return reshape_node;
}

AnfNodePtr GetBroadcastToNode(const FuncGraphPtr &func_graph, const mindspore::AnfNodePtr &node,
                              const ShapeVector &dst_shape) {
  auto bst_prim = std::make_shared<Primitive>(prim::kPrimBroadcastTo->name());
  MS_CHECK_TRUE_MSG(bst_prim != nullptr, nullptr, "create BroadcastTo Primitive failed.");
  bst_prim->AddAttr("primitive_function", MakeValue<bool>(true));
  bst_prim->AddAttr("shape", MakeValue<ShapeVector>(dst_shape));

  std::vector<AnfNodePtr> bst_inputs = {NewValueNode(bst_prim), node};
  auto bst_node = func_graph->NewCNode(bst_inputs);
  MS_CHECK_TRUE_MSG(bst_node != nullptr, nullptr, "create BroadcastTo CNode failed.");
  auto type_id = GetSingleNodeOutputTypeId(node);
  auto bst_abs = std::make_shared<mindspore::abstract::AbstractTensor>(TypeIdToType(type_id), dst_shape);
  bst_node->set_abstract(bst_abs);
  bst_node->set_scope(node->scope());

  return bst_node;
}
}  // namespace mindspore::opt
