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
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "utils/ms_context.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/converter/import/convert_extend_ops/utils.h"
#include "tools/converter/import/convert_extend_ops/convert_extend_ops_pass.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_r.h"

namespace mindspore::opt {
AnfNodePtr ConvertMaxMinPass(const FuncGraphPtr &func_graph, const mindspore::AnfNodePtr &node) {
  auto max_min_cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(max_min_cnode != nullptr, nullptr);
  MS_CHECK_TRUE_RET(max_min_cnode->size() == kInputSizeTwo, nullptr);
  bool is_max = CheckPrimitiveType(max_min_cnode, prim::kPrimMax);
  if (!is_max && !CheckPrimitiveType(max_min_cnode, prim::kPrimMin)) {
    return nullptr;
  }

  auto input = max_min_cnode->input(kIndex1);
  auto input_shape = input->abstract()->GetShape();
  MS_CHECK_TRUE_MSG(input_shape != nullptr, nullptr, "Can't get input shape from " + (is_max ? "Max" : "Min") + ".");
  auto input_shape_vec = input_shape->GetShapeVector();

  auto max_min_prim_value_node = max_min_cnode->input(kIndex0)->cast<ValueNodePtr>();
  MS_CHECK_TRUE_MSG(max_min_prim_value_node != nullptr, nullptr, "primitive cannot be converted to a ValueNode.");
  auto max_min_prim_value = max_min_prim_value_node->value();
  MS_CHECK_TRUE_MSG(max_min_prim_value != nullptr, nullptr, "primitive cannot get value.");
  auto max_min_prim = max_min_prim_value->cast<PrimitivePtr>();
  MS_CHECK_TRUE_MSG(max_min_prim != nullptr, nullptr, "primitive cannot cast to Primitive.");

  auto new_prim_name = is_max ? prim::kPrimReduceMax->name() : prim::kPrimReduceMin->name();
  auto new_prim = std::make_shared<Primitive>(new_prim_name);
  MS_CHECK_TRUE_MSG(new_prim != nullptr, nullptr, "create Primitive " + new_prim_name + " failed.");
  (void)new_prim->SetAttrs(max_min_prim->attrs());
  new_prim->AddAttr("keep_dims", MakeValue(false));

  std::vector<int64_t> axis = {};
  if (!input_shape->IsDynamic()) {
    for (int64_t i = 0; i < SizeToLong(input_shape_vec.size()); ++i) {
      axis.emplace_back(i);
    }
  }

  auto axis_value = MakeValue(axis);
  auto axis_node = NewValueNode(axis_value);
  axis_node->set_abstract(axis_value->ToAbstract());

  std::vector<AnfNodePtr> new_inputs = {NewValueNode(new_prim), input, axis_node};
  auto reduce_max_min_node = func_graph->NewCNode(new_inputs);
  MS_CHECK_TRUE_MSG(reduce_max_min_node != nullptr, nullptr, "create CNode " + new_prim_name + " failed.");
  reduce_max_min_node->set_abstract(max_min_cnode->abstract());

  return reduce_max_min_node;
}
}  // namespace mindspore::opt
