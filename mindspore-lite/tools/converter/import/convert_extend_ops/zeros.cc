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
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "utils/ms_context.h"
#include "mindspore/ops/op_def/array_ops.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/converter/import/convert_extend_ops/utils.h"
#include "tools/converter/import/convert_extend_ops/convert_extend_ops_pass.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_z.h"

namespace mindspore::opt {
namespace {
AnfNodePtr ZerosGetValueByDtype(const FuncGraphPtr &func_graph, const mindspore::CNodePtr &cnode) {
  auto dtype = cnode->input(kInputIndexTwo);
  if (IsValueNode<None>(dtype)) {
    return NewValueNode(MakeValue<float>(0));
  }
  auto dtype_value_node_ptr = dtype->cast<ValueNodePtr>();
  MS_CHECK_TRUE_MSG(dtype_value_node_ptr != nullptr, nullptr, "dtype cannot be converted to a ValueNode.");
  auto dtype_value_ptr = utils::cast<ValuePtr>(dtype_value_node_ptr->value());
  auto dst_type_id = static_cast<TypeId>(GetValue<int64_t>(dtype_value_ptr));
  return GetCastedScalar(0, dst_type_id);
}
}  // namespace

AnfNodePtr ConvertZerosPass(const FuncGraphPtr &func_graph, const mindspore::AnfNodePtr &node) {
  auto zeros_cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(zeros_cnode != nullptr, nullptr);
  MS_CHECK_TRUE_RET(zeros_cnode->size() == kInputSizeThree, nullptr);
  if (!CheckPrimitiveType(zeros_cnode, prim::kPrimZeros)) {
    return nullptr;
  }

  auto zeros_prim_value_node = zeros_cnode->input(kIndex0)->cast<ValueNodePtr>();
  MS_CHECK_TRUE_MSG(zeros_prim_value_node != nullptr, nullptr, "primitive cannot be converted to a ValueNode.");
  auto zeros_prim_value = zeros_prim_value_node->value();
  MS_CHECK_TRUE_MSG(zeros_prim_value != nullptr, nullptr, "primitive cannot get value.");
  auto zeros_prim = zeros_prim_value->cast<PrimitivePtr>();
  MS_CHECK_TRUE_MSG(zeros_prim != nullptr, nullptr, "primitive cannot cast to Primitive.");

  auto new_prim_name = prim::kPrimFillV2->name();
  auto new_prim = std::make_shared<Primitive>(new_prim_name);
  MS_CHECK_TRUE_MSG(new_prim != nullptr, nullptr, "create Primitive " + new_prim_name + " failed.");

  auto value_node = ZerosGetValueByDtype(func_graph, zeros_cnode);

  auto input = zeros_cnode->input(kIndex1);
  std::vector<AnfNodePtr> new_inputs = {NewValueNode(new_prim), input, value_node};
  auto reduce_zeros_node = func_graph->NewCNode(new_inputs);
  MS_CHECK_TRUE_MSG(reduce_zeros_node != nullptr, nullptr, "create CNode " + new_prim_name + " failed.");
  reduce_zeros_node->set_abstract(zeros_cnode->abstract());

  return reduce_zeros_node;
}
}  // namespace mindspore::opt
