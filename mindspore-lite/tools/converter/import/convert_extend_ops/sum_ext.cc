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
#include <vector>
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/converter/import/convert_extend_ops/utils.h"
#include "tools/converter/import/convert_extend_ops/convert_extend_ops_pass.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_r.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"

namespace mindspore::opt {
namespace {
AnfNodePtr ReduceExtendGetCastInputByDtype(const FuncGraphPtr &func_graph, const mindspore::CNodePtr &cnode) {
  auto input = cnode->input(kInputIndexOne);
  auto dtype = cnode->input(kInputIndexFour);

  auto src_type_id = GetSingleNodeOutputTypeId(input);
  MS_CHECK_TRUE_MSG(src_type_id != kTypeUnknown, nullptr, "get src_type_id failed.");
  if (IsValueNode<None>(dtype)) {
    const std::set<TypeId> kIntergralSet = {kNumberTypeBool, kNumberTypeUInt8, kNumberTypeInt8, kNumberTypeInt16,
                                            kNumberTypeInt32};
    if (kIntergralSet.find(src_type_id) != kIntergralSet.end()) {
      MS_LOG(INFO) << "For SumExt, when dtype of 'input' is [bool, uint8, int8, int16, int32, int64] and 'dtype' is "
                   << "None, then dtype of output will be int64.";
      return GetCastNode(func_graph, input, kNumberTypeInt64);
    }
    return input;
  }

  auto dtype_value_node_ptr = dtype->cast<ValueNodePtr>();
  MS_CHECK_TRUE_MSG(dtype_value_node_ptr != nullptr, nullptr, "dtype cannot be converted to a ValueNode.");
  auto dtype_value_ptr = utils::cast<ValuePtr>(dtype_value_node_ptr->value());
  auto dst_type_id = static_cast<TypeId>(GetValue<int64_t>(dtype_value_ptr));
  return dst_type_id != src_type_id ? GetCastNode(func_graph, input, dst_type_id) : input;
}
}  // namespace

AnfNodePtr ConvertSumExtPass(const FuncGraphPtr &func_graph, const mindspore::AnfNodePtr &node) {
  auto sum_ext_cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(sum_ext_cnode != nullptr, nullptr);
  MS_CHECK_TRUE_RET(sum_ext_cnode->size() == kInputSizeFive, nullptr);
  if (!CheckPrimitiveType(sum_ext_cnode, prim::kPrimSumExt)) {
    return nullptr;
  }

  auto input = ReduceExtendGetCastInputByDtype(func_graph, sum_ext_cnode);
  MS_CHECK_TRUE_MSG(input != nullptr, nullptr, "input is invalid.");
  auto axis = sum_ext_cnode->input(kInputIndexTwo);
  if (IsValueNode<None>(axis)) {
    auto new_axis_value = MakeValue<std::vector<int64_t>>({});
    auto new_axis = NewValueNode(new_axis_value);
    new_axis->set_scope(axis->scope());
    new_axis->set_abstract(new_axis_value->ToAbstract());
    axis = new_axis;
  }

  const auto &keep_dims = sum_ext_cnode->input(kInputIndexThree);
  auto keep_dims_value_node_ptr = keep_dims->cast<ValueNodePtr>();
  MS_CHECK_TRUE_MSG(keep_dims_value_node_ptr != nullptr, nullptr, "keep_dims cannot be converted to a ValueNode.");

  auto sum_ext_prim_value_node = sum_ext_cnode->input(kInputIndexZero)->cast<ValueNodePtr>();
  MS_CHECK_TRUE_MSG(sum_ext_prim_value_node != nullptr, nullptr, "primitive cannot be converted to a ValueNode.");
  auto sum_ext_prim_value = sum_ext_prim_value_node->value();
  MS_CHECK_TRUE_MSG(sum_ext_prim_value != nullptr, nullptr, "primitive cannot get value.");
  auto sum_ext_prim = sum_ext_prim_value->cast<PrimitivePtr>();
  MS_CHECK_TRUE_MSG(sum_ext_prim != nullptr, nullptr, "primitive cannot cast to Primitive.");

  auto reduce_sum_prim = std::make_shared<Primitive>(prim::kPrimReduceSum->name());
  MS_CHECK_TRUE_MSG(reduce_sum_prim != nullptr, nullptr, "create Primitive ReduceSum failed.");
  (void)reduce_sum_prim->SetAttrs(sum_ext_prim->attrs());
  reduce_sum_prim->AddAttr("keep_dims", keep_dims_value_node_ptr->value());
  reduce_sum_prim->AddAttr("skip_mode", MakeValue<bool>(false));

  std::vector<AnfNodePtr> reduce_sum_inputs = {NewValueNode(reduce_sum_prim), input, axis};
  auto reduce_sum_node = func_graph->NewCNode(reduce_sum_inputs);
  MS_CHECK_TRUE_MSG(reduce_sum_node != nullptr, nullptr, "create CNode reduce_sum failed.");
  reduce_sum_node->set_abstract(sum_ext_cnode->abstract());

  return reduce_sum_node;
}
}  // namespace mindspore::opt
