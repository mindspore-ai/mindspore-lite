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
#include "ir/dtype.h"
#include "utils/ms_context.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/converter/import/convert_extend_ops/utils.h"
#include "tools/converter/import/convert_extend_ops/convert_extend_ops_pass.h"
#include "mindspore/ccsrc/include/common/utils/convert_utils.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"

namespace mindspore::opt {
namespace {
TypeId GetMulsPromoteType(TypeId input_type, TypeId other_type) {
  std::set<TypeId> kSet = {kNumberTypeUInt8, kNumberTypeInt8, kNumberTypeInt16, kNumberTypeInt32, kNumberTypeInt64};
  auto promote_type = kNumberTypeFloat32;
  if ((kSet.find(input_type) != kSet.end()) && other_type == kNumberTypeFloat32) {
    promote_type = kNumberTypeFloat32;
  } else if (input_type == kNumberTypeBool && (other_type == kNumberTypeFloat32 || other_type == kNumberTypeInt64)) {
    promote_type = other_type;
  } else {
    promote_type = input_type;
  }
  return promote_type;
}
}  // namespace

AnfNodePtr ConvertMulsPass(const FuncGraphPtr &func_graph, const mindspore::AnfNodePtr &node) {
  auto muls_cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(muls_cnode != nullptr, nullptr);
  MS_CHECK_TRUE_RET(muls_cnode->size() == kInputSizeThree, nullptr);
  if (!CheckPrimitiveType(muls_cnode, prim::kPrimMuls)) {
    return nullptr;
  }

  auto input = muls_cnode->input(kIndex1);
  auto input_type = input->abstract()->GetType();
  MS_CHECK_TRUE_MSG(input_type != nullptr, nullptr, "Can't get 'input' type from Muls.");
  auto input_tensor_type = input_type->cast<TensorTypePtr>();
  MS_CHECK_TRUE_MSG(input_tensor_type != nullptr, nullptr, "Can't get 'input' tensor type from Muls.");
  auto input_typeid = input_tensor_type->element()->type_id();
  auto other = muls_cnode->input(kIndex2);
  auto other_type = other->abstract()->GetType();
  MS_CHECK_TRUE_MSG(other_type != nullptr, nullptr, "Can't get 'other' type from Muls.");
  auto other_typeid = other_type->type_id();

  if ((input_typeid == kNumberTypeUInt16 || input_typeid == kNumberTypeUInt32 || input_typeid == kNumberTypeUInt64) &&
      (other_typeid == kNumberTypeFloat32 || other_typeid == kNumberTypeInt64)) {
    MS_LOG(ERROR) << "Type implicit conversion between Tensor[" << TypeIdToString(input_typeid) << "] and "
                  << TypeIdToString(other_typeid) << " is not supported for Muls.";
    return nullptr;
  }
  auto promote_type = GetMulsPromoteType(input_typeid, other_typeid);

  auto mul_prim = std::make_shared<Primitive>(prim::kPrimMul->name());
  MS_CHECK_TRUE_MSG(mul_prim != nullptr, nullptr, "create Mul Primitive failed.");
  mul_prim->AddAttr("primitive_function", MakeValue<bool>(true));
  auto cast_input = GetCastNode(func_graph, input, promote_type);

  auto other_value_node = other->cast<ValueNodePtr>();
  MS_CHECK_TRUE_MSG(other_value_node != nullptr, nullptr, "Convert to ValueNode failed.");
  auto other_value = other_value_node->value();
  MS_CHECK_TRUE_MSG(other_value != nullptr, nullptr, "Can't get Value from ValueNode.");
  auto other_scalar = other_value->cast<ScalarPtr>();
  MS_CHECK_TRUE_MSG(other_scalar != nullptr, nullptr, "Can't get Scalar from Value.");
  auto other_tensor = ScalarToTensor(other_scalar, TypeIdToType(promote_type));
  auto cast_other = NewValueNode(other_tensor);
  cast_other->set_abstract(other_tensor->ToAbstract());
  cast_other->set_scope(node->scope());

  std::vector<AnfNodePtr> mul_inputs = {NewValueNode(mul_prim), cast_input, cast_other};
  auto mul_node = func_graph->NewCNode(mul_inputs);
  MS_CHECK_TRUE_MSG(mul_node != nullptr, nullptr, "create Mul CNode failed.");
  auto mul_abs = std::make_shared<mindspore::abstract::AbstractTensor>(TypeIdToType(promote_type), node->Shape());
  mul_node->set_abstract(mul_abs);
  mul_node->set_scope(node->scope());

  return mul_node;
}
}  // namespace mindspore::opt
