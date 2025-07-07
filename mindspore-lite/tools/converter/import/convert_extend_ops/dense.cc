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
#include "infer/ops_func_impl/add.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/converter/import/convert_extend_ops/utils.h"
#include "tools/converter/import/convert_extend_ops/convert_extend_ops_pass.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_a.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_d.h"

namespace mindspore::opt {
namespace {
ShapeArray CalcDenseReshapeVector(const ShapeVector &input_shape_vec, const ShapeVector &weight_shape_vec) {
  ShapeVector input_reshape_vec = {-1, input_shape_vec.back()};
  ShapeVector weight_reshape_vec = {-1, weight_shape_vec.back()};
  ShapeVector output_reshape_vec = input_shape_vec;

  if (weight_shape_vec.size() == 1) {
    output_reshape_vec.erase(output_reshape_vec.end() - 1);
  } else {
    output_reshape_vec.back() = weight_shape_vec[0];
  }

  return ShapeArray{input_reshape_vec, weight_reshape_vec, output_reshape_vec};
}

AnfNodePtr GetAddNode(const FuncGraphPtr &func_graph, const mindspore::AnfNodePtr &input,
                      const mindspore::AnfNodePtr &other) {
  auto add_prim = std::make_shared<Primitive>(prim::kPrimAdd->name());
  MS_CHECK_TRUE_MSG(add_prim != nullptr, nullptr, "create Add Primitive failed.");
  add_prim->AddAttr("primitive_function", MakeValue<bool>(true));

  auto add_shape = input->abstract()->GetShape();
  MS_CHECK_TRUE_MSG(add_shape != nullptr, nullptr, "Can't get input shape.");

  auto add_infer_impl = std::make_shared<ops::AddFuncImpl>();
  auto add_type = add_infer_impl->InferType(add_prim, {input->abstract(), other->abstract()});
  auto add_tensor_type = add_type->cast<TensorTypePtr>();
  MS_CHECK_TRUE_MSG(add_tensor_type != nullptr, nullptr, "cast add_tensor_type failed.");

  std::vector<AnfNodePtr> add_inputs = {NewValueNode(add_prim), input, other};
  auto add_node = func_graph->NewCNode(add_inputs);
  MS_CHECK_TRUE_MSG(add_node != nullptr, nullptr, "create Add CNode failed.");
  auto add_abs = std::make_shared<mindspore::abstract::AbstractTensor>(add_tensor_type->element(), add_shape);
  add_node->set_abstract(add_abs);
  add_node->set_scope(input->scope());

  return add_node;
}
}  // namespace

AnfNodePtr ConvertDensePass(const FuncGraphPtr &func_graph, const mindspore::AnfNodePtr &node) {
  auto dense_cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(dense_cnode != nullptr, nullptr);
  MS_CHECK_TRUE_RET(dense_cnode->size() == kInputSizeFour, nullptr);
  if (!CheckPrimitiveType(dense_cnode, prim::kPrimDense)) {
    return nullptr;
  }

  auto input = dense_cnode->input(kIndex1);
  auto input_shape = input->abstract()->GetShape();
  MS_CHECK_TRUE_MSG(input_shape != nullptr, nullptr, "Can't get input shape from Dense.");
  auto weight = dense_cnode->input(kIndex2);
  auto weight_shape = weight->abstract()->GetShape();
  MS_CHECK_TRUE_MSG(weight_shape != nullptr, nullptr, "Can't get weight shape from Dense.");
  MS_CHECK_TRUE_MSG(!input_shape->IsDynamic() && !weight_shape->IsDynamic(), nullptr, "Dense got dynamic shape.");

  auto input_shape_vec = input_shape->GetShapeVector();
  auto weight_shape_vec = weight_shape->GetShapeVector();
  constexpr const size_t kRank2 = 2;
  bool need_reshape = input_shape_vec.size() != kRank2 || weight_shape_vec.size() != kRank2;
  ShapeArray reshape_array;
  if (need_reshape) {
    reshape_array = CalcDenseReshapeVector(input_shape_vec, weight_shape_vec);
    input = GetReshapeNode(func_graph, input, reshape_array[kIndex0]);
    if (weight_shape_vec.size() != kRank2) {
      weight = GetReshapeNode(func_graph, weight, reshape_array[kIndex1]);
    }
  }

  auto output = GetMatMulNode(func_graph, input, weight, false, true);
  MS_CHECK_TRUE_MSG(output != nullptr, nullptr, "Can't get MatMul output from Dense.");

  auto bias = dense_cnode->input(kIndex3);
  if (!IsValueNode<None>(bias)) {
    output = GetAddNode(func_graph, output, bias);
    MS_CHECK_TRUE_MSG(output != nullptr, nullptr, "Can't get Add output from Dense.");
  }

  if (need_reshape) {
    output = GetReshapeNode(func_graph, output, reshape_array[kIndex2]);
    MS_CHECK_TRUE_MSG(output != nullptr, nullptr, "Can't get Reshape output from Dense.");
  }

  return output;
}
}  // namespace mindspore::opt
