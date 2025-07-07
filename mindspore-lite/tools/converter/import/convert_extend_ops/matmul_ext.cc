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
#include "utils/ms_context.h"
#include "infer/ops_func_impl/matmul_ext.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/converter/import/convert_extend_ops/utils.h"
#include "tools/converter/import/convert_extend_ops/convert_extend_ops_pass.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_b.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"

namespace mindspore::opt {
namespace {
constexpr size_t kMatMulRank = 2;

ShapeVector GetMatMulShapes(const ShapeVector &input_shape_vec, const ShapeVector &other_shape_vec) {
  ShapeVector shape_out;
  int len_diff = std::abs(static_cast<int>(input_shape_vec.size()) - static_cast<int>(other_shape_vec.size()));
  ShapeVector input_shape_padded;
  ShapeVector other_shape_padded;
  if (input_shape_vec.size() < other_shape_vec.size()) {
    input_shape_padded = ShapeVector(len_diff, 1);
    input_shape_padded.insert(input_shape_padded.end(), input_shape_vec.begin(), input_shape_vec.end());
    other_shape_padded = other_shape_vec;
  } else {
    other_shape_padded = ShapeVector(len_diff, 1);
    other_shape_padded.insert(other_shape_padded.end(), other_shape_vec.begin(), other_shape_vec.end());
    input_shape_padded = input_shape_vec;
  }
  int max_len = std::max(static_cast<int>(input_shape_padded.size()) - kIndex2,
                         static_cast<int>(other_shape_padded.size()) - kIndex2);
  for (int i = 0; i < max_len; ++i) {
    int64_t dim1 = i < static_cast<int>(input_shape_padded.size() - kIndex2) ? input_shape_padded[i] : 1;
    int64_t dim2 = i < static_cast<int>(other_shape_padded.size() - kIndex2) ? other_shape_padded[i] : 1;
    shape_out.push_back(std::max(dim1, dim2));
  }
  return shape_out;
}

ShapeVector GetNodeShape(const mindspore::AnfNodePtr &node) {
  auto shape = node->abstract()->GetShape();
  auto shape_vec = shape->GetShapeVector();
  return shape_vec;
}

ShapeVector ReduceTo3D(const ShapeVector &shape) {
  ShapeVector ret;

  int64_t dim0 = 1;
  for (size_t i = 0; i < shape.size() - kDim2; ++i) {
    dim0 *= shape[i];
  }
  ret.push_back(dim0);
  ret.push_back(shape[shape.size() - kDim2]);
  ret.push_back(shape[shape.size() - kDim1]);
  return ret;
}

AnfNodePtr GetExpandNode(const FuncGraphPtr &func_graph, const mindspore::AnfNodePtr &node, const size_t &ndims) {
  auto shape = node->abstract()->GetShape();
  MS_CHECK_TRUE_MSG(shape != nullptr, nullptr, "Can't get node shape.");
  auto shape_vec = shape->GetShapeVector();

  if (shape_vec.size() >= ndims) {
    return node;
  }

  while (shape_vec.size() < ndims) {
    shape_vec.insert(shape_vec.begin(), 1);
  }
  return GetReshapeNode(func_graph, node, shape_vec);
}

AnfNodePtr GetBatchMatMulNode(const FuncGraphPtr &func_graph, const mindspore::AnfNodePtr &input,
                              const mindspore::AnfNodePtr &other, const bool &transpose_a = false,
                              const bool &transpose_b = false) {
  auto bmm_prim = std::make_shared<Primitive>(prim::kPrimBatchMatMul->name());
  MS_CHECK_TRUE_MSG(bmm_prim != nullptr, nullptr, "create BatchMatMul Primitive failed.");
  bmm_prim->AddAttr("primitive_function", MakeValue<bool>(true));
  bmm_prim->AddAttr("transpose_a", MakeValue<bool>(transpose_a));
  bmm_prim->AddAttr("transpose_b", MakeValue<bool>(transpose_b));
  std::vector<AnfNodePtr> bmm_inputs = {NewValueNode(bmm_prim), input, other};
  auto bmm_node = func_graph->NewCNode(bmm_inputs);
  MS_CHECK_TRUE_MSG(bmm_node != nullptr, nullptr, "create BatchMatMul CNode failed.");

  constexpr size_t kBatchMatMulRank = 3;
  auto input_shape_vec = GetNodeShape(input);
  auto other_shape_vec = GetNodeShape(other);
  ShapeVector out_shape_vec(kBatchMatMulRank, abstract::TensorShape::kShapeDimAny);
  out_shape_vec[kIndex0] = input_shape_vec[kIndex0];
  out_shape_vec[kIndex1] = transpose_a ? input_shape_vec[kIndex2] : input_shape_vec[kIndex1];
  out_shape_vec[kIndex2] = transpose_b ? other_shape_vec[kIndex1] : other_shape_vec[kIndex2];

  auto input_type_id = GetSingleNodeOutputTypeId(input);
  MS_CHECK_TRUE_MSG(input_type_id != kTypeUnknown, nullptr, "get input_type_id failed.");
  auto bmm_type_id = input_type_id;

  auto context_ptr = MsContext::GetInstance();
  MS_CHECK_TRUE_MSG(context_ptr != nullptr, nullptr, "get context failed.");
  std::string device_target = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (input_type_id == kNumberTypeInt8 && device_target == kAscendDevice) {
    bmm_type_id = kNumberTypeInt32;
  }

  auto bmm_abs = std::make_shared<mindspore::abstract::AbstractTensor>(TypeIdToType(bmm_type_id), out_shape_vec);
  bmm_node->set_abstract(bmm_abs);

  if (input_type_id != bmm_type_id) {
    return GetCastNode(func_graph, bmm_node, input_type_id);
  }

  return bmm_node;
}
}  // namespace

AnfNodePtr GetMatMulNode(const FuncGraphPtr &func_graph, const mindspore::AnfNodePtr &input,
                         const mindspore::AnfNodePtr &other, const bool &transpose_a = false,
                         const bool &transpose_b = false) {
  auto matmul_prim = std::make_shared<Primitive>(prim::kPrimMatMul->name());
  MS_CHECK_TRUE_MSG(matmul_prim != nullptr, nullptr, "create MatMul Primitive failed.");
  matmul_prim->AddAttr("primitive_function", MakeValue<bool>(true));
  matmul_prim->AddAttr("transpose_a", MakeValue<bool>(transpose_a));
  matmul_prim->AddAttr("transpose_b", MakeValue<bool>(transpose_b));
  std::vector<AnfNodePtr> matmul_inputs = {NewValueNode(matmul_prim), input, other};
  auto matmul_node = func_graph->NewCNode(matmul_inputs);
  MS_CHECK_TRUE_MSG(matmul_node != nullptr, nullptr, "create MatMul CNode failed.");

  auto input_shape_vec = GetNodeShape(input);
  auto other_shape_vec = GetNodeShape(other);
  ShapeVector out_shape_vec(kMatMulRank, abstract::TensorShape::kShapeDimAny);
  out_shape_vec[kIndex0] = transpose_a ? input_shape_vec[kIndex1] : input_shape_vec[kIndex0];
  out_shape_vec[kIndex1] = transpose_b ? other_shape_vec[kIndex0] : other_shape_vec[kIndex1];

  auto input_type_id = GetSingleNodeOutputTypeId(input);
  MS_CHECK_TRUE_MSG(input_type_id != kTypeUnknown, nullptr, "get input_type_id failed.");
  auto matmul_type_id = input_type_id;

  auto context_ptr = MsContext::GetInstance();
  MS_CHECK_TRUE_MSG(context_ptr != nullptr, nullptr, "get context failed.");
  std::string device_target = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (input_type_id == kNumberTypeInt8 && device_target == kAscendDevice) {
    matmul_type_id = kNumberTypeInt32;
  }

  auto matmul_abs = std::make_shared<mindspore::abstract::AbstractTensor>(TypeIdToType(matmul_type_id), out_shape_vec);
  matmul_node->set_abstract(matmul_abs);

  if (input_type_id != matmul_type_id) {
    return GetCastNode(func_graph, matmul_node, input_type_id);
  }

  return matmul_node;
}

AnfNodePtr ConvertMatMulExtPass(const FuncGraphPtr &func_graph, const mindspore::AnfNodePtr &node) {
  auto matmul_ext_cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(matmul_ext_cnode != nullptr, nullptr);
  MS_CHECK_TRUE_RET(matmul_ext_cnode->size() == kInputSizeThree, nullptr);
  if (!CheckPrimitiveType(matmul_ext_cnode, prim::kPrimMatMulExt)) {
    return nullptr;
  }

  auto input = matmul_ext_cnode->input(kInputIndexOne);
  auto other = matmul_ext_cnode->input(kInputIndexTwo);
  auto input_shape = input->abstract()->GetShape();
  MS_CHECK_TRUE_MSG(input_shape != nullptr, nullptr, "Can't get input shape from MatMulExt.");
  auto other_shape = other->abstract()->GetShape();
  MS_CHECK_TRUE_MSG(other_shape != nullptr, nullptr, "Can't get other shape from MatMulExt.");
  MS_CHECK_TRUE_MSG(!input_shape->IsDynamic() && !other_shape->IsDynamic(), nullptr, "MatMulExt got dynamic shape.");

  auto input_shape_vec = input_shape->GetShapeVector();
  auto other_shape_vec = other_shape->GetShapeVector();
  if (input_shape_vec.size() == kMatMulRank && other_shape_vec.size() == kMatMulRank) {
    auto matmul_node = GetMatMulNode(func_graph, input, other);
    MS_CHECK_TRUE_MSG(matmul_node != nullptr, nullptr, "Can't create MatMul node.");
    return matmul_node;
  }

  bool transpose_b = other_shape_vec.size() == 1;
  ShapeVector shape_backbone = GetMatMulShapes(input_shape_vec, other_shape_vec);
  ShapeVector shape_out = ops::InferShapeRem(shape_backbone, input_shape_vec, other_shape_vec, transpose_b);

  input = GetExpandNode(func_graph, input, kMatMulRank);
  MS_CHECK_TRUE_MSG(input != nullptr, nullptr, "Can't get Expand input from MatMulExt.");
  other = GetExpandNode(func_graph, other, kMatMulRank);
  MS_CHECK_TRUE_MSG(other != nullptr, nullptr, "Can't get Expand other from MatMulExt.");

  AnfNodePtr output;
  if (GetNodeShape(other).size() == kMatMulRank) {
    if (GetNodeShape(input).size() > kMatMulRank) {
      int64_t product_dim = 1;
      for (size_t i = 0; i < input_shape_vec.size() - 1; ++i) {
        product_dim *= input_shape_vec[i];
      }
      std::vector<int64_t> reshape_vec = {product_dim, input_shape_vec.back()};
      input = GetReshapeNode(func_graph, input, reshape_vec);
      MS_CHECK_TRUE_MSG(input != nullptr, nullptr, "Can't get Reshape input from MatMulExt.");
    }
    output = GetMatMulNode(func_graph, input, other, false, transpose_b);
    MS_CHECK_TRUE_MSG(output != nullptr, nullptr, "Can't get MatMul input from MatMulExt.");
  } else {
    size_t ndim_aligned = std::max(input_shape_vec.size(), other_shape_vec.size());
    input = GetExpandNode(func_graph, input, ndim_aligned);
    MS_CHECK_TRUE_MSG(input != nullptr, nullptr, "Can't get Expand input from MatMulExt.");
    other = GetExpandNode(func_graph, other, ndim_aligned);
    MS_CHECK_TRUE_MSG(other != nullptr, nullptr, "Can't get Expand other from MatMulExt.");
    ShapeVector input_shape_aligned = GetNodeShape(input);
    ShapeVector other_shape_aligned = GetNodeShape(other);
    const ShapeVector &broadcast_input_shape = ops::GetMatMulExtBroadcastShape(shape_backbone, input_shape_vec);
    const ShapeVector &broadcast_other_shape = ops::GetMatMulExtBroadcastShape(shape_backbone, other_shape_vec);
    if (input_shape_aligned != broadcast_input_shape) {
      input = GetBroadcastToNode(func_graph, input, broadcast_input_shape);
      MS_CHECK_TRUE_MSG(input != nullptr, nullptr, "Can't get BroadcastTo input from MatMulExt.");
    }
    if (other_shape_aligned != broadcast_other_shape) {
      other = GetBroadcastToNode(func_graph, other, broadcast_other_shape);
      MS_CHECK_TRUE_MSG(other != nullptr, nullptr, "Can't get BroadcastTo other from MatMulExt.");
    }
    input = GetReshapeNode(func_graph, input, ReduceTo3D(GetNodeShape(input)));
    MS_CHECK_TRUE_MSG(input != nullptr, nullptr, "Can't get Reshape input from MatMulExt.");
    other = GetReshapeNode(func_graph, other, ReduceTo3D(GetNodeShape(other)));
    MS_CHECK_TRUE_MSG(other != nullptr, nullptr, "Can't get Reshape other from MatMulExt.");
    output = GetBatchMatMulNode(func_graph, input, other, false, transpose_b);
    MS_CHECK_TRUE_MSG(output != nullptr, nullptr, "Can't get BatchMatMul input from MatMulExt.");
  }

  output = GetReshapeNode(func_graph, output, shape_out);
  MS_CHECK_TRUE_MSG(output != nullptr, nullptr, "Can't get Reshape output from MatMulExt.");
  return output;
}
}  // namespace mindspore::opt
