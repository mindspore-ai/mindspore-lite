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
#include "tools/converter/parser/einsum_adjust.h"
#include <string>
#include <vector>
#include <memory>
#include "op_def/auto_generate/gen_lite_ops.h"
#include "ops/primitive_c.h"
#include "infer/cxx_api/scale_fusion.h"
#include "infer/cxx_api/mat_mul_fusion.h"
#include "tools/converter/ops/ops_def.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "src/common/utils.h"
#include "nnacl_c/op_base.h"
#include "infer/unsqueeze.h"

namespace mindspore::lite {
namespace {
constexpr const char *DELIM_COMMA = ",";
constexpr const char *DELIM_ARROW = "->";
constexpr const char *DELIM_BLANK = " ";
constexpr const int kIndex0 = 0;
constexpr const int kIndex1 = 1;
constexpr const int kIndex2 = 2;
constexpr const int kIndex3 = 3;
constexpr const int kIndex4 = 4;
constexpr const size_t kLen1 = 1;
constexpr const size_t kLen2 = 2;
constexpr const size_t kLen3 = 3;
constexpr const size_t kLen4 = 4;

lite::STATUS CheckSubdims(const std::string &first_subdims, const std::string &second_subdims,
                          const std::string &output_subdims) {
  auto min_dim = first_subdims.length() < second_subdims.length() ? first_subdims.length() : second_subdims.length();
  min_dim = min_dim < output_subdims.length() ? min_dim : output_subdims.length();
  auto max_subdims = first_subdims.length() > second_subdims.length() ? first_subdims : second_subdims;
  if (first_subdims.substr(first_subdims.length() - min_dim) !=
      second_subdims.substr(second_subdims.length() - min_dim) ||
      first_subdims.substr(first_subdims.length() - min_dim) !=
      output_subdims.substr(output_subdims.length() - min_dim) ||
      max_subdims.substr(0, 1) != output_subdims.substr(0, 1)) {
    return RET_ERROR;
  }
  return RET_OK;
}

lite::STATUS CheckCanConvertToMatmul(const std::string &first_dims, const std::string &second_dims,
                                     const std::string &output_dims, bool *trans_a, bool *trans_b, bool *trans_out) {
  MS_CHECK_TRUE_RET(trans_a != nullptr && trans_b != nullptr && trans_out != nullptr, RET_NULL_PTR);
  // dimensions other than the last two dimensions and not common dimension from the right should be the same.
  // e.g. "bdn,bdm->bnm"/"bnm,bdm->bdn"/"bhid,bhjd->bhij"/"bhid,hjd->bhij"
  auto first_subdims = first_dims.substr(0, first_dims.length() - DIMENSION_2D);
  auto second_subdims = second_dims.substr(0, second_dims.length() - DIMENSION_2D);
  auto output_subdims = output_dims.substr(0, output_dims.length() - DIMENSION_2D);
  if (CheckSubdims(first_subdims, second_subdims, output_subdims) != RET_OK) {
    MS_LOG(ERROR) << "Check subdim failed!";
    return RET_ERROR;
  }

  std::function<std::string(std::string)> get_reversed_string = [](std::string str) {
    std::reverse(str.begin(), str.end());
    return str;
  };
  std::function<bool(std::string, std::string, std::string)> matched_matmul = [](std::string dim_a, std::string dim_b,
                                                                                 std::string dim_out) -> bool {
    MS_CHECK_TRUE_RET(dim_a.size() >= kLen2 && dim_b.size() >= kLen2 && dim_out.size() >= kLen2, false);
    return dim_a.at(1) == dim_b.at(0) && dim_a.at(0) == dim_out.at(0) && dim_b.at(1) == dim_out.at(1);
  };

  auto first_dim = first_dims.substr(first_dims.length() - DIMENSION_2D);
  auto second_dim = second_dims.substr(second_dims.length() - DIMENSION_2D);
  auto output_dim = output_dims.substr(output_dims.length() - DIMENSION_2D);
  std::vector<bool> trans{false, true};
  for (size_t i = 0; i < trans.size(); i++) {
    *trans_a = trans.at(i);
    auto dim_a = *trans_a ? get_reversed_string(first_dim) : first_dim;
    for (size_t j = 0; j < trans.size(); j++) {
      *trans_b = trans.at(j);
      auto dim_b = *trans_b ? get_reversed_string(second_dim) : second_dim;
      for (size_t k = 0; k < trans.size(); k++) {
        *trans_out = trans.at(k);
        auto dim_out = *trans_out ? get_reversed_string(output_dim) : output_dim;
        if (matched_matmul(dim_a, dim_b, dim_out)) {
          return RET_OK;
        }
      }
    }
  }
  return RET_ERROR;
}

lite::STATUS CreateUnsqueezeNode(AnfNodePtrList &&unsqueeze_inputs, std::vector<int64_t> axis,
                                 const FuncGraphPtr &func_graph, const CNodePtr &cnode, const std::string &suffix,
                                 AnfNodePtr *unsqueeze_param) {
  MS_CHECK_TRUE_RET(cnode != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(func_graph != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(unsqueeze_param != nullptr, RET_ERROR);
  auto unsqueeze_prim = std::make_shared<ops::Unsqueeze>();
  MS_CHECK_TRUE_RET(unsqueeze_prim != nullptr, RET_ERROR);
  auto unsqueeze_prim_c = unsqueeze_prim->GetPrim();
  MS_CHECK_TRUE_RET(unsqueeze_prim_c != nullptr, RET_ERROR);
  unsqueeze_prim_c->AddAttr("axis", MakeValue(axis));
  auto unsqueeze_cnode = func_graph->NewCNode(unsqueeze_prim_c, unsqueeze_inputs);
  if ((unsqueeze_cnode) == nullptr) {
    MS_LOG(ERROR) << "New unsqueeze cnode failed!";
    return RET_ERROR;
  }
  (unsqueeze_cnode)->set_fullname_with_scope(cnode->fullname_with_scope() + suffix);
  if (!utils::isa<AnfNodePtr>(unsqueeze_cnode)) {
    MS_LOG(ERROR) << "unsqueeze cnode is not AnfNodePtr!";
    return RET_ERROR;
  }
  *unsqueeze_param = (unsqueeze_cnode)->cast<AnfNodePtr>();
  if (cnode->abstract() != nullptr) {
    (unsqueeze_cnode)->set_abstract(cnode->abstract()->Clone());
  }
  return RET_OK;
}

lite::STATUS CreateMulNode(AnfNodePtrList &&mul_inputs, const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                           const std::string &suffix, AnfNodePtr *mul_param) {
  MS_CHECK_TRUE_RET(cnode != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(func_graph != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(mul_param != nullptr, RET_ERROR);
  auto mul_prim = std::make_shared<ops::Mul>();
  MS_CHECK_TRUE_RET(mul_prim != nullptr, RET_ERROR);
  auto mul_prim_c = mul_prim->GetPrim();
  MS_CHECK_TRUE_RET(mul_prim_c != nullptr, RET_ERROR);
  auto mul_cnode = func_graph->NewCNode(mul_prim_c, mul_inputs);
  if (mul_cnode == nullptr) {
    MS_LOG(ERROR) << "New mul cnode failed!";
    return RET_ERROR;
  }
  mul_cnode->set_fullname_with_scope(cnode->fullname_with_scope() + suffix);
  if (!utils::isa<AnfNodePtr>(mul_cnode)) {
    MS_LOG(ERROR) << "Mul cnode is not AnfNodePtr!";
    return RET_ERROR;
  }
  *mul_param = mul_cnode->cast<AnfNodePtr>();
  if (cnode->abstract() != nullptr) {
    mul_cnode->set_abstract(cnode->abstract()->Clone());
  }
  return RET_OK;
}

lite::STATUS CreateReduceSumNode(AnfNodePtrList &&reduce_sum_inputs, const FuncGraphPtr &func_graph,
                                 const CNodePtr &cnode, const std::string &suffix, CNodePtr *reduce_sum_cnode) {
  MS_CHECK_TRUE_RET(cnode != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(func_graph != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(reduce_sum_cnode != nullptr, RET_ERROR);
  auto reduce_sum = std::make_shared<ops::ReduceSum>();
  MS_CHECK_TRUE_RET(reduce_sum != nullptr, RET_ERROR);
  auto reduce_sum_prim_c = reduce_sum->GetPrim();
  MS_CHECK_TRUE_RET(reduce_sum_prim_c != nullptr, RET_ERROR);
  *reduce_sum_cnode = func_graph->NewCNode(reduce_sum_prim_c, reduce_sum_inputs);
  if ((*reduce_sum_cnode) == nullptr) {
    MS_LOG(ERROR) << "new reduce sum node failed!";
    return RET_ERROR;
  }
  (*reduce_sum_cnode)->set_fullname_with_scope(cnode->fullname_with_scope() + suffix);
  if (!utils::isa<AnfNodePtr>(*reduce_sum_cnode)) {
    MS_LOG(ERROR) << "matmul weight is not constant, can not update weight!";
    return RET_ERROR;
  }
  if (cnode->abstract() != nullptr) {
    (*reduce_sum_cnode)->set_abstract(cnode->abstract()->Clone());
  }
  return RET_OK;
}

lite::STATUS CreateTransposeNode(AnfNodePtrList &&transpose_inputs, const FuncGraphPtr &func_graph,
                                 const CNodePtr &cnode, const std::string &suffix, CNodePtr *transpose_cnode) {
  MS_CHECK_TRUE_RET(cnode != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(func_graph != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(transpose_cnode != nullptr, RET_ERROR);
  auto transpose = std::make_shared<ops::Transpose>();
  MS_CHECK_TRUE_RET(transpose != nullptr, RET_ERROR);
  auto transpose_prim_c = transpose->GetPrim();
  MS_CHECK_TRUE_RET(transpose_prim_c != nullptr, RET_ERROR);
  *transpose_cnode = func_graph->NewCNode(transpose_prim_c, transpose_inputs);
  if ((*transpose_cnode) == nullptr) {
    MS_LOG(ERROR) << "new reduce sum node failed!";
    return RET_ERROR;
  }
  (*transpose_cnode)->set_fullname_with_scope(cnode->fullname_with_scope() + suffix);
  if (!utils::isa<AnfNodePtr>(*transpose_cnode)) {
    MS_LOG(ERROR) << "matmul weight is not constant, can not update weight!";
    return RET_ERROR;
  }
  if (cnode->abstract() != nullptr) {
    (*transpose_cnode)->set_abstract(cnode->abstract()->Clone());
  }
  return RET_OK;
}

bool CheckCanConvertToMul(const std::string &first_dims, const std::string &second_dims,
                          const std::string &output_dims) {
  // abc,cde->abde
  MS_CHECK_TRUE_RET(first_dims.size() > 0 && second_dims.size() > 0 && output_dims.size() > 0, false);
  return (
    first_dims[first_dims.size() - 1] == second_dims[kIndex0] &&
    (first_dims.substr(kIndex0, first_dims.size() - 1) + second_dims.substr(1, second_dims.size() - 1) == output_dims));
}

bool CheckCanConvertToMulTrans(const std::string &first_dims, const std::string &second_dims,
                               const std::string &output_dims) {
  // aecd,abcd->acbe
  if (first_dims.size() != kLen4 || second_dims.size() != kLen4 || output_dims.size() != kLen4) {
    return false;
  }
  if (first_dims[kIndex0] != second_dims[kIndex0] || first_dims[kIndex1] == second_dims[kIndex1] ||
      first_dims[kIndex2] != second_dims[kIndex2] || first_dims[kIndex3] != second_dims[kIndex3]) {
    return false;
  }
  if (output_dims[kIndex0] != first_dims[kIndex0] || output_dims[kIndex1] != first_dims[kIndex2] ||
      output_dims[kIndex2] != second_dims[kIndex1] || output_dims[kIndex3] != first_dims[kIndex1]) {
    return false;
  }
  return true;
}

bool CheckCanConvertToTransMul(const std::string &first_dims, const std::string &second_dims,
                               const std::string &output_dims) {
  // acbe,aecd->abcd
  if (first_dims.size() != kLen4 || second_dims.size() != kLen4 || output_dims.size() != kLen4) {
    return false;
  }
  if (first_dims[kIndex0] != second_dims[kIndex0] || first_dims[kIndex1] != second_dims[kIndex2] ||
      first_dims[kIndex3] != second_dims[kIndex1]) {
    return false;
  }
  if (output_dims[kIndex0] != first_dims[kIndex0] || output_dims[kIndex1] != first_dims[kIndex2] ||
      output_dims[kIndex2] != first_dims[kIndex1] || output_dims[kIndex3] != second_dims[kIndex3]) {
    return false;
  }
  return true;
}

bool CheckCanConvertToMulReduce(const std::string &first_dims, const std::string &second_dims,
                                const std::string &output_dims) {
  // abcd,cde->abe
  if (first_dims.size() != kLen4 || second_dims.size() != kLen3 || output_dims.size() != kLen3) {
    return false;
  }
  if (first_dims[kIndex2] != second_dims[kIndex0] || first_dims[kIndex3] != second_dims[kIndex1]) {
    return false;
  }
  if (output_dims[kIndex0] != first_dims[kIndex0] || output_dims[kIndex1] != first_dims[kIndex1] ||
      output_dims[kIndex2] != second_dims[kIndex2]) {
    return false;
  }
  return true;
}

int InsertMulNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  // abc,cde->abde
  MS_CHECK_TRUE_RET(cnode != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(func_graph != nullptr, RET_ERROR);
  if (cnode->inputs().size() < kLen3) {
    MS_LOG(ERROR) << "Input size must larger than 2!";
    return RET_ERROR;
  }
  AnfNodePtr input1 = cnode->input(kIndex1);
  AnfNodePtr input2 = cnode->input(kIndex2);
  std::vector<int64_t> axis_1 = {kIndex3, kIndex4};
  std::vector<int64_t> axis_2 = {kIndex0, kIndex1};
  AnfNodePtr axes_param =
    opt::BuildIntValueParameterNode(func_graph, kIndex2, cnode->fullname_with_scope() + "_reduce_sum_axes", true);
  MS_CHECK_TRUE_MSG(axes_param != nullptr, RET_NULL_PTR, "axes_param is nullptr!");
  auto unsqueeze1_input = {input1};
  AnfNodePtr unsqueeze_param1 = nullptr;
  auto ret = CreateUnsqueezeNode(unsqueeze1_input, axis_1, func_graph, cnode, "_unsqueeze_1", &unsqueeze_param1);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Create unsqueeze node failed!";
    return ret;
  }
  auto unsqueeze2_input = {input2};
  AnfNodePtr unsqueeze_param2 = nullptr;
  ret = CreateUnsqueezeNode(unsqueeze2_input, axis_2, func_graph, cnode, "_unsqueeze_2", &unsqueeze_param2);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Create unsqueeze node failed!";
    return ret;
  }
  auto mul_input = {unsqueeze_param1, unsqueeze_param2};
  AnfNodePtr mul_param = nullptr;
  ret = CreateMulNode(mul_input, func_graph, cnode, "_mul", &mul_param);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Create mul node failed!";
    return ret;
  }
  auto reducesum_input = {mul_param, axes_param};
  CNodePtr reduce_sum_cnode = nullptr;
  ret = CreateReduceSumNode(reducesum_input, func_graph, cnode, "_reduce_sum", &reduce_sum_cnode);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Create reducesum node failed!";
    return ret;
  }
  auto manager = Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr!";
    return lite::RET_ERROR;
  }

  if (!manager->Replace(cnode, reduce_sum_cnode)) {
    MS_LOG(ERROR) << "Replace node failed!";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

int InsertMulNodeTrans(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  // aecd,abcd->acbe
  MS_CHECK_TRUE_RET(cnode != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(func_graph != nullptr, RET_ERROR);
  if (cnode->inputs().size() < kLen3) {
    MS_LOG(ERROR) << "Input size must larger than 2!";
    return RET_ERROR;
  }
  AnfNodePtr input1 = cnode->input(kIndex1);
  AnfNodePtr input2 = cnode->input(kIndex2);
  std::vector<int64_t> axis_1 = {kIndex1};
  std::vector<int64_t> axis_2 = {kIndex2};
  std::vector<int32_t> perm = {kIndex0, kIndex3, kIndex1, kIndex2};
  AnfNodePtr perm_param = opt::BuildIntVecParameterNode(func_graph, perm, cnode->fullname_with_scope() + "_perm");
  MS_CHECK_TRUE_MSG(perm_param != nullptr, RET_NULL_PTR, "perm_param is nullptr!");
  AnfNodePtr axes_param =
    opt::BuildIntValueParameterNode(func_graph, kIndex4, cnode->fullname_with_scope() + "_reduce_sum_axes", true);
  MS_CHECK_TRUE_MSG(axes_param != nullptr, RET_NULL_PTR, "axes_param is nullptr!");
  auto unsqueeze1_input = {input1};
  AnfNodePtr unsqueeze_param1 = nullptr;
  auto ret = CreateUnsqueezeNode(unsqueeze1_input, axis_1, func_graph, cnode, "_unsqueeze_1", &unsqueeze_param1);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Create unsqueeze node failed!";
    return ret;
  }
  auto unsqueeze2_input = {input2};
  AnfNodePtr unsqueeze_param2 = nullptr;
  ret = CreateUnsqueezeNode(unsqueeze2_input, axis_2, func_graph, cnode, "_unsqueeze_2", &unsqueeze_param2);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Create unsqueeze node failed!";
    return ret;
  }
  auto mul_input = {unsqueeze_param1, unsqueeze_param2};
  AnfNodePtr mul_param = nullptr;
  ret = CreateMulNode(mul_input, func_graph, cnode, "_mul", &mul_param);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Create mul node failed!";
    return ret;
  }
  auto reducesum_input = {mul_param, axes_param};
  CNodePtr reduce_sum_cnode = nullptr;
  ret = CreateReduceSumNode(reducesum_input, func_graph, cnode, "_reduce_sum", &reduce_sum_cnode);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Create reducesum node failed!";
    return ret;
  }
  if (!utils::isa<AnfNodePtr>(reduce_sum_cnode)) {
    MS_LOG(ERROR) << "unsqueeze cnode is not AnfNodePtr!";
    return RET_ERROR;
  }
  auto reduce_sum_param = (reduce_sum_cnode)->cast<AnfNodePtr>();
  auto transpose_input = {reduce_sum_param, perm_param};
  CNodePtr transpose_cnode = nullptr;
  ret = CreateTransposeNode(transpose_input, func_graph, cnode, "_transpose", &transpose_cnode);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Create transpose node failed!";
    return ret;
  }
  auto manager = Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr!";
    return lite::RET_ERROR;
  }
  if (!manager->Replace(cnode, transpose_cnode)) {
    MS_LOG(ERROR) << "Replace node failed!";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

int InsertTransMulNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  // acbe,aecd->abcd
  MS_CHECK_TRUE_RET(cnode != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(func_graph != nullptr, RET_ERROR);
  if (cnode->inputs().size() < kLen3) {
    MS_LOG(ERROR) << "Input size must larger than 2!";
    return RET_ERROR;
  }
  AnfNodePtr input1 = cnode->input(kIndex1);
  AnfNodePtr input2 = cnode->input(kIndex2);

  std::vector<int64_t> axis_1 = {kIndex3};
  std::vector<int64_t> axis_2 = {kIndex1};
  std::vector<int32_t> perm1 = {kIndex0, kIndex2, kIndex1, kIndex3};
  std::vector<int32_t> perm2 = {kIndex0, kIndex2, kIndex3, kIndex1};
  AnfNodePtr perm1_param = opt::BuildIntVecParameterNode(func_graph, perm1, cnode->fullname_with_scope() + "_perm1");
  MS_CHECK_TRUE_MSG(perm1_param != nullptr, RET_NULL_PTR, "perm_param is nullptr!");
  AnfNodePtr perm2_param = opt::BuildIntVecParameterNode(func_graph, perm2, cnode->fullname_with_scope() + "_perm2");
  MS_CHECK_TRUE_MSG(perm2_param != nullptr, RET_NULL_PTR, "perm_param is nullptr!");
  AnfNodePtr axes_param =
    opt::BuildIntValueParameterNode(func_graph, kIndex4, cnode->fullname_with_scope() + "_reduce_sum_axes", true);
  MS_CHECK_TRUE_MSG(axes_param != nullptr, RET_NULL_PTR, "axes_param is nullptr!");

  auto transpose1_input = {input1, perm1_param};
  CNodePtr transpose1_cnode = nullptr;
  auto ret = CreateTransposeNode(transpose1_input, func_graph, cnode, "_transpose_1", &transpose1_cnode);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Create unsqueeze node failed!";
    return ret;
  }
  if (!utils::isa<AnfNodePtr>(transpose1_cnode)) {
    MS_LOG(ERROR) << "transpose cnode is not AnfNodePtr!";
    return RET_ERROR;
  }
  auto transpose1_param = (transpose1_cnode)->cast<AnfNodePtr>();

  auto transpose2_input = {input2, perm2_param};
  CNodePtr transpose2_cnode = nullptr;
  ret = CreateTransposeNode(transpose2_input, func_graph, cnode, "_transpose_2", &transpose2_cnode);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Create unsqueeze node failed!";
    return ret;
  }
  if (!utils::isa<AnfNodePtr>(transpose2_cnode)) {
    MS_LOG(ERROR) << "transpose cnode is not AnfNodePtr!";
    return RET_ERROR;
  }
  auto transpose2_param = (transpose2_cnode)->cast<AnfNodePtr>();
  auto unsqueeze1_input = {transpose1_param};
  AnfNodePtr unsqueeze_param1 = nullptr;
  ret = CreateUnsqueezeNode(unsqueeze1_input, axis_1, func_graph, cnode, "_unsqueeze_1", &unsqueeze_param1);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Create unsqueeze node failed!";
    return ret;
  }
  auto unsqueeze2_input = {transpose2_param};
  AnfNodePtr unsqueeze_param2 = nullptr;
  ret = CreateUnsqueezeNode(unsqueeze2_input, axis_2, func_graph, cnode, "_unsqueeze_2", &unsqueeze_param2);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Create unsqueeze node failed!";
    return ret;
  }
  auto mul_input = {unsqueeze_param1, unsqueeze_param2};
  AnfNodePtr mul_param = nullptr;
  ret = CreateMulNode(mul_input, func_graph, cnode, "_mul", &mul_param);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Create mul node failed!";
    return ret;
  }
  auto reducesum_input = {mul_param, axes_param};
  CNodePtr reduce_sum_cnode = nullptr;
  ret = CreateReduceSumNode(reducesum_input, func_graph, cnode, "_reduce_sum", &reduce_sum_cnode);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Create reducesum node failed!";
    return ret;
  }
  auto manager = Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr!";
    return lite::RET_ERROR;
  }
  if (!manager->Replace(cnode, reduce_sum_cnode)) {
    MS_LOG(ERROR) << "Replace node failed!";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

int InsertMulReduceNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  // abcd,cde->abe
  MS_CHECK_TRUE_RET(cnode != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(func_graph != nullptr, RET_ERROR);
  if (cnode->inputs().size() < kLen3) {
    MS_LOG(ERROR) << "Input size must larger than 2!";
    return RET_ERROR;
  }
  AnfNodePtr input1 = cnode->input(kIndex1);
  AnfNodePtr input2 = cnode->input(kIndex2);

  std::vector<int64_t> axis_1 = {kIndex4};
  std::vector<int64_t> axis_2 = {kIndex0, kIndex1};
  AnfNodePtr axes_param =
    opt::BuildIntValueParameterNode(func_graph, kIndex2, cnode->fullname_with_scope() + "_reduce_sum_axes", true);
  MS_CHECK_TRUE_MSG(axes_param != nullptr, RET_NULL_PTR, "axes_param is nullptr!");

  auto unsqueeze1_input = {input1};
  AnfNodePtr unsqueeze_param1 = nullptr;
  auto ret = CreateUnsqueezeNode(unsqueeze1_input, axis_1, func_graph, cnode, "_unsqueeze_1", &unsqueeze_param1);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Create unsqueeze node failed!";
    return ret;
  }
  auto unsqueeze2_input = {input2};
  AnfNodePtr unsqueeze_param2 = nullptr;
  ret = CreateUnsqueezeNode(unsqueeze2_input, axis_2, func_graph, cnode, "_unsqueeze_2", &unsqueeze_param2);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Create unsqueeze node failed!";
    return ret;
  }
  auto mul_input = {unsqueeze_param1, unsqueeze_param2};
  AnfNodePtr mul_param = nullptr;
  ret = CreateMulNode(mul_input, func_graph, cnode, "_mul", &mul_param);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Create mul node failed!";
    return ret;
  }
  auto reducesum1_input = {mul_param, axes_param};
  CNodePtr reduce_sum1_cnode = nullptr;
  ret = CreateReduceSumNode(reducesum1_input, func_graph, cnode, "_reduce_sum1", &reduce_sum1_cnode);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Create reducesum node failed!";
    return ret;
  }
  if (!utils::isa<AnfNodePtr>(reduce_sum1_cnode)) {
    MS_LOG(ERROR) << "reduce sum cnode is not AnfNodePtr!";
    return RET_ERROR;
  }
  auto reducesum1_param = reduce_sum1_cnode->cast<AnfNodePtr>();

  auto reducesum2_input = {reducesum1_param, axes_param};
  CNodePtr reduce_sum2_cnode = nullptr;
  ret = CreateReduceSumNode(reducesum2_input, func_graph, cnode, "_reduce_sum1", &reduce_sum2_cnode);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Create reducesum node failed!";
    return ret;
  }
  auto manager = Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr!";
    return lite::RET_ERROR;
  }
  if (!manager->Replace(cnode, reduce_sum2_cnode)) {
    MS_LOG(ERROR) << "Replace node failed!";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

lite::STATUS InsertReshapeMulNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  // ops.outer is created by reshape and mul
  MS_CHECK_TRUE_RET(cnode != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(func_graph != nullptr, RET_ERROR);
  if (cnode->inputs().size() != kLen3) {
    MS_LOG(ERROR) << "Input size must be 3!";
    return RET_ERROR;
  }
  AnfNodePtr input1 = cnode->input(kIndex1);
  AnfNodePtr input2 = cnode->input(kIndex2);

  std::vector<int32_t> shape = {-1, 1};
  auto reshape_node = opt::GenReshapeNode(func_graph, input1, shape, cnode->fullname_with_scope() + "_reshape");
  if (reshape_node == nullptr) {
    MS_LOG(ERROR) << "Create outer node failed!";
    return RET_ERROR;
  }
  if (cnode->abstract() != nullptr) {
    reshape_node->set_abstract(cnode->abstract()->Clone());
  }

  auto mul_node = opt::CreateMulNode(func_graph, reshape_node, input2);
  if (mul_node == nullptr) {
    MS_LOG(ERROR) << "Create mul node failed!";
    return RET_ERROR;
  }
  auto manager = Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr!";
    return lite::RET_ERROR;
  }
  if (!manager->Replace(cnode, mul_node)) {
    MS_LOG(ERROR) << "Replace node failed!";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

int CheckAndConvertEinsum(const FuncGraphPtr &func_graph, const CNodePtr &cnode, const std::string &first_dims,
                          const std::string &second_dims, const std::string &output_dims) {
  if (CheckCanConvertToMul(first_dims, second_dims, output_dims)) {
    return InsertMulNode(func_graph, cnode);
  } else if (CheckCanConvertToMulTrans(first_dims, second_dims, output_dims)) {
    return InsertMulNodeTrans(func_graph, cnode);
  } else if (CheckCanConvertToTransMul(first_dims, second_dims, output_dims)) {
    return InsertTransMulNode(func_graph, cnode);
  } else if (CheckCanConvertToMulReduce(first_dims, second_dims, output_dims)) {
    return InsertMulReduceNode(func_graph, cnode);
  }
  return lite::RET_ERROR;
}
}  // namespace

bool EinsumAdjust::Adjust(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, false);
  auto cnodes = func_graph->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    if (!opt::CheckPrimitiveType(cnode, std::make_shared<Primitive>(lite::kNameEinsum))) {
      continue;
    }
    // get the second input node whose output is the padding parameter of pad.
    auto src_prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    MS_CHECK_TRUE_RET(src_prim != nullptr, false);
    auto equation_value = src_prim->GetAttr("equation");
    MS_CHECK_TRUE_RET(equation_value != nullptr, false);
    auto equation = GetValue<std::string>(equation_value);
    MS_CHECK_TRUE_RET(!equation.empty(), false);
    size_t index = 0;
    while ((index = equation.find(DELIM_BLANK, index)) != std::string::npos) {
      (void)equation.erase(index, 1);
    }
    auto in_out_dims = StrSplit(equation, DELIM_ARROW);
    MS_CHECK_TRUE_MSG(in_out_dims.size() == DIMENSION_2D, false, "The equation of einsum must have input and output!");
    auto inputs = StrSplit(in_out_dims.front(), DELIM_COMMA);
    MS_CHECK_TRUE_MSG(inputs.size() == DIMENSION_2D, false, "Only einsum with two inputs is supported!");
    auto first_dims = inputs.front();
    auto second_dims = inputs.at(kIndex1);
    auto output_dims = in_out_dims.at(kIndex1);
    MS_CHECK_TRUE_RET(!first_dims.empty() && !second_dims.empty() && !output_dims.empty(), false);
    // check can convert to scale. e.g. "bdn,d->bdn"
    if (output_dims == first_dims && first_dims.find(second_dims) != std::string::npos) {
      auto value_node = cnode->input(kIndex0)->cast<ValueNodePtr>();
      MS_CHECK_TRUE_RET(value_node != nullptr, false);
      ops::ScaleFusion scale_node;
      auto scale_prim = scale_node.GetPrim();
      MS_CHECK_TRUE_MSG(scale_prim != nullptr, RET_NULL_PTR, "dst_prim is nullptr.");
      auto axis = first_dims.find(second_dims);
      scale_node.set_axis(static_cast<int64_t>(axis));
      value_node->set_value(scale_prim);
      continue;
    }
    // check can convert to outer. e.g. "i,j->ij"
    if (output_dims.length() == kLen2 && output_dims == first_dims + second_dims) {
      // outer is implemented by reshape and mul in MindSpore
      if (InsertReshapeMulNode(func_graph, cnode) == RET_OK) {
        continue;
      } else {
        MS_LOG(ERROR) << "Convert einsum to outer failed!";
        return false;
      }
    }
    // convert to matmul
    bool trans_a = false;
    bool trans_b = false;
    bool trans_out = false;
    if (CheckCanConvertToMatmul(first_dims, second_dims, output_dims, &trans_a, &trans_b, &trans_out) == RET_OK) {
      auto value_node = cnode->input(kIndex0)->cast<ValueNodePtr>();
      MS_CHECK_TRUE_RET(value_node != nullptr, false);
      ops::MatMulFusion matmul_node;
      // ops::Mul matmul_node;
      auto matmul_prim = matmul_node.GetPrim();
      MS_CHECK_TRUE_MSG(matmul_prim != nullptr, RET_NULL_PTR, "dst_prim is nullptr.");
      matmul_node.set_transpose_a(trans_a);
      matmul_node.set_transpose_b(trans_b);
      value_node->set_value(matmul_prim);
      if (trans_out) {
        std::vector<int> perm(output_dims.size());
        std::iota(perm.begin(), perm.end(), 0);
        std::reverse(perm.end() - DIMENSION_2D, perm.end());
        auto transpose = opt::GenTransposeNode(func_graph, cnode, perm, cnode->fullname_with_scope() + "_transpose");
        MS_CHECK_TRUE_MSG(transpose != nullptr, false, "create transpose failed!");
        auto manager = Manage(func_graph, true);
        MS_CHECK_TRUE_MSG(manager != nullptr, false, "manager is nullptr!");
        if (!manager->Replace(cnode, transpose)) {
          MS_LOG(ERROR) << "Replace node failed!";
          return false;
        }
      }
      continue;
    }
    // convert to other operations
    if (CheckAndConvertEinsum(func_graph, cnode, first_dims, second_dims, output_dims) == RET_OK) {
      continue;
    }
    MS_LOG(ERROR) << "Convert einsum failed!";
    return false;
  }
  return true;
}
}  // namespace mindspore::lite
