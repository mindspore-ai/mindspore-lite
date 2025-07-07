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
#include "tools/converter/import/convert_extend_ops/convert_extend_ops_pass.h"
#include <memory>
#include <set>
#include <unordered_map>
#include <vector>
#include "tools/optimizer/common/gllo_utils.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_d.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_z.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_o.h"

namespace mindspore::opt {
namespace {
constexpr auto kNameSumExtPatternName = "SumExtPatternName";
constexpr auto kNameMatMulExtPatternName = "MatMulExtPatternName";
constexpr auto kNameMaxPatternName = "MaxPatternName";
constexpr auto kNameMinPatternName = "MinPatternName";
constexpr auto kNameDensePatternName = "DensePatternName";
constexpr auto kNameOnesPatternName = "OnesPatternName";
constexpr auto kNameZerosPatternName = "ZerosPatternName";
constexpr auto kNameMulsPatternName = "MulsPatternName";
}  // namespace

VectorRef ConvertExtendOpsPass::DefineSumExtPattern() const {
  auto is_sum_ext = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimSumExt>);
  MS_CHECK_TRUE_RET(is_sum_ext != nullptr, {});
  auto input = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(input != nullptr, {});
  auto axis = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(axis != nullptr, {});
  auto keep_dims = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(keep_dims != nullptr, {});
  auto dtype = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(dtype != nullptr, {});
  VectorRef sum_ext_ref = VectorRef({is_sum_ext, input, axis, keep_dims, dtype});
  return sum_ext_ref;
}

VectorRef ConvertExtendOpsPass::DefineMatMulExtPattern() const {
  auto is_matmul_ext = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMatMulExt>);
  MS_CHECK_TRUE_RET(is_matmul_ext != nullptr, {});
  auto input = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(input != nullptr, {});
  auto other = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(other != nullptr, {});
  VectorRef matmul_ext_ref = VectorRef({is_matmul_ext, input, other});
  return matmul_ext_ref;
}

VectorRef ConvertExtendOpsPass::DefineMaxPattern() const {
  auto is_max = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMax>);
  MS_CHECK_TRUE_RET(is_max != nullptr, {});
  auto input = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(input != nullptr, {});
  VectorRef max_ref = VectorRef({is_max, input});
  return max_ref;
}

VectorRef ConvertExtendOpsPass::DefineMinPattern() const {
  auto is_min = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMin>);
  MS_CHECK_TRUE_RET(is_min != nullptr, {});
  auto input = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(input != nullptr, {});
  VectorRef min_ref = VectorRef({is_min, input});
  return min_ref;
}

VectorRef ConvertExtendOpsPass::DefineDensePattern() const {
  auto is_dense = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimDense>);
  MS_CHECK_TRUE_RET(is_dense != nullptr, {});
  auto input = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(input != nullptr, {});
  auto weight = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(weight != nullptr, {});
  auto bias = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(bias != nullptr, {});
  VectorRef dense_ref = VectorRef({is_dense, input, weight, bias});
  return dense_ref;
}

VectorRef ConvertExtendOpsPass::DefineOnesPattern() const {
  auto is_ones = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimOnes>);
  MS_CHECK_TRUE_RET(is_ones != nullptr, {});
  auto input = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(input != nullptr, {});
  auto dtype = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(dtype != nullptr, {});
  VectorRef ones_ref = VectorRef({is_ones, input, dtype});
  return ones_ref;
}

VectorRef ConvertExtendOpsPass::DefineZerosPattern() const {
  auto is_zeros = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimZeros>);
  MS_CHECK_TRUE_RET(is_zeros != nullptr, {});
  auto input = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(input != nullptr, {});
  auto dtype = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(dtype != nullptr, {});
  VectorRef zeros_ref = VectorRef({is_zeros, input, dtype});
  return zeros_ref;
}

VectorRef ConvertExtendOpsPass::DefineMulsPattern() const {
  auto is_muls = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMuls>);
  MS_CHECK_TRUE_RET(is_muls != nullptr, {});
  auto input = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(input != nullptr, {});
  auto other = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(other != nullptr, {});
  VectorRef muls_ref = VectorRef({is_muls, input, other});
  return muls_ref;
}

std::unordered_map<std::string, VectorRef> ConvertExtendOpsPass::DefinePatterns() const {
  std::unordered_map<std::string, VectorRef> patterns;
  patterns[kNameSumExtPatternName] = DefineSumExtPattern();
  patterns[kNameMatMulExtPatternName] = DefineMatMulExtPattern();
  patterns[kNameMaxPatternName] = DefineMaxPattern();
  patterns[kNameMinPatternName] = DefineMinPattern();
  patterns[kNameDensePatternName] = DefineDensePattern();
  patterns[kNameOnesPatternName] = DefineOnesPattern();
  patterns[kNameZerosPatternName] = DefineZerosPattern();
  patterns[kNameMulsPatternName] = DefineMulsPattern();
  return patterns;
}

using ConvertExtendOpsSubPass = AnfNodePtr (*)(const FuncGraphPtr &, const mindspore::AnfNodePtr &);

AnfNodePtr ConvertExtendOpsPass::Process(const std::string &pattern_name, const mindspore::FuncGraphPtr &func_graph,
                                         const mindspore::AnfNodePtr &node, const mindspore::EquivPtr &equiv) const {
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }

  static std::unordered_map<std::string, ConvertExtendOpsSubPass> sub_pass_map = {
    {kNameSumExtPatternName, ConvertSumExtPass}, {kNameMatMulExtPatternName, ConvertMatMulExtPass},
    {kNameMaxPatternName, ConvertMaxMinPass},    {kNameMinPatternName, ConvertMaxMinPass},
    {kNameDensePatternName, ConvertDensePass},   {kNameOnesPatternName, ConvertOnesPass},
    {kNameZerosPatternName, ConvertZerosPass},   {kNameMulsPatternName, ConvertMulsPass}};

  if (sub_pass_map.find(pattern_name) != sub_pass_map.end()) {
    MS_LOG(INFO) << "The node " << node->fullname_with_scope() << " is matched pattern[" << pattern_name
                 << "] in ConvertExtendOpsPass.";
    return sub_pass_map.at(pattern_name)(func_graph, node);
  }
  return nullptr;
}
}  // namespace mindspore::opt
