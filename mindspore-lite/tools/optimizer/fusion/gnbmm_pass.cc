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
#include "tools/optimizer/fusion/gnbmm_pass.h"
#include <memory>
#include <utility>
#include "op_def/array_ops.h"
#include "op_def/nn_ops.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "mindspore/ops/op_def/lite_ops.h"
#include "op_def/auto_generate/gen_lite_ops.h"
#include "infer/custom.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_a.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_i.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_r.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_t.h"

namespace mindspore::opt {
namespace {

constexpr auto kNameGNBMMPatternForSDXL = "GNBMMPatternForSDXL";
constexpr size_t kNumIndex1 = 1;
constexpr size_t kNumIndex2 = 2;

int64_t GetInstanceNormGroups(const AnfNodePtr &instance_norm_node) {
  auto instance_norm_cnode = instance_norm_node->cast<CNodePtr>();
  if (instance_norm_cnode == nullptr) {
    MS_LOG(ERROR) << "instance_norm_cnode is nullptr!";
    return -1;
  }
  if (instance_norm_cnode->inputs().size() <= kNumIndex2) {
    MS_LOG(WARNING) << "instance_norm_cnode's inputs size less than " << kNumIndex2 << ".";
    return -1;
  }
  auto instance_norm_input2 = instance_norm_cnode->input(kNumIndex2);
  if (instance_norm_input2 == nullptr) {
    MS_LOG(ERROR) << "instance_norm_input2 is nullptr!";
    return -1;
  }
  auto scale_param = instance_norm_input2->cast<ParameterPtr>();
  if (scale_param == nullptr) {
    MS_LOG(ERROR) << "scale_param is nullptr!";
    return -1;
  }
  auto scale_default_param = scale_param->default_param();
  auto scale_value = std::dynamic_pointer_cast<tensor::Tensor>(scale_default_param);
  if (scale_value == nullptr) {
    MS_LOG(ERROR) << "Scale_value is nullptr!";
    return -1;
  }
  return static_cast<int64_t>(scale_value->ElementsNum());
}
}  // namespace

std::unordered_map<std::string, VectorRef> GNBMMPass::DefinePatterns() const {
  std::unordered_map<std::string, VectorRef> patterns;
  patterns[kNameGNBMMPatternForSDXL] = DefineGNBMMPatternForSDXL();
  return patterns;
}

const VectorRef GNBMMPass::DefineGNBMMPatternForSDXL() const {
  // reshape
  auto reshape_1_input_1 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(reshape_1_input_1 != nullptr, {});
  auto reshape_1_input_2 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(reshape_1_input_2 != nullptr, {});
  auto is_reshape_1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimReshape>);
  MS_CHECK_TRUE_RET(is_reshape_1 != nullptr, {});
  auto reshape_1 = VectorRef({is_reshape_1, reshape_1_input_1, reshape_1_input_2});

  // instanceNormalization
  auto instance_norm_input_2 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(instance_norm_input_2 != nullptr, {});
  auto instance_norm_input_3 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(instance_norm_input_3 != nullptr, {});
  auto is_instance_norm = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimInstanceNorm>);
  MS_CHECK_TRUE_RET(is_instance_norm != nullptr, {});
  auto instance_norm = VectorRef({is_instance_norm, reshape_1, instance_norm_input_2, instance_norm_input_3});

  // reshape
  auto reshape_2_input_2 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(reshape_2_input_2 != nullptr, {});
  auto is_reshape_2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimReshape>);
  MS_CHECK_TRUE_RET(is_reshape_2 != nullptr, {});
  auto reshape_2 = VectorRef({is_reshape_2, instance_norm, reshape_2_input_2});

  // mul
  auto mul_1_input_2 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(mul_1_input_2 != nullptr, {});
  auto is_mul_1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul_1 != nullptr, {});
  auto mul_1 = VectorRef({is_mul_1, reshape_2, mul_1_input_2});

  // add
  auto add_input_2 = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(add_input_2 != nullptr, {});
  auto is_add = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>);
  MS_CHECK_TRUE_RET(is_add != nullptr, {});
  auto add = VectorRef({is_add, mul_1, add_input_2});

  // transpose
  auto transpose_param = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(transpose_param != nullptr, {});
  auto is_transpose = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimTranspose>);
  MS_CHECK_TRUE_RET(is_transpose != nullptr, {});
  auto transpose = VectorRef({is_transpose, add, transpose_param});

  // reshape
  auto reshape_3_input_2 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(reshape_3_input_2 != nullptr, {});
  auto is_reshape_3 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimReshape>);
  MS_CHECK_TRUE_RET(is_reshape_3 != nullptr, {});
  auto reshape_3 = VectorRef({is_reshape_3, transpose, reshape_3_input_2});

  // matmul
  auto matmul_input_2 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(matmul_input_2 != nullptr, {});
  auto is_matmul = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMatMulFusion>);
  MS_CHECK_TRUE_RET(is_matmul != nullptr, {});
  auto matmul = VectorRef({is_matmul, reshape_3, matmul_input_2});

  // add
  auto add_1_input_1 = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(add_1_input_1 != nullptr, {});
  auto is_add_1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>);
  MS_CHECK_TRUE_RET(is_add_1 != nullptr, {});
  auto add_1 = VectorRef({is_add_1, add_1_input_1, matmul});

  return add_1;
}

CNodePtr GNBMMPass::ReshapeCNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node) const {
  std::vector<int32_t> shape_1d = {0};
  auto shape_parm_node = BuildIntVecParameterNode(func_graph, shape_1d, node->fullname_with_scope() + "_shape_param");
  MS_CHECK_TRUE_MSG(shape_parm_node != nullptr, nullptr, "create shape_parm_node return nullptr");

  std::vector<AnfNodePtr> op_inputs;
  if (utils::isa<ParameterPtr>(node)) {
    auto reshape_input_1 = node->cast<ParameterPtr>();
    op_inputs = {reshape_input_1, shape_parm_node};
  } else {
    MS_LOG(ERROR) << "node is not ParameterPtr!";
    return nullptr;
  }

  auto reshape_prim = std::make_shared<ops::Reshape>();
  MS_CHECK_TRUE_MSG(reshape_prim != nullptr, nullptr, "create reshape_prim return nullptr");

  auto reshape_prim_c = reshape_prim->GetPrim();
  MS_CHECK_TRUE_MSG(reshape_prim_c != nullptr, nullptr, "create prim_c return nullptr");

  auto reshape_node = func_graph->NewCNode(reshape_prim_c, op_inputs);
  MS_CHECK_TRUE_MSG(reshape_node != nullptr, nullptr, "create node return nullptr");

  reshape_node->set_fullname_with_scope(node->fullname_with_scope() + "_GNBMM_reshape");
  if (node->abstract() != nullptr) {
    reshape_node->set_abstract(node->abstract()->Clone());
  }

  auto manager = Manage(func_graph);
  (void)manager->Replace(node, reshape_node);
  return reshape_node;
}

CNodePtr GNBMMPass::CreateGNBMMNodeForSDXL(const std::string &pattern_name, const FuncGraphPtr &func_graph,
                                           const AnfNodePtr &node, const EquivPtr &equiv) const {
  MS_CHECK_TRUE_RET(node != nullptr, nullptr);
  auto cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(cnode != nullptr, nullptr);

  auto add_2 = cnode;

  auto matmul = add_2->input(kNumIndex2)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(matmul != nullptr, nullptr);

  auto reshape_3 = matmul->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(reshape_3 != nullptr, nullptr);

  auto transpose = reshape_3->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(transpose != nullptr, nullptr);

  auto add_1 = transpose->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(add_1 != nullptr, nullptr);

  auto mul = add_1->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(mul != nullptr, nullptr);

  auto reshape_2 = mul->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(reshape_2 != nullptr, nullptr);

  auto instance_norm = reshape_2->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(instance_norm != nullptr, nullptr);

  auto reshape_1 = instance_norm->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(reshape_1 != nullptr, nullptr);

  auto subgraph_input = reshape_1->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(subgraph_input != nullptr, nullptr);

  // gamma and beta reshape 1D: C
  auto gamma_3D = mul->input(kNumIndex2);
  MS_CHECK_TRUE_RET(gamma_3D != nullptr, nullptr);
  auto gamma_1D = ReshapeCNode(func_graph, gamma_3D);
  MS_CHECK_TRUE_RET(gamma_1D != nullptr, nullptr);

  auto beta_3D = add_1->input(kNumIndex2);
  MS_CHECK_TRUE_RET(beta_3D != nullptr, nullptr);
  auto beta_1D = ReshapeCNode(func_graph, beta_3D);
  MS_CHECK_TRUE_RET(beta_1D != nullptr, nullptr);

  auto weight = matmul->input(kNumIndex2);
  MS_CHECK_TRUE_RET(weight != nullptr, nullptr);
  auto bias = add_2->input(kNumIndex1);
  MS_CHECK_TRUE_RET(bias != nullptr, nullptr);

  auto instance_norm_prim = GetCNodePrimitive(instance_norm);
  if (instance_norm_prim == nullptr) {
    MS_LOG(ERROR) << "instance_norm_prim is nullptr!";
    return nullptr;
  }
  auto num_groups = GetInstanceNormGroups(instance_norm);
  auto gnbmm_prim = std::make_shared<ops::Custom>();
  if (gnbmm_prim == nullptr) {
    MS_LOG(ERROR) << "new GNBMM prim failed.";
    return nullptr;
  }
  float eps_value = GetValue<float>(instance_norm_prim->GetAttr(kAttrEpsilon));
  MS_CHECK_TRUE_RET(gnbmm_prim != nullptr, nullptr);
  std::vector<std::string> input_names = {"x", "gamma", "beta", "weight", "bias"};
  std::vector<std::string> output_names = {"z"};
  gnbmm_prim->set_type("GNBMM");
  gnbmm_prim->AddAttr("input_names", api::MakeValue(input_names));
  gnbmm_prim->AddAttr("output_names", api::MakeValue(output_names));
  gnbmm_prim->AddAttr("num_groups", api::MakeValue(static_cast<int32_t>(num_groups)));
  gnbmm_prim->AddAttr("eps", api::MakeValue(eps_value));
  gnbmm_prim->AddAttr("reg_op_name", api::MakeValue("GNBMM"));

  auto GNBMM_prim_c = gnbmm_prim->GetPrim();
  if (GNBMM_prim_c == nullptr) {
    MS_LOG(ERROR) << "GNBMM_prim_c is nullptr!";
    return nullptr;
  }

  auto gnbmm_cnode = func_graph->NewCNode(GNBMM_prim_c, {subgraph_input, gamma_1D, beta_1D, weight, bias});
  if (gnbmm_cnode == nullptr) {
    MS_LOG(ERROR) << "new GNBMM cnode failed!";
    return nullptr;
  }

  gnbmm_cnode->set_fullname_with_scope(node->fullname_with_scope() + "_gnbmm_sd");
  if (node->abstract() != nullptr) {
    gnbmm_cnode->set_abstract(node->abstract()->Clone());
  }

  auto manager = Manage(func_graph);
  (void)manager->Replace(cnode, gnbmm_cnode);
  return gnbmm_cnode;
}

AnfNodePtr GNBMMPass::Process(const std::string &patten_name, const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                              const EquivPtr &equiv) const {
  MS_LOG(INFO) << "GNBMM fusion started.";
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    MS_LOG(ERROR) << "function graph, node or equiv is nullptr!";
    return nullptr;
  }

  if (!utils::isa<CNodePtr>(node)) {
    MS_LOG(ERROR) << "this node is not cnode, node name: " << node->fullname_with_scope();
    return nullptr;
  }

  if (IsMarkedTrainOp(utils::cast<CNodePtr>(node))) {
    MS_LOG(ERROR) << "node is train op, can not fusion!";
    return nullptr;
  }

  auto manager = Manage(func_graph);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr!";
    return nullptr;
  }

  CNodePtr gnbmm_node = nullptr;
  if (patten_name == kNameGNBMMPatternForSDXL) {
    gnbmm_node = CreateGNBMMNodeForSDXL(patten_name, func_graph, node, equiv);
  } else {
    MS_LOG(ERROR) << " not pattern matched!";
  }
  if (gnbmm_node == nullptr) {
    MS_LOG(ERROR) << "Create GNBMM is nullptr!";
    return nullptr;
  }
  MS_LOG(INFO) << "GNBMM fusion success.";
  return gnbmm_node;
}

}  // namespace mindspore::opt
