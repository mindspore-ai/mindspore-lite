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
#include "tools/optimizer/fusion/gnsnz_pass.h"
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

namespace mindspore::opt {
namespace {
constexpr auto kNameGNSNZPatternForSD15 = "GNSNZPatternForSD15";
constexpr auto kNameGNSNZPatternForSD15WithoutSilu = "GNSNZPatternForSD15WithoutSilu";

constexpr size_t kNumIndex1 = 1;
constexpr size_t kNumIndex2 = 2;
constexpr float kNumEps = 0.00001;

int64_t GetInstanceNormGroups(const AnfNodePtr &instance_norm_node) {
  auto instance_norm_cnode = instance_norm_node->cast<CNodePtr>();
  if (instance_norm_cnode == nullptr) {
    MS_LOG(ERROR) << "instance_norm_cnode is nullptr!";
    return -1;
  }
  if (instance_norm_cnode->inputs().size() <= kNumIndex2) {
    MS_LOG(INFO) << "instance_norm_cnode's input size less than " << kNumIndex2 << ".";
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

static const VectorRef DefineGroupNormSiluPatternForSD15WithoutSilu() {
  // reshape
  auto reshape_1_input_1 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(reshape_1_input_1 != nullptr, {});
  auto reshape_1_input_2 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(reshape_1_input_2 != nullptr, {});
  auto is_reshape_1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimReshape>);
  MS_CHECK_TRUE_RET(is_reshape_1 != nullptr, {});
  auto reshape_1 = VectorRef({is_reshape_1, reshape_1_input_1, reshape_1_input_2});

  // instanceNormlization
  auto instance_normlization_input_2 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(instance_normlization_input_2 != nullptr, {});
  auto instance_normlization_input_3 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(instance_normlization_input_3 != nullptr, {});
  auto is_instance_norm = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimInstanceNorm>);
  MS_CHECK_TRUE_RET(is_instance_norm != nullptr, {});
  auto instance_norm =
    VectorRef({is_instance_norm, reshape_1, instance_normlization_input_2, instance_normlization_input_3});

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

  return add;
}

static const VectorRef DefineGroupNormSiluPatternForSD15() {
  auto add = DefineGroupNormSiluPatternForSD15WithoutSilu();

  // sigmoid
  auto is_sigmoid = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimActivation>);
  MS_CHECK_TRUE_RET(is_sigmoid != nullptr, {});
  auto sigmoid = VectorRef({is_sigmoid, add});

  // mul
  auto is_mul_2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul_2 != nullptr, {});
  auto mul_2 = VectorRef({is_mul_2, add, sigmoid});
  return mul_2;
}

static CNodePtr ReshapeCNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  std::vector<int32_t> shape_1d = {0};
  auto shape_parm_node = BuildIntVecParameterNode(func_graph, shape_1d, node->fullname_with_scope() + "_shape_param");
  MS_CHECK_TRUE_MSG(shape_parm_node != nullptr, nullptr, "create shape_parm_node return nullptr");

  std::vector<AnfNodePtr> op_inputs;
  if (utils::isa<ParameterPtr>(node)) {
    auto reshape_input_1 = node->cast<ParameterPtr>();
    op_inputs = {reshape_input_1, shape_parm_node};
  } else {
    MS_LOG(ERROR) << "Node is not ParameterPtr!";
    return nullptr;
  }

  auto reshape_prim = std::make_shared<ops::Reshape>();
  MS_CHECK_TRUE_MSG(reshape_prim != nullptr, nullptr, "create reshape_prim return nullptr");

  auto reshape_prim_c = reshape_prim->GetPrim();
  MS_CHECK_TRUE_MSG(reshape_prim_c != nullptr, nullptr, "create prim_c return nullptr");

  auto reshape_node = func_graph->NewCNode(reshape_prim_c, op_inputs);
  MS_CHECK_TRUE_MSG(reshape_node != nullptr, nullptr, "create node return nullptr");

  reshape_node->set_fullname_with_scope(node->fullname_with_scope() + "_GNS_reshape");
  if (node->abstract() != nullptr) {
    reshape_node->set_abstract(node->abstract()->Clone());
  }
  auto manager = Manage(func_graph);
  (void)manager->Replace(node, reshape_node);
  MS_LOG(INFO) << "Create reshape node end.";
  return reshape_node;
}

static CNodePtr CreateGroupNormSiluNodeForSD15(const std::string &pattern_name, const FuncGraphPtr &func_graph,
                                               const AnfNodePtr &node, const EquivPtr &equiv, bool use_silu) {
  MS_CHECK_TRUE_RET(node != nullptr, nullptr);
  auto cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(cnode != nullptr, nullptr);
  auto mul_2 = cnode;
  auto add = mul_2;
  if (use_silu) {
    add = mul_2->input(kNumIndex1)->cast<CNodePtr>();
  }
  MS_CHECK_TRUE_RET(add != nullptr, nullptr);

  auto mul_1 = add->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(mul_1 != nullptr, nullptr);

  auto reshape_2 = mul_1->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(reshape_2 != nullptr, nullptr);

  auto instance_normalization = reshape_2->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(instance_normalization != nullptr, nullptr);

  auto reshape_1 = instance_normalization->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(reshape_1 != nullptr, nullptr);

  auto conv = reshape_1->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(conv != nullptr, nullptr);

  // gamma and beta reshape 1D: C
  auto gamma_3D = mul_1->input(kNumIndex2);
  MS_CHECK_TRUE_RET(gamma_3D != nullptr, nullptr);
  auto gamma_1D = ReshapeCNode(func_graph, gamma_3D);
  MS_CHECK_TRUE_RET(gamma_1D != nullptr, nullptr);

  auto beta_3D = add->input(kNumIndex2);
  MS_CHECK_TRUE_RET(beta_3D != nullptr, nullptr);
  auto beta_1D = ReshapeCNode(func_graph, beta_3D);
  MS_CHECK_TRUE_RET(beta_1D != nullptr, nullptr);

  // get instancenorm input2 scale shape
  auto num_groups = GetInstanceNormGroups(instance_normalization);
  if (num_groups == -1) {
    MS_LOG(ERROR) << "Get num_groups failed!";
    return nullptr;
  }
  // 32
  MS_LOG(INFO) << "Num_groups: " << num_groups;

  // create op
  auto groupnorm_silu_prim = std::make_shared<ops::Custom>();
  if (groupnorm_silu_prim == nullptr) {
    MS_LOG(ERROR) << "New GroupNormSilu prim failed!";
    return nullptr;
  }
  std::vector<std::string> input_names = {"x", "gamma", "beta"};
  std::vector<std::string> output_names = {"z"};
  groupnorm_silu_prim->set_type("GNSNZ");
  groupnorm_silu_prim->AddAttr("input_names", api::MakeValue(input_names));
  groupnorm_silu_prim->AddAttr("output_names", api::MakeValue(output_names));
  groupnorm_silu_prim->AddAttr("num_groups", api::MakeValue(static_cast<int32_t>(num_groups)));
  groupnorm_silu_prim->AddAttr("eps", api::MakeValue(kNumEps));
  groupnorm_silu_prim->AddAttr("activate_silu", api::MakeValue(use_silu));
  groupnorm_silu_prim->AddAttr("reg_op_name", api::MakeValue("GNSNZ"));

  auto GNS_prim_c = groupnorm_silu_prim->GetPrim();
  if (GNS_prim_c == nullptr) {
    MS_LOG(ERROR) << "GNS_prim_c is nullptr!";
    return nullptr;
  }
  // 2, 320, 64, 64
  auto groupnorm_silu_cnode = func_graph->NewCNode(GNS_prim_c, {conv, gamma_1D, beta_1D});
  if (groupnorm_silu_cnode == nullptr) {
    MS_LOG(ERROR) << "New cnode failed!";
    return nullptr;
  }

  auto name_suffix = use_silu ? "_gnsnz_sd" : "_gnnz_sd";
  groupnorm_silu_cnode->set_fullname_with_scope(node->fullname_with_scope() + name_suffix);
  if (node->abstract() != nullptr) {
    groupnorm_silu_cnode->set_abstract(node->abstract()->Clone());
  }

  if (groupnorm_silu_cnode == nullptr) {
    MS_LOG(ERROR) << "Groupnorm_silu_cnode failed!";
    return nullptr;
  }
  auto manager = Manage(func_graph);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "Create Manage object failed!";
    return nullptr;
  }
  (void)manager->Replace(cnode, groupnorm_silu_cnode);
  MS_LOG(INFO) << "Create GroupNormSilu NZ success.";
  return groupnorm_silu_cnode;
}

std::unordered_map<std::string, VectorRef> GNSNZPass::DefinePatterns() const {
  MS_LOG(INFO) << "Start define gnsnz fusion patterns.";
  std::unordered_map<std::string, VectorRef> patterns;
  patterns[kNameGNSNZPatternForSD15] = DefineGroupNormSiluPatternForSD15();
  patterns[kNameGNSNZPatternForSD15WithoutSilu] = DefineGroupNormSiluPatternForSD15WithoutSilu();
  return patterns;
}

AnfNodePtr GNSNZPass::Process(const std::string &patten_name, const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                              const EquivPtr &equiv) const {
  MS_CHECK_TRUE_RET(node != nullptr, nullptr);
  MS_LOG(INFO) << "Do groupnorm silu nz fusion, pattern name: " << patten_name << "   " << node->fullname_with_scope();
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    MS_LOG(ERROR) << "Function graph, node or equiv is nullptr!";
    return nullptr;
  }
  if (!utils::isa<CNodePtr>(node)) {
    MS_LOG(ERROR) << "This node is not cnode, node name: " << node->fullname_with_scope();
    return nullptr;
  }
  if (IsMarkedTrainOp(utils::cast<CNodePtr>(node))) {
    MS_LOG(ERROR) << "Node is train op, can not fusion!";
    return nullptr;
  }
  auto manager = Manage(func_graph);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "Manager is nullptr!";
    return nullptr;
  }
  CNodePtr groupnormsilu_node = nullptr;
  if (patten_name == kNameGNSNZPatternForSD15) {
    MS_LOG(INFO) << "Start create GNSNZ node for SD15.";
    groupnormsilu_node = CreateGroupNormSiluNodeForSD15(patten_name, func_graph, node, equiv, true);
  } else if (patten_name == kNameGNSNZPatternForSD15WithoutSilu) {
    MS_LOG(INFO) << "Start create GNSNZ node for SD15.";
    groupnormsilu_node = CreateGroupNormSiluNodeForSD15(patten_name, func_graph, node, equiv, false);
  } else {
    MS_LOG(ERROR) << " Not pattern!";
  }
  if (groupnormsilu_node == nullptr) {
    MS_LOG(ERROR) << "Create GroupNormSiluNodeForSD15 is nullptr!";
    return nullptr;
  }
  MS_LOG(INFO) << "GroupNormSilu node fusion success, fusion node name: " << groupnormsilu_node->fullname_with_scope();
  return groupnormsilu_node;
}

}  // namespace mindspore::opt
