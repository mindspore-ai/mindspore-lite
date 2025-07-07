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
#include "tools/optimizer/fusion/groupnormsilu_fusion.h"
#include <memory>
#include "mindspore/ops/op_def/array_ops.h"
#include "mindspore/ops/op_def/nn_ops.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "mindspore/ops/op_def/lite_ops.h"
#include "mindspore/ops/op_def/auto_generate/gen_lite_ops.h"
#include "infer/custom.h"
#include "infer/group_norm_silu.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_a.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_c.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_g.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_i.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_r.h"

namespace mindspore::opt {
namespace {
constexpr auto kNameGroupNormSiluPatternForSD15 = "GroupNormSiluPatternForSD15";
constexpr auto kNameGroupNormSiluPatternForSDWithCast = "GroupNormSiluPatternForSDWithCast";
constexpr auto kNameGroupNormSiluPatternForSDWithoutSilu = "GroupNormSiluPatternForSDWithoutSilu";
constexpr auto kNameGroupNormSiluPatternForGroupNorm = "GroupNormSiluPatternForGroupNorm";
constexpr size_t kNumIndex1 = 1;
constexpr size_t kNumIndex2 = 2;
constexpr size_t kNumIndex3 = 3;
constexpr size_t kNumIndex4 = 4;
constexpr size_t kNumIndex5 = 5;
constexpr float kNumEps = 0.00001;

std::vector<int64_t> GetTensorShape(CNodePtr cnode, size_t input_index) {
  auto abstract = GetCNodeInputAbstract(cnode, input_index);
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "GetCNodeInputAbstract in GroupNormSilu fusion.";
    return {};
  }
  std::vector<int64_t> shape = {};
  if (FetchShapeFromAbstract(abstract, &shape) != lite::RET_OK) {
    MS_LOG(ERROR) << "FetchShapeFromAbstract failed.";
    return {};
  }
  return shape;
}

int64_t GetInstanceNormGroups(const AnfNodePtr &instance_norm_node) {
  auto instance_norm_cnode = instance_norm_node->cast<CNodePtr>();
  if (instance_norm_cnode == nullptr) {
    MS_LOG(WARNING) << "instance_norm_cnode is nullptr.";
    return -1;
  }
  auto instance_norm_input2 = instance_norm_cnode->input(kNumIndex2);
  if (instance_norm_input2 == nullptr) {
    MS_LOG(WARNING) << "instance_norm_input2 is nullptr.";
    return -1;
  }
  auto scale_param = instance_norm_input2->cast<ParameterPtr>();
  if (scale_param == nullptr) {
    MS_LOG(WARNING) << "scale_param is nullptr.";
    return -1;
  }
  auto scale_default_param = scale_param->default_param();
  auto scale_value = std::dynamic_pointer_cast<tensor::Tensor>(scale_default_param);
  if (scale_value == nullptr) {
    MS_LOG(WARNING) << "scale_value is nullptr.";
    return -1;
  }
  return static_cast<int64_t>(scale_value->ElementsNum());
}
}  // namespace

std::unordered_map<std::string, VectorRef> GroupNormSiluFusion::DefinePatterns() const {
  MS_LOG(INFO) << "start define flash attention fusion patterns.";
  std::unordered_map<std::string, VectorRef> patterns;
  patterns[kNameGroupNormSiluPatternForSD15] = DefineGroupNormSiluPatternForSD15();
  patterns[kNameGroupNormSiluPatternForSDWithCast] = DefineGroupNormSiluPatternForSDWithCast();
  patterns[kNameGroupNormSiluPatternForSDWithoutSilu] = DefineGroupNormSiluPatternForSDWithoutSilu();
  patterns[kNameGroupNormSiluPatternForGroupNorm] = DefineGroupNormSiluPatternForGroupNorm();
  return patterns;
}

const VectorRef GroupNormSiluFusion::DefineGroupNormSiluPatternForSD15() const {
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

const VectorRef GroupNormSiluFusion::DefineGroupNormSiluPatternForSDWithCast() const {
  // cast
  auto cast_1_input = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(cast_1_input != nullptr, {});
  auto is_cast_1_param = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_cast_1_param != nullptr, {});
  auto is_cast_1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimCast>);
  MS_CHECK_TRUE_RET(is_cast_1 != nullptr, {});
  auto cast_1 = VectorRef({is_cast_1, cast_1_input, is_cast_1_param});

  // reshape
  auto reshape_1_input_2 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(reshape_1_input_2 != nullptr, {});
  auto is_reshape_1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimReshape>);
  MS_CHECK_TRUE_RET(is_reshape_1 != nullptr, {});
  auto reshape_1 = VectorRef({is_reshape_1, cast_1, reshape_1_input_2});

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

  // cast
  auto is_cast_2_param = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_cast_2_param != nullptr, {});
  auto is_cast_2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimCast>);
  MS_CHECK_TRUE_RET(is_cast_2 != nullptr, {});
  auto cast_2 = VectorRef({is_cast_2, add, is_cast_2_param});

  // sigmoid
  auto is_sigmoid = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimActivation>);
  MS_CHECK_TRUE_RET(is_sigmoid != nullptr, {});
  auto sigmoid = VectorRef({is_sigmoid, cast_2});

  // mul
  auto is_mul_2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul_2 != nullptr, {});
  auto mul_2 = VectorRef({is_mul_2, cast_2, sigmoid});
  return mul_2;
}

const VectorRef GroupNormSiluFusion::DefineGroupNormSiluPatternForSDWithoutSilu() const {
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

  return add;
}

const VectorRef GroupNormSiluFusion::DefineGroupNormSiluPatternForGroupNorm() const {
  auto groupnorm_input_1 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(groupnorm_input_1 != nullptr, {});
  auto groupnorm_input_2 = std::make_shared<Var>();  // num_groups
  MS_CHECK_TRUE_RET(groupnorm_input_2 != nullptr, {});
  auto groupnorm_input_3 = std::make_shared<Var>();  // gamma
  MS_CHECK_TRUE_RET(groupnorm_input_3 != nullptr, {});
  auto groupnorm_input_4 = std::make_shared<Var>();  // beta
  MS_CHECK_TRUE_RET(groupnorm_input_4 != nullptr, {});
  auto groupnorm_input_5 = std::make_shared<Var>();  // eps
  MS_CHECK_TRUE_RET(groupnorm_input_5 != nullptr, {});
  auto is_groupnorm = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimGroupNorm>);
  MS_CHECK_TRUE_RET(is_groupnorm != nullptr, {});
  auto groupnorm = VectorRef(
    {is_groupnorm, groupnorm_input_1, groupnorm_input_2, groupnorm_input_3, groupnorm_input_4, groupnorm_input_5});

  return groupnorm;
}

CNodePtr GroupNormSiluFusion::ReshapeCNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                           const string &node_name) const {
  // reshape [x, 1, 1] to [x], node_name is unique and used to avoid repeat name
  std::vector<int32_t> shape_1d = {0};
  auto shape_param_node =
    BuildIntVecParameterNode(func_graph, shape_1d, node_name + node->fullname_with_scope() + "_shape_param");
  MS_CHECK_TRUE_MSG(shape_param_node != nullptr, nullptr, "create shape_param_node return nullptr");

  std::vector<AnfNodePtr> op_inputs;
  if (utils::isa<ParameterPtr>(node)) {
    auto reshape_input_1 = node->cast<ParameterPtr>();
    op_inputs = {reshape_input_1, shape_param_node};
  } else {
    MS_LOG(ERROR) << "node is not ParameterPtr.";
    return nullptr;
  }

  auto reshape_prim = std::make_shared<ops::Reshape>();
  MS_CHECK_TRUE_MSG(reshape_prim != nullptr, nullptr, "create reshape_prim return nullptr");
  auto reshape_prim_c = reshape_prim->GetPrim();
  MS_CHECK_TRUE_MSG(reshape_prim_c != nullptr, nullptr, "create prim_c return nullptr");
  auto reshape_node = func_graph->NewCNode(reshape_prim_c, op_inputs);
  MS_CHECK_TRUE_MSG(reshape_node != nullptr, nullptr, "create node return nullptr");

  reshape_node->set_fullname_with_scope(node_name + node->fullname_with_scope() + "_GNS_reshape");
  if (node->abstract() != nullptr) {
    reshape_node->set_abstract(node->abstract()->Clone());
  }
  MS_LOG(INFO) << "GroupNormSiluFusion create reshape node end.";
  return reshape_node;
}

CNodePtr GroupNormSiluFusion::CreateGroupNormSiluNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                      const AnfNodePtr &conv, const CNodePtr &mul, const CNodePtr &add,
                                                      int64_t num_groups, bool activate_silu) const {
  MS_LOG(INFO) << "create GroupNormSilu node";
  auto gamma_3D = mul->input(kNumIndex2);
  MS_CHECK_TRUE_RET(gamma_3D != nullptr, nullptr);
  auto beta_3D = add->input(kNumIndex2);
  MS_CHECK_TRUE_RET(beta_3D != nullptr, nullptr);
  auto gamma_1D = ReshapeCNode(func_graph, gamma_3D, mul->fullname_with_scope());
  MS_CHECK_TRUE_RET(gamma_1D != nullptr, nullptr);
  auto beta_1D = ReshapeCNode(func_graph, beta_3D, add->fullname_with_scope());
  MS_CHECK_TRUE_RET(beta_1D != nullptr, nullptr);

  auto groupnorm_silu_prim = std::make_shared<ops::GroupNormSilu>();
  if (groupnorm_silu_prim == nullptr) {
    MS_LOG(ERROR) << "new GroupNormSilu prim failed.";
    return nullptr;
  }
  groupnorm_silu_prim->AddAttr("num_groups", api::MakeValue(num_groups));
  groupnorm_silu_prim->AddAttr("eps", api::MakeValue(kNumEps));
  groupnorm_silu_prim->AddAttr("activate_silu", api::MakeValue(activate_silu));

  auto GNS_prim_c = groupnorm_silu_prim->GetPrim();
  if (GNS_prim_c == nullptr) {
    MS_LOG(ERROR) << "GNS_prim_c is nullptr.";
    return nullptr;
  }

  auto groupnorm_silu_cnode = func_graph->NewCNode(GNS_prim_c, {conv, gamma_1D, beta_1D});
  if (groupnorm_silu_cnode == nullptr) {
    MS_LOG(ERROR) << "new groupnormsilu cnode failed.";
    return nullptr;
  }
  groupnorm_silu_cnode->set_fullname_with_scope(node->fullname_with_scope() + "_groupnormsilu");
  if (node->abstract() != nullptr) {
    groupnorm_silu_cnode->set_abstract(node->abstract()->Clone());
  }
  return groupnorm_silu_cnode;
}

CNodePtr GroupNormSiluFusion::CreateGroupNormSiluNodeForSD15(const std::string &pattern_name,
                                                             const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                             const EquivPtr &equiv) const {
  MS_LOG(INFO) << "GroupNormSilu for SD15";
  MS_CHECK_TRUE_RET(node != nullptr, nullptr);
  auto cnode = node->cast<CNodePtr>();  // mul_2
  MS_CHECK_TRUE_RET(cnode != nullptr, nullptr);

  auto add = cnode->input(kNumIndex1)->cast<CNodePtr>();
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

  // get instanceNorm input2 scale shape
  auto num_groups = GetInstanceNormGroups(instance_normalization);
  if (num_groups == -1) {
    MS_LOG(ERROR) << "get num_groups failed";
    return nullptr;
  }
  auto conv_output_shape = GetTensorShape(reshape_1, kNumIndex1);
  MS_LOG(INFO) << "num_groups: " << num_groups << ", conv_output_shape: " << conv_output_shape;

  auto groupnorm_silu_cnode = CreateGroupNormSiluNode(func_graph, node, conv, mul_1, add, num_groups, true);
  if (groupnorm_silu_cnode == nullptr) {
    MS_LOG(WARNING) << "create groupnorm_silu_cnode failed";
    return nullptr;
  }

  auto manager = Manage(func_graph);
  (void)manager->Replace(cnode, groupnorm_silu_cnode);
  MS_LOG(INFO) << "create GroupNormSilu for SD15 success.";
  return groupnorm_silu_cnode;
}

CNodePtr GroupNormSiluFusion::CreateGroupNormSiluNodeForSDWithCast(const std::string &pattern_name,
                                                                   const FuncGraphPtr &func_graph,
                                                                   const AnfNodePtr &node,
                                                                   const EquivPtr &equiv) const {
  MS_LOG(INFO) << "GroupNormSilu with cast";
  MS_CHECK_TRUE_RET(node != nullptr, nullptr);
  auto cnode = node->cast<CNodePtr>();  // mul_2
  MS_CHECK_TRUE_RET(cnode != nullptr, nullptr);

  auto cast = cnode->input(1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(cast != nullptr, nullptr);

  auto add = cast->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(add != nullptr, nullptr);

  auto mul_1 = add->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(mul_1 != nullptr, nullptr);

  auto reshape_2 = mul_1->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(reshape_2 != nullptr, nullptr);

  auto instance_normalization = reshape_2->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(instance_normalization != nullptr, nullptr);

  auto reshape_1 = instance_normalization->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(reshape_1 != nullptr, nullptr);

  auto cast_1 = reshape_1->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(cast_1 != nullptr, nullptr);

  auto conv = cast_1->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(conv != nullptr, nullptr);

  // get instanceNorm input2 scale shape
  auto num_groups = GetInstanceNormGroups(instance_normalization);
  if (num_groups == -1) {
    MS_LOG(ERROR) << "get num_groups failed";
    return nullptr;
  }
  auto conv_output_shape = GetTensorShape(reshape_1, kNumIndex1);
  MS_LOG(INFO) << "num_groups: " << num_groups << ", conv_output_shape: " << conv_output_shape;

  auto groupnorm_silu_cnode = CreateGroupNormSiluNode(func_graph, node, conv, mul_1, add, num_groups, true);
  if (groupnorm_silu_cnode == nullptr) {
    MS_LOG(WARNING) << "create groupnorm_silu_cnode failed";
    return nullptr;
  }

  auto manager = Manage(func_graph);
  (void)manager->Replace(cnode, groupnorm_silu_cnode);
  MS_LOG(INFO) << "create GroupNormSilu with cast success.";
  return groupnorm_silu_cnode;
}

CNodePtr GroupNormSiluFusion::CreateGroupNormSiluNodeForSDWithoutSilu(const std::string &pattern_name,
                                                                      const FuncGraphPtr &func_graph,
                                                                      const AnfNodePtr &node,
                                                                      const EquivPtr &equiv) const {
  MS_LOG(INFO) << "GroupNormSilu without silu";
  MS_CHECK_TRUE_RET(node != nullptr, nullptr);
  auto cnode = node->cast<CNodePtr>();  // add
  MS_CHECK_TRUE_RET(cnode->size() >= kInputSizeTwo, nullptr);

  auto mul = cnode->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(mul != nullptr, nullptr);
  MS_CHECK_TRUE_RET(mul->size() >= kInputSizeTwo, nullptr);

  auto reshape_2 = mul->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(reshape_2 != nullptr, nullptr);
  MS_CHECK_TRUE_RET(reshape_2->size() >= kInputSizeTwo, nullptr);

  auto instance_normalization = reshape_2->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(instance_normalization != nullptr, nullptr);
  MS_CHECK_TRUE_RET(instance_normalization->size() >= kInputSizeTwo, nullptr);

  auto reshape_1 = instance_normalization->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(reshape_1 != nullptr, nullptr);
  MS_CHECK_TRUE_RET(reshape_1->size() >= kInputSizeTwo, nullptr);

  auto conv = reshape_1->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(conv != nullptr, nullptr);

  // get instanceNorm input2 scale shape
  auto num_groups = GetInstanceNormGroups(instance_normalization);
  if (num_groups == -1) {
    MS_LOG(ERROR) << "get num_groups failed";
    return nullptr;
  }
  auto conv_output_shape = GetTensorShape(reshape_1, kNumIndex1);
  MS_LOG(INFO) << "num_groups: " << num_groups << ", conv_output_shape: " << conv_output_shape;

  auto groupnorm_silu_cnode = CreateGroupNormSiluNode(func_graph, node, conv, mul, cnode, num_groups, false);
  if (groupnorm_silu_cnode == nullptr) {
    MS_LOG(WARNING) << "create groupnorm_silu_cnode failed";
    return nullptr;
  }

  auto manager = Manage(func_graph);
  (void)manager->Replace(cnode, groupnorm_silu_cnode);
  MS_LOG(INFO) << "create GroupNormSilu with cast success.";
  return groupnorm_silu_cnode;
}

int64_t GetNumGroupsFromGroupNorm(const AnfNodePtr &groupnorm_input_2) {
  ValuePtr value_node = nullptr;
  if (groupnorm_input_2->isa<ValueNode>()) {
    auto value_node_ptr = groupnorm_input_2->cast<ValueNodePtr>();
    MS_CHECK_TRUE_RET(value_node_ptr != nullptr, -1);
    value_node = GetValueNode(value_node_ptr);
    MS_CHECK_TRUE_RET(value_node != nullptr && value_node->isa<Int64Imm>(), -1);
    return GetValue<int64_t>(value_node);
  } else if (groupnorm_input_2->isa<Parameter>()) {
    auto param_node = groupnorm_input_2->cast<ParameterPtr>();
    MS_CHECK_TRUE_RET(param_node != nullptr, -1);
    auto num_groups_param = param_node->default_param();
    MS_CHECK_TRUE_RET(num_groups_param != nullptr, -1);
    auto num_groups_value = std::dynamic_pointer_cast<tensor::Tensor>(num_groups_param);
    MS_CHECK_TRUE_RET(num_groups_value != nullptr, -1);
    if (num_groups_value->ElementsNum() != 1) {
      MS_LOG(WARNING) << "num_groups value elements num is not 1, ElementsNum is: " << num_groups_value->ElementsNum();
      return -1;
    }
    if (num_groups_value->data_type() == kNumberTypeInt64) {
      auto num_groups_data = static_cast<int64_t *>(num_groups_value->data_c());
      if (num_groups_data == nullptr) {
        MS_LOG(WARNING) << "num_groups_data is nullptr.";
        return -1;
      }
      return static_cast<int64_t>(num_groups_data[0]);
    } else {
      MS_LOG(WARNING) << "num_groups data type must be int64, but got " << num_groups_value->data_type();
      return -1;
    }
  } else {
    MS_LOG(WARNING) << "groupnorm_input_2 is not ValueNode or Parameter";
    return -1;
  }
}

float GetEpsFromGroupNorm(const AnfNodePtr &groupnorm_input_5) {
  ValuePtr value_node = nullptr;
  if (groupnorm_input_5->isa<ValueNode>()) {
    auto value_node_ptr = groupnorm_input_5->cast<ValueNodePtr>();
    MS_CHECK_TRUE_RET(value_node_ptr != nullptr, -1);
    value_node = GetValueNode(value_node_ptr);
    MS_CHECK_TRUE_RET(value_node != nullptr && value_node->isa<FP32Imm>(), -1);
    return GetValue<float>(value_node);
  } else if (groupnorm_input_5->isa<Parameter>()) {
    auto param_node = groupnorm_input_5->cast<ParameterPtr>();
    MS_CHECK_TRUE_RET(param_node != nullptr, -1);
    auto eps_param = param_node->default_param();
    MS_CHECK_TRUE_RET(eps_param != nullptr, -1);
    auto eps_value = std::dynamic_pointer_cast<tensor::Tensor>(eps_param);
    MS_CHECK_TRUE_RET(eps_value != nullptr, -1);
    if (eps_value->ElementsNum() != 1) {
      MS_LOG(WARNING) << "eps value elements num is not 1, ElementsNum is: " << eps_value->ElementsNum();
      return -1;
    }
    if (eps_value->data_type() == kNumberTypeFloat) {
      auto eps_data = static_cast<float *>(eps_value->data_c());
      if (eps_data == nullptr) {
        MS_LOG(WARNING) << "eps_data is nullptr.";
        return -1;
      }
      return static_cast<float>(eps_data[0]);
    } else {
      MS_LOG(WARNING) << "eps data type must be float, but got " << eps_value->data_type();
      return -1;
    }
  } else {
    MS_LOG(WARNING) << "groupnorm_input_5 is not ValueNode or Parameter";
    return -1;
  }
}

CNodePtr GroupNormSiluFusion::CreateGroupNormSiluNodeForGroupNorm(const std::string &pattern_name,
                                                                  const FuncGraphPtr &func_graph,
                                                                  const AnfNodePtr &node, const EquivPtr &equiv) const {
  MS_LOG(INFO) << "replace GroupNorm with GroupNormSilu";
  MS_CHECK_TRUE_RET(node != nullptr, nullptr);
  auto cnode = node->cast<CNodePtr>();  // GroupNorm
  MS_CHECK_TRUE_RET(cnode != nullptr, nullptr);
  auto groupnorm_input_1 = cnode->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(groupnorm_input_1 != nullptr, nullptr);
  auto groupnorm_input_2 = cnode->input(kNumIndex2);  // num_groups
  MS_CHECK_TRUE_RET(groupnorm_input_2 != nullptr, nullptr);
  auto groupnorm_input_3 = cnode->input(kNumIndex3);  // gamma
  MS_CHECK_TRUE_RET(groupnorm_input_3 != nullptr, nullptr);
  auto groupnorm_input_4 = cnode->input(kNumIndex4);  // beta
  MS_CHECK_TRUE_RET(groupnorm_input_4 != nullptr, nullptr);
  auto groupnorm_input_5 = cnode->input(kNumIndex5);  // eps
  MS_CHECK_TRUE_RET(groupnorm_input_5 != nullptr, nullptr);

  auto num_groups = GetNumGroupsFromGroupNorm(groupnorm_input_2);
  if (num_groups == -1) {
    MS_LOG(ERROR) << "get num_groups failed!";
    return nullptr;
  }
  auto eps = GetEpsFromGroupNorm(groupnorm_input_5);
  if (eps < 0) {
    MS_LOG(ERROR) << "get eps failed!";
    return nullptr;
  }
  MS_LOG(INFO) << "num_groups: " << num_groups << ", eps: " << eps;

  auto groupnorm_silu_prim = std::make_shared<ops::GroupNormSilu>();
  if (groupnorm_silu_prim == nullptr) {
    MS_LOG(ERROR) << "new GroupNormSilu prim failed!";
    return nullptr;
  }
  groupnorm_silu_prim->AddAttr("num_groups", api::MakeValue(num_groups));
  groupnorm_silu_prim->AddAttr("eps", api::MakeValue(eps));
  groupnorm_silu_prim->AddAttr("activate_silu", api::MakeValue(false));

  auto GNS_prim_c = groupnorm_silu_prim->GetPrim();
  if (GNS_prim_c == nullptr) {
    MS_LOG(ERROR) << "GNS_prim_c is nullptr!";
    return nullptr;
  }
  auto groupnorm_silu_cnode =
    func_graph->NewCNode(GNS_prim_c, {groupnorm_input_1, groupnorm_input_3, groupnorm_input_4});
  if (groupnorm_silu_cnode == nullptr) {
    MS_LOG(ERROR) << "new groupnormsilu cnode failed!";
    return nullptr;
  }
  groupnorm_silu_cnode->set_fullname_with_scope(node->fullname_with_scope() + "_groupnormsilu");
  if (node->abstract() != nullptr) {
    groupnorm_silu_cnode->set_abstract(node->abstract()->Clone());
  }

  auto manager = Manage(func_graph);
  (void)manager->Replace(cnode, groupnorm_silu_cnode);
  MS_LOG(INFO) << "create GroupNormSilu with cast success.";
  return groupnorm_silu_cnode;
}

AnfNodePtr GroupNormSiluFusion::Process(const std::string &patten_name, const FuncGraphPtr &func_graph,
                                        const AnfNodePtr &node, const EquivPtr &equiv) const {
  MS_LOG(INFO) << "do GroupNormSilu fusion, pattern name: " << patten_name;
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    MS_LOG(ERROR) << "function graph, node or equiv is nullptr.";
    return nullptr;
  }
  if (!utils::isa<CNodePtr>(node)) {
    MS_LOG(ERROR) << "this node is not cnode, node name: " << node->fullname_with_scope();
    return nullptr;
  }
  if (IsMarkedTrainOp(utils::cast<CNodePtr>(node))) {
    MS_LOG(ERROR) << "node is train op, can not fusion.";
    return nullptr;
  }
  auto manager = Manage(func_graph);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr.";
    return nullptr;
  }
  CNodePtr groupnormsilu_node = nullptr;
  if (patten_name == kNameGroupNormSiluPatternForSD15) {
    MS_LOG(INFO) << "start create GroupNormSilu node for SD15.";
    groupnormsilu_node = CreateGroupNormSiluNodeForSD15(patten_name, func_graph, node, equiv);
  } else if (patten_name == kNameGroupNormSiluPatternForSDWithCast) {
    MS_LOG(INFO) << "start create GroupNormSilu node for SD15 with cast.";
    groupnormsilu_node = CreateGroupNormSiluNodeForSDWithCast(patten_name, func_graph, node, equiv);
  } else if (patten_name == kNameGroupNormSiluPatternForSDWithoutSilu) {
    MS_LOG(INFO) << "start create GroupNormSilu node for SD15 without silu.";
    groupnormsilu_node = CreateGroupNormSiluNodeForSDWithoutSilu(patten_name, func_graph, node, equiv);
  } else if (patten_name == kNameGroupNormSiluPatternForGroupNorm) {
    MS_LOG(INFO) << "start create GroupNormSilu node to replace GroupNorm.";
    groupnormsilu_node = CreateGroupNormSiluNodeForGroupNorm(patten_name, func_graph, node, equiv);
  } else {
    MS_LOG(ERROR) << " not pattern.";
  }
  if (groupnormsilu_node == nullptr) {
    MS_LOG(INFO) << "GroupNormSilu not fusion.";
    return nullptr;
  }
  MS_LOG(INFO) << "GroupNormSilu node fusion success, fusion node name: " << groupnormsilu_node->fullname_with_scope();
  return groupnormsilu_node;
}
}  // namespace mindspore::opt
