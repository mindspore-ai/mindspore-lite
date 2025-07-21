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
#include "tools/optimizer/fusion/ffn_custom_pass.h"
#include <memory>
#include <utility>
#include <string>
#include <climits>
#include "tools/optimizer/common/gllo_utils.h"
#include "mindspore/ops/op_def/lite_ops.h"
#include "mindspore/ops/infer/custom.h"
#include "nnacl_c/op_base.h"
#include "tools/common/string_util.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_a.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_d.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_e.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_g.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"

namespace mindspore::opt {
namespace {
constexpr auto kNameFFNNameConf = "FFNCust";
constexpr auto kNameFFNPatternForSD = "FFNCustPatternForSD";
constexpr auto kNameFFNPatternForSDConst = "FFNCustPatternForSDConst";

constexpr size_t kNumIndex0 = 0;
constexpr size_t kNumIndex1 = 1;
constexpr size_t kNumIndex2 = 2;
constexpr size_t kNumIndex3 = 3;

}  // namespace

static std::vector<int64_t> GetTensorShape(const CNodePtr &cnode, size_t input_index) {
  auto abstract = GetCNodeInputAbstract(cnode, input_index);
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "GetCNodeInputAbstract failed in FFN CUST!";
    return {};
  }
  std::vector<int64_t> shape = {};
  if (FetchShapeFromAbstract(abstract, &shape) != lite::RET_OK) {
    MS_LOG(ERROR) << "FetchShapeFromAbstract failed!";
    return {};
  }
  return shape;
}

static std::vector<int64_t> GetTensorShapeParam(const AnfNodePtr &cnode) {
  auto parameter = cnode->cast<ParameterPtr>();
  auto abstract = parameter->abstract();
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "GetCNodeInputAbstract failed in FFN CUST!";
    return {};
  }
  std::vector<int64_t> shape = {};
  if (FetchShapeFromAbstract(abstract, &shape) != lite::RET_OK) {
    MS_LOG(ERROR) << "FetchShapeFromAbstract failed!";
    return {};
  }
  return shape;
}

bool FFNCustomPass::CheckInputShpae(const CNodePtr &input_x, const AnfNodePtr &weight1,
                                    const AnfNodePtr &weight2) const {
  if (op_attrs_map_.find(kNameFFNNameConf) != op_attrs_map_.end()) {
    auto input_shape = GetTensorShape(input_x, kNumIndex1);
    auto weight1_shape = GetTensorShapeParam(weight1);
    auto weight2_shape = GetTensorShapeParam(weight2);
    MS_LOG(INFO) << "input shape:" << input_shape << " weight1 shape:" << weight1_shape
                 << " weight2_shape:" << weight2_shape;
    if (input_shape.size() != kNumIndex3 || weight1_shape.size() != kNumIndex2 || weight2_shape.size() != kNumIndex2) {
      MS_LOG(ERROR) << "input shapes is not correct!";
      return false;
    }
    auto attr_map = op_attrs_map_.at("FFNCust");
    for (const auto &attr : attr_map) {
      auto shape_vec = mindspore::lite::SplitStringToVector(attr.second, ",");
      if (shape_vec.size() != kNumIndex2) {
        continue;
      }
      auto a = std::stoi(shape_vec[kNumIndex0]);
      auto b = std::stoi(shape_vec[kNumIndex1]);
      auto a_1 = INT_MAX;
      auto b_1 = INT_MAX;
      if (attr.first == "x_thresh") {
        a_1 = input_shape[kNumIndex1];
        b_1 = input_shape[kNumIndex2];
      }
      if (attr.first == "w1_thresh") {
        a_1 = weight1_shape[kNumIndex0];
        b_1 = weight1_shape[kNumIndex1];
      }
      if (attr.first == "w2_thresh") {
        a_1 = weight2_shape[kNumIndex0];
        b_1 = weight2_shape[kNumIndex1];
      }
      if (a > a_1 || b > b_1) {
        MS_LOG(WARNING) << attr.first << " shapes: [" << a_1 << " " << b_1 << "] is not match threshold [" << a << " "
                        << b << "].";
        return false;
      }
    }
  }
  return true;
}

static const VectorRef DefineFFNPatterbForSD() {
  // reshape
  MS_LOG(INFO) << "Start define FFN fusion pattern for dynamic.";
  const size_t param_num = 8;
  std::vector<CondVarPtr> params(param_num);
  for (size_t i = 0; i < params.size(); ++i) {
    params[i] = std::make_shared<CondVar>(IsParamNode);
    MS_CHECK_TRUE_RET(params[i] != nullptr, {});
  }
  size_t index = 0;
  auto input_x = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(input_x != nullptr, {});
  auto matmul1 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(matmul1 != nullptr, {});
  auto is_matmul1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMatMulFusion>);
  MS_CHECK_TRUE_RET(is_matmul1 != nullptr, {});
  VectorRef matmul1_ref({is_matmul1, input_x, matmul1});

  auto add1 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(add1 != nullptr, {});
  auto is_add1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>);
  MS_CHECK_TRUE_RET(is_add1 != nullptr, {});
  VectorRef add1_ref({is_add1, add1, matmul1_ref});

  auto is_shape = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimShape>);
  MS_CHECK_TRUE_RET(is_shape != nullptr, {});
  VectorRef shape_ref({is_shape, add1_ref});

  auto gather = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(gather != nullptr, {});
  auto is_gather = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimGather>);
  MS_CHECK_TRUE_RET(is_gather != nullptr, {});
  VectorRef gather_ref({is_gather, shape_ref, gather, params[index++]});

  auto add2 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(add2 != nullptr, {});
  auto is_add2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>);
  MS_CHECK_TRUE_RET(is_add2 != nullptr, {});
  VectorRef add2_ref({is_add2, gather_ref, add2});

  auto div1 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(div1 != nullptr, {});
  auto is_div1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimDivFusion>);
  MS_CHECK_TRUE_RET(is_div1 != nullptr, {});
  VectorRef div1_ref({is_div1, add2_ref, div1});

  auto mul1 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(mul1 != nullptr, {});
  auto is_mul1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul1 != nullptr, {});
  VectorRef mul1_ref({is_mul1, div1_ref, mul1});

  auto is_stridedslice1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimStridedSlice>);
  MS_CHECK_TRUE_RET(is_stridedslice1 != nullptr, {});
  VectorRef stridedslice1_ref(
    {is_stridedslice1, add1_ref, params[index++], mul1_ref, params[index++], params[index++]});

  auto mul2 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(mul2 != nullptr, {});
  auto is_mul2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul2 != nullptr, {});
  VectorRef mul2_ref({is_mul2, div1_ref, mul2});

  auto is_stridedslice2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimStridedSlice>);
  MS_CHECK_TRUE_RET(is_stridedslice2 != nullptr, {});
  VectorRef stridedslice2_ref({is_stridedslice2, add1_ref, mul1_ref, mul2_ref, params[index++], params[index++]});

  auto div2 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(div2 != nullptr, {});
  auto is_div2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimDivFusion>);
  MS_CHECK_TRUE_RET(is_div2 != nullptr, {});
  VectorRef div2_ref({is_div2, stridedslice2_ref, div2});

  auto is_erf = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimErf>);
  MS_CHECK_TRUE_RET(is_erf != nullptr, {});
  VectorRef erf_ref({is_erf, div2_ref});

  auto add3 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(add3 != nullptr, {});
  auto is_add3 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>);
  MS_CHECK_TRUE_RET(is_add3 != nullptr, {});
  VectorRef add3_ref({is_add3, erf_ref, add3});

  auto is_mul3 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul3 != nullptr, {});
  VectorRef mul3_ref({is_mul3, stridedslice2_ref, add3_ref});

  auto mul4 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(mul4 != nullptr, {});
  auto is_mul4 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul4 != nullptr, {});
  VectorRef mul4_ref({is_mul4, mul3_ref, mul4});

  auto is_mul5 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul5 != nullptr, {});
  VectorRef mul5_ref({is_mul5, stridedslice1_ref, mul4_ref});

  auto matmul2 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(matmul2 != nullptr, {});
  auto is_matmul2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMatMulFusion>);
  MS_CHECK_TRUE_RET(is_matmul2 != nullptr, {});
  VectorRef matmul2_ref({is_matmul2, mul5_ref, matmul2});

  auto add4 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(add4 != nullptr, {});
  auto is_add4 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>);
  MS_CHECK_TRUE_RET(is_add4 != nullptr, {});
  VectorRef add4_ref({is_add4, add4, matmul2_ref});

  MS_LOG(INFO) << "Finish define FFN fusion pattern for dynamic.";
  return add4_ref;
}

static const VectorRef DefineFFNPatterbForSDConst() {
  // reshape
  MS_LOG(INFO) << "Start define FFN fusion pattern for const.";
  const size_t param_num = 8;
  std::vector<CondVarPtr> params(param_num);
  for (size_t i = 0; i < params.size(); ++i) {
    params[i] = std::make_shared<CondVar>(IsParamNode);
    MS_CHECK_TRUE_RET(params[i] != nullptr, {});
  }
  size_t index = 0;
  auto input_x = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(input_x != nullptr, {});
  auto matmul1 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(matmul1 != nullptr, {});
  auto is_matmul1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMatMulFusion>);
  MS_CHECK_TRUE_RET(is_matmul1 != nullptr, {});
  VectorRef matmul1_ref({is_matmul1, input_x, matmul1});

  auto add1 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(add1 != nullptr, {});
  auto is_add1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>);
  MS_CHECK_TRUE_RET(is_add1 != nullptr, {});
  VectorRef add1_ref({is_add1, add1, matmul1_ref});

  auto is_stridedslice1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimStridedSlice>);
  MS_CHECK_TRUE_RET(is_stridedslice1 != nullptr, {});
  VectorRef stridedslice1_ref(
    {is_stridedslice1, add1_ref, params[index++], params[index++], params[index++], params[index++]});

  auto is_stridedslice2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimStridedSlice>);
  MS_CHECK_TRUE_RET(is_stridedslice2 != nullptr, {});
  VectorRef stridedslice2_ref(
    {is_stridedslice2, add1_ref, params[index++], params[index++], params[index++], params[index++]});

  auto div2 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(div2 != nullptr, {});
  auto is_div2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimDivFusion>);
  MS_CHECK_TRUE_RET(is_div2 != nullptr, {});
  VectorRef div2_ref({is_div2, stridedslice2_ref, div2});

  auto is_erf = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimErf>);
  MS_CHECK_TRUE_RET(is_erf != nullptr, {});
  VectorRef erf_ref({is_erf, div2_ref});

  auto add3 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(add3 != nullptr, {});
  auto is_add3 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>);
  MS_CHECK_TRUE_RET(is_add3 != nullptr, {});
  VectorRef add3_ref({is_add3, erf_ref, add3});

  auto is_mul3 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul3 != nullptr, {});
  VectorRef mul3_ref({is_mul3, stridedslice2_ref, add3_ref});

  auto mul4 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(mul4 != nullptr, {});
  auto is_mul4 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul4 != nullptr, {});
  VectorRef mul4_ref({is_mul4, mul3_ref, mul4});

  auto is_mul5 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul5 != nullptr, {});
  VectorRef mul5_ref({is_mul5, stridedslice1_ref, mul4_ref});

  auto matmul2 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(matmul2 != nullptr, {});
  auto is_matmul2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMatMulFusion>);
  MS_CHECK_TRUE_RET(is_matmul2 != nullptr, {});
  VectorRef matmul2_ref({is_matmul2, mul5_ref, matmul2});

  auto add4 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(add4 != nullptr, {});
  auto is_add4 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>);
  MS_CHECK_TRUE_RET(is_add4 != nullptr, {});
  VectorRef add4_ref({is_add4, add4, matmul2_ref});

  MS_LOG(INFO) << "Finish define FFN fusion pattern for const.";
  return add4_ref;
}

CNodePtr FFNCustomPass::CreateFFNFusionNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                            const EquivPtr &equiv) const {
  MS_LOG(INFO) << "Start create FFN cust fusion node.";
  MS_CHECK_TRUE_RET(node != nullptr, nullptr);
  auto cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(cnode != nullptr, nullptr);
  auto add4 = cnode;
  auto bias2 = add4->input(kNumIndex1);
  MS_CHECK_TRUE_RET(bias2 != nullptr, nullptr);

  auto matmul2 = add4->input(kNumIndex2)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(matmul2 != nullptr, nullptr);

  auto weight2 = matmul2->input(kNumIndex2);
  MS_CHECK_TRUE_RET(weight2 != nullptr, nullptr);

  auto mul5 = matmul2->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(mul5 != nullptr, nullptr);
  auto slice1 = mul5->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(slice1 != nullptr, nullptr);
  auto add1 = slice1->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(add1 != nullptr, nullptr);

  auto bias1 = add1->input(kNumIndex1);
  MS_CHECK_TRUE_RET(bias1 != nullptr, nullptr);

  auto matmul1 = add1->input(kNumIndex2)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(matmul1 != nullptr, nullptr);
  auto input_x = matmul1->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(input_x != nullptr, nullptr);
  auto weight1 = matmul1->input(kNumIndex2);
  MS_CHECK_TRUE_RET(weight1 != nullptr, nullptr);

  if (CheckInputShpae(input_x, weight1, weight2) != true) {
    return nullptr;
  }

  // create op
  auto ffn_fusion_prim = std::make_shared<ops::Custom>();
  MS_CHECK_TRUE_RET(ffn_fusion_prim != nullptr, nullptr);
  std::vector<std::string> input_names = {"x", "weight1", "weight2", "bias1", "bias2"};
  std::vector<std::string> output_names = {"y"};
  ffn_fusion_prim->set_type("FFNPro");
  ffn_fusion_prim->AddAttr("input_names", api::MakeValue(input_names));
  ffn_fusion_prim->AddAttr("output_names", api::MakeValue(output_names));
  ffn_fusion_prim->AddAttr("reg_op_name", api::MakeValue("FFNPro"));
  ffn_fusion_prim->AddAttr("activation", api::MakeValue("geglu"));
  ffn_fusion_prim->AddAttr("inner_precise", api::MakeValue(1));

  auto ffn_prim_c = ffn_fusion_prim->GetPrim();
  if (ffn_prim_c == nullptr) {
    MS_LOG(ERROR) << "ffn_prim_c is nullptr!";
    return nullptr;
  }
  auto ffn_cnode = func_graph->NewCNode(ffn_prim_c, {input_x, weight1, weight2, bias1, bias2});
  if (ffn_cnode == nullptr) {
    MS_LOG(ERROR) << "New FFN cnode failed!";
    return nullptr;
  }
  ffn_cnode->set_fullname_with_scope(node->fullname_with_scope() + "_ffn_cust_fusion");
  if (node->abstract() != nullptr) {
    ffn_cnode->set_abstract(node->abstract()->Clone());
  }

  auto manager = Manage(func_graph);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "Create Manage object failed!";
    return nullptr;
  }
  (void)manager->Replace(cnode, ffn_cnode);
  MS_LOG(INFO) << "Finish create FFN cust fusion node success.";
  return ffn_cnode;
}

std::unordered_map<std::string, VectorRef> FFNCustomPass::DefinePatterns() const {
  MS_LOG(INFO) << "Start define FFN cust fusion patterns.";
  std::unordered_map<std::string, VectorRef> patterns;
  patterns[kNameFFNPatternForSD] = DefineFFNPatterbForSD();
  patterns[kNameFFNPatternForSDConst] = DefineFFNPatterbForSDConst();
  MS_LOG(INFO) << "Finish define FFN cust fusion patterns.";
  return patterns;
}

AnfNodePtr FFNCustomPass::Process(const std::string &patten_name, const FuncGraphPtr &func_graph,
                                  const AnfNodePtr &node, const EquivPtr &equiv) const {
  MS_CHECK_TRUE_MSG(node != nullptr, nullptr, "node is nullptr!");
  MS_LOG(INFO) << "FFN cust fusion start, pattern name: " << patten_name << "   " << node->fullname_with_scope();
  if (func_graph == nullptr || equiv == nullptr) {
    MS_LOG(ERROR) << "Function graph or equiv is nullptr!";
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
  auto cnode = CreateFFNFusionNode(func_graph, node, equiv);
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "Create FFN cust fusion node return nullptr!";
    return nullptr;
  }
  MS_LOG(INFO) << "FFN cust node fusion success, fusion node name: " << cnode->fullname_with_scope();
  return cnode;
}

}  // namespace mindspore::opt
