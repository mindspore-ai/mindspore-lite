/**
 * Copyright 2026 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define USE_DEPRECATED_API
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "common/common_test.h"
#include "include/registry/converter_context.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/func_graph.h"
#include "ir/tensor.h"
#include "mindspore/ops/op_def/auto_generate/gen_lite_ops.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"
#include "mindspore/ops/include/primitive/lite_ops.h"
#include "infer/eltwise.h"
#include "src/common/ops/primitive/add_fusion.h"
#include "test/ut/utils/build_func_graph.h"
#include "tools/converter/parser/onnx/onnx_inputs_adjust.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore {
namespace lite {
namespace {
ParameterPtr AddFp32Parameter(const FuncGraphPtr &graph, const std::string &name) {
  auto param = graph->add_parameter();
  param->set_name(name);
  param->set_abstract(std::make_shared<abstract::AbstractTensor>(kFloat32, std::vector<int64_t>{3, 5}));
  return param;
}

CNodePtr AddVariadicNode(const FuncGraphPtr &graph, const PrimitivePtr &prim,
                         const std::vector<AnfNodePtr> &data_inputs, const std::string &name) {
  std::vector<AnfNodePtr> inputs{NewValueNode(prim)};
  inputs.insert(inputs.end(), data_inputs.begin(), data_inputs.end());
  auto cnode = graph->NewCNode(inputs);
  if (cnode == nullptr) {
    return nullptr;
  }
  cnode->set_fullname_with_scope(name);
  cnode->set_abstract(std::make_shared<abstract::AbstractTensor>(kFloat32, std::vector<int64_t>{3, 5}));
  return cnode;
}

CNodePtr GraphOutput(const FuncGraphPtr &graph) { return graph->get_return()->input(1)->cast<CNodePtr>(); }

size_t CountNodes(const FuncGraphPtr &graph, const PrimitivePtr &prim) {
  size_t count = 0;
  std::vector<AnfNodePtr> queue{graph->get_return()};
  while (!queue.empty()) {
    auto node = queue.back();
    queue.pop_back();
    if (opt::CheckPrimitiveType(node, prim)) {
      count++;
    }
    if (node == nullptr || !utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    for (size_t i = 1; i < cnode->size(); i++) {
      queue.push_back(cnode->input(i));
    }
  }
  return count;
}

// Guards against the silent data-input drop of N-ary ops: every original input
// parameter must still be reachable from the graph output after the adjust pass.
std::set<AnfNodePtr> CollectReachableParameters(const FuncGraphPtr &graph) {
  std::set<AnfNodePtr> params;
  std::set<AnfNodePtr> visited;
  std::vector<AnfNodePtr> queue{graph->get_return()};
  while (!queue.empty()) {
    auto node = queue.back();
    queue.pop_back();
    if (node == nullptr || !visited.insert(node).second) {
      continue;
    }
    if (utils::isa<ParameterPtr>(node)) {
      params.insert(node);
      continue;
    }
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    for (size_t i = 1; i < cnode->size(); i++) {
      queue.push_back(cnode->input(i));
    }
  }
  return params;
}
}  // namespace

class OnnxInputsAdjustTest : public mindspore::CommonTest {
 public:
  OnnxInputsAdjustTest() {
    flag_.fmk = converter::kFmkTypeOnnx;
    flag_.save_type = kMindIR_Lite;
  }

  FuncGraphPtr BuildGraph(const PrimitivePtr &prim, size_t data_input_num) {
    auto graph = std::make_shared<FuncGraph>();
    graph->set_attr("fmk", MakeValue(static_cast<int>(converter::kFmkTypeOnnx)));
    std::vector<AnfNodePtr> data_inputs;
    for (size_t i = 0; i < data_input_num; i++) {
      auto param = AddFp32Parameter(graph, "input_" + std::to_string(i));
      if (i == 0) {
        first_param_ = param;
      }
      all_params_.push_back(param);
      data_inputs.push_back(param);
    }
    auto node = AddVariadicNode(graph, prim, data_inputs, "Max");
    if (node == nullptr) {
      return nullptr;
    }
    if (AddReturn(graph, {node}) == nullptr) {
      return nullptr;
    }
    return graph;
  }

  static PrimitivePtr MakeSumPrim() {
    auto eltwise = std::make_shared<ops::Eltwise>();
    eltwise->set_mode(mindspore::EltwiseMode::SUM);
    return eltwise->GetPrim();
  }

  converter::ConverterParameters flag_{};
  AnfNodePtr first_param_ = nullptr;
  std::vector<AnfNodePtr> all_params_{};
};

// ONNX Max with a single data input must become binary Max(X, X) so that
// ArithmeticInferShape (requires >=2 inputs) succeeds during conversion.
TEST_F(OnnxInputsAdjustTest, SingleInputMaxIsDuplicatedToBinary) {
  auto graph = BuildGraph(std::make_shared<ops::Maximum>()->GetPrim(), 1);
  ASSERT_NE(graph, nullptr);
  ASSERT_TRUE(OnnxInputAdjust::Adjust(graph, flag_));

  auto output = GraphOutput(graph);
  ASSERT_NE(output, nullptr);
  ASSERT_TRUE(opt::CheckPrimitiveType(output, prim::kPrimMaximum));
  ASSERT_EQ(output->size(), 3);  // primitive + 2 duplicated data inputs
  EXPECT_EQ(output->input(1), output->input(2));
}

TEST_F(OnnxInputsAdjustTest, SingleInputMinIsDuplicatedToBinary) {
  auto graph = BuildGraph(std::make_shared<ops::Minimum>()->GetPrim(), 1);
  ASSERT_NE(graph, nullptr);
  ASSERT_TRUE(OnnxInputAdjust::Adjust(graph, flag_));

  auto output = GraphOutput(graph);
  ASSERT_NE(output, nullptr);
  ASSERT_TRUE(opt::CheckPrimitiveType(output, prim::kPrimMinimum));
  ASSERT_EQ(output->size(), 3);
  EXPECT_EQ(output->input(1), output->input(2));
}

// Two data inputs are natively supported and must stay untouched.
TEST_F(OnnxInputsAdjustTest, TwoInputMaxIsUnchanged) {
  auto graph = BuildGraph(std::make_shared<ops::Maximum>()->GetPrim(), 2);
  ASSERT_NE(graph, nullptr);
  ASSERT_TRUE(OnnxInputAdjust::Adjust(graph, flag_));

  auto output = GraphOutput(graph);
  ASSERT_NE(output, nullptr);
  ASSERT_TRUE(opt::CheckPrimitiveType(output, prim::kPrimMaximum));
  ASSERT_EQ(output->size(), 3);
  EXPECT_EQ(output->input(1), first_param_);
}

// N-ary Max must be chained into binary nodes; the binary kernels drop data
// inputs beyond the 2nd, so a 4-input node needs 3 chained binary ops.
TEST_F(OnnxInputsAdjustTest, FourInputMaxIsChainedToBinaryNodes) {
  auto graph = BuildGraph(std::make_shared<ops::Maximum>()->GetPrim(), 4);
  ASSERT_NE(graph, nullptr);
  ASSERT_TRUE(OnnxInputAdjust::Adjust(graph, flag_));

  auto output = GraphOutput(graph);
  ASSERT_NE(output, nullptr);
  ASSERT_TRUE(opt::CheckPrimitiveType(output, prim::kPrimMaximum));
  ASSERT_EQ(output->size(), 3);
  auto inner = output->input(1)->cast<CNodePtr>();
  ASSERT_NE(inner, nullptr);
  ASSERT_TRUE(opt::CheckPrimitiveType(inner, prim::kPrimMaximum));
  ASSERT_EQ(inner->size(), 3);
  EXPECT_EQ(CountNodes(graph, prim::kPrimMaximum), 3);
}

// Single-input Sum is an identity: Sum(X) == X + 0. A scalar-zero const is
// appended instead of duplicating the input (Sum(X, X) == 2X would be wrong).
TEST_F(OnnxInputsAdjustTest, SingleInputSumGetsScalarZeroConst) {
  auto graph = BuildGraph(MakeSumPrim(), 1);
  ASSERT_NE(graph, nullptr);
  ASSERT_TRUE(OnnxInputAdjust::Adjust(graph, flag_));

  auto output = GraphOutput(graph);
  ASSERT_NE(output, nullptr);
  ASSERT_TRUE(opt::CheckPrimitiveType(output, prim::kPrimEltwise));
  ASSERT_EQ(output->size(), 3);  // primitive + data input + scalar zero
  auto zero_param = output->input(2)->cast<ParameterPtr>();
  ASSERT_NE(zero_param, nullptr);
  auto zero_tensor = std::dynamic_pointer_cast<tensor::Tensor>(zero_param->default_param());
  ASSERT_NE(zero_tensor, nullptr);
  ASSERT_EQ(zero_tensor->ElementsNum(), 1);
  EXPECT_EQ(static_cast<const float *>(zero_tensor->data_c())[0], 0.0f);
}

// N-ary Sum is chained into binary AddFusion nodes.
TEST_F(OnnxInputsAdjustTest, ThreeInputSumIsChainedToAddFusion) {
  auto graph = BuildGraph(MakeSumPrim(), 3);
  ASSERT_NE(graph, nullptr);
  ASSERT_TRUE(OnnxInputAdjust::Adjust(graph, flag_));

  auto output = GraphOutput(graph);
  ASSERT_NE(output, nullptr);
  ASSERT_TRUE(opt::CheckPrimitiveType(output, prim::kPrimAddFusion));
  ASSERT_EQ(output->size(), 3);
  auto inner = output->input(1)->cast<CNodePtr>();
  ASSERT_NE(inner, nullptr);
  ASSERT_TRUE(opt::CheckPrimitiveType(inner, prim::kPrimAddFusion));
  ASSERT_EQ(inner->size(), 3);
}

// Two data inputs of Eltwise SUM are natively supported and must stay untouched.
TEST_F(OnnxInputsAdjustTest, TwoInputSumIsUnchanged) {
  auto graph = BuildGraph(MakeSumPrim(), 2);
  ASSERT_NE(graph, nullptr);
  ASSERT_TRUE(OnnxInputAdjust::Adjust(graph, flag_));

  auto output = GraphOutput(graph);
  ASSERT_NE(output, nullptr);
  ASSERT_TRUE(opt::CheckPrimitiveType(output, prim::kPrimEltwise));
  ASSERT_EQ(output->size(), 3);
  EXPECT_EQ(output->input(1), all_params_[0]);
  EXPECT_EQ(output->input(2), all_params_[1]);
}

// N-ary Min is chained the same way as Max, and every original data input
// stays reachable from the output (guards the silent-drop regression).
TEST_F(OnnxInputsAdjustTest, FourInputMinIsChainedAndKeepsAllInputsReachable) {
  auto graph = BuildGraph(std::make_shared<ops::Minimum>()->GetPrim(), 4);
  ASSERT_NE(graph, nullptr);
  ASSERT_TRUE(OnnxInputAdjust::Adjust(graph, flag_));

  auto output = GraphOutput(graph);
  ASSERT_NE(output, nullptr);
  ASSERT_TRUE(opt::CheckPrimitiveType(output, prim::kPrimMinimum));
  ASSERT_EQ(output->size(), 3);
  EXPECT_EQ(CountNodes(graph, prim::kPrimMinimum), 3);
  auto reachable = CollectReachableParameters(graph);
  ASSERT_EQ(reachable.size(), all_params_.size());
  for (auto &param : all_params_) {
    EXPECT_NE(reachable.find(param), reachable.end());
  }
}

// The chained rewrite of N-ary Max must keep every original data input
// reachable from the graph output, otherwise the 3rd+ inputs would be
// silently dropped by the binary kernels.
TEST_F(OnnxInputsAdjustTest, FiveInputMaxKeepsAllDataInputsReachable) {
  auto graph = BuildGraph(std::make_shared<ops::Maximum>()->GetPrim(), 5);
  ASSERT_NE(graph, nullptr);
  ASSERT_TRUE(OnnxInputAdjust::Adjust(graph, flag_));

  auto output = GraphOutput(graph);
  ASSERT_NE(output, nullptr);
  ASSERT_TRUE(opt::CheckPrimitiveType(output, prim::kPrimMaximum));
  ASSERT_EQ(output->size(), 3);
  EXPECT_EQ(CountNodes(graph, prim::kPrimMaximum), 4);  // 5 data inputs -> 4 binary nodes
  auto reachable = CollectReachableParameters(graph);
  ASSERT_EQ(reachable.size(), all_params_.size());
  for (auto &param : all_params_) {
    EXPECT_NE(reachable.find(param), reachable.end());
  }
}

// Only Eltwise SUM is adjusted; other Eltwise modes must be left untouched
// (guards the mode check that was moved ahead of the size check).
TEST_F(OnnxInputsAdjustTest, NonSumEltwiseSingleInputIsNotAdjusted) {
  auto eltwise = std::make_shared<ops::Eltwise>();
  eltwise->set_mode(mindspore::EltwiseMode::PROD);
  auto graph = BuildGraph(eltwise->GetPrim(), 1);
  ASSERT_NE(graph, nullptr);
  ASSERT_TRUE(OnnxInputAdjust::Adjust(graph, flag_));

  auto output = GraphOutput(graph);
  ASSERT_NE(output, nullptr);
  ASSERT_TRUE(opt::CheckPrimitiveType(output, prim::kPrimEltwise));
  ASSERT_EQ(output->size(), 2);  // primitive + single data input, unchanged
  EXPECT_EQ(output->input(1), first_param_);
}
}  // namespace lite
}  // namespace mindspore
