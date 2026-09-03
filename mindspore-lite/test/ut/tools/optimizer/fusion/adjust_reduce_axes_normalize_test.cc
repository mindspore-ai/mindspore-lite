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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define USE_DEPRECATED_API
#include <memory>
#include <vector>
#include "common/common_test.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "base/base.h"
#include "ir/dtype/number.h"
#include "ir/tensor.h"
#include "src/common/ops/primitive/reduce_fusion.h"
#include "include/registry/converter_context.h"
#include "tools/optimizer/fusion/adjust_reducesum_pass.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "test/ut/utils/build_func_graph.h"
#include "tools/converter/parser/onnx/onnx_reduce_parser.h"

namespace mindspore {
namespace lite {
namespace {
// Reads the reduce axes carried by the node's second input after a pass run: FillAxesForNode
// installs an int-sequence ValueNode, while an untouched graph keeps the original Parameter.
std::vector<int> GetAxesInputValues(const CNodePtr &cnode) {
  auto axes_input = cnode->input(kIndex2);
  auto value_node = axes_input->cast<ValueNodePtr>();
  if (value_node != nullptr && value_node->value() != nullptr) {
    return opt::CastToInt(value_node->value());
  }
  auto param = axes_input->cast<ParameterPtr>();
  if (param != nullptr && param->has_default()) {
    auto tensor = param->default_param()->cast<tensor::TensorPtr>();
    if (tensor != nullptr && tensor->ElementsNum() > 0) {
      std::vector<int> values;
      for (int64_t i = 0; i < tensor->ElementsNum(); ++i) {
        values.push_back(tensor->data_type() == kNumberTypeInt64
                           ? static_cast<int>(static_cast<const int64_t *>(tensor->data_c())[i])
                           : static_cast<const int32_t *>(tensor->data_c())[i]);
      }
      return values;
    }
  }
  return {};
}
}  // namespace

class AdjustReduceAxesNormalizeTest : public mindspore::CommonTest {
 public:
  AdjustReduceAxesNormalizeTest() = default;

  // Builds input-as-axes graphs (opset >= 13 ONNX style): the axes constant is the reduce
  // node's second input, which is the form both regressions arrive in.
  CNodePtr BuildReduceProdWithAxesInput(const FuncGraphPtr &func_graph, const ShapeVector &shape,
                                        const std::vector<int64_t> &axes_values, bool skip_mode = false) {
    auto prim = std::make_unique<ops::ReduceFusion>();
    prim->set_mode(ReduceMode::Reduce_Prod);
    prim->set_keep_dims(true);
    if (skip_mode) {
      prim->set_skip_mode(true);
    }
    auto data_param = func_graph->add_parameter();
    data_param->set_abstract(std::make_shared<abstract::AbstractTensor>(kFloat32, shape));
    auto axes_param = opt::BuildInt64VecParameterNode(func_graph, axes_values, "axes");
    auto cnode = func_graph->NewCNode({NewValueNode(prim->GetPrim()), data_param, axes_param->cast<AnfNodePtr>()});
    cnode->set_abstract(data_param->abstract()->Clone());
    return cnode;
  }

  // The empty-axes ONNX form: the second input is a zero-element constant tensor.
  CNodePtr BuildReduceProdWithEmptyAxesInput(const FuncGraphPtr &func_graph, const ShapeVector &shape,
                                             bool skip_mode = false) {
    auto prim = std::make_unique<ops::ReduceFusion>();
    prim->set_mode(ReduceMode::Reduce_Prod);
    prim->set_keep_dims(true);
    if (skip_mode) {
      prim->set_skip_mode(true);
    }
    auto data_param = func_graph->add_parameter();
    data_param->set_abstract(std::make_shared<abstract::AbstractTensor>(kFloat32, shape));
    auto axes_param = func_graph->add_parameter();
    axes_param->set_name("axes");
    auto axes_tensor = std::make_shared<tensor::Tensor>(kNumberTypeInt64, ShapeVector{0});
    axes_param->set_default_param(axes_tensor);
    axes_param->set_abstract(axes_tensor->ToAbstract());
    auto cnode = func_graph->NewCNode({NewValueNode(prim->GetPrim()), data_param, axes_param->cast<AnfNodePtr>()});
    cnode->set_abstract(data_param->abstract()->Clone());
    return cnode;
  }
};

// Invalid input, input form: duplicate axes must reach the kernel-side validation unchanged —
// they are rejected there with a detailed message (num of reduce axes larger than input rank).
TEST_F(AdjustReduceAxesNormalizeTest, DuplicateAxesInputPassthrough) {
  auto func_graph = std::make_shared<FuncGraph>();
  auto reduce = BuildReduceProdWithAxesInput(func_graph, {1, 4, 4, 3}, {0, 1, 1, 2, 0});
  ASSERT_TRUE(lite::AddReturn(func_graph, {reduce}) != nullptr);
  func_graph->set_attr("fmk", MakeValue(static_cast<int>(converter::kFmkTypeOnnx)));

  ASSERT_TRUE(opt::AdjustReduceSumPass().Run(func_graph));
  auto out_cnode = func_graph->output()->cast<CNodePtr>();
  ASSERT_TRUE(out_cnode != nullptr);
  std::vector<int> expected{0, 1, 1, 2, 0};
  ASSERT_EQ(GetAxesInputValues(out_cnode), expected);
}

// Invalid input, negative duplicate axes: also passed through unchanged.
TEST_F(AdjustReduceAxesNormalizeTest, DuplicateNegativeAxesInputPassthrough) {
  auto func_graph = std::make_shared<FuncGraph>();
  auto reduce = BuildReduceProdWithAxesInput(func_graph, {2, 3, 4}, {-1, -1, 2});
  ASSERT_TRUE(lite::AddReturn(func_graph, {reduce}) != nullptr);
  func_graph->set_attr("fmk", MakeValue(static_cast<int>(converter::kFmkTypeOnnx)));

  ASSERT_TRUE(opt::AdjustReduceSumPass().Run(func_graph));
  auto out_cnode = func_graph->output()->cast<CNodePtr>();
  ASSERT_TRUE(out_cnode != nullptr);
  std::vector<int> expected{-1, -1, 2};
  ASSERT_EQ(GetAxesInputValues(out_cnode), expected);
}

// Bug 2: a zero-element constant axes input means reduce-all (noop_with_empty_axes=false), but
// the empty tensor deadlocks the MindRT scheduler in the quantizer's DoInference and breaks the
// micro coder; it must be replaced with an explicit full axis list.
TEST_F(AdjustReduceAxesNormalizeTest, EmptyAxesInputFilled) {
  auto func_graph = std::make_shared<FuncGraph>();
  auto reduce = BuildReduceProdWithEmptyAxesInput(func_graph, {2, 3, 4});
  ASSERT_TRUE(lite::AddReturn(func_graph, {reduce}) != nullptr);
  func_graph->set_attr("fmk", MakeValue(static_cast<int>(converter::kFmkTypeOnnx)));

  ASSERT_TRUE(opt::AdjustReduceSumPass().Run(func_graph));
  auto out_cnode = func_graph->output()->cast<CNodePtr>();
  ASSERT_TRUE(out_cnode != nullptr);
  std::vector<int> expected{0, 1, 2};
  ASSERT_EQ(GetAxesInputValues(out_cnode), expected);
}

// Bug 2 guard rail: empty axes with noop_with_empty_axes=true is an identity, not reduce-all;
// the pass must leave it to the identity rewrite instead of filling a full axis list.
TEST_F(AdjustReduceAxesNormalizeTest, EmptyAxesInputNoopIdentity) {
  auto func_graph = std::make_shared<FuncGraph>();
  auto reduce = BuildReduceProdWithEmptyAxesInput(func_graph, {2, 3, 4}, true);
  ASSERT_TRUE(lite::AddReturn(func_graph, {reduce}) != nullptr);
  func_graph->set_attr("fmk", MakeValue(static_cast<int>(converter::kFmkTypeOnnx)));

  ASSERT_TRUE(opt::AdjustReduceSumPass().Run(func_graph));
  auto out_cnode = func_graph->output()->cast<CNodePtr>();
  ASSERT_TRUE(out_cnode != nullptr);
  auto prim = GetCNodePrimitive(out_cnode);
  ASSERT_TRUE(prim != nullptr);
  ASSERT_EQ(prim->name(), "Reshape");
}

// Control: a well-formed constant axes input must pass through untouched.
TEST_F(AdjustReduceAxesNormalizeTest, CleanAxesInputUntouched) {
  auto func_graph = std::make_shared<FuncGraph>();
  auto reduce = BuildReduceProdWithAxesInput(func_graph, {2, 3, 4}, {1, 2});
  ASSERT_TRUE(lite::AddReturn(func_graph, {reduce}) != nullptr);
  func_graph->set_attr("fmk", MakeValue(static_cast<int>(converter::kFmkTypeOnnx)));

  ASSERT_TRUE(opt::AdjustReduceSumPass().Run(func_graph));
  auto out_cnode = func_graph->output()->cast<CNodePtr>();
  ASSERT_TRUE(out_cnode != nullptr);
  std::vector<int> expected{1, 2};
  ASSERT_EQ(GetAxesInputValues(out_cnode), expected);
}

namespace {
// Invalid input, attribute form: duplicates on the legacy axes attribute are kept as-is by the
// parser; they reach the kernel-side validation which rejects them with a detailed message.
onnx::NodeProto BuildOnnxReduceProdNode(const std::vector<int64_t> &axes, bool with_axes_attr) {
  onnx::NodeProto node;
  node.set_op_type("ReduceProd");
  node.add_input("input");
  if (with_axes_attr) {
    auto attr = node.add_attribute();
    attr->set_name("axes");
    for (auto axis : axes) {
      attr->add_ints(axis);
    }
  }
  return node;
}

onnx::GraphProto BuildOnnxGraphWithInput(const ShapeVector &shape) {
  onnx::GraphProto graph;
  auto value_info = graph.add_input();
  value_info->set_name("input");
  auto tensor_type = value_info->mutable_type()->mutable_tensor_type();
  auto shape_proto = tensor_type->mutable_shape();
  for (auto dim : shape) {
    shape_proto->add_dim()->set_dim_value(dim);
  }
  return graph;
}
}  // namespace

TEST_F(AdjustReduceAxesNormalizeTest, OnnxParserDuplicateAxesAttrKept) {
  auto graph = BuildOnnxGraphWithInput({1, 4, 4, 3});
  auto node = BuildOnnxReduceProdNode({0, 1, 1, 2, 0}, true);
  auto prim = lite::OnnxReduceParser().Parse(graph, node);
  ASSERT_TRUE(prim != nullptr);
  std::vector<int> expected{0, 1, 1, 2, 0};
  ASSERT_EQ(opt::CastToInt(prim->GetAttr("axes")), expected);
}

TEST_F(AdjustReduceAxesNormalizeTest, OnnxParserEmptyAxesAttrFillsFull) {
  auto graph = BuildOnnxGraphWithInput({2, 3, 4});
  auto node = BuildOnnxReduceProdNode({}, false);
  auto prim = lite::OnnxReduceParser().Parse(graph, node);
  ASSERT_TRUE(prim != nullptr);
  std::vector<int> expected{0, 1, 2};
  ASSERT_EQ(opt::CastToInt(prim->GetAttr("axes")), expected);
}
}  // namespace lite
}  // namespace mindspore
