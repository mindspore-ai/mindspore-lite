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
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>
#include "common/common_test.h"
#include "ir/func_graph.h"
#include "ir/manager.h"
#include "mindspore/ops/op_def/nn_ops.h"
#include "mindspore/ops/op_def/lite_ops.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "tools/optimizer/common/fusion_trace_utils.h"

namespace mindspore {
namespace lite {
namespace {
// Build a connected node so GetOrderedCnodes() (a topo sort from the graph output) reaches it.
CNodePtr NewNode(const FuncGraphPtr &graph, const std::string &prim_name, const std::string &name,
                 const AnfNodePtr &input = nullptr) {
  // A private clone per node: tracing attrs live on the Primitive, so sharing the
  // process-wide kPrim* singletons would leak records across test cases.
  auto prim = std::make_shared<Primitive>(prim_name);
  AnfNodePtr operand = input;
  if (operand == nullptr) {
    auto param = graph->add_parameter();
    param->set_name(name + "_in");
    operand = param;
  }
  auto cnode = graph->NewCNode({NewValueNode(prim), operand});
  cnode->set_fullname_with_scope(name);
  graph->set_output(cnode);
  return cnode;
}
}  // namespace

class FusionTraceUtilsTest : public ::testing::Test {
 public:
  FusionTraceUtilsTest() {
    graph_ = std::make_shared<FuncGraph>();
    conv_ = NewNode(graph_, "Conv2DFusion", "Conv2DFusion-3");
    bn_ = NewNode(graph_, "AddFusion", "BN_3", conv_);
    act_ = NewNode(graph_, "Activation", "Relu_3", bn_);
  }
  void SetUp() override { ASSERT_TRUE(Manage(graph_, true)) << "failed to create the graph manager"; }

  FuncGraphPtr graph_ = nullptr;
  CNodePtr conv_ = nullptr;
  CNodePtr bn_ = nullptr;
  CNodePtr act_ = nullptr;
};

// TC-TRC: import-time recording makes every node carry its origin name/type.
TEST_F(FusionTraceUtilsTest, RecordAndQueryOriginInfo) {
  opt::TraceOriginInfo(conv_, "Conv_3", "Conv");
  auto names = opt::GetSelfAndOriginNames(conv_);
  ASSERT_EQ(names.size(), 2);
  EXPECT_EQ(names[0], "Conv2DFusion-3");  // current name first
  EXPECT_EQ(names[1], "Conv_3");
  auto types = opt::GetOriginOpTypes(conv_);
  ASSERT_EQ(types.size(), 1);
  EXPECT_EQ(types[0], "Conv");
  EXPECT_EQ(opt::GetOriginTensors(conv_), "");
}

// TC-TRC-02/05: absorbing a component carries its records; chained absorbs accumulate.
TEST_F(FusionTraceUtilsTest, AbsorbAccumulatesRecords) {
  opt::TraceOriginInfo(conv_, "Conv_3", "Conv");
  opt::TraceOriginInfo(bn_, "BN_3", "BatchNormalization", "in:x -> out:y");
  opt::TraceAbsorbNames(conv_, bn_);
  // Conv absorbs activation next: records accumulate along the fusion chain.
  opt::TraceOriginInfo(act_, "Relu_3", "Relu");
  opt::TraceAbsorbNames(conv_, act_);

  auto names = opt::GetSelfAndOriginNames(conv_);
  EXPECT_EQ(names, (std::vector<std::string>{"Conv2DFusion-3", "Conv_3", "BN_3", "Relu_3"}));
  auto types = opt::GetOriginOpTypes(conv_);
  EXPECT_EQ(types, (std::vector<std::string>{"Conv", "BatchNormalization", "Relu"}));
  // TFLite anchor falls back to the absorbed node's anchor when the survivor has none.
  EXPECT_EQ(opt::GetOriginTensors(conv_), "in:x -> out:y");

  // Dedup: absorbing twice must not duplicate entries.
  opt::TraceAbsorbNames(conv_, bn_);
  EXPECT_EQ(opt::GetSelfAndOriginNames(conv_).size(), 4);
}

// TC-TRC-04: new-node fusion collects records of every participant.
TEST_F(FusionTraceUtilsTest, NewFusionNodeCollectsParts) {
  opt::TraceOriginInfo(bn_, "BN_3", "BatchNormalization");
  opt::TraceOriginInfo(act_, "Relu_3", "Relu");
  auto fused = NewNode(graph_, "LayerNormFusion", "layer_norm_x");
  opt::TraceNewFusionNode(fused, {bn_, act_});
  auto names = opt::GetSelfAndOriginNames(fused);
  EXPECT_EQ(names, (std::vector<std::string>{"layer_norm_x", "BN_3", "Relu_3"}));
}

// TC-TRC-09/10: stripping removes every tracing attribute so the exported .ms is unchanged.
TEST_F(FusionTraceUtilsTest, StripRemovesAllRecords) {
  opt::TraceOriginInfo(conv_, "Conv_3", "Conv", "in:a -> out:b");
  opt::StripOriginNames(graph_);
  EXPECT_EQ(opt::GetSelfAndOriginNames(conv_).size(), 1);  // only the current name remains
  EXPECT_TRUE(opt::GetOriginOpTypes(conv_).empty());
  EXPECT_TRUE(opt::GetOriginTensors(conv_).empty());
  auto prim = GetValueNode<PrimitivePtr>(conv_->input(0));
  ASSERT_NE(prim, nullptr);
  EXPECT_EQ(prim->GetAttr(opt::kOriginNamesAttr), nullptr);
  EXPECT_EQ(prim->GetAttr(opt::kOriginOpTypesAttr), nullptr);
  EXPECT_EQ(prim->GetAttr(opt::kOriginTensorsAttr), nullptr);
}
}  // namespace lite
}  // namespace mindspore
