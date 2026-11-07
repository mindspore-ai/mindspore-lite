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
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "common/common_test.h"
#include "ir/func_graph.h"
#include "ir/manager.h"
#include "mindspore/ops/op_def/nn_ops.h"
#include "mindspore/ops/op_def/lite_ops.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "tools/optimizer/common/fusion_trace_utils.h"
#include "tools/optimizer/common/multiple_pattern_process_pass.h"
#include "tools/optimizer/common/gllo_utils.h"

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
  FusionTraceUtilsTest() { graph_ = std::make_shared<FuncGraph>(); }
  void SetUp() override {
    // Hold the manager strongly (FuncGraph keeps only a weak reference) and build the graph
    // after Manage(): pre-attach edges never enter node_users(), which the region policy and
    // the consume-input fake Process bodies both rely on.
    manager_ = Manage(graph_, true);
    ASSERT_NE(manager_, nullptr) << "failed to create the graph manager";
    conv_ = NewNode(graph_, "Conv2DFusion", "Conv2DFusion-3");
    bn_ = NewNode(graph_, "AddFusion", "BN_3", conv_);
    act_ = NewNode(graph_, "Activation", "Relu_3", bn_);
  }

  FuncGraphPtr graph_ = nullptr;
  FuncGraphManagerPtr manager_ = nullptr;
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
}

// TC-TRC-02/05: absorbing a component carries its records; chained absorbs accumulate.
TEST_F(FusionTraceUtilsTest, AbsorbAccumulatesRecords) {
  opt::TraceOriginInfo(conv_, "Conv_3", "Conv");
  opt::TraceOriginInfo(bn_, "BN_3", "BatchNormalization");
  opt::TraceAbsorbNames(conv_, bn_);
  // Conv absorbs activation next: records accumulate along the fusion chain.
  opt::TraceOriginInfo(act_, "Relu_3", "Relu");
  opt::TraceAbsorbNames(conv_, act_);

  auto names = opt::GetSelfAndOriginNames(conv_);
  EXPECT_EQ(names, (std::vector<std::string>{"Conv2DFusion-3", "Conv_3", "BN_3", "Relu_3"}));
  auto types = opt::GetOriginOpTypes(conv_);
  EXPECT_EQ(types, (std::vector<std::string>{"Conv", "BatchNormalization", "Relu"}));

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
  opt::TraceOriginInfo(conv_, "Conv_3", "Conv");
  opt::StripOriginNames(graph_);
  EXPECT_EQ(opt::GetSelfAndOriginNames(conv_).size(), 1);  // only the current name remains
  EXPECT_TRUE(opt::GetOriginOpTypes(conv_).empty());
  auto prim = GetValueNode<PrimitivePtr>(conv_->input(0));
  ASSERT_NE(prim, nullptr);
  EXPECT_EQ(prim->GetAttr(opt::kOriginNamesAttr), nullptr);
  EXPECT_EQ(prim->GetAttr(opt::kOriginOpTypesAttr), nullptr);
}

namespace {
// Fake fusion: Process returns a preset survivor, so only the Run() exit hook is under test.
// Regression lock: the hook used to live only in LitePatternProcessPass::Run while its
// MultiplePatternProcessPass sibling silently dropped every multi-pattern fusion's root identity.
class HookMultiPatternPass : public opt::MultiplePatternProcessPass {
 public:
  explicit HookMultiPatternPass(const AnfNodePtr &survivor, bool auto_trace = true, bool consume_input = false,
                                bool region = false)
      : MultiplePatternProcessPass("hook_multi_pattern_test"), survivor_(survivor), consume_input_(consume_input) {
    if (!auto_trace) {
      SetTracePolicy(TracePolicy::kManual);  // squeeze_fusion-style construction-time opt-out
    } else if (region) {
      SetTracePolicy(TracePolicy::kAutoRegion);
    }
  }
  std::unordered_map<std::string, VectorRef> DefinePatterns() const override {
    auto is_act =
      std::make_shared<CondVar>(std::bind(opt::IsSpecifiedNode<&prim::kPrimActivation>, std::placeholders::_1));
    // Anonymous CondVar on the input slot: proves equiv captures it for the region policy.
    auto is_add =
      std::make_shared<CondVar>(std::bind(opt::IsSpecifiedNode<&prim::kPrimAddFusion>, std::placeholders::_1));
    return {{"hook_pattern", VectorRef({is_act, is_add, std::make_shared<SeqVar>()})}};
  }
  AnfNodePtr Process(const std::string &, const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                     const EquivPtr &) const override {
    if (consume_input_) {
      // In-place fusion shape: bypass the root's input CNode so it loses all its users.
      auto consumed = node->cast<CNodePtr>()->input(1);
      if (consumed != nullptr && consumed->isa<CNode>()) {
        (void)func_graph->manager()->Replace(consumed, consumed->cast<CNodePtr>()->input(1));
      }
    }
    return survivor_;
  }

 private:
  AnfNodePtr survivor_;
  bool consume_input_;
};
}  // namespace

// TC-TRC-11: the multi-pattern Run() exit carries the matched root's records onto the survivor,
// and the NeedAutoFusionTrace opt-out (squeeze_fusion-style multi-survivor passes) keeps it silent.
TEST_F(FusionTraceUtilsTest, MultiPatternPassRunHookCarriesRootRecords) {
  opt::TraceOriginInfo(act_, "Relu_3", "Relu");
  auto survivor = NewNode(graph_, "Activation", "Relu_3-fused", conv_);
  HookMultiPatternPass tracer(survivor);
  ASSERT_EQ(tracer.Run(graph_, act_), survivor);
  EXPECT_EQ(opt::GetSelfAndOriginNames(survivor), (std::vector<std::string>{"Relu_3-fused", "Relu_3"}));

  auto survivor2 = NewNode(graph_, "Activation", "Relu_3-fused2", conv_);
  HookMultiPatternPass optout(survivor2, false);
  ASSERT_EQ(optout.Run(graph_, act_), survivor2);
  EXPECT_EQ(opt::GetSelfAndOriginNames(survivor2).size(), 1);  // only its own current name
}

// Region policy (kAutoRegion): equiv-matched parts consumed inside Process carry onto the
// survivor via absorb-by-disappearance; a consumed part keeps silent under the default
// kAutoRoot policy. Locks the single-survivor multi-part contract (add_layernorm/norm/...).
TEST_F(FusionTraceUtilsTest, RegionPolicyCarriesConsumedParts) {
  opt::TraceOriginInfo(act_, "Relu_3", "Relu");
  opt::TraceOriginInfo(bn_, "BN_3", "BatchNormalization");

  auto survivor = graph_->NewCNode({NewValueNode(std::make_shared<Primitive>("Activation")), conv_});
  survivor->set_fullname_with_scope("Relu_3-fused");

  // Default kAutoRoot: the root is carried, the consumed region part is NOT.
  HookMultiPatternPass root_only(survivor, true, true, false);
  ASSERT_EQ(root_only.Run(graph_, act_), survivor);
  EXPECT_EQ(opt::GetSelfAndOriginNames(survivor), (std::vector<std::string>{"Relu_3-fused", "Relu_3"}));

  // Fresh graph state per scenario: bn_ was already bypassed above, rebuild the chain.
  auto bn2 = graph_->NewCNode({NewValueNode(std::make_shared<Primitive>("AddFusion")), conv_});
  bn2->set_fullname_with_scope("BN_4");
  auto act2 = graph_->NewCNode({NewValueNode(std::make_shared<Primitive>("Activation")), bn2});
  act2->set_fullname_with_scope("Relu_4");
  opt::TraceOriginInfo(bn2, "BN_4", "BatchNormalization");
  opt::TraceOriginInfo(act2, "Relu_4", "Relu");
  graph_->set_output(act2);
  auto survivor2 = graph_->NewCNode({NewValueNode(std::make_shared<Primitive>("Activation")), conv_});
  survivor2->set_fullname_with_scope("Relu_4-fused");

  HookMultiPatternPass region(survivor2, true, true, true);
  ASSERT_EQ(region.Run(graph_, act2), survivor2);
  // Both the matched root (Relu_4) and the consumed region part (BN_4) land on the survivor.
  EXPECT_EQ(opt::GetSelfAndOriginNames(survivor2), (std::vector<std::string>{"Relu_4-fused", "Relu_4", "BN_4"}));
}

// Graph-level: the NodePass::Run(func_graph) framework loop drives the pass end to end - the
// framework Replace AND the Run-exit tracing both fire for a kAutoRegion fusion. The pattern's
// is_add slot keeps the freshly returned survivor from re-matching on the next visit.
TEST_F(FusionTraceUtilsTest, GraphLevelRunReplacesAndCarries) {
  opt::TraceOriginInfo(act_, "Relu_3", "Relu");
  opt::TraceOriginInfo(bn_, "BN_3", "BatchNormalization");
  auto survivor = graph_->NewCNode({NewValueNode(std::make_shared<Primitive>("Activation")), conv_});
  survivor->set_fullname_with_scope("Relu_3-fused");
  graph_->set_output(act_);
  HookMultiPatternPass pass(survivor, true, true, true);
  // Qualified: the derived Run(fg, node) declaration hides the whole-graph overload.
  ASSERT_TRUE(pass.opt::LiteNodePass::Run(graph_));
  EXPECT_EQ(graph_->output(), survivor);  // framework replaced the matched root
  EXPECT_EQ(opt::GetSelfAndOriginNames(survivor), (std::vector<std::string>{"Relu_3-fused", "Relu_3", "BN_3"}));
}

// Subgraph cleanup: records on nodes inside a subgraph referenced through a
// ValueNode<FuncGraphPtr> input are stripped together with the main graph.
TEST_F(FusionTraceUtilsTest, StripReachesSubgraphs) {
  auto sub = std::make_shared<FuncGraph>();
  auto sub_act = NewNode(sub, "Activation", "SubAct_1");
  opt::TraceOriginInfo(sub_act, "Sub_1", "SubOp");
  auto holder = graph_->NewCNode({NewValueNode(sub)});
  holder->set_fullname_with_scope("sub_holder");
  graph_->set_output(holder);
  opt::StripOriginNames(graph_);
  EXPECT_TRUE(opt::GetOriginNames(sub_act).empty());
}

}  // namespace lite
}  // namespace mindspore
