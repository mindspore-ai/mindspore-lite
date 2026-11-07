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
#include "tools/optimizer/common/pass_manager_extends.h"
#include "tools/converter/ms_depend/pass_manager.h"
#include "tools/converter/ms_depend/graph_optimizer.h"

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
    opt::SetFusionTraceEnabled(true);  // the switch is process-global; other suites may close it
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
// Fake Pass-framework fusion (custom Run loop, returns bool) driven through the LitePassManager:
// exactly the family only the graph-diff transport can observe. It bypasses the recorded bn_
// (the conv+add shape) and replaces the root act_ by a new survivor, then reports changed=false
// on purpose - upstream MatMulAddFusion-style dishonesty the diff must survive.
class HookMultiPatternPass : public opt::Pass {
 public:
  static constexpr const char *kPassName = "DiffFusionPass";
  explicit HookMultiPatternPass(const CNodePtr &root, const CNodePtr &consumed, const AnfNodePtr &survivor,
                                bool report_changed = false)
      : Pass(kPassName), root_(root), consumed_(consumed), survivor_(survivor), report_changed_(report_changed) {}
  bool Run(const FuncGraphPtr &func_graph) override {
    auto manager = func_graph->manager();
    if (manager == nullptr) {
      return false;
    }
    (void)manager->Replace(consumed_, consumed_->input(1));  // bn_ bypassed in favor of conv_
    (void)manager->Replace(root_, survivor_);                // act_ replaced by the new node
    return report_changed_;
  }

 private:
  CNodePtr root_;
  CNodePtr consumed_;
  AnfNodePtr survivor_;
  bool report_changed_;
};
}  // namespace

// TC-TRC-11 (plan B): the pass-level graph-diff transport carries the vanished nodes' records
// onto the replacement - even when the pass dishonestly reports changed=false, and not at all
// when tracing is disabled by the converter (plain conversions carry zero overhead).
TEST_F(FusionTraceUtilsTest, DiffTransportCarriesVanishedRecords) {
  opt::TraceOriginInfo(act_, "Relu_3", "Relu");
  opt::TraceOriginInfo(bn_, "BN_3", "BatchNormalization");
  auto survivor = graph_->NewCNode({NewValueNode(std::make_shared<Primitive>("Activation")), conv_});
  survivor->set_fullname_with_scope("Relu_3-fused");
  graph_->set_output(act_);

  auto pm = std::make_shared<opt::LitePassManager>("diff_transport_test", false);
  pm->AddPass(std::make_shared<HookMultiPatternPass>(act_, bn_, survivor));
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  optimizer->AddPassManager(pm);
  ASSERT_TRUE(optimizer->Optimize(graph_) != nullptr);
  EXPECT_EQ(graph_->output(), survivor);
  // Unified receiver rule: bn_ -> its replacement conv_, act_ -> its replacement survivor.
  EXPECT_EQ(opt::GetSelfAndOriginNames(survivor), (std::vector<std::string>{"Relu_3-fused", "Relu_3"}));
  EXPECT_EQ(opt::GetSelfAndOriginNames(conv_), (std::vector<std::string>{"Conv2DFusion-3", "BN_3"}));

  // Tracing disabled: the same fusion carries nothing.
  opt::TraceOriginInfo(bn_, "BN_3", "BatchNormalization");
  opt::SetFusionTraceEnabled(false);
  auto bn2 = graph_->NewCNode({NewValueNode(std::make_shared<Primitive>("AddFusion")), conv_});
  bn2->set_fullname_with_scope("BN_4");
  auto act2 = graph_->NewCNode({NewValueNode(std::make_shared<Primitive>("Activation")), bn2});
  act2->set_fullname_with_scope("Relu_4");
  graph_->set_output(act2);
  auto survivor2 = graph_->NewCNode({NewValueNode(std::make_shared<Primitive>("Activation")), conv_});
  survivor2->set_fullname_with_scope("Relu_4-fused");
  opt::SetFusionTraceEnabled(true);
  opt::TraceOriginInfo(bn2, "BN_4", "BatchNormalization");
  opt::SetFusionTraceEnabled(false);
  auto pm2 = std::make_shared<opt::LitePassManager>("diff_transport_off", false);
  pm2->AddPass(std::make_shared<HookMultiPatternPass>(act2, bn2, survivor2));
  auto optimizer2 = std::make_shared<opt::GraphOptimizer>();
  optimizer2->AddPassManager(pm2);
  (void)optimizer2->Optimize(graph_);
  opt::SetFusionTraceEnabled(true);
  EXPECT_EQ(opt::GetSelfAndOriginNames(survivor2).size(), 1);  // only its own current name
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
