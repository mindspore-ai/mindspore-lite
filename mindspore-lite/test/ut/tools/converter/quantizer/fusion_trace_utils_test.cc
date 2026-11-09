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
  // Chained collapse: the whole folded chain's identity lands on the FINAL fusion node - the
  // bypassed bn_ resolves through the memoized chain (its consumer act_ vanished into survivor)
  // rather than to the intermediate conv_, keeping "one region, one absorber" semantics.
  EXPECT_EQ(opt::GetSelfAndOriginNames(survivor), (std::vector<std::string>{"Relu_3-fused", "Relu_3", "BN_3"}));
  EXPECT_EQ(opt::GetSelfAndOriginNames(conv_), (std::vector<std::string>{"Conv2DFusion-3"}));

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

// ---- flash_attention surgery replay: the 10 ReplaceWithFaNode sites reduce to two graph
// shapes (ROOT replaced by the new fa node; INNER replaced while the matched root survives as
// a wrapper), plus two aggravations (extra new helper CNodes; a dead region chain whose
// records must propagate through the memoized resolution). Replay each shape through
// LitePassManager and verify the diff transport lands records where the explicit
// ReplaceWithFaNode would - the empirical answer to "can diff take over flash_attention".
namespace {
class FAReplayPass : public opt::Pass {
 public:
  explicit FAReplayPass(int shape) : Pass("FAReplayPass"), shape_(shape) {}
  bool Run(const FuncGraphPtr &fg) override {
    auto manager = fg->manager();
    auto make = [&fg](const std::string &prim, const AnfNodePtr &in) {
      return fg->NewCNode({NewValueNode(std::make_shared<Primitive>(prim)), in});
    };
    auto root = fg->output()->cast<CNodePtr>();
    switch (shape_) {
      case 0: {  // ROOT shape: Replace(root, fa), fa takes root's input
        auto fa = make("PromptFlashAttention", root->input(1));
        (void)manager->Replace(root, fa);
        break;
      }
      case 1: {  // INNER shape: root survives as wrapper, its input (inner) replaced by fa
        auto inner = root->input(1)->cast<CNodePtr>();
        auto fa = make("PromptFlashAttention", inner->input(1));
        (void)manager->Replace(inner, fa);
        break;
      }
      case 2: {  // INNER + extra new helper CNodes (q/k trans style) consumed by fa
        auto inner = root->input(1)->cast<CNodePtr>();
        auto q_trans = make("Transpose", inner->input(1));
        auto k_trans = make("Transpose", inner->input(1));
        auto fa = fg->NewCNode({NewValueNode(std::make_shared<Primitive>("PromptFlashAttention")), q_trans, k_trans});
        (void)manager->Replace(inner, fa);
        break;
      }
      case 3: {  // ROOT + dead region chain: fa bypasses the dead middle node to the source
        auto inner = root->input(1)->cast<CNodePtr>();
        auto src = inner->input(1)->cast<CNodePtr>();
        auto fa = make("PromptFlashAttention", src->input(1));
        (void)manager->Replace(root, fa);  // inner AND src die with the root
        break;
      }
      case 4: {  // INNER replaced + direct set_input bypass on another node's param slot
                 // (PreMulInBNSDOutBSND: k_trans->set_input(2, perm) then Replace(output_trans, fa))
        auto inner = root->input(1)->cast<CNodePtr>();
        auto conv_node = inner->input(1)->cast<CNodePtr>();
        auto perm = fg->add_parameter();                  // new param swapped in via direct set_input
        conv_node->cast<CNodePtr>()->set_input(1, perm);  // bypass the manager on purpose
        auto fa = make("PromptFlashAttention", perm);
        (void)manager->Replace(inner, fa);
        break;
      }
      case 5: {  // INNER replaced (wrapper root survives) + a dead region sibling dies with it
        auto inner = root->input(1)->cast<CNodePtr>();
        auto src = inner->input(1)->cast<CNodePtr>();
        auto fa = make("PromptFlashAttention", src->input(1));
        (void)manager->Replace(inner, fa);  // inner AND src die; root survives wrapping fa
        break;
      }
      default:
        break;
    }
    return false;  // MatMulAddFusion-style: changed but not reported
  }

 private:
  int shape_;
};

CNodePtr BuildChainForReplay(const FuncGraphPtr &fg, const std::vector<std::string> &prims) {
  CNodePtr prev = nullptr;
  for (const auto &prim : prims) {
    AnfNodePtr input = prev;
    if (input == nullptr) {
      input = fg->add_parameter();
    }
    prev = fg->NewCNode({NewValueNode(std::make_shared<Primitive>(prim)), input});
    prev->set_fullname_with_scope(prim);
  }
  fg->set_output(prev);
  return prev;
}
}  // namespace

TEST_F(FusionTraceUtilsTest, DiffTransportCoversFlashAttentionSurgeryShapes) {
  auto run = [](int shape) {
    auto fg = std::make_shared<FuncGraph>();
    (void)Manage(fg, true);
    opt::SetFusionTraceEnabled(true);
    // Shape 3 uses a 4-node chain (conv -> softmax -> inner -> root) so the softmax branch is
    // REACHABLE before the pass and dies only because the pass rewires the root - the real
    // flash-attention region-collapse shape. Other shapes use the 3-node chain.
    auto root = BuildChainForReplay(
      fg, shape >= 3 ? std::vector<std::string>{"Conv2DFusion", "Softmax", "AddFusion", "Activation"}
                     : std::vector<std::string>{"Conv2DFusion", "AddFusion", "Activation"});
    opt::TraceOriginInfo(root, "Act_root", "Activation");
    auto inner = root->input(1)->cast<CNodePtr>();
    opt::TraceOriginInfo(inner, "Inner_node", "AddFusion");
    auto conv = inner->input(1)->cast<CNodePtr>();
    if (shape >= 3) {
      opt::TraceOriginInfo(conv->cast<CNodePtr>(), "Softmax_src", "Softmax");  // conv node plays the softmax role
      auto src = conv->input(1)->cast<CNodePtr>();
      opt::TraceOriginInfo(src, "Conv_src", "Conv2DFusion");
    } else {
      opt::TraceOriginInfo(conv, "Conv_src", "Conv2DFusion");
    }
    auto pm = std::make_shared<opt::LitePassManager>("fa_replay", false);
    pm->AddPass(std::make_shared<FAReplayPass>(shape));
    auto optimizer = std::make_shared<opt::GraphOptimizer>();
    optimizer->AddPassManager(pm);
    (void)optimizer->Optimize(fg);
    std::map<std::string, std::vector<std::string>> out;
    for (auto &cnode : fg->GetOrderedCnodes()) {
      auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
      out[prim != nullptr ? prim->name() : "return"] = opt::GetOriginNames(cnode);
    }
    return out;
  };

  // Shape 0 (ROOT): fa carries the root's records; conv/inner keep their own.
  auto r0 = run(0);
  EXPECT_EQ(r0["PromptFlashAttention"], (std::vector<std::string>{"Act_root"}));
  EXPECT_EQ(r0["Conv2DFusion"], (std::vector<std::string>{"Conv_src"}));
  EXPECT_EQ(r0["AddFusion"], (std::vector<std::string>{"Inner_node"}));

  // Shape 1 (INNER, root survives as wrapper): fa carries inner's records; root/conv keep own.
  auto r1 = run(1);
  EXPECT_EQ(r1["PromptFlashAttention"], (std::vector<std::string>{"Inner_node"}));
  EXPECT_EQ(r1["Activation"], (std::vector<std::string>{"Act_root"}));
  EXPECT_EQ(r1["Conv2DFusion"], (std::vector<std::string>{"Conv_src"}));

  // Shape 2 (INNER + extra new helper nodes): helpers carry nothing, fa still gets inner's.
  auto r2 = run(2);
  EXPECT_EQ(r2["PromptFlashAttention"], (std::vector<std::string>{"Inner_node"}));
  EXPECT_TRUE(r2["Transpose"].empty());
  EXPECT_EQ(r2["Activation"], (std::vector<std::string>{"Act_root"}));

  // Shape 3 (ROOT + dead sibling chain created inside the pass): the sibling's records
  // propagate to fa via the memoized chain (fa absorbed the region) - a superset of the
  // explicit behavior, never a loss.
  auto r3 = run(3);
  // Act_root via h-replace; Softmax_src via the memoized chain (its user - the inner Add -
  // also vanished and resolved to fa); the source conv keeps its own record.
  EXPECT_EQ(r3["PromptFlashAttention"], (std::vector<std::string>{"Act_root", "Inner_node", "Softmax_src"}));
  EXPECT_EQ(r3["Conv2DFusion"], (std::vector<std::string>{"Conv_src"}));

  // Shape 4 (set_input bypass): the direct input swap must not confuse the transport - fa
  // still absorbs the inner chain, the swapped-in param carries nothing.
  auto r4 = run(4);
  // The set_input bypass drops the conv out of the graph - it is part of the collapsed region,
  // so its record lands on fa too (a user naming that conv must still hit the fused layer).
  EXPECT_EQ(r4["PromptFlashAttention"], (std::vector<std::string>{"Inner_node", "Softmax_src", "Conv_src"}));
  EXPECT_EQ(r4["Activation"], (std::vector<std::string>{"Act_root"}));

  // Shape 5 (wrapper survives + dead region): root keeps its own, fa absorbs inner + dead src.
  auto r5 = run(5);
  EXPECT_EQ(r5["PromptFlashAttention"], (std::vector<std::string>{"Inner_node", "Softmax_src"}));
  EXPECT_EQ(r5["Activation"], (std::vector<std::string>{"Act_root"}));
  EXPECT_EQ(r5["Conv2DFusion"], (std::vector<std::string>{"Conv_src"}));
}

}  // namespace lite
}  // namespace mindspore
