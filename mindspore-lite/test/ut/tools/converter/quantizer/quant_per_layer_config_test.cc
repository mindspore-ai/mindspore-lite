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
#include <set>
#include <string>
#include <vector>
#include "common/common_test.h"
#include "ir/func_graph.h"
#include "ir/manager.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "src/common/ops/primitive/activation.h"
#include "mindspore/ops/op_def/lite_ops.h"
#include "mindspore/ops/op_def/nn_ops.h"
#include "ir/dtype/type_id.h"
#include "include/errorcode.h"
#include "tools/converter/config_parser/quant_param_parser.h"
#include "tools/converter/quantizer/layer_decision_reporter.h"
#include "tools/converter/quantizer/quant_strategy.h"
#include "tools/optimizer/common/fusion_trace_utils.h"

namespace mindspore {
namespace lite {
namespace {
CNodePtr NewFp32Node(const FuncGraphPtr &graph, const PrimitivePtr &prim, const std::string &name) {
  auto param = graph->add_parameter();
  param->set_name(name + "_in");
  param->set_abstract(std::make_shared<abstract::AbstractTensor>(kFloat32, std::vector<int64_t>{3, 5}));
  auto cnode = graph->NewCNode({NewValueNode(prim), param});
  cnode->set_fullname_with_scope(name);
  cnode->set_abstract(std::make_shared<abstract::AbstractTensor>(kFloat32, std::vector<int64_t>{3, 5}));
  return cnode;
}
}  // namespace

// TC-CFG: parsing of the per-layer config lists and the whitelist/blacklist mutual exclusion.
class QuantPerLayerConfigTest : public ::testing::Test {
 protected:
  quant::CommonQuantParam param_;
  int Parse(const CommonQuantString &str) { return lite::QuantParamParser::ParseCommonQuant(str, &param_); }
};

TEST_F(QuantPerLayerConfigTest, ParseListsAndReportPath) {
  CommonQuantString str;
  str.quant_type = "FULL_QUANT";
  str.skip_quant_node = "Conv_3,BN-8";
  str.skip_quant_op_type = "Gather";
  str.layer_decision_report = "./layer_decision.csv";
  EXPECT_EQ(Parse(str), lite::RET_OK);
  EXPECT_EQ(param_.skip_quant_node, (std::set<std::string>{"Conv_3", "BN-8"}));
  EXPECT_EQ(param_.skip_quant_op_type, (std::set<std::string>{"Gather"}));
  EXPECT_TRUE(param_.quant_node.empty());
  EXPECT_EQ(param_.layer_decision_report, "./layer_decision.csv");
}

TEST_F(QuantPerLayerConfigTest, WhitelistBlacklistMutuallyExclusive) {
  CommonQuantString str;
  str.quant_node = "Conv_3";
  str.skip_quant_node = "BN-8";
  EXPECT_EQ(Parse(str), lite::RET_INPUT_PARAM_INVALID);

  quant::CommonQuantParam second;
  CommonQuantString str2;
  str2.quant_op_type = "Conv";
  str2.skip_quant_op_type = "Gather";
  EXPECT_EQ(lite::QuantParamParser::ParseCommonQuant(str2, &second), lite::RET_INPUT_PARAM_INVALID);

  // The remaining two cross combinations: any whitelist key x any blacklist key is rejected.
  quant::CommonQuantParam third;
  CommonQuantString str3;
  str3.quant_node = "Conv_3";
  str3.skip_quant_op_type = "Gather";
  EXPECT_EQ(lite::QuantParamParser::ParseCommonQuant(str3, &third), lite::RET_INPUT_PARAM_INVALID);

  quant::CommonQuantParam fourth;
  CommonQuantString str4;
  str4.quant_op_type = "Conv";
  str4.skip_quant_node = "BN-8";
  EXPECT_EQ(lite::QuantParamParser::ParseCommonQuant(str4, &fourth), lite::RET_INPUT_PARAM_INVALID);

  // Empty entries from "a,,b" style lists are dropped and never activate a list.
  quant::CommonQuantParam fifth;
  CommonQuantString str5;
  str5.quant_node = ",";
  EXPECT_EQ(lite::QuantParamParser::ParseCommonQuant(str5, &fifth), lite::RET_OK);
  EXPECT_TRUE(fifth.quant_node.empty());
}

TEST_F(QuantPerLayerConfigTest, OneSideAloneIsValid) {
  CommonQuantString whitelist;
  whitelist.quant_type = "FULL_QUANT";
  whitelist.quant_node = "Conv_3";
  whitelist.quant_op_type = "Conv";
  EXPECT_EQ(Parse(whitelist), lite::RET_OK);
  EXPECT_TRUE(param_.skip_quant_node.empty());
}

// TC-CFG: new list keys (whitelist pair + the type blacklist key) are full-quant-only — pairing
// any of them with another quant_type is rejected instead of being silently ignored; only the
// legacy skip_quant_node keeps its three-path behavior.
TEST_F(QuantPerLayerConfigTest, NewKeysRequireFullQuant) {
  // Rejected entries stay inserted into the parsed param, so every sub-case parses into a FRESH
  // param object (same pattern as EntryCountLimits) to avoid leaking keys across sub-cases.
  auto parse_fresh = [](const CommonQuantString &str) {
    quant::CommonQuantParam param;
    return std::make_pair(lite::QuantParamParser::ParseCommonQuant(str, &param), param);
  };

  CommonQuantString weight;
  weight.quant_type = "WEIGHT_QUANT";
  weight.quant_node = "Conv_3";
  EXPECT_EQ(parse_fresh(weight).first, lite::RET_INPUT_PARAM_INVALID);

  CommonQuantString dynamic;
  dynamic.quant_type = "DYNAMIC_QUANT";
  dynamic.quant_op_type = "Conv";
  EXPECT_EQ(parse_fresh(dynamic).first, lite::RET_INPUT_PARAM_INVALID);

  CommonQuantString type_blacklist;
  type_blacklist.quant_type = "WEIGHT_QUANT";
  type_blacklist.skip_quant_op_type = "MatMul";
  EXPECT_EQ(parse_fresh(type_blacklist).first, lite::RET_INPUT_PARAM_INVALID);

  // The legacy name blacklist keeps working on every path.
  CommonQuantString legacy_name;
  legacy_name.quant_type = "WEIGHT_QUANT";
  legacy_name.skip_quant_node = "Conv_3";
  EXPECT_EQ(parse_fresh(legacy_name).first, lite::RET_OK);

  CommonQuantString full_whitelist;
  full_whitelist.quant_type = "FULL_QUANT";
  full_whitelist.quant_node = "Conv_3";
  full_whitelist.quant_op_type = "Conv";
  EXPECT_EQ(parse_fresh(full_whitelist).first, lite::RET_OK);

  CommonQuantString full_type_blacklist;
  full_type_blacklist.quant_type = "FULL_QUANT";
  full_type_blacklist.skip_quant_op_type = "MatMul";
  EXPECT_EQ(parse_fresh(full_type_blacklist).first, lite::RET_OK);
}

// TC-CFG: entry-count guards — name lists cap at 1000 raw fields, op-type lists at 200, single
// entry length at 1024. The count is the raw comma-separated field count checked BEFORE set
// deduplication, so a list of duplicates can still be rejected. Every failure path leaves the
// parsed set untouched (all checks run before any insert), asserted on the too-long case below.
// Every sub-case still parses into a FRESH param object so the boundary sub-cases stay
// independent.
TEST_F(QuantPerLayerConfigTest, EntryCountLimits) {
  auto join_n = [](size_t n, const std::string &prefix) {
    std::string list;
    for (size_t i = 0; i < n; ++i) {
      if (i != 0) list += ",";
      list += prefix + std::to_string(i);
    }
    return list;
  };
  auto join_n_dup = [](size_t n, const std::string &entry) {
    std::string list;
    for (size_t i = 0; i < n; ++i) {
      if (i != 0) list += ",";
      list += entry;
    }
    return list;
  };
  auto parse_fresh = [](const CommonQuantString &str) {
    quant::CommonQuantParam param;
    auto ret = lite::QuantParamParser::ParseCommonQuant(str, &param);
    return std::make_pair(ret, param);
  };

  CommonQuantString over_names;
  over_names.quant_type = "FULL_QUANT";
  over_names.quant_node = join_n(1001, "Conv_");
  auto names_over = parse_fresh(over_names);
  EXPECT_EQ(names_over.first, lite::RET_INPUT_PARAM_INVALID);

  CommonQuantString over_types;
  over_types.quant_type = "FULL_QUANT";
  over_types.skip_quant_op_type = join_n(201, "Type");
  auto types_over = parse_fresh(over_types);
  EXPECT_EQ(types_over.first, lite::RET_INPUT_PARAM_INVALID);

  CommonQuantString ok_names;  // exactly at the cap passes
  ok_names.quant_type = "FULL_QUANT";
  ok_names.quant_node = join_n(1000, "Conv_");
  auto names_ok = parse_fresh(ok_names);
  EXPECT_EQ(names_ok.first, lite::RET_OK);
  EXPECT_EQ(names_ok.second.quant_node.size(), 1000u);

  CommonQuantString ok_types;
  ok_types.quant_type = "FULL_QUANT";
  ok_types.skip_quant_op_type = join_n(200, "Type");
  auto types_ok = parse_fresh(ok_types);
  EXPECT_EQ(types_ok.first, lite::RET_OK);
  EXPECT_EQ(types_ok.second.skip_quant_op_type.size(), 200u);

  // The other two keys carry their own cap constants: legacy name blacklist 1000/1001,
  // whitelist op types 200/201.
  CommonQuantString over_skip_names;
  over_skip_names.quant_type = "FULL_QUANT";
  over_skip_names.skip_quant_node = join_n(1001, "Skip_");
  EXPECT_EQ(parse_fresh(over_skip_names).first, lite::RET_INPUT_PARAM_INVALID);

  CommonQuantString ok_skip_names;
  ok_skip_names.quant_type = "FULL_QUANT";
  ok_skip_names.skip_quant_node = join_n(1000, "Skip_");
  auto skip_names_ok = parse_fresh(ok_skip_names);
  EXPECT_EQ(skip_names_ok.first, lite::RET_OK);
  EXPECT_EQ(skip_names_ok.second.skip_quant_node.size(), 1000u);

  CommonQuantString over_quant_types;
  over_quant_types.quant_type = "FULL_QUANT";
  over_quant_types.quant_op_type = join_n(201, "QType");
  EXPECT_EQ(parse_fresh(over_quant_types).first, lite::RET_INPUT_PARAM_INVALID);

  CommonQuantString ok_quant_types;
  ok_quant_types.quant_type = "FULL_QUANT";
  ok_quant_types.quant_op_type = join_n(200, "QType");
  auto quant_types_ok = parse_fresh(ok_quant_types);
  EXPECT_EQ(quant_types_ok.first, lite::RET_OK);
  EXPECT_EQ(quant_types_ok.second.quant_op_type.size(), 200u);

  // Pre-dedup counting: 1001 duplicated entries deduplicate to one, but the raw field count
  // still breaks the name-list cap; 1000 duplicates stay legal.
  CommonQuantString dup_over;
  dup_over.quant_type = "FULL_QUANT";
  dup_over.quant_node = join_n_dup(1001, "Conv_0");
  auto dup_over_ret = parse_fresh(dup_over);
  EXPECT_EQ(dup_over_ret.first, lite::RET_INPUT_PARAM_INVALID);

  CommonQuantString dup_ok;
  dup_ok.quant_type = "FULL_QUANT";
  dup_ok.quant_node = join_n_dup(1000, "Conv_0");
  auto dup_ok_ret = parse_fresh(dup_ok);
  EXPECT_EQ(dup_ok_ret.first, lite::RET_OK);
  EXPECT_EQ(dup_ok_ret.second.quant_node.size(), 1u);

  // Single-entry length cap: 1025 characters rejected, 1024 accepted.
  CommonQuantString long_over;
  long_over.quant_type = "FULL_QUANT";
  long_over.skip_quant_node = join_n(2, "ok_") + "," + std::string(1025, 'x');
  auto long_over_ret = parse_fresh(long_over);
  EXPECT_EQ(long_over_ret.first, lite::RET_INPUT_PARAM_INVALID);
  // Atomic failure: the valid entries before the too-long one must not be left behind.
  EXPECT_TRUE(long_over_ret.second.skip_quant_node.empty());

  CommonQuantString long_ok;
  long_ok.quant_type = "FULL_QUANT";
  long_ok.skip_quant_node = std::string(1024, 'x');
  auto long_ok_ret = parse_fresh(long_ok);
  EXPECT_EQ(long_ok_ret.first, lite::RET_OK);
  EXPECT_EQ(long_ok_ret.second.skip_quant_node.size(), 1u);
}

TEST_F(QuantPerLayerConfigTest, ParseStrictFlag) {
  CommonQuantString str;
  str.layer_decision_strict = "true";
  EXPECT_EQ(Parse(str), lite::RET_OK);
  EXPECT_TRUE(param_.layer_decision_strict);

  CommonQuantString str2;
  str2.layer_decision_strict = "false";
  EXPECT_EQ(Parse(str2), lite::RET_OK);
  EXPECT_FALSE(param_.layer_decision_strict);

  CommonQuantString str3;
  str3.layer_decision_strict = "not-a-bool";
  EXPECT_EQ(Parse(str3), lite::RET_INPUT_PARAM_INVALID);
}

// Pre-optimization validation: entries naming no node of the original model warn in default mode
// and fail the run under layer_decision_strict; name and type domains validate separately.
TEST_F(QuantPerLayerConfigTest, ValidatePerLayerConfigAgainstSnapshot) {
  auto param = std::make_shared<ConverterPara>();
  param->commonQuantParam.skip_quant_node = {"Conv_3"};
  const std::set<std::string> names = {"Conv_3", "MatMul_7"};
  const std::set<std::string> types = {"Conv", "Gather"};
  EXPECT_TRUE(quant::LayerDecisionReporter::ValidatePerLayerConfig(*param, names, types));

  param->commonQuantParam.skip_quant_node = {"Conv_3", "Reshape-88"};  // Reshape-88 matches nothing
  EXPECT_TRUE(quant::LayerDecisionReporter::ValidatePerLayerConfig(*param, names, types));
  param->commonQuantParam.layer_decision_strict = true;
  EXPECT_FALSE(quant::LayerDecisionReporter::ValidatePerLayerConfig(*param, names, types));

  param->commonQuantParam.layer_decision_strict = false;
  param->commonQuantParam.skip_quant_node.clear();
  param->commonQuantParam.skip_quant_op_type = {"Gather", "gather"};  // types are case-sensitive too
  EXPECT_TRUE(quant::LayerDecisionReporter::ValidatePerLayerConfig(*param, names, types));
  param->commonQuantParam.layer_decision_strict = true;
  EXPECT_FALSE(quant::LayerDecisionReporter::ValidatePerLayerConfig(*param, names, types));
}

// TC-TRC: origin-domain matching of the per-layer lists.
class QuantPerLayerMatchTest : public ::testing::Test {
 public:
  QuantPerLayerMatchTest() {
    graph_ = std::make_shared<FuncGraph>();
    conv_ = NewFp32Node(graph_, prim::kPrimConv2DFusion, "Conv2DFusion-3");
    bn_ = NewFp32Node(graph_, prim::kPrimAddFusion, "BN_3");
    opt::TraceOriginInfo(conv_, "Conv_3", "Conv");
    opt::TraceOriginInfo(bn_, "BN_3", "BatchNormalization");
    opt::TraceAbsorbNames(conv_, bn_);
    manager_ = Manage(graph_, true);
  }

  FuncGraphPtr graph_ = nullptr;
  FuncGraphManagerPtr manager_ = nullptr;
  CNodePtr conv_ = nullptr;
  CNodePtr bn_ = nullptr;
};

// TC-TRC-04: name and type domains hitting the same layer count once in the layer row, while
// every hitting entry is consumed (no false CONFIG_UNMATCHED status row).
TEST_F(QuantPerLayerMatchTest, MultiHitLayerConsumesAllEntries) {
  quant::QuantStrategy strategy(0, 0, {"BN_3"});
  strategy.SetPerLayerConfig({}, {}, {"Conv"});
  auto match = strategy.MatchPerLayerList(conv_);
  EXPECT_EQ(match.hit, quant::ListHitType::kSkipNode);
  EXPECT_EQ(match.matched_entry, "BN_3");
  auto entries = strategy.CollectMatchedEntries(conv_);
  EXPECT_EQ(entries.size(), 2);
  EXPECT_NE(entries.count("skip_quant_node=BN_3"), 0);
  EXPECT_NE(entries.count("skip_quant_op_type=Conv"), 0);
}

// TC-TRC-02: configuring the absorbed BN name keeps the whole fused conv layer in fp32.
TEST_F(QuantPerLayerMatchTest, AbsorbedNameHitsSurvivor) {
  // An explicitly empty config hits nothing.
  quant::QuantStrategy empty_strategy(0, 0, {});
  empty_strategy.SetPerLayerConfig({}, {}, {});
  EXPECT_EQ(empty_strategy.MatchPerLayerList(conv_).hit, quant::ListHitType::kNone);

  // skip_quant_node passed through the legacy ctor slot.
  quant::QuantStrategy skip_strategy(0, 0, {"BN_3"});
  auto match = skip_strategy.MatchPerLayerList(conv_);
  EXPECT_EQ(match.hit, quant::ListHitType::kSkipNode);
  EXPECT_EQ(match.matched_entry, "BN_3");

  // A node without tracing records (MindIR/MSLITE style) still matches by its current name:
  // the name domain is self+origin, so untraced frontends keep the legacy behavior.
  auto plain = NewFp32Node(graph_, prim::kPrimConv2DFusion, "Plain-0");
  quant::QuantStrategy plain_strategy(0, 0, {"Plain-0"});
  auto plain_match = plain_strategy.MatchPerLayerList(plain);
  EXPECT_EQ(plain_match.hit, quant::ListHitType::kSkipNode);
  EXPECT_EQ(plain_match.matched_entry, "Plain-0");
}

// TC-CFG-11: legacy current-name matching keeps working (survivor keeps its name).
// Legacy compatibility: the converted (current) name still hits the BLACKLIST name domain
// (upstream matched skip_quant_node by fullname only); the WHITELIST accepts origin names only.
TEST_F(QuantPerLayerMatchTest, CurrentNameHitsBlacklistOnly) {
  quant::QuantStrategy skip_strategy(0, 0, {"Conv2DFusion-3"});
  EXPECT_EQ(skip_strategy.MatchPerLayerList(conv_).hit, quant::ListHitType::kSkipNode);

  quant::QuantStrategy quant_strategy(0, 0, {});
  quant_strategy.SetPerLayerConfig({"Conv2DFusion-3"}, {}, {});
  EXPECT_EQ(quant_strategy.MatchPerLayerList(conv_).hit, quant::ListHitType::kNone);
  quant_strategy.SetPerLayerConfig({"Conv_3"}, {}, {});
  EXPECT_EQ(quant_strategy.MatchPerLayerList(conv_).hit, quant::ListHitType::kQuantNode);
}

// TC-CFG-06/07/TC-TRC-08: matching is exact and case-sensitive.
TEST_F(QuantPerLayerMatchTest, CaseSensitiveMatching) {
  quant::QuantStrategy strategy(0, 0, {"conv_3"});
  EXPECT_EQ(strategy.MatchPerLayerList(conv_).hit, quant::ListHitType::kNone);
  quant::QuantStrategy type_strategy(0, 0, {});
  type_strategy.SetPerLayerConfig({}, {}, {"conv"});
  EXPECT_EQ(type_strategy.MatchPerLayerList(conv_).hit, quant::ListHitType::kNone);
}

// TC-TRC-03: original op types hit (whitelist and blacklist alike).
TEST_F(QuantPerLayerMatchTest, OriginTypeHits) {
  quant::QuantStrategy strategy(0, 0, {});
  strategy.SetPerLayerConfig({}, {"Conv"}, {});
  auto match = strategy.MatchPerLayerList(conv_);
  EXPECT_EQ(match.hit, quant::ListHitType::kQuantOpType);
  EXPECT_TRUE(strategy.WhiteListActive());

  quant::QuantStrategy skip_type(0, 0, {});
  skip_type.SetPerLayerConfig({}, {}, {"BatchNormalization"});
  EXPECT_EQ(skip_type.MatchPerLayerList(conv_).hit, quant::ListHitType::kSkipOpType);
}

// TC-TRC-11: fused framework types are not config entries (origin domain only).
TEST_F(QuantPerLayerMatchTest, FrameworkTypeNotMatched) {
  quant::QuantStrategy strategy(0, 0, {});
  strategy.SetPerLayerConfig({}, {"Conv2DFusion"}, {});
  EXPECT_EQ(strategy.MatchPerLayerList(conv_).hit, quant::ListHitType::kNone);
}

// R4: reason codes for nodes rejected by the quantizer's own checks.
TEST_F(QuantPerLayerMatchTest, ReasonCodes) {
  quant::QuantStrategy strategy(0, 0, {});
  auto support = std::set<PrimitivePtr>{prim::kPrimConv2DFusion};

  std::string reason;
  EXPECT_TRUE(strategy.CanOpFullQuantizedWithReason(manager_, conv_, support, {}, {}, &reason));
  EXPECT_TRUE(reason.empty());

  // Structural node: Return is never quantized.
  auto ret_node = NewFp32Node(graph_, prim::kPrimReturn, "Return-0");
  EXPECT_FALSE(strategy.CanOpFullQuantizedWithReason(manager_, ret_node, support, {}, {}, &reason));
  EXPECT_EQ(reason, "STRUCTURAL");

  // Activation-typed node in the op support set whose activation type is outside the activation
  // support set (E2E-unreachable from onnx/tflite: every activation maps to its own primitive, so
  // this defensive branch is guarded at UT level).
  auto act_node = NewFp32Node(graph_, prim::kPrimActivation, "Act_0");
  auto act_prim = std::make_shared<mindspore::ops::Activation>();
  act_prim->set_activation_type(mindspore::GELU);
  act_node->set_input(0, NewValueNode(act_prim->GetPrim()));
  EXPECT_FALSE(strategy.CanOpFullQuantizedWithReason(manager_, act_node, {prim::kPrimActivation}, {},
                                                     {mindspore::RELU, mindspore::RELU6}, &reason));
  EXPECT_EQ(reason, "ACTIVATION_NOT_SUPPORTED");

  // Op outside the support set (DSP-style empty support set quantizes everything; a non-empty set
  // without the op rejects it).
  auto bn_check = NewFp32Node(graph_, prim::kPrimAddFusion, "BN_chk");

  // Nodes added after the fixture's Manage() must be synced into a rebuilt manager, otherwise
  // the strategy's manager queries (users/outputs) miss them.
  graph_->set_manager(nullptr);
  manager_ = Manage(graph_, true);
  EXPECT_FALSE(strategy.CanOpFullQuantizedWithReason(manager_, bn_check, support, {}, {}, &reason));
  EXPECT_EQ(reason, "OP_NOT_IN_SUPPORT_SET");
}
}  // namespace lite
}  // namespace mindspore
