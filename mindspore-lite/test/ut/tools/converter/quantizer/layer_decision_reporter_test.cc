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
#include <algorithm>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include "common/common_test.h"
#include "ir/func_graph.h"
#include "ir/manager.h"
#include "mindspore/ops/op_def/nn_ops.h"
#include "mindspore/ops/op_def/lite_ops.h"
#include "mindspore/ops/op_def/array_ops.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_r.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "tools/converter/quantizer/layer_decision_reporter.h"
#include "tools/converter/quantizer/quant_params.h"
#include "tools/optimizer/common/fusion_trace_utils.h"

namespace mindspore {
namespace lite {
namespace {
CNodePtr NewFp32Node(const FuncGraphPtr &graph, const PrimitivePtr &prim, const std::string &name,
                     const AnfNodePtr &input = nullptr) {
  auto param = graph->add_parameter();
  param->set_name(name + "_in");
  param->set_abstract(std::make_shared<abstract::AbstractTensor>(kFloat32, std::vector<int64_t>{3, 5}));
  auto cnode =
    (input == nullptr) ? graph->NewCNode({NewValueNode(prim), param}) : graph->NewCNode({NewValueNode(prim), input});
  cnode->set_fullname_with_scope(name);
  cnode->set_abstract(std::make_shared<abstract::AbstractTensor>(kFloat32, std::vector<int64_t>{3, 5}));
  graph->set_output(cnode);
  return cnode;
}

std::string ReadFile(const std::string &path) {
  std::ifstream in(path);
  std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  return content;
}
}  // namespace

// TC-RPT-01: the 12-column order is frozen, origin domain first.
TEST(LayerDecisionReporterTest, CsvHeaderFrozen) {
  auto header = quant::LayerDecisionReporter::CsvHeader();
  ASSERT_EQ(header.size(), 12);
  EXPECT_EQ(header[0], "OriginNames");
  EXPECT_EQ(header[1], "OriginOpTypes");
  EXPECT_EQ(header[2], "NodeName");
  EXPECT_EQ(header[3], "NodeType");
  EXPECT_EQ(header[4], "OriginDataType");
  EXPECT_EQ(header[8], "MicroCoderAvailable");
}

// TC-RPT-08: formula-injection escaping plus standard CSV quoting.
TEST(LayerDecisionReporterTest, EscapeCsvField) {
  EXPECT_EQ(quant::LayerDecisionReporter::EscapeCsvField("=cmd"), "'=cmd");
  EXPECT_EQ(quant::LayerDecisionReporter::EscapeCsvField("+1"), "'+1");
  EXPECT_EQ(quant::LayerDecisionReporter::EscapeCsvField("-x"), "'-x");
  EXPECT_EQ(quant::LayerDecisionReporter::EscapeCsvField("@f"), "'@f");
  EXPECT_EQ(quant::LayerDecisionReporter::EscapeCsvField("plain"), "plain");
  EXPECT_EQ(quant::LayerDecisionReporter::EscapeCsvField("a,b"), "\"a,b\"");
  EXPECT_EQ(quant::LayerDecisionReporter::EscapeCsvField("a\"b"), "\"a\"\"b\"");
}

TEST(LayerDecisionReporterTest, BuildCsvContentPadsToColumnCount) {
  std::vector<std::vector<std::string>> rows{{"a", "b"}};
  auto content = quant::LayerDecisionReporter::BuildCsvContent(rows);
  auto line = content.substr(0, content.find('\n'));
  // 12 columns: 2 given + 10 empty.
  EXPECT_EQ(std::count(line.begin(), line.end(), ','), 11);
}

// TC-CFG-10: multi-model report naming inserts the model name before the extension.
TEST(LayerDecisionReporterTest, InsertModelName) {
  using quant::LayerDecisionReporter;
  EXPECT_EQ(LayerDecisionReporter::InsertModelName("./out/a.csv", "net"), "./out/a_net.csv");
  EXPECT_EQ(LayerDecisionReporter::InsertModelName("a.csv", "net"), "a_net.csv");
  EXPECT_EQ(LayerDecisionReporter::InsertModelName("dir/x", "net"), "dir/x_net");
  EXPECT_EQ(LayerDecisionReporter::InsertModelName("a.csv", ""), "a.csv");
}

TEST(LayerDecisionReporterTest, RecorderStoresDecisionsAndSnapshot) {
  auto &recorder = quant::LayerDecisionRecorder::GetInstance();
  recorder.Clear();
  quant::LayerDecisionInfo info;
  info.quantized = false;
  info.reason = quant::QuantReportReason::kSkipNode;
  info.matched_config = "skip_quant_node=BN-8";
  recorder.Record("Conv2DFusion-8", info);
  quant::LayerDecisionInfo got;
  EXPECT_TRUE(recorder.Get("Conv2DFusion-8", &got));
  EXPECT_EQ(got.reason, "SKIP_NODE");
  recorder.SetPreOptSnapshot({"gelu-3"});
  EXPECT_TRUE(recorder.InPreOptSnapshot("gelu-3"));
  EXPECT_FALSE(recorder.InPreOptSnapshot("Reshape-88"));
  recorder.Clear();
  EXPECT_FALSE(recorder.Get("Conv2DFusion-8", &got));
}

// TC-CON-01/02/03/05: the prediction column is a registry lookup that mirrors codegen's key:
// quantized layers look up the int8 coder, weight_fp16 folds unquantized fp32 layers to fp16,
// ConstantOfShape is registered by its OUTPUT dtype, and without micro the column is N/A.
TEST(LayerDecisionReporterTest, PredictCoderAvailableContract) {
  auto make_param = []() {
    auto p = std::make_shared<ConverterPara>();
    p->microParam.enable_micro = true;
    p->microParam.target = "x86";
    return p;
  };
  auto graph = std::make_shared<FuncGraph>();

  // TC-CON-01 baseline: a QUANT_ALL conv looks up the x86 int8 conv coder -> true.
  auto quant_conv = NewFp32Node(graph, prim::kPrimConv2DFusion, "ConvQ");
  GetValueNode<PrimitivePtr>(quant_conv->input(0))
    ->AddAttr(quant::kQuantType, MakeValue(static_cast<int>(quant::QUANT_ALL)));
  quant::LayerDecisionReporter baseline(make_param());
  EXPECT_EQ(baseline.PredictCoderAvailable(quant_conv), "true");

  // TC-CON-05 degenerate: micro disabled -> N/A for every layer.
  auto no_micro = make_param();
  no_micro->microParam.enable_micro = false;
  quant::LayerDecisionReporter na(no_micro);
  EXPECT_EQ(na.PredictCoderAvailable(quant_conv), "N/A");

  // TC-CON-02 fp16 fold: Softmax fp16 is registered only for ARM, so on x86 an unquantized fp32
  // Softmax folds to fp16 and predicts false; the delegated fp16 Conv still predicts true —
  // together the two lock the fold itself AND the registry per dtype.
  auto fp32_softmax = NewFp32Node(graph, prim::kPrimSoftmax, "SoftmaxFp32");
  auto fp32_conv = NewFp32Node(graph, prim::kPrimConv2DFusion, "ConvFp32");
  auto fp16_param = make_param();
  fp16_param->weight_fp16 = true;
  quant::LayerDecisionReporter folded(fp16_param);
  EXPECT_EQ(folded.PredictCoderAvailable(fp32_softmax), "false");
  EXPECT_EQ(folded.PredictCoderAvailable(fp32_conv), "true");

  // TC-CON-03 ConstantOfShape is registered by OUTPUT dtype (fp32) -> true.
  auto cos = NewFp32Node(graph, prim::kPrimConstantOfShape, "COS_0");
  quant::LayerDecisionReporter by_output(make_param());
  EXPECT_EQ(by_output.PredictCoderAvailable(cos), "true");
}

// TC-RPT-10/12 + TC-TRC-06: full report generation including status rows.
TEST(LayerDecisionReporterTest, GenerateReportWithStatusRows) {
  auto &recorder = quant::LayerDecisionRecorder::GetInstance();
  recorder.Clear();

  auto graph = std::make_shared<FuncGraph>();
  auto conv = NewFp32Node(graph, prim::kPrimConv2DFusion, "Conv2DFusion-3");
  opt::TraceOriginInfo(conv, "Conv_3", "Conv");
  auto reshape = NewFp32Node(graph, prim::kPrimReshape, "Reshape-20", conv);
  (void)reshape;
  (void)Manage(graph, true);

  // Conv: hit the blacklist via the absorbed component name and is not quantized.
  quant::LayerDecisionInfo conv_info;
  conv_info.quantized = false;
  conv_info.reason = quant::QuantReportReason::kSkipNode;
  conv_info.matched_config = "skip_quant_node=BN-8";
  recorder.Record("Conv2DFusion-3", conv_info);
  // Reshape: no record and no quant_type attribute -> NOT_QUANTIZED with the out-of-pass default
  // reason (asserted below), so every layer row still explains itself.

  recorder.SetPreOptSnapshot({"Conv_3", "gelu-3"});

  auto param = std::make_shared<ConverterPara>();
  param->model_file = "/tmp/net.onnx";
  param->commonQuantParam.quant_type = quant::QUANT_ALL;
  param->commonQuantParam.skip_quant_node = {"Reshape-88", "gelu-3", "BN-8"};
  std::string path = "/tmp/layer_decision_reporter_test.csv";
  remove(path.c_str());
  // The template gets the model name inserted; compute the same final path for reading.
  param->commonQuantParam.layer_decision_report = path;

  quant::LayerDecisionReporter reporter(param);
  EXPECT_EQ(reporter.Generate(graph), lite::RET_OK);

  auto final_path = quant::LayerDecisionReporter::InsertModelName(path, "net");
  auto content = ReadFile(final_path);
  remove(final_path.c_str());

  // Header first.
  EXPECT_NE(content.find("OriginNames,OriginOpTypes,NodeName"), std::string::npos);
  // Layer row: origin columns populated, decision and matched config recorded.
  EXPECT_NE(content.find("Conv_3"), std::string::npos);
  EXPECT_NE(content.find("Conv2DFusion-3"), std::string::npos);
  EXPECT_NE(content.find("SKIP_NODE"), std::string::npos);
  EXPECT_NE(content.find("skip_quant_node=BN-8"), std::string::npos);
  // Micro not enabled: prediction column is N/A.
  EXPECT_NE(content.find("N/A"), std::string::npos);
  // Unrecorded layer still reports a reason instead of an empty one, and the Reshape row is
  // the NOT_QUANTIZED one carrying it (boundary casts are QUANTIZED).
  EXPECT_NE(content.find("OUTSIDE_MARK_PASS"), std::string::npos);
  EXPECT_NE(content.find("Reshape-20"), std::string::npos);
  EXPECT_NE(content.find(",NOT_QUANTIZED,OUTSIDE_MARK_PASS,"), std::string::npos);
  // Status rows: never-existed entry -> CONFIG_UNMATCHED; swallowed by uncovered fusion ->
  // FUSED_INVALID. Matched entries (BN-8) get no status row.
  EXPECT_NE(content.find("CONFIG_UNMATCHED"), std::string::npos);
  EXPECT_NE(content.find("skip_quant_node=Reshape-88"), std::string::npos);
  EXPECT_NE(content.find("FUSED_INVALID"), std::string::npos);
  EXPECT_NE(content.find("skip_quant_node=gelu-3"), std::string::npos);
  // Status row layout: layer description columns empty, status in NodeType (4th), entry in
  // MatchedConfig (8th): ",,," prefix then empties up to the 8th column.
  EXPECT_NE(content.find(",,,FUSED_INVALID,,,,skip_quant_node=gelu-3"), std::string::npos);

  recorder.Clear();
}
}  // namespace lite
}  // namespace mindspore
