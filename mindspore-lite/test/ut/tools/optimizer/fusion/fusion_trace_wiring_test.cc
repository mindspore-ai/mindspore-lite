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
// Wiring guards for fusion tracing: run the REAL fusion passes on small graphs and assert the
// survivor actually carries the absorbed node's name (the per-layer quant matching depends on it).
#include <memory>
#include <string>
#include <vector>
#include "common/common_test.h"
#include "ir/func_graph.h"
#include "ir/graph_utils.h"
#include "ir/manager.h"
#include "schema/inner/model_generated.h"
#include "test/common/import_from_meta_graphT.h"
#include "tools/optimizer/common/fusion_trace_utils.h"
#include "tools/optimizer/fusion/matmul_add_fusion.h"
#include "tools/optimizer/fusion/conv_scale_fusion.h"
#include "tools/optimizer/fusion/conv_biasadd_fusion.h"
#include "tools/optimizer/fusion/conv_activation_fusion.h"
#include "tools/converter/optimizer_manager.h"
#include "tools/converter/ms_depend/pass_manager.h"
#include "tools/converter/ms_depend/graph_optimizer.h"
#include "tools/optimizer/common/pass_manager_extends.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "mindspore/ops/op_def/lite_ops.h"
#include "ops_utils/op_utils.h"

namespace mindspore {
namespace {
using MetaGraphTptr = std::shared_ptr<schema::MetaGraphT>;
using CNodeTptr = std::unique_ptr<schema::CNodeT>;

std::unique_ptr<schema::TensorT> MakeTensor(const std::vector<int32_t> &dims, bool with_data) {
  auto tensor = std::make_unique<schema::TensorT>();
  tensor->nodeType = with_data ? lite::NodeType_ValueNode : lite::NodeType_Parameter;
  tensor->format = schema::Format_NHWC;
  tensor->dataType = TypeId::kNumberTypeFloat32;
  tensor->dims = dims;
  if (with_data) {
    size_t element_num = 1;
    for (int32_t dim : dims) {
      element_num *= static_cast<size_t>(dim);
    }
    tensor->data.resize(sizeof(float) * element_num);  // full-size const data, as the fusions expect
  }
  tensor->offset = -1;
  return tensor;
}

// ---------------- matmul + add ----------------
constexpr int kInputDimsM = 128;
constexpr int kInputDimsK = 225;
constexpr int kInputDimsN = 98;

CNodeTptr BuildMatMul() {
  auto matmul_node = std::make_unique<schema::CNodeT>();
  matmul_node->inputIndex = {0, 1};
  matmul_node->outputIndex = {2};
  matmul_node->primitive = std::make_unique<schema::PrimitiveT>();
  matmul_node->primitive->value.type = schema::PrimitiveType_MatMulFusion;
  auto prim = new schema::MatMulFusionT;
  prim->transpose_a = false;
  prim->transpose_b = false;
  matmul_node->primitive->value.value = prim;
  matmul_node->name = "MatMul";
  return matmul_node;
}

CNodeTptr BuildAdd() {
  auto add_node = std::make_unique<schema::CNodeT>();
  add_node->inputIndex = {2, 3};
  add_node->outputIndex = {4};
  add_node->primitive = std::make_unique<schema::PrimitiveT>();
  add_node->primitive->value.type = schema::PrimitiveType_AddFusion;
  auto prim = new schema::AddFusionT;
  prim->activation_type = schema::ActivationType_NO_ACTIVATION;
  add_node->primitive->value.value = prim;
  add_node->name = "Add";
  return add_node;
}

MetaGraphTptr BuildMatMulAddGraph() {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";
  meta_graph->nodes.emplace_back(BuildMatMul());
  meta_graph->nodes.emplace_back(BuildAdd());
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {4};
  meta_graph->allTensors.emplace_back(MakeTensor({kInputDimsM, kInputDimsK}, false));  // 0: x
  meta_graph->allTensors.emplace_back(MakeTensor({kInputDimsK, kInputDimsN}, true));   // 1: w (const)
  meta_graph->allTensors.emplace_back(MakeTensor({kInputDimsM, kInputDimsN}, false));  // 2: matmul out
  meta_graph->allTensors.emplace_back(MakeTensor({kInputDimsN}, true));                // 3: add const bias
  meta_graph->allTensors.emplace_back(MakeTensor({kInputDimsM, kInputDimsN}, false));  // 4: final out
  return meta_graph;
}

// ---------------- conv + scale (BN merged into ScaleFusion by batchnorm_to_scale) ----------------
constexpr int kConvInChannel = 3;
constexpr int kConvOutChannel = 8;

CNodeTptr BuildConv2D() {
  auto conv_node = std::make_unique<schema::CNodeT>();
  conv_node->inputIndex = {0, 1};
  conv_node->outputIndex = {2};
  conv_node->primitive = std::make_unique<schema::PrimitiveT>();
  conv_node->primitive->value.type = schema::PrimitiveType_Conv2DFusion;
  auto prim = new schema::Conv2DFusionT;
  prim->format = schema::Format_NHWC;
  prim->kernel_size = {3, 3};
  prim->stride = {1, 1};
  prim->dilation = {1, 1};
  prim->pad_list = {0, 0, 0, 0};
  prim->group = 1;
  prim->in_channel = kConvInChannel;
  prim->out_channel = kConvOutChannel;
  prim->activation_type = schema::ActivationType_NO_ACTIVATION;
  conv_node->primitive->value.value = prim;
  conv_node->name = "Conv2D";
  return conv_node;
}

CNodeTptr BuildScale() {
  auto scale_node = std::make_unique<schema::CNodeT>();
  scale_node->inputIndex = {2, 3, 4};
  scale_node->outputIndex = {5};
  scale_node->primitive = std::make_unique<schema::PrimitiveT>();
  scale_node->primitive->value.type = schema::PrimitiveType_ScaleFusion;
  auto prim = new schema::ScaleFusionT;
  prim->axis = -1;
  prim->activation_type = schema::ActivationType_NO_ACTIVATION;
  scale_node->primitive->value.value = prim;
  scale_node->name = "Scale";
  return scale_node;
}

MetaGraphTptr BuildConvScaleGraph() {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";
  meta_graph->nodes.emplace_back(BuildConv2D());
  meta_graph->nodes.emplace_back(BuildScale());
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {5};
  meta_graph->allTensors.emplace_back(MakeTensor({1, 56, 56, kConvInChannel}, false));             // 0: x
  meta_graph->allTensors.emplace_back(MakeTensor({3, 3, kConvInChannel, kConvOutChannel}, true));  // 1: w
  meta_graph->allTensors.emplace_back(MakeTensor({1, 54, 54, kConvOutChannel}, false));            // 2: conv out
  meta_graph->allTensors.emplace_back(MakeTensor({kConvOutChannel}, true));                        // 3: scale const
  meta_graph->allTensors.emplace_back(MakeTensor({kConvOutChannel}, true));                        // 4: scale bias
  meta_graph->allTensors.emplace_back(MakeTensor({1, 54, 54, kConvOutChannel}, false));            // 5: final out
  return meta_graph;
}

CNodeTptr BuildRelu() {
  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {2};
  node->outputIndex = {3};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_Activation;
  auto prim = new schema::ActivationT;
  prim->activation_type = schema::ActivationType_RELU;
  node->primitive->value.value = prim;
  node->name = "Relu";
  return node;
}

MetaGraphTptr BuildConvActGraph() {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";
  meta_graph->nodes.emplace_back(BuildConv2D());
  meta_graph->nodes.emplace_back(BuildRelu());
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {3};
  meta_graph->allTensors.emplace_back(MakeTensor({1, 56, 56, kConvInChannel}, false));             // 0: x
  meta_graph->allTensors.emplace_back(MakeTensor({3, 3, kConvInChannel, kConvOutChannel}, true));  // 1: w
  meta_graph->allTensors.emplace_back(MakeTensor({1, 54, 54, kConvOutChannel}, false));            // 2: conv out
  meta_graph->allTensors.emplace_back(MakeTensor({1, 54, 54, kConvOutChannel}, false));            // 3: final out
  return meta_graph;
}

CNodeTptr BuildBiasAdd() {
  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {2, 4};
  node->outputIndex = {5};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_AddFusion;
  auto prim = new schema::AddFusionT;
  prim->activation_type = schema::ActivationType_NO_ACTIVATION;
  node->primitive->value.value = prim;
  node->name = "Add";
  return node;
}

MetaGraphTptr BuildConvBiasGraph() {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";
  meta_graph->nodes.emplace_back(BuildConv2D());
  meta_graph->nodes.emplace_back(BuildBiasAdd());
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {5};
  meta_graph->allTensors.emplace_back(MakeTensor({1, 56, 56, kConvInChannel}, false));             // 0: x
  meta_graph->allTensors.emplace_back(MakeTensor({3, 3, kConvInChannel, kConvOutChannel}, true));  // 1: w
  meta_graph->allTensors.emplace_back(MakeTensor({1, 54, 54, kConvOutChannel}, false));            // 2: conv out
  meta_graph->allTensors.emplace_back(MakeTensor({1, 54, 54, kConvOutChannel}, false));            // 3(unused)
  meta_graph->allTensors.emplace_back(MakeTensor({kConvOutChannel}, true));                        // 4: bias const
  meta_graph->allTensors.emplace_back(MakeTensor({1, 54, 54, kConvOutChannel}, false));            // 5: final out
  return meta_graph;
}

// Find the first cnode (excluding Return) whose carried names contain `token`.
CNodePtr FindNodeByCarriedName(const FuncGraphPtr &func_graph, const std::string &token) {
  for (auto &cnode : func_graph->GetOrderedCnodes()) {
    for (const auto &name : opt::GetSelfAndOriginNames(cnode)) {
      if (name.find(token) != std::string::npos) {
        return cnode;
      }
    }
  }
  return nullptr;
}
}  // namespace

class FusionTraceWiringTest : public mindspore::CommonTest {
 public:
  FusionTraceWiringTest() = default;
};

// MatMul+Add (in-place absorb): after the real fusion runs, the surviving MatMul must carry the
// Add node's name, so a per-layer list entry naming the absorbed Add still hits the layer.
TEST_F(FusionTraceWiringTest, MatMulAddCarriesAbsorbedName) {
  auto meta_graph = BuildMatMulAddGraph();
  auto func_graph = lite::AnfImporterFromMetaGraphT::Fb2Anf(meta_graph.get());
  ASSERT_NE(func_graph, nullptr);
  auto manager = Manage(func_graph, true);
  ASSERT_NE(manager, nullptr);
  opt::MatMulAddFusion fusion;
  (void)fusion.Run(func_graph);

  // The Add is folded: exactly one compute cnode (the MatMul survivor) remains.
  size_t compute_nodes = 0;
  for (auto &cnode : func_graph->GetOrderedCnodes()) {
    if (!opt::CheckPrimitiveType(cnode, prim::kPrimReturn)) {
      compute_nodes++;
    }
  }
  ASSERT_EQ(compute_nodes, 1);

  auto survivor = FindNodeByCarriedName(func_graph, "MatMul");
  ASSERT_NE(survivor, nullptr);
  bool carries_add = false;
  for (const auto &name : opt::GetSelfAndOriginNames(survivor)) {
    if (name.find("Add") != std::string::npos) {
      carries_add = true;
    }
  }
  EXPECT_TRUE(carries_add);
}

// Conv+bias (ConvBiasaddFusion): the surviving Conv must carry the absorbed Add's name.
TEST_F(FusionTraceWiringTest, ConvBiasCarriesAbsorbedName) {
  auto meta_graph = BuildConvBiasGraph();
  auto func_graph = lite::AnfImporterFromMetaGraphT::Fb2Anf(meta_graph.get());
  ASSERT_NE(func_graph, nullptr);
  auto manager = Manage(func_graph, true);
  ASSERT_NE(manager, nullptr);
  auto fusion_pm = std::make_shared<opt::LitePassManager>("conv_bias_wiring", false);
  ASSERT_NE(fusion_pm, nullptr);
  fusion_pm->AddPass(std::make_shared<opt::ConvBiasaddFusion>());
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  ASSERT_NE(optimizer, nullptr);
  optimizer->AddPassManager(fusion_pm);
  ASSERT_NE(optimizer->Optimize(func_graph), nullptr);

  auto survivor = FindNodeByCarriedName(func_graph, "Conv2D");
  ASSERT_NE(survivor, nullptr);
  bool carries_add = false;
  for (const auto &name : opt::GetSelfAndOriginNames(survivor)) {
    if (name.find("Add") != std::string::npos) {
      carries_add = true;
    }
  }
  EXPECT_TRUE(carries_add);
}

// Conv+activation (ConvActivationFusion): the surviving Conv must carry the absorbed Relu's name.
TEST_F(FusionTraceWiringTest, ConvActivationCarriesAbsorbedName) {
  auto meta_graph = BuildConvActGraph();
  auto func_graph = lite::AnfImporterFromMetaGraphT::Fb2Anf(meta_graph.get());
  ASSERT_NE(func_graph, nullptr);
  auto manager = Manage(func_graph, true);
  ASSERT_NE(manager, nullptr);
  auto fusion_pm = std::make_shared<opt::LitePassManager>("conv_act_wiring", false);
  ASSERT_NE(fusion_pm, nullptr);
  fusion_pm->AddPass(std::make_shared<opt::ConvActivationFusion>(std::make_shared<ConverterPara>()));
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  ASSERT_NE(optimizer, nullptr);
  optimizer->AddPassManager(fusion_pm);
  ASSERT_NE(optimizer->Optimize(func_graph), nullptr);

  auto survivor = FindNodeByCarriedName(func_graph, "Conv2D");
  ASSERT_NE(survivor, nullptr);
  bool carries_relu = false;
  for (const auto &name : opt::GetSelfAndOriginNames(survivor)) {
    if (name.find("Relu") != std::string::npos) {
      carries_relu = true;
    }
  }
  EXPECT_TRUE(carries_relu);
}

// Conv+Scale (the ConvTransformFusion family; a real BN reaches here as ScaleFusion after
// batchnorm_to_scale): after the real fusion runs, the surviving Conv must carry the Scale
// node's name, so a per-layer list entry naming the absorbed component still hits the layer.
TEST_F(FusionTraceWiringTest, ConvScaleCarriesAbsorbedName) {
  auto meta_graph = BuildConvScaleGraph();
  auto func_graph = lite::AnfImporterFromMetaGraphT::Fb2Anf(meta_graph.get());
  ASSERT_NE(func_graph, nullptr);
  auto manager = Manage(func_graph, true);
  ASSERT_NE(manager, nullptr);
  // Drive the fusion exactly like AnfTransform does: GraphOptimizer + LitePassManager, whose
  // per-node traversal routes PrimitiveC nodes through the name-matching pattern engine.
  auto fusion_pm = std::make_shared<opt::LitePassManager>("conv_scale_wiring", false);
  ASSERT_NE(fusion_pm, nullptr);
  fusion_pm->AddPass(std::make_shared<opt::ConvScaleFusion>(converter::kFmkTypeTflite));
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  ASSERT_NE(optimizer, nullptr);
  optimizer->AddPassManager(fusion_pm);
  ASSERT_NE(optimizer->Optimize(func_graph), nullptr);

  // The Scale is folded into the conv weight: exactly one compute cnode remains.
  size_t compute_nodes = 0;
  for (auto &cnode : func_graph->GetOrderedCnodes()) {
    if (!opt::CheckPrimitiveType(cnode, prim::kPrimReturn)) {
      compute_nodes++;
    }
  }
  ASSERT_EQ(compute_nodes, 1);

  auto survivor = FindNodeByCarriedName(func_graph, "Conv2D");
  ASSERT_NE(survivor, nullptr);
  bool carries_scale = false;
  for (const auto &name : opt::GetSelfAndOriginNames(survivor)) {
    if (name.find("Scale") != std::string::npos) {
      carries_scale = true;
    }
  }
  EXPECT_TRUE(carries_scale);
}
}  // namespace mindspore
