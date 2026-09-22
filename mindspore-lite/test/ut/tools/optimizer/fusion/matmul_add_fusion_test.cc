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
#include "schema/inner/model_generated.h"
#include "common/common_test.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "test/common/import_from_meta_graphT.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "ir/func_graph.h"
#include "ir/graph_utils.h"
#include "mindspore/ops/op_def/array_ops.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "ops_utils/op_utils.h"
#include "tools/optimizer/fusion/matmul_add_fusion.h"

namespace mindspore {
namespace {
constexpr int kInputDimsM = 128;
constexpr int kInputDimsK = 225;
constexpr int kInputDimsN = 98;
}  // namespace

class MatMulAddFusionTest : public mindspore::CommonTest {
 public:
  MatMulAddFusionTest() = default;
};

using MetaGraphTptr = std::shared_ptr<schema::MetaGraphT>;
using CNodeTptr = std::unique_ptr<schema::CNodeT>;

namespace {
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

std::unique_ptr<schema::TensorT> MakeTensor(const std::vector<int32_t> &dims, bool with_data) {
  auto tensor = std::make_unique<schema::TensorT>();
  tensor->nodeType = with_data ? lite::NodeType_ValueNode : lite::NodeType_Parameter;
  tensor->format = schema::Format_NHWC;
  tensor->dataType = TypeId::kNumberTypeFloat32;
  tensor->dims = dims;
  if (with_data) {
    tensor->data.resize(sizeof(float));
    for (int32_t i = 0; i < dims[0]; i++) {
      (void)tensor->data.emplace_back(0);
    }
  }
  tensor->offset = -1;
  return tensor;
}

// weight_with_data == false builds the ticket topology: the matmul weight is a graph input
// (no const data), e.g. the online second input of a batch-matmul.
MetaGraphTptr BuildGraph(bool weight_with_data) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";
  meta_graph->nodes.emplace_back(BuildMatMul());
  meta_graph->nodes.emplace_back(BuildAdd());

  // graph input x (tensor 0) is always variable; tensor 1 (weight) is variable or const.
  meta_graph->inputIndex = {0};
  if (!weight_with_data) {
    meta_graph->inputIndex.push_back(1);
  }
  meta_graph->outputIndex = {4};

  meta_graph->allTensors.emplace_back(MakeTensor({kInputDimsM, kInputDimsK}, false));             // 0: x
  meta_graph->allTensors.emplace_back(MakeTensor({kInputDimsK, kInputDimsN}, weight_with_data));  // 1: w
  meta_graph->allTensors.emplace_back(MakeTensor({kInputDimsM, kInputDimsN}, false));             // 2: matmul out
  meta_graph->allTensors.emplace_back(MakeTensor({kInputDimsN}, true));                           // 3: add const bias
  meta_graph->allTensors.emplace_back(MakeTensor({kInputDimsM, kInputDimsN}, false));             // 4: final out
  return meta_graph;
}

size_t CNodeCount(const FuncGraphPtr &func_graph) {
  size_t count = 0;
  for (auto &node : TopoSort(func_graph->get_return())) {
    if (utils::isa<CNode>(node) && !opt::CheckPrimitiveType(node, prim::kPrimReturn)) {
      count++;
    }
  }
  return count;
}
}  // namespace

// A variable matmul weight (no const data) must NOT absorb the Add: the folded bias can only
// be quantized offline with the weight scales, which do not exist for online weights.
TEST_F(MatMulAddFusionTest, TestNotFusedWithVariableWeight) {
  auto meta_graph = BuildGraph(false);
  auto func_graph = lite::AnfImporterFromMetaGraphT::Fb2Anf(meta_graph.get());
  ASSERT_NE(func_graph, nullptr);
  auto manager = Manage(func_graph, true);
  opt::MatMulAddFusion fusion;
  (void)fusion.Run(func_graph);
  ASSERT_EQ(CNodeCount(func_graph), 2);
  MS_LOG(INFO) << "Passed";
}

// The classic constant-weight matmul + add still fuses into a single matmul-with-bias node.
TEST_F(MatMulAddFusionTest, TestFusedWithConstantWeight) {
  auto meta_graph = BuildGraph(true);
  auto func_graph = lite::AnfImporterFromMetaGraphT::Fb2Anf(meta_graph.get());
  ASSERT_NE(func_graph, nullptr);
  auto manager = Manage(func_graph, true);
  opt::MatMulAddFusion fusion;
  (void)fusion.Run(func_graph);
  ASSERT_EQ(CNodeCount(func_graph), 1);
  MS_LOG(INFO) << "Passed";
}
}  // namespace mindspore
