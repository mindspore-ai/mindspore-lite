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

#include <cstring>
#include <memory>
#include <vector>
#include "common/common_test.h"
#include "include/errorcode.h"
#include "schema/inner/model_generated.h"
#include "src/common/utils.h"
#include "src/train/optimizer/fusion/conv_activation_fusion_pass.h"
#include "src/train/optimizer/fusion/conv_batchnorm_fusion_pass.h"

namespace mindspore {
namespace lite {
namespace {
constexpr int kChannel = 4;
constexpr int kInChannel = 3;
constexpr int kHeight = 5;
constexpr int kWidth = 5;
constexpr float kEps = 0.0f;
// BN params: scale 2, offset 1, mean 0.5, var 1 => scale_norm = 2.
constexpr float kScale = 2.0f;
constexpr float kOffset = 1.0f;
constexpr float kMean = 0.5f;
constexpr float kVar = 1.0f;
constexpr float kConvBias = 3.0f;

void SetTensorData(schema::TensorT *tensor, const std::vector<float> &values) {
  if (values.empty()) {
    return;
  }
  tensor->data.resize(values.size() * sizeof(float));
  std::memcpy(tensor->data.data(), values.data(), tensor->data.size());
}

std::unique_ptr<schema::TensorT> MakeTensor(const std::vector<int32_t> &dims, int node_type,
                                            const std::vector<float> &values = {}) {
  auto tensor = std::make_unique<schema::TensorT>();
  tensor->nodeType = node_type;
  tensor->format = schema::Format_NHWC;
  tensor->dataType = static_cast<int>(TypeId::kNumberTypeFloat32);
  tensor->dims = dims;
  SetTensorData(tensor.get(), values);
  return tensor;
}

const float *TensorData(const schema::MetaGraphT &graph, size_t index) {
  return reinterpret_cast<const float *>(graph.allTensors.at(index)->data.data());
}

// Tensors: 0=input, 1=weight, 2=conv output, 3=scale, 4=offset, 5=mean,
// 6=var, 7=BN output, and 8=conv bias when `with_bias` is set.
schema::MetaGraphT BuildConvBnGraph(bool with_bias, bool conv_has_activation, bool bad_bias) {
  schema::MetaGraphT graph;
  graph.name = "conv_bn";
  graph.inputIndex = {0};
  graph.outputIndex = {7};
  graph.allTensors.emplace_back(MakeTensor({1, kHeight, kWidth, kInChannel}, NodeType_Parameter));
  graph.allTensors.emplace_back(MakeTensor({kChannel, kInChannel, kInChannel, kInChannel}, NodeType_ValueNode,
                                           std::vector<float>(kChannel * kInChannel * kInChannel * kInChannel, 1.0f)));
  graph.allTensors.emplace_back(MakeTensor({1, kHeight, kWidth, kChannel}, NodeType_Parameter));
  graph.allTensors.emplace_back(MakeTensor({kChannel}, NodeType_ValueNode, std::vector<float>(kChannel, kScale)));
  graph.allTensors.emplace_back(MakeTensor({kChannel}, NodeType_ValueNode, std::vector<float>(kChannel, kOffset)));
  graph.allTensors.emplace_back(MakeTensor({kChannel}, NodeType_ValueNode, std::vector<float>(kChannel, kMean)));
  graph.allTensors.emplace_back(MakeTensor({kChannel}, NodeType_ValueNode, std::vector<float>(kChannel, kVar)));
  graph.allTensors.emplace_back(MakeTensor({1, kHeight, kWidth, kChannel}, NodeType_Parameter));
  if (with_bias) {
    const std::vector<int32_t> bias_dims = bad_bias ? std::vector<int32_t>{2} : std::vector<int32_t>{kChannel};
    const int bias_size = bad_bias ? 2 : kChannel;
    graph.allTensors.emplace_back(MakeTensor(bias_dims, NodeType_ValueNode, std::vector<float>(bias_size, kConvBias)));
  }

  auto conv = std::make_unique<schema::CNodeT>();
  conv->name = "Conv2D";
  conv->inputIndex = with_bias ? std::vector<uint32_t>{0, 1, 8} : std::vector<uint32_t>{0, 1};
  conv->outputIndex = {2};
  conv->primitive = std::make_unique<schema::PrimitiveT>();
  conv->primitive->value.type = schema::PrimitiveType_Conv2DFusion;
  auto *conv_value = new schema::Conv2DFusionT();
  conv_value->format = schema::Format_NHWC;
  conv_value->kernel_size = {3, 3};
  conv_value->stride = {1, 1};
  conv_value->dilation = {1, 1};
  conv_value->pad_mode = schema::PadMode_SAME;
  conv_value->group = 1;
  conv_value->in_channel = kInChannel;
  conv_value->out_channel = kChannel;
  conv_value->activation_type =
    conv_has_activation ? schema::ActivationType_RELU : schema::ActivationType_NO_ACTIVATION;
  conv->primitive->value.value = conv_value;
  graph.nodes.emplace_back(std::move(conv));

  auto bn = std::make_unique<schema::CNodeT>();
  bn->name = "BN";
  bn->inputIndex = {2, 3, 4, 5, 6};
  bn->outputIndex = {7};
  bn->primitive = std::make_unique<schema::PrimitiveT>();
  bn->primitive->value.type = schema::PrimitiveType_FusedBatchNorm;
  auto *bn_value = new schema::FusedBatchNormT();
  bn_value->epsilon = kEps;
  bn_value->momentum = 0.9f;
  bn_value->mode = 0;
  bn->primitive->value.value = bn_value;
  graph.nodes.emplace_back(std::move(bn));
  return graph;
}

// Tensors: 0=input, 1=weight, 2=conv output, 3=activation output.
schema::MetaGraphT BuildConvActGraph(schema::ActivationType act_type) {
  schema::MetaGraphT graph;
  graph.name = "conv_act";
  graph.inputIndex = {0};
  graph.outputIndex = {3};
  graph.allTensors.emplace_back(MakeTensor({1, kHeight, kWidth, kInChannel}, NodeType_Parameter));
  graph.allTensors.emplace_back(MakeTensor({kChannel, kInChannel, kInChannel, kInChannel}, NodeType_ValueNode,
                                           std::vector<float>(kChannel * kInChannel * kInChannel * kInChannel, 1.0f)));
  graph.allTensors.emplace_back(MakeTensor({1, kHeight, kWidth, kChannel}, NodeType_Parameter));
  graph.allTensors.emplace_back(MakeTensor({1, kHeight, kWidth, kChannel}, NodeType_Parameter));

  auto conv = std::make_unique<schema::CNodeT>();
  conv->name = "Conv2D";
  conv->inputIndex = {0, 1};
  conv->outputIndex = {2};
  conv->primitive = std::make_unique<schema::PrimitiveT>();
  conv->primitive->value.type = schema::PrimitiveType_Conv2DFusion;
  auto *conv_value = new schema::Conv2DFusionT();
  conv_value->format = schema::Format_NHWC;
  conv_value->kernel_size = {3, 3};
  conv_value->stride = {1, 1};
  conv_value->dilation = {1, 1};
  conv_value->pad_mode = schema::PadMode_SAME;
  conv_value->group = 1;
  conv_value->in_channel = kInChannel;
  conv_value->out_channel = kChannel;
  conv_value->activation_type = schema::ActivationType_NO_ACTIVATION;
  conv->primitive->value.value = conv_value;
  graph.nodes.emplace_back(std::move(conv));

  auto act = std::make_unique<schema::CNodeT>();
  act->name = "Activation";
  act->inputIndex = {2};
  act->outputIndex = {3};
  act->primitive = std::make_unique<schema::PrimitiveT>();
  act->primitive->value.type = schema::PrimitiveType_Activation;
  auto *act_value = new schema::ActivationT();
  act_value->activation_type = act_type;
  act->primitive->value.value = act_value;
  graph.nodes.emplace_back(std::move(act));
  return graph;
}
}  // namespace

class ConvFusionPassTest : public mindspore::CommonTest {};

TEST_F(ConvFusionPassTest, FoldConvBnWithoutBias) {
  auto graph = BuildConvBnGraph(false, false, false);
  ConvBatchNormFusionPass pass;
  EXPECT_EQ(pass.Run(&graph), RET_OK);

  ASSERT_EQ(graph.nodes.size(), 2U);
  ASSERT_EQ(graph.allTensors.size(), 9U);  // a bias tensor is appended
  const auto &conv = graph.nodes.at(0);
  ASSERT_EQ(conv->inputIndex.size(), 3U);
  EXPECT_EQ(conv->inputIndex.at(2), 8U);
  ASSERT_EQ(conv->outputIndex.size(), 1U);
  EXPECT_EQ(conv->outputIndex.at(0), 7U);
  EXPECT_TRUE(graph.nodes.at(1)->inputIndex.empty());
  EXPECT_TRUE(graph.nodes.at(1)->outputIndex.empty());

  const size_t weight_size = graph.allTensors.at(1)->data.size() / sizeof(float);
  for (size_t i = 0; i < weight_size; ++i) {
    EXPECT_FLOAT_EQ(TensorData(graph, 1)[i], kScale) << "weight index " << i;
  }
  // b' = offset - mean * scale_norm = 1 - 0.5 * 2 = 0
  for (int c = 0; c < kChannel; ++c) {
    EXPECT_FLOAT_EQ(TensorData(graph, 8)[c], 0.0f) << "bias index " << c;
  }
}

TEST_F(ConvFusionPassTest, FoldConvBnWithBias) {
  auto graph = BuildConvBnGraph(true, false, false);
  ConvBatchNormFusionPass pass;
  EXPECT_EQ(pass.Run(&graph), RET_OK);

  ASSERT_EQ(graph.nodes.size(), 2U);
  ASSERT_EQ(graph.allTensors.size(), 9U);
  ASSERT_EQ(graph.nodes.at(0)->inputIndex.size(), 3U);
  EXPECT_EQ(graph.nodes.at(0)->inputIndex.at(2), 8U);
  // b' = conv_bias * scale_norm + offset - mean * scale_norm = 3 * 2 + 0 = 6
  for (int c = 0; c < kChannel; ++c) {
    EXPECT_FLOAT_EQ(TensorData(graph, 8)[c], 6.0f) << "bias index " << c;
  }
}

TEST_F(ConvFusionPassTest, InvalidBiasLeavesGraphUntouched) {
  auto graph = BuildConvBnGraph(true, false, true);
  const size_t tensor_count = graph.allTensors.size();
  const size_t node_count = graph.nodes.size();

  ConvBatchNormFusionPass pass;
  EXPECT_EQ(pass.Run(&graph), RET_NO_CHANGE);

  // A late validation failure must not have scaled the weight or removed the BN.
  ASSERT_EQ(graph.allTensors.size(), tensor_count);
  ASSERT_EQ(graph.nodes.size(), node_count);
  const size_t weight_size = graph.allTensors.at(1)->data.size() / sizeof(float);
  for (size_t i = 0; i < weight_size; ++i) {
    EXPECT_FLOAT_EQ(TensorData(graph, 1)[i], 1.0f) << "weight index " << i;
  }
  EXPECT_EQ(graph.nodes.at(1)->inputIndex.size(), 5U);
  EXPECT_EQ(graph.nodes.at(1)->outputIndex.size(), 1U);
  EXPECT_EQ(graph.nodes.at(0)->outputIndex.at(0), 2U);
}

TEST_F(ConvFusionPassTest, ConvWithFusedActivationIsNotFolded) {
  auto graph = BuildConvBnGraph(false, true, false);
  ConvBatchNormFusionPass pass;
  EXPECT_EQ(pass.Run(&graph), RET_NO_CHANGE);

  ASSERT_EQ(graph.nodes.size(), 2U);
  EXPECT_EQ(graph.nodes.at(1)->inputIndex.size(), 5U);
  EXPECT_EQ(graph.nodes.at(0)->outputIndex.at(0), 2U);
}

TEST_F(ConvFusionPassTest, FuseSupportedActivation) {
  auto graph = BuildConvActGraph(schema::ActivationType_RELU6);
  ConvActivationFusionPass pass;
  EXPECT_EQ(pass.Run(&graph), RET_OK);

  ASSERT_EQ(graph.nodes.size(), 2U);
  const auto &conv = graph.nodes.at(0);
  ASSERT_EQ(conv->outputIndex.size(), 1U);
  EXPECT_EQ(conv->outputIndex.at(0), 3U);
  EXPECT_EQ(conv->primitive->value.AsConv2DFusion()->activation_type, schema::ActivationType_RELU6);
  EXPECT_TRUE(graph.nodes.at(1)->inputIndex.empty());
  EXPECT_TRUE(graph.nodes.at(1)->outputIndex.empty());
}

TEST_F(ConvFusionPassTest, UnsupportedActivationIsNotFused) {
  auto graph = BuildConvActGraph(schema::ActivationType_HSWISH);
  ConvActivationFusionPass pass;
  EXPECT_EQ(pass.Run(&graph), RET_NO_CHANGE);

  ASSERT_EQ(graph.nodes.size(), 2U);
  const auto &conv = graph.nodes.at(0);
  ASSERT_EQ(conv->outputIndex.size(), 1U);
  EXPECT_EQ(conv->outputIndex.at(0), 2U);
  EXPECT_EQ(conv->primitive->value.AsConv2DFusion()->activation_type, schema::ActivationType_NO_ACTIVATION);
  ASSERT_EQ(graph.nodes.at(1)->inputIndex.size(), 1U);
  EXPECT_EQ(graph.nodes.at(1)->inputIndex.at(0), 2U);
  EXPECT_EQ(graph.nodes.at(1)->primitive->value.AsActivation()->activation_type, schema::ActivationType_HSWISH);
}

TEST_F(ConvFusionPassTest, ConvOutputSharedWithGraphOutputIsNotFolded) {
  // The conv output is also a model output, so handing BN's output to conv would leave
  // that graph output without a producer.
  auto graph = BuildConvBnGraph(false, false, false);
  graph.outputIndex.push_back(2);
  const auto weight_before = graph.allTensors.at(1)->data;

  ConvBatchNormFusionPass pass;
  EXPECT_EQ(pass.Run(&graph), RET_NO_CHANGE);

  ASSERT_EQ(graph.nodes.at(0)->outputIndex.size(), 1U);
  EXPECT_EQ(graph.nodes.at(0)->outputIndex.at(0), 2U);
  EXPECT_EQ(graph.nodes.at(1)->inputIndex.size(), 5U);
  EXPECT_EQ(graph.allTensors.at(1)->data, weight_before);
}

TEST_F(ConvFusionPassTest, ConvOutputSharedWithGraphOutputIsNotActivationFused) {
  auto graph = BuildConvActGraph(schema::ActivationType_RELU);
  graph.outputIndex.push_back(2);

  ConvActivationFusionPass pass;
  EXPECT_EQ(pass.Run(&graph), RET_NO_CHANGE);

  ASSERT_EQ(graph.nodes.at(0)->outputIndex.size(), 1U);
  EXPECT_EQ(graph.nodes.at(0)->outputIndex.at(0), 2U);
  EXPECT_EQ(graph.nodes.at(0)->primitive->value.AsConv2DFusion()->activation_type,
            schema::ActivationType_NO_ACTIVATION);
  ASSERT_EQ(graph.nodes.at(1)->inputIndex.size(), 1U);
  EXPECT_EQ(graph.nodes.at(1)->inputIndex.at(0), 2U);
}
}  // namespace lite
}  // namespace mindspore
