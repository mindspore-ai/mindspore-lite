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

#include <iostream>
#include <memory>
#include <vector>
#include "ut/src/runtime/kernel/dsp/pnna/pnna_test.h"
#include "include/api/context.h"
#include "include/api/data_type.h"
#include "include/api/model.h"
#include "schema/inner/model_generated.h"
#include "src/litert/kernel_registry.h"

namespace mindspore::lite::dsp::test {
class TestPNNA_Reshape : public CommonTest {};

TEST_F(TestPNNA_Reshape, reshape_2_3_to_1_6) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";

  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1};
  node->outputIndex = {2};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_Reshape;
  auto primitive = new schema::ReshapeT;
  node->primitive->value.value = primitive;
  node->name = "Reshape";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {2};

  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = lite::NodeType_Parameter;
  input0->dataType = TypeId::kNumberTypeFloat32;
  input0->dims = {2, 3};
  input0->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input0));

  auto shape = std::make_unique<schema::TensorT>();
  shape->nodeType = lite::NodeType_ValueNode;
  shape->dataType = TypeId::kNumberTypeInt32;
  shape->dims = {2};
  shape->offset = -1;
  std::vector<int32_t> shape_data = {1, 6};
  shape->data.resize(sizeof(int32_t) * 2);
  memcpy(shape->data.data(), shape_data.data(), 2 * sizeof(int32_t));
  meta_graph->allTensors.emplace_back(std::move(shape));

  auto output = std::make_unique<schema::TensorT>();
  output->nodeType = lite::NodeType_Parameter;
  output->dataType = TypeId::kNumberTypeFloat32;
  output->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(output));

  flatbuffers::FlatBufferBuilder builder(1024);
  auto offset = schema::MetaGraph::Pack(builder, meta_graph.get());
  builder.Finish(offset);
  schema::FinishMetaGraphBuffer(builder, offset);
  size_t size = builder.GetSize();
  const char *content = reinterpret_cast<char *>(builder.GetBufferPointer());

  // create a context
  auto context = std::make_shared<mindspore::Context>();
  context->SetBuiltInDelegate(mindspore::DelegateMode::kPNNA);
  auto &device_list = context->MutableDeviceInfo();
  std::shared_ptr<DSPDeviceInfo> device_info = std::make_shared<DSPDeviceInfo>();
  device_list.push_back(device_info);

  // build a model
  auto model = std::make_shared<mindspore::Model>();
  auto ret = model->Build(content, size, kMindIR_Lite, context);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  auto inputs = model->GetInputs();
  ASSERT_EQ(inputs.size(), 1);
  auto inTensor = inputs.front();
  auto impl = inTensor.impl();
  ASSERT_NE(nullptr, impl);
  float *in0_data = static_cast<float *>(inTensor.MutableData());
  std::vector<float> input_data = {
    1, 2, 3, 1, 2, 3,
  };
  std::vector<float> expect = {
    1, 2, 3, 1, 2, 3,
  };
  for (size_t i = 0; i < input_data.size(); ++i) in0_data[i] = input_data[i];
  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);
  auto *outData = reinterpret_cast<const float *>(outputs.front().Data().get());
  ASSERT_EQ(0, CompareOutputData(outData, expect.data(), expect.size()));
}

TEST_F(TestPNNA_Reshape, reshape_2_3_to_1_6_int8) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";

  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1};
  node->outputIndex = {2};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_Reshape;
  auto primitive = new schema::ReshapeT;
  node->primitive->value.value = primitive;
  node->name = "Reshape";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {2};

  float input_min = 0.0, input_max = 3.0, output_min = 0.0, output_max = 3.0;
  float input_scale, output_scale;
  int input_zero_point, output_zero_point;
  std::vector<float> input_data = {
    1, 2, 3, 1, 2, 3,
  };
  std::vector<int8_t> input_quant_data(input_data.size());
  QuantProcess(input_data.data(), input_data.size(), input_min, input_max, &input_scale, &input_zero_point,
               input_quant_data.data());
  auto input_quant = std::make_unique<schema::QuantParamT>();
  input_quant->scale = input_scale;
  input_quant->zeroPoint = input_zero_point;

  std::vector<float> expect = {
    1, 2, 3, 1, 2, 3,
  };
  QuantProcess(expect.data(), expect.size(), output_min, output_max, &output_scale, &output_zero_point, nullptr);
  auto output_quant = std::make_unique<schema::QuantParamT>();
  output_quant->scale = output_scale;
  output_quant->zeroPoint = output_zero_point;

  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = lite::NodeType_Parameter;
  input0->dataType = TypeId::kNumberTypeInt8;
  input0->dims = {2, 3};
  input0->offset = -1;
  input0->quantParams.emplace_back(std::move(input_quant));
  meta_graph->allTensors.emplace_back(std::move(input0));

  auto shape = std::make_unique<schema::TensorT>();
  shape->nodeType = lite::NodeType_ValueNode;
  shape->dataType = TypeId::kNumberTypeInt32;
  shape->dims = {2};
  shape->offset = -1;
  std::vector<int32_t> shape_data = {1, 6};
  shape->data.resize(sizeof(int32_t) * 2);
  memcpy(shape->data.data(), shape_data.data(), 2 * sizeof(int32_t));
  meta_graph->allTensors.emplace_back(std::move(shape));

  auto output = std::make_unique<schema::TensorT>();
  output->nodeType = lite::NodeType_Parameter;
  output->dataType = TypeId::kNumberTypeInt8;
  output->offset = -1;
  output->quantParams.emplace_back(std::move(output_quant));
  meta_graph->allTensors.emplace_back(std::move(output));

  flatbuffers::FlatBufferBuilder builder(1024);
  auto offset = schema::MetaGraph::Pack(builder, meta_graph.get());
  builder.Finish(offset);
  schema::FinishMetaGraphBuffer(builder, offset);
  size_t size = builder.GetSize();
  const char *content = reinterpret_cast<char *>(builder.GetBufferPointer());

  // create a context
  auto context = std::make_shared<mindspore::Context>();
  context->SetBuiltInDelegate(mindspore::DelegateMode::kPNNA);
  auto &device_list = context->MutableDeviceInfo();
  std::shared_ptr<DSPDeviceInfo> device_info = std::make_shared<DSPDeviceInfo>();
  device_list.push_back(device_info);

  // build a model
  auto model = std::make_shared<mindspore::Model>();
  auto ret = model->Build(content, size, kMindIR_Lite, context);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  auto inputs = model->GetInputs();
  ASSERT_EQ(inputs.size(), 1);
  auto inTensor = inputs.front();
  auto impl = inTensor.impl();
  ASSERT_NE(nullptr, impl);
  int8_t *in0_data = static_cast<int8_t *>(inTensor.MutableData());
  for (size_t i = 0; i < inputs[0].ElementNum(); ++i) in0_data[i] = input_quant_data[i];
  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);
  std::vector<float> output_dequant(outputs[0].ElementNum());
  Dequantize(static_cast<const int8_t *>(outputs[0].MutableData()), outputs[0].ElementNum(), output_scale,
             output_zero_point, output_dequant.data());
  ASSERT_EQ(0, CompareOutputData(output_dequant.data(), expect.data(), expect.size()));
}
}  // namespace mindspore::lite::dsp::test
