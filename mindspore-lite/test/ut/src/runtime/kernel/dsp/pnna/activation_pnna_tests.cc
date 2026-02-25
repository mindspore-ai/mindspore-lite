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
class TestPNNA_Activation : public CommonTest {};

TEST_F(TestPNNA_Activation, Relu) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";

  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0};
  node->outputIndex = {1};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_Activation;
  auto primitive = new schema::ActivationT;
  primitive->activation_type = schema::ActivationType_RELU;
  node->primitive->value.value = primitive;
  node->name = "relu";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {1};

  auto input = std::make_unique<schema::TensorT>();
  input->nodeType = lite::NodeType_Parameter;
  input->dataType = TypeId::kNumberTypeFloat32;
  input->dims = {8};
  input->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input));

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
  float *in_data = static_cast<float *>(inputs[0].MutableData());
  std::vector<float> input_data = {-3, -2, -1, 0, 1, 5, -6, 7};
  std::vector<float> expect = {0, 0, 0, 0, 1, 5, 0, 7};
  for (size_t i = 0; i < input_data.size(); ++i) in_data[i] = input_data[i];
  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);
  auto *outData = reinterpret_cast<const float *>(outputs.front().Data().get());
  ASSERT_EQ(0, CompareOutputData(outData, expect.data(), expect.size()));
}

TEST_F(TestPNNA_Activation, Relu6) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";

  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0};
  node->outputIndex = {1};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_Activation;
  auto primitive = new schema::ActivationT;
  primitive->activation_type = schema::ActivationType_RELU6;
  node->primitive->value.value = primitive;
  node->name = "relu6";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {1};

  auto input = std::make_unique<schema::TensorT>();
  input->nodeType = lite::NodeType_Parameter;
  input->dataType = TypeId::kNumberTypeFloat32;
  input->dims = {8};
  input->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input));

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
  float *in_data = static_cast<float *>(inputs[0].MutableData());
  std::vector<float> input_data = {-3, -2, -1, 0, 1, 5, 6, 7};
  std::vector<float> expect = {0, 0, 0, 0, 1, 5, 6, 6};
  for (size_t i = 0; i < input_data.size(); ++i) in_data[i] = input_data[i];
  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);
  auto *outData = reinterpret_cast<const float *>(outputs.front().Data().get());
  ASSERT_EQ(0, CompareOutputData(outData, expect.data(), expect.size()));
}

TEST_F(TestPNNA_Activation, Sigmoid) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";

  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0};
  node->outputIndex = {1};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_Activation;
  auto primitive = new schema::ActivationT;
  primitive->activation_type = schema::ActivationType_SIGMOID;
  node->primitive->value.value = primitive;
  node->name = "sigmoid";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {1};

  auto input = std::make_unique<schema::TensorT>();
  input->nodeType = lite::NodeType_Parameter;
  input->dataType = TypeId::kNumberTypeFloat32;
  input->dims = {8};
  input->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input));

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
  float *in_data = static_cast<float *>(inputs[0].MutableData());
  std::vector<float> input_data = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> expect = {0.5, 0.731059, 0.880797, 0.952574, 0.982014, 0.993307, 0.997527, 0.999089};
  for (size_t i = 0; i < input_data.size(); ++i) in_data[i] = input_data[i];
  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);
  auto *outData = reinterpret_cast<const float *>(outputs.front().Data().get());
  ASSERT_EQ(0, CompareOutputData(outData, expect.data(), expect.size()));
}

TEST_F(TestPNNA_Activation, Swish) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";

  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0};
  node->outputIndex = {1};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_Activation;
  auto primitive = new schema::ActivationT;
  primitive->activation_type = schema::ActivationType_SWISH;
  node->primitive->value.value = primitive;
  node->name = "swish";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {1};

  auto input = std::make_unique<schema::TensorT>();
  input->nodeType = lite::NodeType_Parameter;
  input->dataType = TypeId::kNumberTypeFloat32;
  input->dims = {8};
  input->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input));

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
  float *in_data = static_cast<float *>(inputs[0].MutableData());
  std::vector<float> input_data = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> expect = {0, 0.731059, 1.761594, 2.857722, 3.928056, 4.966535, 5.985162, 6.993623};
  for (size_t i = 0; i < input_data.size(); ++i) in_data[i] = input_data[i];
  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);
  auto *outData = reinterpret_cast<const float *>(outputs.front().Data().get());
  ASSERT_EQ(0, CompareOutputData(outData, expect.data(), expect.size()));
}

TEST_F(TestPNNA_Activation, Tanh) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";

  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0};
  node->outputIndex = {1};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_Activation;
  auto primitive = new schema::ActivationT;
  primitive->activation_type = schema::ActivationType_TANH;
  node->primitive->value.value = primitive;
  node->name = "tanh";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {1};

  auto input = std::make_unique<schema::TensorT>();
  input->nodeType = lite::NodeType_Parameter;
  input->dataType = TypeId::kNumberTypeFloat32;
  input->dims = {7};
  input->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input));

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
  float *in_data = static_cast<float *>(inputs[0].MutableData());
  std::vector<float> input_data = {-3, -2, -1, 0, 1, 2, 3};
  std::vector<float> expect = {-0.995055, -0.964028, -0.761594, 0.000000, 0.761594, 0.964028, 0.995055};
  for (size_t i = 0; i < input_data.size(); ++i) in_data[i] = input_data[i];
  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);
  auto *outData = reinterpret_cast<const float *>(outputs.front().Data().get());
  ASSERT_EQ(0, CompareOutputData(outData, expect.data(), expect.size()));
}

TEST_F(TestPNNA_Activation, LeakyRelu) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";

  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0};
  node->outputIndex = {1};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_Activation;
  auto primitive = new schema::ActivationT;
  primitive->activation_type = schema::ActivationType_LEAKY_RELU;
  primitive->alpha = 0.001;
  node->primitive->value.value = primitive;
  node->name = "leakyrelu";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {1};

  auto input = std::make_unique<schema::TensorT>();
  input->nodeType = lite::NodeType_Parameter;
  input->dataType = TypeId::kNumberTypeFloat32;
  input->dims = {7};
  input->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input));

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
  float *in_data = static_cast<float *>(inputs[0].MutableData());
  std::vector<float> input_data = {-3, -2, -1, 0, 1, 2, 3};
  std::vector<float> expect = {-0.003, -0.002, -0.001, 0.000000, 1, 2, 3};
  for (size_t i = 0; i < input_data.size(); ++i) in_data[i] = input_data[i];
  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);
  auto *outData = reinterpret_cast<const float *>(outputs.front().Data().get());
  ASSERT_EQ(0, CompareOutputData(outData, expect.data(), expect.size()));
}

TEST_F(TestPNNA_Activation, Relu_int8) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";

  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0};
  node->outputIndex = {1};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_Activation;
  auto primitive = new schema::ActivationT;
  primitive->activation_type = schema::ActivationType_RELU;
  node->primitive->value.value = primitive;
  node->name = "relu";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {1};

  std::vector<float> input_data = {-3, -2, -1, 0, 1, 5, -6, 7};
  float input_min = -3, input_max = 7;
  std::vector<int8_t> input_quant_data(input_data.size());
  float in0_scale;
  int in0_zp;
  QuantProcess(input_data.data(), input_data.size(), input_min, input_max, &in0_scale, &in0_zp,
               input_quant_data.data());
  auto input_quant = std::make_unique<schema::QuantParamT>();
  input_quant->scale = in0_scale;
  input_quant->zeroPoint = in0_zp;

  std::vector<float> expect = {0, 0, 0, 0, 1, 5, 0, 7};
  float output_min = -7, output_max = 7;
  float out0_scale;
  int out0_zp;
  QuantProcess(expect.data(), expect.size(), output_min, output_max, &out0_scale, &out0_zp, nullptr);
  auto output_quant = std::make_unique<schema::QuantParamT>();
  output_quant->scale = out0_scale;
  output_quant->zeroPoint = out0_zp;

  auto input = std::make_unique<schema::TensorT>();
  input->nodeType = lite::NodeType_Parameter;
  input->dataType = TypeId::kNumberTypeInt8;
  input->dims = {8};
  input->offset = -1;
  input->quantParams.emplace_back(std::move(input_quant));
  meta_graph->allTensors.emplace_back(std::move(input));

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
  int8_t *in_data = static_cast<int8_t *>(inputs[0].MutableData());
  for (size_t i = 0; i < input_data.size(); ++i) in_data[i] = input_quant_data[i];

  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);
  std::vector<float> out(expect.size());
  auto *outData = reinterpret_cast<const int8_t *>(outputs.front().Data().get());
  Dequantize(outData, outputs.front().ElementNum(), out0_scale, out0_zp, out.data());
  ASSERT_EQ(0, CompareOutputData(out.data(), expect.data(), expect.size(), 1e-2));
}

TEST_F(TestPNNA_Activation, Relu6_int8) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";

  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0};
  node->outputIndex = {1};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_Activation;
  auto primitive = new schema::ActivationT;
  primitive->activation_type = schema::ActivationType_RELU6;
  node->primitive->value.value = primitive;
  node->name = "relu6";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {1};

  std::vector<float> input_data = {-3, -2, -1, 0, 1, 5, 6, 7};
  float input_min = -3, input_max = 7;
  std::vector<int8_t> input_quant_data(input_data.size());
  float in0_scale;
  int in0_zp;
  QuantProcess(input_data.data(), input_data.size(), input_min, input_max, &in0_scale, &in0_zp,
               input_quant_data.data());
  auto input_quant = std::make_unique<schema::QuantParamT>();
  input_quant->scale = in0_scale;
  input_quant->zeroPoint = in0_zp;

  std::vector<float> expect = {0, 0, 0, 0, 1, 5, 6, 6};
  float output_min = 0, output_max = 6;
  float out0_scale;
  int out0_zp;
  QuantProcess(expect.data(), expect.size(), output_min, output_max, &out0_scale, &out0_zp, nullptr);
  auto output_quant = std::make_unique<schema::QuantParamT>();
  output_quant->scale = out0_scale;
  output_quant->zeroPoint = out0_zp;

  auto input = std::make_unique<schema::TensorT>();
  input->nodeType = lite::NodeType_Parameter;
  input->dataType = TypeId::kNumberTypeInt8;
  input->dims = {8};
  input->offset = -1;
  input->quantParams.emplace_back(std::move(input_quant));
  meta_graph->allTensors.emplace_back(std::move(input));

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
  int8_t *in_data = static_cast<int8_t *>(inputs[0].MutableData());
  for (size_t i = 0; i < input_data.size(); ++i) in_data[i] = input_quant_data[i];
  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);
  std::vector<float> out(expect.size());
  auto *outData = reinterpret_cast<const int8_t *>(outputs.front().Data().get());
  Dequantize(outData, outputs.front().ElementNum(), out0_scale, out0_zp, out.data());
  ASSERT_EQ(0, CompareOutputData(out.data(), expect.data(), expect.size(), 1e-2));
}

TEST_F(TestPNNA_Activation, Sigmoid_int8) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";

  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0};
  node->outputIndex = {1};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_Activation;
  auto primitive = new schema::ActivationT;
  primitive->activation_type = schema::ActivationType_SIGMOID;
  node->primitive->value.value = primitive;
  node->name = "sigmoid";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {1};

  std::vector<float> input_data = {0, 1, 2, 3, 4, 5, 6, 7};
  float input_min = 0, input_max = 7;
  std::vector<int8_t> input_quant_data(input_data.size());
  float in0_scale;
  int in0_zp;
  QuantProcess(input_data.data(), input_data.size(), input_min, input_max, &in0_scale, &in0_zp,
               input_quant_data.data());
  auto input_quant = std::make_unique<schema::QuantParamT>();
  input_quant->scale = in0_scale;
  input_quant->zeroPoint = in0_zp;

  std::vector<float> expect = {0.5, 0.731059, 0.880797, 0.952574, 0.982014, 0.993307, 0.997527, 0.999089};
  float output_min = 0, output_max = 1;
  float out0_scale;
  int out0_zp;
  QuantProcess(expect.data(), expect.size(), output_min, output_max, &out0_scale, &out0_zp, nullptr);
  auto output_quant = std::make_unique<schema::QuantParamT>();
  output_quant->scale = out0_scale;
  output_quant->zeroPoint = out0_zp;

  auto input = std::make_unique<schema::TensorT>();
  input->nodeType = lite::NodeType_Parameter;
  input->dataType = TypeId::kNumberTypeInt8;
  input->dims = {8};
  input->offset = -1;
  input->quantParams.emplace_back(std::move(input_quant));
  meta_graph->allTensors.emplace_back(std::move(input));

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
  int8_t *in_data = static_cast<int8_t *>(inputs[0].MutableData());
  for (size_t i = 0; i < input_data.size(); ++i) in_data[i] = input_quant_data[i];
  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);
  std::vector<float> out(expect.size());
  auto *outData = reinterpret_cast<const int8_t *>(outputs.front().Data().get());
  Dequantize(outData, outputs.front().ElementNum(), out0_scale, out0_zp, out.data());
  ASSERT_EQ(0, CompareOutputData(out.data(), expect.data(), expect.size(), 1e-2));
}

TEST_F(TestPNNA_Activation, Tanh_int8) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";

  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0};
  node->outputIndex = {1};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_Activation;
  auto primitive = new schema::ActivationT;
  primitive->activation_type = schema::ActivationType_TANH;
  node->primitive->value.value = primitive;
  node->name = "tanh";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {1};

  std::vector<float> input_data = {-3, -2, -1, 0, 1, 2, 3};
  float input_min = -3, input_max = 3;
  std::vector<int8_t> input_quant_data(input_data.size());
  float in0_scale;
  int in0_zp;
  QuantProcess(input_data.data(), input_data.size(), input_min, input_max, &in0_scale, &in0_zp,
               input_quant_data.data());
  auto input_quant = std::make_unique<schema::QuantParamT>();
  input_quant->scale = in0_scale;
  input_quant->zeroPoint = in0_zp;

  std::vector<float> expect = {-0.995055, -0.964028, -0.761594, 0.000000, 0.761594, 0.964028, 0.995055};
  float output_min = -1, output_max = 1;
  float out0_scale;
  int out0_zp;
  QuantProcess(expect.data(), expect.size(), output_min, output_max, &out0_scale, &out0_zp, nullptr);
  auto output_quant = std::make_unique<schema::QuantParamT>();
  output_quant->scale = out0_scale;
  output_quant->zeroPoint = out0_zp;

  auto input = std::make_unique<schema::TensorT>();
  input->nodeType = lite::NodeType_Parameter;
  input->dataType = TypeId::kNumberTypeInt8;
  input->dims = {7};
  input->offset = -1;
  input->quantParams.emplace_back(std::move(input_quant));
  meta_graph->allTensors.emplace_back(std::move(input));

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
  int8_t *in_data = static_cast<int8_t *>(inputs[0].MutableData());
  for (size_t i = 0; i < input_data.size(); ++i) in_data[i] = input_quant_data[i];
  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);
  std::vector<float> out(expect.size());
  auto *outData = reinterpret_cast<const int8_t *>(outputs.front().Data().get());
  Dequantize(outData, outputs.front().ElementNum(), out0_scale, out0_zp, out.data());
  ASSERT_EQ(0, CompareOutputData(out.data(), expect.data(), expect.size(), 1e-2));
}

TEST_F(TestPNNA_Activation, LeakyRelu_int8) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";

  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0};
  node->outputIndex = {1};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_Activation;
  node->quantType = schema::QuantType_QUANT_ALL;
  auto primitive = new schema::ActivationT;
  primitive->activation_type = schema::ActivationType_LEAKY_RELU;
  primitive->alpha = 0.001;
  node->primitive->value.value = primitive;
  node->name = "leakyrelu";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {1};

  std::vector<float> input_data = {-3, -2, -1, 0, 1, 2, 3};
  float input_min = -3, input_max = 3;
  std::vector<int8_t> input_quant_data(input_data.size());
  float in0_scale;
  int in0_zp;
  QuantProcess(input_data.data(), input_data.size(), input_min, input_max, &in0_scale, &in0_zp,
               input_quant_data.data());
  auto input_quant = std::make_unique<schema::QuantParamT>();
  input_quant->scale = in0_scale;
  input_quant->zeroPoint = in0_zp;

  std::vector<float> expect = {-0.003, -0.002, -0.001, 0.000000, 1, 2, 3};
  float output_min = -1, output_max = 3;
  float out0_scale;
  int out0_zp;
  QuantProcess(expect.data(), expect.size(), output_min, output_max, &out0_scale, &out0_zp, nullptr);
  auto output_quant = std::make_unique<schema::QuantParamT>();
  output_quant->scale = out0_scale;
  output_quant->zeroPoint = out0_zp;

  auto input = std::make_unique<schema::TensorT>();
  input->nodeType = lite::NodeType_Parameter;
  input->dataType = TypeId::kNumberTypeInt8;
  input->dims = {7};
  input->offset = -1;
  input->quantParams.emplace_back(std::move(input_quant));
  meta_graph->allTensors.emplace_back(std::move(input));

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
  int8_t *in_data = static_cast<int8_t *>(inputs[0].MutableData());
  for (size_t i = 0; i < input_data.size(); ++i) in_data[i] = input_quant_data[i];
  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);
  std::vector<float> out(expect.size());
  auto *outData = reinterpret_cast<const int8_t *>(outputs.front().Data().get());
  Dequantize(outData, outputs.front().ElementNum(), out0_scale, out0_zp, out.data());
  ASSERT_EQ(0, CompareOutputData(out.data(), expect.data(), expect.size(), 1e-2));
}
}  // namespace mindspore::lite::dsp::test
