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
class TestPNNA_Resize : public CommonTest {};

TEST_F(TestPNNA_Resize, shape_2_2_to_3_3_fp32) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";

  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1};
  node->outputIndex = {2};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_Resize;
  auto primitive = new schema::ResizeT;
  primitive->method = schema::ResizeMethod_LINEAR;
  primitive->new_height = 2;
  primitive->new_width = 2;
  node->primitive->value.value = primitive;
  node->name = "ResizeBilinear";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {2};

  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = lite::NodeType_Parameter;
  input0->dataType = TypeId::kNumberTypeFloat32;
  input0->format = schema::Format_NHWC;
  input0->dims = {1, 2, 2, 1};
  input0->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input0));

  auto shape = std::make_unique<schema::TensorT>();
  shape->nodeType = lite::NodeType_ValueNode;
  shape->dataType = TypeId::kNumberTypeFloat32;
  shape->dims = {2};
  shape->offset = -1;
  std::vector<float> shape_data = {1.5, 1.5};
  shape->data.resize(sizeof(float) * 2);
  memcpy(shape->data.data(), shape_data.data(), 2 * sizeof(float));
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
  float *in0_data = static_cast<float *>(inputs[0].MutableData());
  std::vector<float> a_data = {0.0, 1.0, 2.0, 3.0};
  std::vector<float> expect = {0.0, 0.6666667, 1.0, 1.3333334, 2.0, 2.3333335, 2.0, 2.6666667, 3.0};  // correct answer
  for (size_t i = 0; i < a_data.size(); ++i) in0_data[i] = a_data[i];
  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);
  auto *outData = reinterpret_cast<const float *>(outputs.front().Data().get());
  ASSERT_EQ(0, CompareOutputData(outData, expect.data(), expect.size()));
}

TEST_F(TestPNNA_Resize, shape_2_2_to_2_4_fp32) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";

  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1};
  node->outputIndex = {2};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_Resize;
  auto primitive = new schema::ResizeT;
  primitive->method = schema::ResizeMethod_NEAREST;
  primitive->nearest_mode = schema::NearestMode_FLOOR;
  primitive->new_height = 2;
  primitive->new_width = 4;
  node->primitive->value.value = primitive;
  node->name = "ResizeNearestNeighbor";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {2};

  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = lite::NodeType_Parameter;
  input0->dataType = TypeId::kNumberTypeFloat32;
  input0->format = schema::Format_NHWC;
  input0->dims = {1, 2, 2, 1};
  input0->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input0));

  auto shape = std::make_unique<schema::TensorT>();
  shape->nodeType = lite::NodeType_ValueNode;
  shape->dataType = TypeId::kNumberTypeFloat32;
  shape->dims = {2};
  shape->offset = -1;
  std::vector<float> shape_data = {1.0, 2.0};
  shape->data.resize(sizeof(float) * 2);
  memcpy(shape->data.data(), shape_data.data(), 2 * sizeof(float));
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
  float *in0_data = static_cast<float *>(inputs[0].MutableData());
  std::vector<float> a_data = {0.0, 1.0, 2.0, 3.0};
  std::vector<float> expect = {0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0};  // correct answer
  for (size_t i = 0; i < a_data.size(); ++i) in0_data[i] = a_data[i];
  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);
  auto *outData = reinterpret_cast<const float *>(outputs.front().Data().get());
  ASSERT_EQ(0, CompareOutputData(outData, expect.data(), expect.size()));
}

TEST_F(TestPNNA_Resize, shape_2_2_to_3_3_int8) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";

  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1};
  node->outputIndex = {2};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_Resize;
  auto primitive = new schema::ResizeT;
  primitive->method = schema::ResizeMethod_LINEAR;
  primitive->new_height = 2;
  primitive->new_width = 2;
  node->primitive->value.value = primitive;
  node->name = "ResizeBilinear";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {2};

  float input_min = 0.0, input_max = 3.0, output_min = 0.0, output_max = 3.0;
  float input_scale, output_scale;
  int input_zero_point, output_zero_point;
  std::vector<float> input_data = {0.0, 1.0, 2.0, 3.0};
  std::vector<int8_t> input_quant_data(input_data.size());
  QuantProcess(input_data.data(), input_data.size(), input_min, input_max, &input_scale, &input_zero_point,
               input_quant_data.data());
  auto input_quant = std::make_unique<schema::QuantParamT>();
  input_quant->scale = input_scale;
  input_quant->zeroPoint = input_zero_point;

  std::vector<float> expect = {0.0, 0.6666667, 1.0, 1.3333334, 2.0, 2.3333335, 2.0, 2.6666667, 3.0};  // correct answer
  QuantProcess(expect.data(), expect.size(), output_min, output_max, &output_scale, &output_zero_point, nullptr);
  auto output_quant = std::make_unique<schema::QuantParamT>();
  output_quant->scale = output_scale;
  output_quant->zeroPoint = output_zero_point;

  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = lite::NodeType_Parameter;
  input0->dataType = TypeId::kNumberTypeInt8;
  input0->format = schema::Format_NHWC;
  input0->dims = {1, 2, 2, 1};
  input0->offset = -1;
  input0->quantParams.emplace_back(std::move(input_quant));
  meta_graph->allTensors.emplace_back(std::move(input0));

  auto shape = std::make_unique<schema::TensorT>();
  shape->nodeType = lite::NodeType_ValueNode;
  shape->dataType = TypeId::kNumberTypeFloat32;
  shape->dims = {2};
  shape->offset = -1;
  std::vector<float> shape_data = {1.5, 1.5};
  shape->data.resize(sizeof(float) * 2);
  memcpy(shape->data.data(), shape_data.data(), 2 * sizeof(float));
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
  int8_t *in0_data = static_cast<int8_t *>(inputs[0].MutableData());
  for (size_t i = 0; i < inputs[0].ElementNum(); ++i) in0_data[i] = input_quant_data[i];
  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);
  std::vector<float> output_dequant(outputs[0].ElementNum());
  Dequantize(static_cast<const int8_t *>(outputs[0].MutableData()), outputs[0].ElementNum(), output_scale,
             output_zero_point, output_dequant.data());
  ASSERT_EQ(0, CompareOutputData(output_dequant.data(), expect.data(), expect.size(), 1e-2));
}

TEST_F(TestPNNA_Resize, shape_2_2_to_2_4_int8) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";

  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1};
  node->outputIndex = {2};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_Resize;
  auto primitive = new schema::ResizeT;
  primitive->method = schema::ResizeMethod_NEAREST;
  primitive->nearest_mode = schema::NearestMode_FLOOR;
  primitive->new_height = 2;
  primitive->new_width = 4;
  node->primitive->value.value = primitive;
  node->name = "ResizeNearestNeighbor";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {2};

  float input_min = 0.0, input_max = 3.0, output_min = 0.0, output_max = 3.0;
  float input_scale, output_scale;
  int input_zero_point, output_zero_point;
  std::vector<float> input_data = {0.0, 1.0, 2.0, 3.0};
  std::vector<int8_t> input_quant_data(input_data.size());
  QuantProcess(input_data.data(), input_data.size(), input_min, input_max, &input_scale, &input_zero_point,
               input_quant_data.data());
  auto input_quant = std::make_unique<schema::QuantParamT>();
  input_quant->scale = input_scale;
  input_quant->zeroPoint = input_zero_point;

  std::vector<float> expect = {0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0};  // correct answer
  QuantProcess(expect.data(), expect.size(), output_min, output_max, &output_scale, &output_zero_point, nullptr);
  auto output_quant = std::make_unique<schema::QuantParamT>();
  output_quant->scale = output_scale;
  output_quant->zeroPoint = output_zero_point;

  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = lite::NodeType_Parameter;
  input0->dataType = TypeId::kNumberTypeInt8;
  input0->format = schema::Format_NHWC;
  input0->dims = {1, 2, 2, 1};
  input0->offset = -1;
  input0->quantParams.emplace_back(std::move(input_quant));
  meta_graph->allTensors.emplace_back(std::move(input0));

  auto shape = std::make_unique<schema::TensorT>();
  shape->nodeType = lite::NodeType_ValueNode;
  shape->dataType = TypeId::kNumberTypeFloat32;
  shape->dims = {2};
  shape->offset = -1;
  std::vector<float> shape_data = {1.0, 2.0};
  shape->data.resize(sizeof(float) * 2);
  memcpy(shape->data.data(), shape_data.data(), 2 * sizeof(float));
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
  int8_t *in0_data = static_cast<int8_t *>(inputs[0].MutableData());
  for (size_t i = 0; i < inputs[0].ElementNum(); ++i) in0_data[i] = input_quant_data[i];
  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);
  std::vector<float> output_dequant(outputs[0].ElementNum());
  Dequantize(static_cast<const int8_t *>(outputs[0].MutableData()), outputs[0].ElementNum(), output_scale,
             output_zero_point, output_dequant.data());
  ASSERT_EQ(0, CompareOutputData(output_dequant.data(), expect.data(), expect.size(), 1e-6));
}
}  // namespace mindspore::lite::dsp::test
