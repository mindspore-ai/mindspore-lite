/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "src/litert/kernel/dsp/dsp_subgraph.h"
#include "src/litert/kernel_registry.h"

namespace mindspore::lite::dsp::test {

class TestPNNA_Pad : public CommonTest {};

TEST_F(TestPNNA_Pad, shape_1_2_2_2_constant_fp32) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";

  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1};
  node->outputIndex = {2};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_PadFusion;
  auto primitive = new schema::PadFusionT();
  auto padding = std::make_unique<schema::Vec2DT>();
  auto vec = std::make_unique<schema::VecT>();
  vec->data = {0, 0};
  padding->data.emplace_back(std::move(vec));

  auto vec1 = std::make_unique<schema::VecT>();
  vec1->data = {0, 0};
  padding->data.emplace_back(std::move(vec1));
  auto vec2 = std::make_unique<schema::VecT>();
  vec2->data = {0, 1};
  padding->data.emplace_back(std::move(vec2));
  auto vec3 = std::make_unique<schema::VecT>();
  vec3->data = {1, 0};
  padding->data.emplace_back(std::move(vec3));
  primitive->paddings = std::move(padding);

  node->primitive->value.value = primitive;
  node->name = "Pad";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {2};

  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = lite::NodeType_Parameter;
  input0->dataType = TypeId::kNumberTypeFloat32;
  input0->dims = {1, 2, 2, 2};
  input0->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input0));

  auto in_t_pad = std::make_unique<schema::TensorT>();
  in_t_pad->nodeType = lite::NodeType_ValueNode;
  in_t_pad->dataType = TypeId::kNumberTypeInt32;
  in_t_pad->dims = {4, 2};
  in_t_pad->offset = -1;
  // bias data
  std::vector<int> pad_data = {0, 0, 0, 0, 0, 1, 1, 0};
  in_t_pad->data.resize(sizeof(int) * 2 * 4);
  memcpy(in_t_pad->data.data(), pad_data.data(), 2 * 4 * sizeof(int));
  meta_graph->allTensors.emplace_back(std::move(in_t_pad));

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
  context->SetThreadNum(1);
  auto &device_list = context->MutableDeviceInfo();
  std::shared_ptr<CPUDeviceInfo> device_info = std::make_shared<CPUDeviceInfo>();
  device_list.push_back(device_info);

  // build a model
  auto model = std::make_shared<mindspore::Model>();
  auto ret = model->Build(content, size, kMindIR_Lite, context);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  auto inputs = model->GetInputs();
  ASSERT_EQ(inputs.size(), 1);
  float *in0_data = static_cast<float *>(inputs[0].MutableData());
  std::vector<float> a_data = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> expect = {0, 0, 1, 0, 2, 3, 0, 0, 0, 0, 4, 5, 0, 6, 7, 0, 0, 0};  // correct answer
  for (size_t i = 0; i < inputs[0].ElementNum(); ++i) in0_data[i] = a_data[i];
  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);
  auto *outData = reinterpret_cast<const float *>(outputs.front().Data().get());
  ASSERT_EQ(0, CompareOutputData(outData, expect.data(), outputs.front().ElementNum()));
}

TEST_F(TestPNNA_Pad, shape_1_2_2_2_constant_int8) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";

  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1};
  node->outputIndex = {2};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_PadFusion;
  auto primitive = new schema::PadFusionT();
  auto padding = std::make_unique<schema::Vec2DT>();
  auto vec = std::make_unique<schema::VecT>();
  vec->data = {0, 0};
  padding->data.emplace_back(std::move(vec));

  auto vec1 = std::make_unique<schema::VecT>();
  vec1->data = {0, 0};
  padding->data.emplace_back(std::move(vec1));
  auto vec2 = std::make_unique<schema::VecT>();
  vec2->data = {0, 1};
  padding->data.emplace_back(std::move(vec2));
  auto vec3 = std::make_unique<schema::VecT>();
  vec3->data = {1, 0};
  padding->data.emplace_back(std::move(vec3));
  primitive->paddings = std::move(padding);

  node->primitive->value.value = primitive;
  node->name = "Pad";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {2};

  std::vector<float> input_data = {0, 1, 2, 3, 4, 5, 6, 7};
  float in_scale;
  int in_zp;
  std::vector<int8_t> a_data(input_data.size());
  QuantProcess(input_data.data(), input_data.size(), 0, 8, &in_scale, &in_zp, a_data.data());

  std::vector<float> expect = {0, 0, 1, 0, 2, 3, 0, 0, 0, 0, 4, 5, 0, 6, 7, 0, 0, 0};
  float out_scale;
  int out_zp;
  QuantProcess(expect.data(), expect.size(), 0, 8, &out_scale, &out_zp, nullptr);

  auto input_quant = std::make_unique<schema::QuantParamT>();
  input_quant->scale = in_scale;
  input_quant->zeroPoint = in_zp;

  auto output_quant0 = std::make_unique<schema::QuantParamT>();
  output_quant0->scale = out_scale;
  output_quant0->zeroPoint = out_zp;

  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = lite::NodeType_Parameter;
  input0->dataType = TypeId::kNumberTypeInt8;
  input0->dims = {1, 2, 2, 2};
  input0->offset = -1;
  input0->quantParams.emplace_back(std::move(input_quant));
  meta_graph->allTensors.emplace_back(std::move(input0));

  auto in_t_pad = std::make_unique<schema::TensorT>();
  in_t_pad->nodeType = lite::NodeType_ValueNode;
  in_t_pad->dataType = TypeId::kNumberTypeInt32;
  in_t_pad->dims = {4, 2};
  in_t_pad->offset = -1;
  // bias data
  std::vector<int> pad_data = {0, 0, 0, 0, 0, 1, 1, 0};
  in_t_pad->data.resize(sizeof(int) * 2 * 4);
  memcpy(in_t_pad->data.data(), pad_data.data(), 2 * 4 * sizeof(int));
  meta_graph->allTensors.emplace_back(std::move(in_t_pad));

  auto output = std::make_unique<schema::TensorT>();
  output->nodeType = lite::NodeType_Parameter;
  output->dataType = TypeId::kNumberTypeInt8;
  output->offset = -1;
  output->quantParams.emplace_back(std::move(output_quant0));
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
  int8_t *in0_data = reinterpret_cast<int8_t *>(inputs[0].MutableData());
  for (size_t i = 0; i < inputs[0].ElementNum(); ++i) in0_data[i] = a_data[i];
  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);
  auto *outData = reinterpret_cast<const int8_t *>(outputs.front().Data().get());
  std::vector<float> out_data(outputs.front().ElementNum());
  Dequantize(outData, outputs.front().ElementNum(), out_scale, in_zp, out_data.data());
  ASSERT_EQ(0, CompareOutputData(out_data.data(), expect.data(), outputs.front().ElementNum(), 1e-2));
}
}  // namespace mindspore::lite::dsp::test
