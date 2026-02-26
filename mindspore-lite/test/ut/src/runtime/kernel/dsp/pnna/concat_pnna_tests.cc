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
class TestPNNA_Concat : public CommonTest {};

TEST_F(TestPNNA_Concat, shape_2_2_fp32) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";

  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1};
  node->outputIndex = {2};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_Concat;
  auto primitive = new schema::ConcatT;
  primitive->axis = 0;
  node->primitive->value.value = primitive;
  node->name = "Concat";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0, 1};
  meta_graph->outputIndex = {2};

  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = lite::NodeType_Parameter;
  input0->dataType = TypeId::kNumberTypeFloat32;
  input0->dims = {2, 2};
  input0->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input0));

  auto input1 = std::make_unique<schema::TensorT>();
  input1->nodeType = lite::NodeType_Parameter;
  input1->dataType = TypeId::kNumberTypeFloat32;
  input1->dims = {2, 2};
  input1->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input1));

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
  ASSERT_EQ(inputs.size(), 2);
  float *in0_data = static_cast<float *>(inputs[0].MutableData());
  float *in1_data = static_cast<float *>(inputs[1].MutableData());
  std::vector<float> a_data = {0, 1, 2, 1};
  std::vector<float> b_data = {0, 1, 2, 1};
  std::vector<float> expect = {0, 1, 2, 1, 0, 1, 2, 1};  // correct answer

  for (size_t i = 0; i < a_data.size(); ++i) in0_data[i] = a_data[i];
  for (size_t i = 0; i < b_data.size(); ++i) in1_data[i] = b_data[i];
  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);
  auto *outData = reinterpret_cast<const float *>(outputs.front().Data().get());
  ASSERT_EQ(0, CompareOutputData(outData, expect.data(), expect.size()));
}

TEST_F(TestPNNA_Concat, shape_2_2_int8) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";

  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1};
  node->outputIndex = {2};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_Concat;
  auto primitive = new schema::ConcatT;
  primitive->axis = 0;
  node->primitive->value.value = primitive;
  node->name = "Concat";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0, 1};
  meta_graph->outputIndex = {2};

  int length = 2 * 2;
  float in0[] = {0, 1, 2, 1};
  float input_min = 0, input_max = 2;
  int8_t *input0_data = new int8_t[length];
  float in0_scale;
  int in0_zp;
  QuantProcess(in0, length, input_min, input_max, &in0_scale, &in0_zp, input0_data);
  float in1[] = {0, 1, 2, 1};
  int8_t *input1_data = new int8_t[length];
  float in1_scale;
  int in1_zp;
  QuantProcess(in1, length, input_min, input_max, &in1_scale, &in1_zp, input1_data);

  float expect[] = {0, 1, 2, 1, 0, 1, 2, 1};
  float output_min = 0, output_max = 2;
  float out_scale;
  int out_zp;
  QuantProcess(expect, length * 2, output_min, output_max, &out_scale, &out_zp, nullptr);

  auto input_quant0 = std::make_unique<schema::QuantParamT>();
  input_quant0->scale = in0_scale;
  input_quant0->zeroPoint = in0_zp;

  auto input_quant1 = std::make_unique<schema::QuantParamT>();
  input_quant1->scale = in1_scale;
  input_quant1->zeroPoint = in1_zp;

  auto out_quant = std::make_unique<schema::QuantParamT>();
  out_quant->scale = out_scale;
  out_quant->zeroPoint = out_zp;

  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = lite::NodeType_Parameter;
  input0->dataType = TypeId::kNumberTypeInt8;
  input0->dims = {2, 2};
  input0->offset = -1;
  input0->quantParams.emplace_back(std::move(input_quant0));
  meta_graph->allTensors.emplace_back(std::move(input0));

  auto input1 = std::make_unique<schema::TensorT>();
  input1->nodeType = lite::NodeType_Parameter;
  input1->dataType = TypeId::kNumberTypeInt8;
  input1->dims = {2, 2};
  input1->offset = -1;
  input1->quantParams.emplace_back(std::move(input_quant1));
  meta_graph->allTensors.emplace_back(std::move(input1));

  auto output = std::make_unique<schema::TensorT>();
  output->nodeType = lite::NodeType_Parameter;
  output->dataType = TypeId::kNumberTypeInt8;
  output->offset = -1;
  output->quantParams.emplace_back(std::move(out_quant));
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
  ASSERT_EQ(inputs.size(), 2);
  int8_t *in0_data = static_cast<int8_t *>(inputs[0].MutableData());
  int8_t *in1_data = static_cast<int8_t *>(inputs[1].MutableData());

  for (size_t i = 0; i < inputs[0].ElementNum(); ++i) in0_data[i] = input0_data[i];
  for (size_t i = 0; i < inputs[1].ElementNum(); ++i) in1_data[i] = input1_data[i];
  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);
  float *out = new float[length * 2];
  auto *outData = reinterpret_cast<const int8_t *>(outputs.front().Data().get());
  Dequantize(outData, outputs.front().ElementNum(), out_scale, out_zp, out);
  ASSERT_EQ(0, CompareOutputData(out, expect, outputs.front().ElementNum(), 1e-2));
  delete[] input0_data;
  delete[] input1_data;
  delete[] out;
}
}  // namespace mindspore::lite::dsp::test
