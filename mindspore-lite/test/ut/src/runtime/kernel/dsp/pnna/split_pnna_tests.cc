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
class TestPNNA_Split : public CommonTest {};

TEST_F(TestPNNA_Split, shape_2_2_fp32) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";

  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0};
  node->outputIndex = {1, 2};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_Split;
  auto primitive = new schema::SplitT;
  primitive->axis = 0;
  primitive->output_num = 2;
  primitive->size_splits = {1, 1};
  node->primitive->value.value = primitive;
  node->name = "Split";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {1, 2};

  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = lite::NodeType_Parameter;
  input0->dataType = TypeId::kNumberTypeFloat32;
  input0->dims = {2, 2};
  input0->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input0));

  auto output0 = std::make_unique<schema::TensorT>();
  output0->nodeType = lite::NodeType_Parameter;
  output0->dataType = TypeId::kNumberTypeFloat32;
  output0->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(output0));

  auto output1 = std::make_unique<schema::TensorT>();
  output1->nodeType = lite::NodeType_Parameter;
  output1->dataType = TypeId::kNumberTypeFloat32;
  output1->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(output1));

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
  std::vector<float> a_data = {0, 1, 2, 1};
  std::vector<float> expect_a = {0, 1};
  std::vector<float> expect_b = {2, 1};

  for (size_t i = 0; i < a_data.size(); ++i) in0_data[i] = a_data[i];
  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 2);
  auto *outData0 = reinterpret_cast<const float *>(outputs.front().Data().get());
  ASSERT_EQ(0, CompareOutputData(outData0, expect_a.data(), expect_a.size()));
  auto *outData1 = reinterpret_cast<const float *>(outputs.back().Data().get());
  ASSERT_EQ(0, CompareOutputData(outData1, expect_b.data(), expect_b.size()));
}

TEST_F(TestPNNA_Split, shape_1_26_26_512_int8) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";

  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0};
  node->outputIndex = {1, 2};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_Split;
  auto primitive = new schema::SplitT;
  primitive->axis = 3;
  primitive->output_num = 2;
  primitive->size_splits = {256, 256};
  node->primitive->value.value = primitive;
  node->name = "Split";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {1, 2};

  float input_min = 0, input_max = 2, output_min = 0, output_max = 2;
  float input_scale, output_scale;
  int input_zero_point, output_zero_point;
  std::vector<float> input_data(1 * 26 * 26 * 512);
  for (size_t i = 0; i < 26; ++i) {
    for (size_t j = 0; j < 26; j++) {
      for (size_t k = 0; k < 256; k++) {
        input_data[i * 26 * 512 + j * 512 + k] = 1;
      }
      for (size_t k = 256; k < 512; k++) {
        input_data[i * 26 * 512 + j * 512 + k] = 2;
      }
    }
  }
  std::vector<int8_t> input_quant_data(input_data.size());
  QuantProcess(input_data.data(), input_data.size(), input_min, input_max, &input_scale, &input_zero_point,
               input_quant_data.data());
  auto input_quant = std::make_unique<schema::QuantParamT>();
  input_quant->scale = input_scale;
  input_quant->zeroPoint = input_zero_point;
  std::vector<float> expect_a(1 * 26 * 26 * 256);
  std::vector<float> expect_b(1 * 26 * 26 * 256);
  for (size_t i = 0; i < expect_a.size(); ++i) expect_a[i] = 1;
  for (size_t i = 0; i < expect_b.size(); ++i) expect_b[i] = 2;
  QuantProcess(expect_a.data(), expect_a.size(), output_min, output_max, &output_scale, &output_zero_point, nullptr);
  auto output_quant0 = std::make_unique<schema::QuantParamT>();
  output_quant0->scale = output_scale;
  output_quant0->zeroPoint = output_zero_point;

  auto output_quant1 = std::make_unique<schema::QuantParamT>();
  output_quant1->scale = output_scale;
  output_quant1->zeroPoint = output_zero_point;

  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = lite::NodeType_Parameter;
  input0->dataType = TypeId::kNumberTypeInt8;
  input0->format = schema::Format_NCHW;
  input0->dims = {1, 26, 26, 512};
  input0->offset = -1;
  input0->quantParams.emplace_back(std::move(input_quant));
  meta_graph->allTensors.emplace_back(std::move(input0));

  auto output0 = std::make_unique<schema::TensorT>();
  output0->nodeType = lite::NodeType_Parameter;
  output0->dataType = TypeId::kNumberTypeInt8;
  output0->offset = -1;
  output0->quantParams.emplace_back(std::move(output_quant0));
  meta_graph->allTensors.emplace_back(std::move(output0));

  auto output1 = std::make_unique<schema::TensorT>();
  output1->nodeType = lite::NodeType_Parameter;
  output1->dataType = TypeId::kNumberTypeInt8;
  output1->offset = -1;
  output1->quantParams.emplace_back(std::move(output_quant1));
  meta_graph->allTensors.emplace_back(std::move(output1));

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
  ASSERT_EQ(outputs.size(), 2);
  std::vector<float> output0_dequant(outputs[0].ElementNum());
  Dequantize(static_cast<const int8_t *>(outputs[0].MutableData()), outputs[0].ElementNum(), output_scale,
             output_zero_point, output0_dequant.data());
  ASSERT_EQ(0, CompareOutputData(output0_dequant.data(), expect_a.data(), expect_a.size(), 1e-2));

  std::vector<float> output1_dequant(outputs[1].ElementNum());
  Dequantize(static_cast<const int8_t *>(outputs[1].MutableData()), outputs[1].ElementNum(), output_scale,
             output_zero_point, output1_dequant.data());
  ASSERT_EQ(0, CompareOutputData(output1_dequant.data(), expect_b.data(), expect_b.size(), 1e-2));
}
}  // namespace mindspore::lite::dsp::test
