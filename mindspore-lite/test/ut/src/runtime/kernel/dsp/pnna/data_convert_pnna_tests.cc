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

#include <vector>
#include "ut/src/runtime/kernel/dsp/pnna/pnna_test.h"
#include "include/api/context.h"
#include "include/api/data_type.h"
#include "include/api/model.h"
#include "schema/inner/model_generated.h"
#include "src/litert/kernel_registry.h"

namespace mindspore::lite::dsp::test {
class TestPNNA_DataConvert : public CommonTest {};

TEST_F(TestPNNA_DataConvert, fp32_to_int8) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";

  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0};
  node->outputIndex = {1};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_QuantDTypeCast;
  node->quantType = schema::QuantType_QUANT_ALL;
  auto primitive = new schema::QuantDTypeCastT;
  primitive->src_t = TypeId::kNumberTypeFloat32;
  primitive->dst_t = TypeId::kNumberTypeInt8;
  node->primitive->value.value = primitive;
  node->name = "DataConvert";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {1};

  float output_min = -57;
  float output_max = 0.449816;
  float out0_scale;
  int out0_zp;
  std::vector<float> expect = {0.449816, 0.449816, 0.449816, 0.449816, 0.449816, -20.0168, 0.449816, 0.449816,
                               0.449816, 0.449816, 0.449816, 0.449816, 0.449816, 0.449816, 0.449816, 0.449816};
  QuantProcess(expect.data(), expect.size(), output_min, output_max, &out0_scale, &out0_zp, nullptr);
  auto output_quant = std::make_unique<schema::QuantParamT>();
  output_quant->scale = out0_scale;
  output_quant->zeroPoint = out0_zp;

  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = lite::NodeType_Parameter;
  input0->dataType = TypeId::kNumberTypeFloat32;
  input0->dims = {1, 4, 4, 1};
  input0->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input0));

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
  float *in0_data = static_cast<float *>(inTensor.MutableData());
  std::vector<float> input_data = {1, 2, 5, 6, 10, -20, 3, 8, 18, 10, 3, 4, 11, 16, 15, 25};
  for (size_t i = 0; i < input_data.size(); ++i) in0_data[i] = input_data[i];
  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);
  std::vector<float> output_dequant(outputs.front().ElementNum());
  auto *outData = reinterpret_cast<const int8_t *>(outputs.front().Data().get());
  Dequantize(outData, outputs.front().ElementNum(), out0_scale, out0_zp, output_dequant.data());
  ASSERT_EQ(0, CompareOutputData(output_dequant.data(), expect.data(), expect.size(), 0.2));
}

TEST_F(TestPNNA_DataConvert, int8_to_fp32) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";

  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0};
  node->outputIndex = {1};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_QuantDTypeCast;
  node->quantType = schema::QuantType_QUANT_ALL;
  auto primitive = new schema::QuantDTypeCastT;
  primitive->src_t = TypeId::kNumberTypeInt8;
  primitive->dst_t = TypeId::kNumberTypeFloat32;
  node->primitive->value.value = primitive;
  node->name = "DataConvert";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {1};

  float input_min = 0.21176, input_max = 5;
  float in0_scale;
  int in0_zp;
  QuantProcess(nullptr, 0, input_min, input_max, &in0_scale, &in0_zp, nullptr);
  auto input_quant = std::make_unique<schema::QuantParamT>();
  input_quant->scale = in0_scale;
  input_quant->zeroPoint = in0_zp;

  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = lite::NodeType_Parameter;
  input0->dataType = TypeId::kNumberTypeInt8;
  input0->dims = {1, 4, 4, 1};
  input0->offset = -1;
  input0->quantParams.emplace_back(std::move(input_quant));
  meta_graph->allTensors.emplace_back(std::move(input0));

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
  int8_t *in0_data = static_cast<int8_t *>(inTensor.MutableData());
  std::vector<int8_t> input_data = {10, 14, 29, 33, 52, 99, 19, 43, 90, 52, 19, 24, 57, 127, 76, 123};
  std::vector<float> expect = {2.59128, 2.66639, 2.94805, 3.02316, 3.37993, 4.26247, 2.76028, 3.21094,
                               4.09348, 3.37993, 2.76028, 2.85417, 3.47382, 4.78824, 3.83059, 4.71313};
  for (size_t i = 0; i < input_data.size(); ++i) in0_data[i] = input_data[i];
  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);
  auto *outData = reinterpret_cast<const float *>(outputs.front().Data().get());
  ASSERT_EQ(0, CompareOutputData(outData, expect.data(), expect.size()));
}
}  // namespace mindspore::lite::dsp::test
