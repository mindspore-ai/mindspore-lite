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
#include <utility>
#include "ut/src/runtime/kernel/dsp/pnna/pnna_test.h"
#include "include/api/context.h"
#include "include/api/data_type.h"
#include "include/api/model.h"
#include "src/litert/kernel/dsp/dsp_subgraph.h"
#include "src/litert/kernel_registry.h"
#include "nnacl_c/int8/quantize.h"

namespace mindspore::lite::dsp::test {
class TestPNNA_Softmax : public CommonTest {};

TEST_F(TestPNNA_Softmax, shape_4_3_axis_1_fp32) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";
  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0};
  node->outputIndex = {1};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_Softmax;
  auto primitive = new schema::SoftmaxT;
  std::vector<int64_t> axis{1};
  primitive->axis = axis;
  node->primitive->value.value = primitive;
  node->name = "Softmax";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {1};

  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = lite::NodeType_Parameter;
  input0->dataType = TypeId::kNumberTypeFloat32;
  input0->dims = {4, 3};
  input0->offset = -1;
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
  auto impl = inTensor.impl();
  ASSERT_NE(nullptr, impl);
  float *in0_data = static_cast<float *>(inTensor.MutableData());
  std::vector<float> input_data = {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3};
  std::vector<float> expect = {0.09003057, 0.24472848, 0.66524094, 0.09003057, 0.24472848, 0.66524094,
                               0.09003057, 0.24472848, 0.66524094, 0.09003057, 0.24472848, 0.66524094};
  for (size_t i = 0; i < input_data.size(); ++i) in0_data[i] = input_data[i];
  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);
  auto *outData = reinterpret_cast<const float *>(outputs.front().Data().get());
  ASSERT_EQ(0, CompareOutputData(outData, expect.data(), expect.size(), 1e-3f));
}

TEST_F(TestPNNA_Softmax, shape_4_3_axis_1_int8) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";
  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0};
  node->outputIndex = {1};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_Softmax;
  node->quantType = schema::QuantType_QUANT_ALL;
  auto primitive = new schema::SoftmaxT;
  std::vector<int64_t> axis{1};
  primitive->axis = axis;
  node->primitive->value.value = primitive;
  node->name = "Softmax";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {1};

  std::vector<float> in_data = {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3};
  auto input0_data = std::make_unique<int8_t[]>(in_data.size());
  float in0_scale;
  int in0_zp;
  QuantProcess(in_data.data(), in_data.size(), 1, 3, &in0_scale, &in0_zp, input0_data.get());
  std::vector<float> expect = {0.09003057, 0.24472848, 0.66524094, 0.09003057, 0.24472848, 0.66524094,
                               0.09003057, 0.24472848, 0.66524094, 0.09003057, 0.24472848, 0.66524094};
  float out_scale;
  int out_zp;
  QuantProcess(expect.data(), expect.size(), 0.09003057f, 0.66524094f, &out_scale, &out_zp, nullptr);
  auto input_quant = std::make_unique<schema::QuantParamT>();
  input_quant->scale = in0_scale;
  input_quant->zeroPoint = in0_zp;
  auto output_quant = std::make_unique<schema::QuantParamT>();
  output_quant->scale = out_scale;
  output_quant->zeroPoint = out_zp;

  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = lite::NodeType_Parameter;
  input0->dataType = TypeId::kNumberTypeInt8;
  input0->dims = {4, 3};
  input0->offset = -1;
  input0->quantParams.emplace_back(std::move(input_quant));
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
  auto impl = inTensor.impl();
  ASSERT_NE(nullptr, impl);
  int8_t *in0_data = static_cast<int8_t *>(inTensor.MutableData());
  for (size_t i = 0; i < inTensor.ElementNum(); ++i) in0_data[i] = input0_data[i];
  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);
  std::vector<int8_t> outData(outputs[0].ElementNum());
  memcpy(outData.data(), outputs[0].MutableData(), outData.size());
  std::vector<float> output_dequant(outputs.front().ElementNum());
  Dequantize(outData.data(), outputs.front().ElementNum(), out_scale, out_zp, output_dequant.data());
  ASSERT_EQ(0, CompareOutputData(output_dequant.data(), expect.data(), expect.size(), 0.2));
}
}  // namespace mindspore::lite::dsp::test
