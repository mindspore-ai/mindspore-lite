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
#include "src/litert/kernel/dsp/dsp_subgraph.h"
#include "src/litert/kernel_registry.h"
#include "nnacl_c/int8/quantize.h"

namespace mindspore::lite::dsp::test {

class TestPNNA_Clip : public CommonTest {};

TEST_F(TestPNNA_Clip, shape_2_4_fp32) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";
  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0};
  node->outputIndex = {1};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_Clip;
  auto primitive = new schema::ClipT();
  primitive->min = 5;
  primitive->max = 20;
  node->primitive->value.value = primitive;
  node->name = "Clip";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {1};
  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = lite::NodeType_Parameter;
  input0->dataType = TypeId::kNumberTypeFloat32;
  input0->dims = {2, 4};
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
  float *in0_data = static_cast<float *>(inputs[0].MutableData());
  std::vector<float> a_data = {1, 25, 5, 7, 4, 11, 6, 21};
  std::vector<float> expect = {5, 20, 5, 7, 5, 11, 6, 20};  // correct answer
  for (size_t i = 0; i < inputs[0].ElementNum(); ++i) {
    in0_data[i] = a_data[i];
  }
  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);
  auto *outData = reinterpret_cast<const float *>(outputs.front().Data().get());
  ASSERT_EQ(0, CompareOutputData(outData, expect.data(), expect.size()));
}

// Unsupported op_type index: 30, op_type: Clip, data_type: 32
TEST_F(TestPNNA_Clip, DISABLED_shape_2_4_int8) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";
  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0};
  node->outputIndex = {1};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_Clip;
  auto primitive = new schema::ClipT();
  primitive->min = 5;
  primitive->max = 20;
  node->primitive->value.value = primitive;
  node->name = "Clip";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {1};
  int length = 2 * 4;
  float in0[] = {1, 25, 5, 7, 4, 11, 6, 21};
  float in0_scale;
  int in0_zp;
  std::vector<int8_t> input0_data(length);
  QuantProcess(in0, length, 0, 25, &in0_scale, &in0_zp, input0_data.data());
  float expect[] = {5, 20, 5, 7, 5, 11, 6, 20};
  float out_scale;
  int out_zp;
  QuantProcess(expect, length, 0, 20, &out_scale, &out_zp, nullptr);
  auto input_quant0 = std::make_unique<schema::QuantParamT>();
  input_quant0->scale = in0_scale;
  input_quant0->zeroPoint = in0_zp;
  auto out_quant = std::make_unique<schema::QuantParamT>();
  out_quant->scale = out_scale;
  out_quant->zeroPoint = out_zp;
  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = lite::NodeType_Parameter;
  input0->dataType = TypeId::kNumberTypeInt8;
  input0->dims = {2, 4};
  input0->offset = -1;
  input0->quantParams.emplace_back(std::move(input_quant0));
  meta_graph->allTensors.emplace_back(std::move(input0));
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
  ASSERT_EQ(inputs.size(), 1);
  int8_t *in0_data = static_cast<int8_t *>(inputs[0].MutableData());
  for (size_t i = 0; i < inputs[0].ElementNum(); ++i) {
    in0_data[i] = input0_data[i];
  }
  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);
  auto *outData = reinterpret_cast<const int8_t *>(outputs.front().Data().get());
  std::vector<float> out(outputs.front().ElementNum());
  Dequantize(outData, outputs.front().ElementNum(), out_scale, out_zp, out.data());
  ASSERT_EQ(0, CompareOutputData(out.data(), expect, outputs.front().ElementNum(), 1e-1));
}
}  // namespace mindspore::lite::dsp::test
