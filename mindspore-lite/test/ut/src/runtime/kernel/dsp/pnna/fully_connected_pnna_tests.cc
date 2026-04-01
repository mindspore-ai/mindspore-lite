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
class TestPNNA_FullyConnected : public CommonTest {};

TEST_F(TestPNNA_FullyConnected, shape_5_2_2_2_fp32) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";
  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1, 2};
  node->outputIndex = {3};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_FullConnection;
  auto primitive = new schema::FullConnectionT();
  primitive->use_axis = false;
  primitive->has_bias = true;
  node->primitive->value.value = primitive;
  node->name = "FullyConnected";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {3};
  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = lite::NodeType_Parameter;
  input0->dataType = TypeId::kNumberTypeFloat32;
  input0->dims = {5, 2, 2, 2};
  input0->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input0));
  auto weight = std::make_unique<schema::TensorT>();
  weight->nodeType = lite::NodeType_ValueNode;
  weight->dataType = TypeId::kNumberTypeFloat32;
  weight->dims = {6, 8};
  weight->offset = -1;
  std::vector<float> weight_data = {
    -0.586269014312498,   0.10845796767603733,  0.8455159907124523,   0.20261291069007226,  0.7564258582027543,
    0.4505005038790615,   -0.607259232240795,   -0.6962171798923924,  0.7967573009922135,   -0.46069496925353715,
    -0.2967638879316592,  -0.7025557337565955,  -0.5313515272071268,  0.07584168670764102,  -0.6860034691410029,
    0.9218806800279316,   -0.07408538201953907, -0.7933652717840096,  0.6636691558029275,   -0.30198695606477477,
    0.790225747868754,    -0.9478140254555916,  0.4537316306461665,   0.1776848732022871,   -0.7492316745474277,
    -0.5825825240770948,  0.5680842804542614,   -0.9255552309192772,  0.20866577718844725,  0.9570928647172854,
    0.18172570688854406,  -0.26442830241827253, -0.24765169216720873, -0.19512285277145702, 0.1120696020054861,
    0.7558578199370625,   -0.15032457481135109, -0.08485585411928809, 0.6343014796699504,   0.026380085222785787,
    -0.40516674259120444, -0.7407588590646037,  -0.28521396461492454, 0.2555841827858194,   0.023640857478332444,
    -0.6540694390119834,  0.7439705499824205,   -0.7579774562590929};
  weight->data.resize(sizeof(float) * 6 * 8);
  memcpy(weight->data.data(), weight_data.data(), 6 * 8 * sizeof(float));
  meta_graph->allTensors.emplace_back(std::move(weight));
  auto bias = std::make_unique<schema::TensorT>();
  bias->nodeType = lite::NodeType_ValueNode;
  bias->dataType = TypeId::kNumberTypeFloat32;
  bias->dims = {6};
  bias->offset = -1;
  // bias data
  std::vector<float> bias_data = {0, 0, 0, 0, 0, 0};
  bias->data.resize(sizeof(float) * 6);
  memcpy(bias->data.data(), bias_data.data(), 6 * sizeof(float));
  meta_graph->allTensors.emplace_back(std::move(bias));

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
  std::vector<float> a_data = {
    4.259103407444801,   5.992151035772917,   -9.495343223733581,  3.0509999931426215, -16.635707833991095,
    -14.72005749234452,  2.8290916795754093,  -15.827977973039049, -16.98208477063347, 2.8801101778935347,
    -0.5905297521382735, 18.042746010536085,  3.913511213700396,   11.571264917136105, 19.084257392926148,
    8.571560238377568,   17.58868010598305,   12.433311533838427,  4.548078598583526,  15.609650071521138,
    6.663372887795717,   17.581323475674594,  1.453277207446778,   -6.119351424589654, -16.87310296820285,
    11.906066592064796,  -13.290100998834653, 19.627129875430548,  16.034262583959162, 10.255738135902781,
    12.134650347811792,  -5.5882066903433305, 15.554050723026322,  15.288481461776783, 17.651080309797287,
    -9.258779162183215,  4.218532791445092,   -6.205309122668545,  1.2220458021156908, 1.6800736573947326};
  std::vector<float> expect = {-19.170732, -7.5019627, -13.015462, -27.760283, 4.1447954, 20.660276, 4.0412164,
                               -33.750015, -4.560128,  7.1035166,  27.976341,  9.75216,   14.383608, -12.87587,
                               -24.688887, -12.185722, 3.7933283,  -19.266382, 17.193876, -49.99205, -15.480089,
                               -3.1659412, 19.470417,  13.758459,  4.0713396,  4.614437,  11.296907, -7.244551,
                               -11.143417, -21.233654};  // correct answer
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

TEST_F(TestPNNA_FullyConnected, shape_5_2_2_2_int8) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";
  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1, 2};
  node->outputIndex = {3};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_FullConnection;
  auto primitive = new schema::FullConnectionT();
  primitive->use_axis = false;
  primitive->has_bias = true;
  node->primitive->value.value = primitive;
  node->name = "FullyConnected";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {3};
  float in[] = {4.259103407444801,   5.992151035772917,   -9.495343223733581,  3.0509999931426215, -16.635707833991095,
                -14.72005749234452,  2.8290916795754093,  -15.827977973039049, -16.98208477063347, 2.8801101778935347,
                -0.5905297521382735, 18.042746010536085,  3.913511213700396,   11.571264917136105, 19.084257392926148,
                8.571560238377568,   17.58868010598305,   12.433311533838427,  4.548078598583526,  15.609650071521138,
                6.663372887795717,   17.581323475674594,  1.453277207446778,   -6.119351424589654, -16.87310296820285,
                11.906066592064796,  -13.290100998834653, 19.627129875430548,  16.034262583959162, 10.255738135902781,
                12.134650347811792,  -5.5882066903433305, 15.554050723026322,  15.288481461776783, 17.651080309797287,
                -9.258779162183215,  4.218532791445092,   -6.205309122668545,  1.2220458021156908, 1.6800736573947326};
  float in0_scale;
  int in0_zp;
  std::vector<int8_t> input_data(40);
  QuantProcess(in, 40, -20, 20, &in0_scale, &in0_zp, input_data.data());
  float weight0[] = {
    -0.586269014312498,   0.10845796767603733,  0.8455159907124523,   0.20261291069007226,  0.7564258582027543,
    0.4505005038790615,   -0.607259232240795,   -0.6962171798923924,  0.7967573009922135,   -0.46069496925353715,
    -0.2967638879316592,  -0.7025557337565955,  -0.5313515272071268,  0.07584168670764102,  -0.6860034691410029,
    0.9218806800279316,   -0.07408538201953907, -0.7933652717840096,  0.6636691558029275,   -0.30198695606477477,
    0.790225747868754,    -0.9478140254555916,  0.4537316306461665,   0.1776848732022871,   -0.7492316745474277,
    -0.5825825240770948,  0.5680842804542614,   -0.9255552309192772,  0.20866577718844725,  0.9570928647172854,
    0.18172570688854406,  -0.26442830241827253, -0.24765169216720873, -0.19512285277145702, 0.1120696020054861,
    0.7558578199370625,   -0.15032457481135109, -0.08485585411928809, 0.6343014796699504,   0.026380085222785787,
    -0.40516674259120444, -0.7407588590646037,  -0.28521396461492454, 0.2555841827858194,   0.023640857478332444,
    -0.6540694390119834,  0.7439705499824205,   -0.7579774562590929};
  float weight_scale;
  int weight_zp;
  std::vector<int8_t> weight_data(48);
  QuantProcess(weight0, 48, -1, 1, &weight_scale, &weight_zp, weight_data.data());
  float expect[] = {-19.170732, -7.5019627, -13.015462, -27.760283, 4.1447954,  20.660276,  4.0412164,  -33.750015,
                    -4.560128,  7.1035166,  27.976341,  9.75216,    14.383608,  -12.87587,  -24.688887, -12.185722,
                    3.7933283,  -19.266382, 17.193876,  -49.99205,  -15.480089, -3.1659412, 19.470417,  13.758459,
                    4.0713396,  4.614437,   11.296907,  -7.244551,  -11.143417, -21.233654};
  float out_scale;
  int out_zp;
  QuantProcess(expect, 30, -50, 50, &out_scale, &out_zp, nullptr);
  auto input_quant0 = std::make_unique<schema::QuantParamT>();
  input_quant0->scale = in0_scale;
  input_quant0->zeroPoint = in0_zp;
  auto weight_quant = std::make_unique<schema::QuantParamT>();
  weight_quant->scale = weight_scale;
  weight_quant->zeroPoint = weight_zp;
  auto bias_quant = std::make_unique<schema::QuantParamT>();
  bias_quant->scale = in0_scale * weight_scale;
  bias_quant->zeroPoint = 0;
  auto out_quant = std::make_unique<schema::QuantParamT>();
  out_quant->scale = out_scale;
  out_quant->zeroPoint = out_zp;
  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = lite::NodeType_Parameter;
  input0->dataType = TypeId::kNumberTypeInt8;
  input0->dims = {5, 2, 2, 2};
  input0->offset = -1;
  input0->quantParams.emplace_back(std::move(input_quant0));
  meta_graph->allTensors.emplace_back(std::move(input0));
  auto weight = std::make_unique<schema::TensorT>();
  weight->nodeType = lite::NodeType_ValueNode;
  weight->dataType = TypeId::kNumberTypeInt8;
  weight->dims = {6, 8};
  weight->offset = -1;
  weight->quantParams.emplace_back(std::move(weight_quant));
  weight->data.resize(sizeof(int8_t) * 48);
  memcpy(weight->data.data(), weight_data.data(), 48 * sizeof(int8_t));
  meta_graph->allTensors.emplace_back(std::move(weight));
  auto bias = std::make_unique<schema::TensorT>();
  bias->nodeType = lite::NodeType_ValueNode;
  bias->dataType = TypeId::kNumberTypeInt32;
  bias->dims = {6};
  bias->offset = -1;
  bias->quantParams.emplace_back(std::move(bias_quant));
  // bias data
  bias->data.resize(sizeof(int32_t) * 6);
  std::vector<int32_t> bias_data = {0, 0, 0, 0, 0, 0};
  memcpy(bias->data.data(), bias_data.data(), 6 * sizeof(int32_t));
  meta_graph->allTensors.emplace_back(std::move(bias));
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
    in0_data[i] = input_data[i];
  }
  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);
  auto *outData = reinterpret_cast<const int8_t *>(outputs.front().Data().get());
  std::vector<float> out(outputs.front().ElementNum());
  Dequantize(outData, outputs.front().ElementNum(), out_scale, out_zp, out.data());
  ASSERT_EQ(0, CompareOutputData(out.data(), expect, outputs.front().ElementNum(), 0.3));
}
}  // namespace mindspore::lite::dsp::test
