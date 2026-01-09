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
class TestPNNA_DeConv2d : public CommonTest {};

TEST_F(TestPNNA_DeConv2d, shape_1_2_2_2_nhwc_fp32) {
  std::vector<int> input_shape = {1, 2, 2, 2};
  std::vector<int> kernel_shape = {2, 1, 1, 2};
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";
  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1, 2};
  node->outputIndex = {3};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_Conv2dTransposeFusion;
  auto primitive = new schema::Conv2dTransposeFusionT;
  // set deconv2d params
  primitive->format = schema::Format_NHWC;
  primitive->group = 1;
  primitive->dilation = {1, 1};
  primitive->stride = {1, 1};
  primitive->pad_mode = schema::PadMode_PAD;
  primitive->pad_list = {0, 0, 0, 0};
  primitive->pad = {0, 0, 0, 0};
  primitive->output_paddings = {1, 1};
  primitive->out_channel = 2;
  primitive->in_channel = 2;
  primitive->kernel_size = {1, 1};
  node->primitive->value.value = primitive;
  node->name = "DeConv2D";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {3};

  auto input = std::make_unique<schema::TensorT>();
  input->nodeType = lite::NodeType_Parameter;
  input->dataType = TypeId::kNumberTypeFloat32;
  input->format = schema::Format_NHWC;
  input->dims = input_shape;
  meta_graph->allTensors.emplace_back(std::move(input));

  auto kernel_input = std::make_unique<schema::TensorT>();
  kernel_input->nodeType = lite::NodeType_ValueNode;
  kernel_input->dataType = TypeId::kNumberTypeFloat32;
  kernel_input->format = schema::Format_NHWC;
  kernel_input->dims = kernel_shape;
  std::vector<float> kernel_data = {1, 1, 1, 1};
  kernel_input->data.resize(sizeof(float) * kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * kernel_shape[3]);
  memcpy(kernel_input->data.data(), kernel_data.data(),
         kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * kernel_shape[3] * sizeof(float));
  meta_graph->allTensors.emplace_back(std::move(kernel_input));

  auto bias = std::make_unique<schema::TensorT>();
  bias->nodeType = lite::NodeType_ValueNode;
  bias->dataType = TypeId::kNumberTypeFloat32;
  bias->format = schema::Format_NHWC;
  bias->dims = {2};
  bias->name = "deconv2d.bias";
  // bias data
  std::vector<float> bias_data = {0, 0};
  bias->data.resize(sizeof(float) * 2);
  memcpy(bias->data.data(), bias_data.data(), 2 * sizeof(float));
  meta_graph->allTensors.emplace_back(std::move(bias));

  auto output = std::make_unique<schema::TensorT>();
  output->nodeType = lite::NodeType_Parameter;
  output->dataType = TypeId::kNumberTypeFloat32;
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
  float *in_data = static_cast<float *>(inTensor.MutableData());
  std::vector<float> input_data = {0, 1, 2, 3, 4, 5, -6, -7};
  for (size_t i = 0; i < input_data.size(); ++i) in_data[i] = input_data[i];
  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);
  auto *outData = reinterpret_cast<const float *>(outputs.front().Data().get());
  std::vector<float> expect = {1, 1, 5, 5, 0, 0, 9, 9, -13, -13, 0, 0, 0, 0, 0, 0, 0, 0};
  ASSERT_EQ(0, CompareOutputData(outData, expect.data(), outputs.front().ElementNum()));
}

TEST_F(TestPNNA_DeConv2d, shape_1_4_4_1_no_bias_nhwc_int8) {
  std::vector<int> input_shape = {1, 4, 4, 1};
  std::vector<int> kernel_shape = {1, 3, 3, 1};
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";
  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1};
  node->outputIndex = {2};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_Conv2dTransposeFusion;
  auto primitive = new schema::Conv2dTransposeFusionT;
  // set deconv2d params
  primitive->format = schema::Format_NHWC;
  primitive->group = 1;
  primitive->dilation = {1, 1};
  primitive->stride = {1, 1};
  primitive->pad_mode = schema::PadMode_SAME;
  primitive->pad_list = {0, 0, 0, 0};
  primitive->pad = {0, 0, 0, 0};
  primitive->output_paddings = {0, 0};
  primitive->out_channel = 1;
  primitive->kernel_size = {kernel_shape[1], kernel_shape[2]};
  node->primitive->value.value = primitive;
  node->name = "DeConv2D";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {2};

  float in_scale, kernel_scale, out_scale;
  int in_zp, kernel_zp, out_zp;
  std::vector<float> input_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<int8_t> input_quant_data(input_data.size());
  QuantProcess(input_data.data(), input_data.size(), 0.0, 16.0, &in_scale, &in_zp, input_quant_data.data());
  std::vector<float> kernel_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int8_t> kernel_quant_data(kernel_data.size());
  QuantProcess(kernel_data.data(), kernel_data.size(), 0.0, 9.0, &kernel_scale, &kernel_zp, kernel_quant_data.data());
  std::vector<float> expect = {29, 62, 83, 75, 99, 192, 237, 198, 207, 372, 417, 330, 263, 446, 485, 365};
  QuantProcess(expect.data(), expect.size(), 0.0, 485, &out_scale, &out_zp, nullptr);
  // quant params
  auto input_quant = std::make_unique<schema::QuantParamT>();
  input_quant->scale = in_scale;
  input_quant->zeroPoint = in_zp;
  auto kernel_quant = std::make_unique<schema::QuantParamT>();
  kernel_quant->scale = kernel_scale;
  kernel_quant->zeroPoint = kernel_zp;
  auto output_quant = std::make_unique<schema::QuantParamT>();
  output_quant->scale = out_scale;
  output_quant->zeroPoint = out_zp;

  auto input = std::make_unique<schema::TensorT>();
  input->nodeType = lite::NodeType_Parameter;
  input->dataType = TypeId::kNumberTypeInt8;
  input->format = schema::Format_NHWC;
  input->dims = input_shape;
  input->quantParams.emplace_back(std::move(input_quant));
  meta_graph->allTensors.emplace_back(std::move(input));

  auto kernel_input = std::make_unique<schema::TensorT>();
  kernel_input->nodeType = lite::NodeType_ValueNode;
  kernel_input->dataType = TypeId::kNumberTypeInt8;
  kernel_input->format = schema::Format_NHWC;
  kernel_input->dims = kernel_shape;
  kernel_input->quantParams.emplace_back(std::move(kernel_quant));
  kernel_input->data.resize(sizeof(int8_t) * kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * kernel_shape[3]);
  memcpy(kernel_input->data.data(), kernel_quant_data.data(),
         kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * kernel_shape[3] * sizeof(int8_t));
  meta_graph->allTensors.emplace_back(std::move(kernel_input));

  auto output = std::make_unique<schema::TensorT>();
  output->nodeType = lite::NodeType_Parameter;
  output->dataType = TypeId::kNumberTypeInt8;
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
  int8_t *in_data = static_cast<int8_t *>(inTensor.MutableData());
  for (size_t i = 0; i < input_data.size(); ++i) in_data[i] = input_quant_data[i];
  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);
  auto outData = reinterpret_cast<const int8_t *>(outputs.front().Data().get());
  std::vector<float> output_dequant(outputs[0].ElementNum());
  Dequantize(outData, output_dequant.size(), out_scale, out_zp, output_dequant.data());
  ASSERT_EQ(0, CompareOutputData(output_dequant.data(), expect.data(), expect.size(), 1.0));
}
}  // namespace mindspore::lite::dsp::test
