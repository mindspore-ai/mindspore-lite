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
class TestPNNA_Conv2d : public CommonTest {};

TEST_F(TestPNNA_Conv2d, shape_1_2_2_2_float32_NHWC_Test) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";

  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1, 2};
  node->outputIndex = {3};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_Conv2DFusion;
  auto primitive = new schema::Conv2DFusionT;
  primitive->format = schema::Format_NHWC;
  primitive->group = 1;
  primitive->kernel_size = {1, 1};
  primitive->stride = {1, 1};
  primitive->dilation = {1, 1};
  primitive->pad_list = {0, 0, 0, 0};
  node->primitive->value.value = primitive;
  node->name = "Conv2D";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {3};

  auto input = std::make_unique<schema::TensorT>();
  input->nodeType = lite::NodeType_Parameter;
  input->dataType = TypeId::kNumberTypeFloat32;
  input->format = schema::Format_NHWC;
  input->dims = {1, 2, 2, 2};
  input->name = "x";
  meta_graph->allTensors.emplace_back(std::move(input));

  auto weight = std::make_unique<schema::TensorT>();
  weight->nodeType = lite::NodeType_ValueNode;
  weight->dataType = TypeId::kNumberTypeFloat32;
  weight->format = schema::Format_NHWC;
  weight->dims = {2, 1, 1, 2};
  weight->offset = -1;
  weight->name = "conv2d.weight";
  std::vector<float> weight_data = {1, 1, 1, 1};
  weight->data.resize(sizeof(float) * 2 * 2);
  memcpy(weight->data.data(), weight_data.data(), 2 * 2 * sizeof(float));
  meta_graph->allTensors.emplace_back(std::move(weight));

  auto bias = std::make_unique<schema::TensorT>();
  bias->nodeType = lite::NodeType_ValueNode;
  bias->dataType = TypeId::kNumberTypeFloat32;
  bias->format = schema::Format_NHWC;
  bias->dims = {2};
  bias->offset = -1;
  bias->name = "conv2d.bias";
  // bias data
  std::vector<float> bias_data = {0, 0};
  bias->data.resize(sizeof(float) * 2);
  memcpy(bias->data.data(), bias_data.data(), 2 * sizeof(float));
  meta_graph->allTensors.emplace_back(std::move(bias));

  auto output = std::make_unique<schema::TensorT>();
  output->nodeType = lite::NodeType_Parameter;
  output->dataType = TypeId::kNumberTypeFloat32;
  output->name = "y";
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
  float *in_data = static_cast<float *>(inTensor.MutableData());
  // Input data
  std::vector<float> input_data = {0, 1, 2, 3, 4, 5, -6, -7};
  std::vector<float> expect = {1, 1, 5, 5, 9, 9, -13, -13};
  for (size_t i = 0; i < input_data.size(); ++i) in_data[i] = input_data[i];
  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);
  auto *outData = reinterpret_cast<const float *>(outputs.front().Data().get());
  ASSERT_EQ(0, CompareOutputData(outData, expect.data(), expect.size()));
}

TEST_F(TestPNNA_Conv2d, shape_1_2_2_3_int8_NHWCTest) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";

  auto float2int_node = std::make_unique<schema::CNodeT>();
  float2int_node->inputIndex = {0};
  float2int_node->outputIndex = {1};
  float2int_node->primitive = std::make_unique<schema::PrimitiveT>();
  float2int_node->primitive->value.type = schema::PrimitiveType_QuantDTypeCast;
  float2int_node->quantType = schema::QuantType_QUANT_ALL;
  auto quant_primitive = new schema::QuantDTypeCastT;
  quant_primitive->src_t = TypeId::kNumberTypeFloat32;
  quant_primitive->dst_t = TypeId::kNumberTypeInt8;
  float2int_node->primitive->value.value = quant_primitive;
  float2int_node->name = "DataConvert";
  meta_graph->nodes.emplace_back(std::move(float2int_node));

  float input_min = -63.5, input_max = 64, output_min = -63.5, output_max = 64;
  float in0_scale;
  int in0_zp;
  // Input data
  std::vector<float> input_data = {3, 1, -2, 4, 2, -3, 2, -1, -3, 3, -2, -4};
  QuantProcess(input_data.data(), input_data.size(), input_min, input_max, &in0_scale, &in0_zp, nullptr);
  auto input_quant = std::make_unique<schema::QuantParamT>();
  input_quant->scale = in0_scale;
  input_quant->zeroPoint = in0_zp;

  auto weight_quant0 = std::make_unique<schema::QuantParamT>();
  weight_quant0->scale = 1.0;
  weight_quant0->zeroPoint = 0;
  auto weight_quant1 = std::make_unique<schema::QuantParamT>();
  weight_quant1->scale = 2.0;
  weight_quant1->zeroPoint = 0;

  auto bias_quant0 = std::make_unique<schema::QuantParamT>();
  bias_quant0->scale = 0.5;
  bias_quant0->zeroPoint = 0;
  auto bias_quant1 = std::make_unique<schema::QuantParamT>();
  bias_quant1->scale = 1;
  bias_quant1->zeroPoint = 0;

  float out0_scale;
  int out0_zp;
  std::vector<float> expect = {23, 46, -28, -4};
  QuantProcess(expect.data(), expect.size(), output_min, output_max, &out0_scale, &out0_zp, nullptr);
  auto output_quant = std::make_unique<schema::QuantParamT>();
  output_quant->scale = out0_scale;
  output_quant->zeroPoint = out0_zp;
  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = lite::NodeType_Parameter;
  input0->dataType = TypeId::kNumberTypeFloat32;
  input0->format = schema::Format_NHWC;
  input0->dims = {1, 2, 3, 2};
  input0->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input0));

  auto float2int_output = std::make_unique<schema::TensorT>();
  float2int_output->nodeType = lite::NodeType_Parameter;
  float2int_output->dataType = TypeId::kNumberTypeInt8;
  float2int_output->offset = -1;
  float2int_output->quantParams.emplace_back(std::move(input_quant));
  meta_graph->allTensors.emplace_back(std::move(float2int_output));

  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {1, 2, 3};
  node->outputIndex = {4};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_Conv2DFusion;
  auto conv2d_primitive = new schema::Conv2DFusionT;
  conv2d_primitive->format = schema::Format_NHWC;
  conv2d_primitive->group = 1;
  conv2d_primitive->stride = {1, 1};
  conv2d_primitive->dilation = {1, 1};
  conv2d_primitive->pad_mode = schema::PadMode_VALID;
  conv2d_primitive->pad_list = {0, 0, 0, 0};
  node->primitive->value.value = conv2d_primitive;
  node->name = "Conv2D";
  meta_graph->nodes.emplace_back(std::move(node));

  auto weight = std::make_unique<schema::TensorT>();
  weight->nodeType = lite::NodeType_ValueNode;
  weight->dataType = TypeId::kNumberTypeInt8;
  weight->format = schema::Format_NHWC;
  weight->dims = {2, 2, 2, 2};
  weight->offset = -1;
  weight->quantParams.emplace_back(std::move(weight_quant0));
  weight->quantParams.emplace_back(std::move(weight_quant1));
  std::vector<int8_t> weight_data = {1, 2, 3, 4, 3, 4, 5, 6, 4, 4, 3, 3, 2, 2, 1, 1};
  weight->data.resize(sizeof(int8_t) * 2 * 2 * 2 * 2);
  memcpy(weight->data.data(), weight_data.data(), 2 * 2 * 2 * 2 * sizeof(int8_t));
  meta_graph->allTensors.emplace_back(std::move(weight));

  auto bias = std::make_unique<schema::TensorT>();
  bias->nodeType = lite::NodeType_ValueNode;
  bias->dataType = TypeId::kNumberTypeInt32;
  bias->format = schema::Format_NHWC;
  bias->dims = {2};
  bias->offset = -1;
  bias->quantParams.emplace_back(std::move(bias_quant0));
  bias->quantParams.emplace_back(std::move(bias_quant1));
  // bias data
  std::vector<int32_t> bias_data = {6, -2};
  bias->data.resize(sizeof(int32_t) * 2);
  memcpy(bias->data.data(), bias_data.data(), 2 * sizeof(int32_t));
  meta_graph->allTensors.emplace_back(std::move(bias));

  auto output = std::make_unique<schema::TensorT>();
  output->nodeType = lite::NodeType_Parameter;
  output->dataType = TypeId::kNumberTypeInt8;
  output->quantParams.emplace_back(std::move(output_quant));
  meta_graph->allTensors.emplace_back(std::move(output));

  auto int2float_node = std::make_unique<schema::CNodeT>();
  int2float_node->inputIndex = {4};
  int2float_node->outputIndex = {5};
  int2float_node->primitive = std::make_unique<schema::PrimitiveT>();
  int2float_node->primitive->value.type = schema::PrimitiveType_QuantDTypeCast;
  int2float_node->quantType = schema::QuantType_QUANT_ALL;
  auto primitive1 = new schema::QuantDTypeCastT;
  primitive1->src_t = TypeId::kNumberTypeInt8;
  primitive1->dst_t = TypeId::kNumberTypeFloat32;
  int2float_node->primitive->value.value = primitive1;
  int2float_node->name = "DataConvert";
  meta_graph->nodes.emplace_back(std::move(int2float_node));

  auto int2float_output = std::make_unique<schema::TensorT>();
  int2float_output->nodeType = lite::NodeType_Parameter;
  int2float_output->dataType = TypeId::kNumberTypeFloat32;
  int2float_output->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(int2float_output));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {5};

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
  for (size_t i = 0; i < input_data.size(); ++i) in_data[i] = input_data[i];
  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);
  auto *outData = reinterpret_cast<const float *>(outputs.front().Data().get());
  ASSERT_EQ(0, CompareOutputData(outData, expect.data(), expect.size()));
}
}  // namespace mindspore::lite::dsp::test
