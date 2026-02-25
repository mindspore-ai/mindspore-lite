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
class TestPNNA_AvgPool : public CommonTest {};

TEST_F(TestPNNA_AvgPool, shape_1_1_3_3_fp32_kernel_2_stride_1_nhwc) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";

  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0};
  node->outputIndex = {1};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_AvgPoolFusion;
  auto primitive = new schema::AvgPoolFusionT;
  std::vector<int64_t> kernel_size{2, 2};
  std::vector<int64_t> strides{1, 1};
  std::vector<int64_t> pad{0, 0, 0, 0};
  primitive->kernel_size = kernel_size;
  primitive->strides = strides;
  primitive->pad = pad;
  node->primitive->value.value = primitive;
  node->name = "AvgPool";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {1};

  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = lite::NodeType_Parameter;
  input0->dataType = TypeId::kNumberTypeFloat32;
  input0->format = schema::Format_NHWC;
  input0->dims = {1, 3, 3, 1};
  meta_graph->allTensors.emplace_back(std::move(input0));

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
  auto impl = inTensor.impl();
  ASSERT_NE(nullptr, impl);
  float *in_data = static_cast<float *>(inTensor.MutableData());
  std::vector<float> input_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> expect = {3, 4, 6, 7};
  for (size_t i = 0; i < input_data.size(); ++i) in_data[i] = input_data[i];
  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);
  auto *outData = reinterpret_cast<const float *>(outputs.front().Data().get());
  ASSERT_EQ(0, CompareOutputData(outData, expect.data(), expect.size()));
}

TEST_F(TestPNNA_AvgPool, shape_1_1_3_3_int8_kernel_2_stride_1_nhwc) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";

  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0};
  node->outputIndex = {1};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_AvgPoolFusion;
  auto primitive = new schema::AvgPoolFusionT;
  std::vector<int64_t> kernel_size{2, 2};
  std::vector<int64_t> strides{1, 1};
  std::vector<int64_t> pad{0, 0, 0, 0};
  primitive->kernel_size = kernel_size;
  primitive->strides = strides;
  primitive->pad = pad;
  node->primitive->value.value = primitive;
  node->name = "AvgPool";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {1};

  std::vector<float> input_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  float input_min = -9, input_max = 9;
  std::vector<int8_t> input_quant_data(input_data.size());
  float in0_scale;
  int in0_zp;
  QuantProcess(input_data.data(), input_data.size(), input_min, input_max, &in0_scale, &in0_zp,
               input_quant_data.data());

  std::vector<float> expect = {3, 4, 6, 7};
  float output_min = -7, output_max = 7;
  float out_scale;
  int out_zp;
  QuantProcess(expect.data(), expect.size(), output_min, output_max, &out_scale, &out_zp, nullptr);
  auto input_quant0 = std::make_unique<schema::QuantParamT>();
  input_quant0->scale = in0_scale;
  input_quant0->zeroPoint = in0_zp;

  auto out_quant = std::make_unique<schema::QuantParamT>();
  out_quant->scale = out_scale;
  out_quant->zeroPoint = out_zp;

  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = lite::NodeType_Parameter;
  input0->dataType = TypeId::kNumberTypeInt8;
  input0->format = schema::Format_NHWC;
  input0->dims = {1, 3, 3, 1};
  input0->quantParams.emplace_back(std::move(input_quant0));
  meta_graph->allTensors.emplace_back(std::move(input0));

  auto output = std::make_unique<schema::TensorT>();
  output->nodeType = lite::NodeType_Parameter;
  output->dataType = TypeId::kNumberTypeInt8;
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
  auto inTensor = inputs.front();
  int8_t *in_data = static_cast<int8_t *>(inTensor.MutableData());
  for (size_t i = 0; i < input_quant_data.size(); ++i) in_data[i] = input_quant_data[i];
  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);
  std::vector<float> output_dequant(outputs.front().ElementNum());
  auto *outData = reinterpret_cast<const int8_t *>(outputs.front().Data().get());
  Dequantize(outData, outputs.front().ElementNum(), out_scale, out_zp, output_dequant.data());
  ASSERT_EQ(0, CompareOutputData(output_dequant.data(), expect.data(), outputs.front().ElementNum(), 1e-1));
}
}  // namespace mindspore::lite::dsp::test
