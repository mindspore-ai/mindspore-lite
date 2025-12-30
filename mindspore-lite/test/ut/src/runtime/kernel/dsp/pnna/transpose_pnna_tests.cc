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
#include "schema/inner/model_generated.h"
#include "src/litert/kernel_registry.h"

namespace mindspore::lite::dsp::test {
class TestPNNA_Transpose : public CommonTest {};

TEST_F(TestPNNA_Transpose, shape_1_3_2_3_float32_1_3_3_2) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";

  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1};
  node->outputIndex = {2};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_Transpose;
  auto primitive = new schema::TransposeT;
  node->primitive->value.value = primitive;
  node->name = "Transpose";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {2};

  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = lite::NodeType_Parameter;
  input0->dataType = TypeId::kNumberTypeFloat32;
  input0->dims = {1, 3, 2, 3};
  input0->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input0));

  auto perm = std::make_unique<schema::TensorT>();
  perm->nodeType = lite::NodeType_ValueNode;
  perm->dataType = TypeId::kNumberTypeInt32;
  perm->dims = {4};
  perm->offset = -1;
  std::vector<int32_t> perm_data = {0, 3, 1, 2};
  perm->data.resize(sizeof(int32_t) * 4);
  memcpy(perm->data.data(), perm_data.data(), 4 * sizeof(int32_t));
  meta_graph->allTensors.emplace_back(std::move(perm));

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
  std::vector<float> input_data = {0.59885779, 0.62662862, 0.63011179, 0.82569427, 0.64772359, 0.42895413,
                                   0.30216458, 0.01351635, 0.32545444, 0.0360674,  0.33967769, 0.18092504,
                                   0.09479915, 0.52258112, 0.46735646, 0.95689111, 0.51619059, 0.82685718};
  std::vector<float> expect = {0.59885776, 0.82569426, 0.30216458, 0.0360674,  0.09479915, 0.9568911,
                               0.62662864, 0.6477236,  0.01351635, 0.3396777,  0.5225811,  0.5161906,
                               0.6301118,  0.4289541,  0.32545444, 0.18092504, 0.46735647, 0.8268572};
  for (size_t i = 0; i < input_data.size(); ++i) in0_data[i] = input_data[i];
  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);
  auto *outData = reinterpret_cast<const float *>(outputs.front().Data().get());
  ASSERT_EQ(0, CompareOutputData(outData, expect.data(), expect.size(), 1e-3f));
}

TEST_F(TestPNNA_Transpose, shape_1_3_3_2_float32_1_3_2_3) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";

  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1};
  node->outputIndex = {2};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_Transpose;
  auto primitive = new schema::TransposeT;
  node->primitive->value.value = primitive;
  node->name = "Transpose";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {2};

  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = lite::NodeType_Parameter;
  input0->dataType = TypeId::kNumberTypeFloat32;
  input0->dims = {1, 3, 3, 2};
  input0->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input0));

  auto perm = std::make_unique<schema::TensorT>();
  perm->nodeType = lite::NodeType_ValueNode;
  perm->dataType = TypeId::kNumberTypeInt32;
  perm->dims = {4};
  perm->offset = -1;
  std::vector<int32_t> perm_data = {0, 2, 3, 1};
  perm->data.resize(sizeof(int32_t) * 4);
  memcpy(perm->data.data(), perm_data.data(), 4 * sizeof(int32_t));
  meta_graph->allTensors.emplace_back(std::move(perm));

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
  std::vector<float> input_data = {0.59885779, 0.62662862, 0.63011179, 0.82569427, 0.64772359, 0.42895413,
                                   0.30216458, 0.01351635, 0.32545444, 0.0360674,  0.33967769, 0.18092504,
                                   0.09479915, 0.52258112, 0.46735646, 0.95689111, 0.51619059, 0.82685718};
  std::vector<float> expect = {0.59885776, 0.30216458, 0.09479915, 0.62662864, 0.01351635, 0.5225811,
                               0.6301118,  0.32545444, 0.46735647, 0.82569426, 0.0360674,  0.9568911,
                               0.6477236,  0.3396777,  0.5161906,  0.42895412, 0.18092504, 0.8268572};
  for (size_t i = 0; i < input_data.size(); ++i) in0_data[i] = input_data[i];
  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);
  auto *outData = reinterpret_cast<const float *>(outputs.front().Data().get());
  ASSERT_EQ(0, CompareOutputData(outData, expect.data(), expect.size(), 1e-3f));
}

TEST_F(TestPNNA_Transpose, shape_1_3_2_3_int8_1_3_3_2) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";

  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1};
  node->outputIndex = {2};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_Transpose;
  auto primitive = new schema::TransposeT;
  node->primitive->value.value = primitive;
  node->name = "Transpose";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {2};

  int length = 3 * 3 * 2;
  std::vector<float> in0 = {0.59885779, 0.62662862, 0.63011179, 0.82569427, 0.64772359, 0.42895413,
                            0.30216458, 0.01351635, 0.32545444, 0.0360674,  0.33967769, 0.18092504,
                            0.09479915, 0.52258112, 0.46735646, 0.95689111, 0.51619059, 0.82685718};
  std::vector<int8_t> input0_data(length);
  float in0_scale;
  int in0_zp;
  QuantProcess(in0.data(), length, 0, 1, &in0_scale, &in0_zp, input0_data.data());

  std::vector<float> expect = {0.59885776, 0.82569426, 0.30216458, 0.0360674,  0.09479915, 0.9568911,
                               0.62662864, 0.6477236,  0.01351635, 0.3396777,  0.5225811,  0.5161906,
                               0.6301118,  0.4289541,  0.32545444, 0.18092504, 0.46735647, 0.8268572};
  float out_scale;
  int out_zp;
  QuantProcess(expect.data(), length, 0, 1, &out_scale, &out_zp, nullptr);

  auto input_quant0 = std::make_unique<schema::QuantParamT>();
  input_quant0->scale = in0_scale;
  input_quant0->zeroPoint = in0_zp;

  auto out_quant = std::make_unique<schema::QuantParamT>();
  out_quant->scale = out_scale;
  out_quant->zeroPoint = out_zp;

  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = lite::NodeType_Parameter;
  input0->dataType = TypeId::kNumberTypeInt8;
  input0->dims = {1, 3, 2, 3};
  input0->offset = -1;
  input0->quantParams.emplace_back(std::move(input_quant0));
  meta_graph->allTensors.emplace_back(std::move(input0));

  auto perm = std::make_unique<schema::TensorT>();
  perm->nodeType = lite::NodeType_ValueNode;
  perm->dataType = TypeId::kNumberTypeInt32;
  perm->dims = {4};
  perm->offset = -1;
  std::vector<int32_t> perm_data = {0, 3, 1, 2};
  perm->data.resize(sizeof(int32_t) * 4);
  memcpy(perm->data.data(), perm_data.data(), 4 * sizeof(int32_t));
  meta_graph->allTensors.emplace_back(std::move(perm));

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

  // build a modelexpect
  auto model = std::make_shared<mindspore::Model>();
  auto ret = model->Build(content, size, kMindIR_Lite, context);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  auto inputs = model->GetInputs();
  ASSERT_EQ(inputs.size(), 1);
  auto inTensor = inputs.front();
  auto impl = inTensor.impl();
  ASSERT_NE(nullptr, impl);
  int8_t *in0_data = static_cast<int8_t *>(inTensor.MutableData());
  for (size_t i = 0; i < inputs[0].ElementNum(); ++i) in0_data[i] = input0_data[i];
  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);
  std::vector<float> out(length);
  auto *outData = reinterpret_cast<const int8_t *>(outputs.front().Data().get());
  Dequantize(outData, length, out_scale, out_zp, out.data());
  ASSERT_EQ(0, CompareOutputData(out.data(), expect.data(), length, 1e-3));
}
}  // namespace mindspore::lite::dsp::test
