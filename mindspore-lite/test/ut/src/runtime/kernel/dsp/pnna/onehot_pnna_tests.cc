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
#include <utility>
#include <vector>
#include <memory>
#include "ut/src/runtime/kernel/dsp/pnna/pnna_test.h"
#include "schema/inner/model_generated.h"
#include "common/common_test.h"
#include "src/litert/kernel_registry.h"
#include "include/api/model.h"
#include "include/api/context.h"
#include "include/api/data_type.h"

namespace mindspore::lite::dsp::test {
class TestPNNA_OneHot : public CommonTest {};

TEST_F(TestPNNA_OneHot, test_axis_fp32) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";

  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1, 2};
  node->outputIndex = {3};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_OneHot;
  auto primitive = new schema::OneHotT;
  primitive->axis = -1;
  node->primitive->value.value = primitive;
  node->name = "OneHot";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {3};

  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = lite::NodeType_Parameter;
  input0->dataType = TypeId::kNumberTypeInt32;
  input0->dims = {3, 3};
  input0->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input0));

  auto depth = std::make_unique<schema::TensorT>();
  depth->nodeType = lite::NodeType_ValueNode;
  depth->dataType = TypeId::kNumberTypeInt32;
  depth->dims = {1};
  depth->offset = -1;
  std::vector<int> depth_data = {4};
  depth->data.resize(sizeof(int) * 1);
  memcpy(depth->data.data(), depth_data.data(), 1 * sizeof(int));
  meta_graph->allTensors.emplace_back(std::move(depth));

  auto off_on_value = std::make_unique<schema::TensorT>();
  off_on_value->nodeType = lite::NodeType_ValueNode;
  off_on_value->dataType = TypeId::kNumberTypeFloat32;
  off_on_value->dims = {2};
  off_on_value->offset = -1;
  std::vector<float> off_on_data = {0, 1};
  off_on_value->data.resize(sizeof(float) * 2);
  memcpy(off_on_value->data.data(), off_on_data.data(), 2 * sizeof(float));
  meta_graph->allTensors.emplace_back(std::move(off_on_value));

  auto output0 = std::make_unique<schema::TensorT>();
  output0->nodeType = lite::NodeType_Parameter;
  output0->dataType = TypeId::kNumberTypeFloat32;
  output0->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(output0));

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
  int *in0_data = static_cast<int *>(inputs[0].MutableData());

  std::vector<int> input_data = {0, 0, 1, 0, 0, 2, 0, 1, 2};

  std::vector<float> expect = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                               1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
                               1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};

  for (size_t i = 0; i < input_data.size(); ++i) {
    in0_data[i] = input_data[i];
  }

  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);
  auto *outData0 = reinterpret_cast<const float *>(outputs.front().Data().get());
  ASSERT_EQ(0, CompareOutputData(outData0, expect.data(), expect.size()));
}

TEST_F(TestPNNA_OneHot, test_axis_int8) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";

  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1, 2};
  node->outputIndex = {3};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_OneHot;
  auto primitive = new schema::OneHotT;
  primitive->axis = -1;
  node->primitive->value.value = primitive;
  node->name = "OneHot";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {3};

  std::vector<int> input_data = {0, 0, 1, 0, 0, 2, 0, 1, 2};
  std::vector<float> expect = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                               1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
                               1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};

  float in0_scale;
  int in0_zp;
  std::vector<float> off_on_data = {0, 1};
  std::vector<int8_t> off_on_data_quant(2);
  QuantProcess(off_on_data.data(), 2, 0, 1, &in0_scale, &in0_zp, off_on_data_quant.data());
  auto off_on_value_quant = std::make_unique<schema::QuantParamT>();
  off_on_value_quant->scale = in0_scale;
  off_on_value_quant->zeroPoint = in0_zp;

  float out_scale;
  int out_zp;
  auto length = expect.size();
  QuantProcess(expect.data(), length, 0, 1, &out_scale, &out_zp, nullptr);
  auto out_quant = std::make_unique<schema::QuantParamT>();
  out_quant->scale = out_scale;
  out_quant->zeroPoint = out_zp;

  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = lite::NodeType_Parameter;
  input0->dataType = TypeId::kNumberTypeInt32;
  input0->dims = {3, 3};
  input0->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input0));

  auto depth = std::make_unique<schema::TensorT>();
  depth->nodeType = lite::NodeType_ValueNode;
  depth->dataType = TypeId::kNumberTypeInt32;
  depth->dims = {1};
  depth->offset = -1;
  std::vector<int> depth_data = {4};
  depth->data.resize(sizeof(int) * 1);
  memcpy(depth->data.data(), depth_data.data(), 1 * sizeof(int));
  meta_graph->allTensors.emplace_back(std::move(depth));

  auto off_on_value = std::make_unique<schema::TensorT>();
  off_on_value->nodeType = lite::NodeType_ValueNode;
  off_on_value->dataType = TypeId::kNumberTypeInt8;
  off_on_value->dims = {2};
  off_on_value->offset = -1;
  off_on_value->quantParams.emplace_back(std::move(off_on_value_quant));
  off_on_value->data.resize(sizeof(int8_t) * 2);
  memcpy(off_on_value->data.data(), off_on_data_quant.data(), 2 * sizeof(int8_t));
  meta_graph->allTensors.emplace_back(std::move(off_on_value));

  auto output0 = std::make_unique<schema::TensorT>();
  output0->nodeType = lite::NodeType_Parameter;
  output0->dataType = TypeId::kNumberTypeInt8;
  output0->offset = -1;
  output0->quantParams.emplace_back(std::move(out_quant));
  meta_graph->allTensors.emplace_back(std::move(output0));

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
  int *in0_data = static_cast<int *>(inputs[0].MutableData());

  for (size_t i = 0; i < input_data.size(); ++i) {
    in0_data[i] = input_data[i];
  }

  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);

  std::vector<float> out(length);
  auto *outData = reinterpret_cast<const int8_t *>(outputs.front().Data().get());
  Dequantize(outData, length, out_scale, out_zp, out.data());

  ASSERT_EQ(0, CompareOutputData(out.data(), expect.data(), expect.size(), 1e-3));
}

}  // namespace mindspore::lite::dsp::test
