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
class TestPNNA_Batchnorm : public CommonTest {
 public:
  std::unique_ptr<schema::QuantParamT> CreateQuantParamAndData(float *data, size_t size, float min, float max,
                                                               std::vector<int8_t> *quant_data = nullptr) {
    float scale;
    int zp;
    if (quant_data) {
      quant_data->resize(size);
      QuantProcess(data, size, min, max, &scale, &zp, quant_data->data());
    } else {
      QuantProcess(data, size, min, max, &scale, &zp, nullptr);
    }
    auto quant_param = std::make_unique<schema::QuantParamT>();
    quant_param->scale = scale;
    quant_param->zeroPoint = zp;
    return quant_param;
  }

  void RunPNNAModel(std::shared_ptr<schema::MetaGraphT> &meta_graph, void *input_data, void *output_data) {
    flatbuffers::FlatBufferBuilder builder(1024);
    auto offset = schema::MetaGraph::Pack(builder, meta_graph.get());
    builder.Finish(offset);
    schema::FinishMetaGraphBuffer(builder, offset);
    size_t size = builder.GetSize();
    const char *content = reinterpret_cast<char *>(builder.GetBufferPointer());
    auto context = std::make_shared<mindspore::Context>();
    context->SetBuiltInDelegate(mindspore::DelegateMode::kPNNA);
    auto &device_list = context->MutableDeviceInfo();
    device_list.push_back(std::make_shared<DSPDeviceInfo>());
    auto model = std::make_shared<mindspore::Model>();
    auto ret = model->Build(content, size, kMindIR_Lite, context);
    ASSERT_EQ(kSuccess, ret.StatusCode());
    auto inputs = model->GetInputs();
    ASSERT_EQ(inputs.size(), 1);
    auto inTensor = inputs.front();
    auto impl = inTensor.impl();
    ASSERT_NE(nullptr, impl);
    void *in_data = inTensor.MutableData();
    ASSERT_NE(nullptr, in_data);
    ASSERT_NE(nullptr, input_data);
    memcpy(in_data, input_data, inTensor.DataSize());
    std::vector<mindspore::MSTensor> outputs;
    ret = model->Predict(inputs, &outputs);
    ASSERT_EQ(kSuccess, ret.StatusCode());
    ASSERT_EQ(1, outputs.size());
    ASSERT_NE(nullptr, output_data);
    memcpy(output_data, outputs.front().Data().get(), outputs.front().DataSize());
  }
};
TEST_F(TestPNNA_Batchnorm, shape_1_3_3_2_nhwc_fp32) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";
  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1, 2, 3, 4};
  node->outputIndex = {5};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_FusedBatchNorm;
  auto primitive = new schema::FusedBatchNormT;
  primitive->epsilon = 0.001;
  node->primitive->value.value = primitive;
  node->name = "Batchnorm";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {5};
  auto input = std::make_unique<schema::TensorT>();
  input->nodeType = lite::NodeType_Parameter;
  input->dataType = TypeId::kNumberTypeFloat32;
  input->format = schema::Format_NHWC;
  input->dims = {1, 3, 3, 2};
  meta_graph->allTensors.emplace_back(std::move(input));
  auto scale = std::make_unique<schema::TensorT>();
  scale->nodeType = lite::NodeType_ValueNode;
  scale->dataType = TypeId::kNumberTypeFloat32;
  scale->format = schema::Format_NHWC;
  scale->dims = {2};
  std::vector<float> scale_data = {1.0, 1.0};
  scale->data.resize(sizeof(float) * 2);
  memcpy(scale->data.data(), scale_data.data(), 2 * sizeof(float));
  meta_graph->allTensors.emplace_back(std::move(scale));
  auto offset_tensor = std::make_unique<schema::TensorT>();
  offset_tensor->nodeType = lite::NodeType_ValueNode;
  offset_tensor->dataType = TypeId::kNumberTypeFloat32;
  offset_tensor->format = schema::Format_NHWC;
  offset_tensor->dims = {2};
  std::vector<float> offset_data = {0.0, 0.0};
  offset_tensor->data.resize(sizeof(float) * 2);
  memcpy(offset_tensor->data.data(), offset_data.data(), 2 * sizeof(float));
  meta_graph->allTensors.emplace_back(std::move(offset_tensor));
  auto mean = std::make_unique<schema::TensorT>();
  mean->nodeType = lite::NodeType_ValueNode;
  mean->dataType = TypeId::kNumberTypeFloat32;
  mean->format = schema::Format_NHWC;
  mean->dims = {2};
  std::vector<float> mean_data = {0.43581513, 0.49090168};
  mean->data.resize(sizeof(float) * 2);
  memcpy(mean->data.data(), mean_data.data(), 2 * sizeof(float));
  meta_graph->allTensors.emplace_back(std::move(mean));
  auto var = std::make_unique<schema::TensorT>();
  var->nodeType = lite::NodeType_ValueNode;
  var->dataType = TypeId::kNumberTypeFloat32;
  var->format = schema::Format_NHWC;
  var->dims = {2};
  std::vector<float> var_data = {0.03025229, 0.11069085};
  var->data.resize(sizeof(float) * 2);
  memcpy(var->data.data(), var_data.data(), 2 * sizeof(float));
  meta_graph->allTensors.emplace_back(std::move(var));
  auto output = std::make_unique<schema::TensorT>();
  output->nodeType = lite::NodeType_Parameter;
  output->dataType = TypeId::kNumberTypeFloat32;
  meta_graph->allTensors.emplace_back(std::move(output));
  std::vector<float> input_data = {0.59885779, 0.62662862, 0.63011179, 0.82569427, 0.64772359, 0.42895413,
                                   0.30216458, 0.01351635, 0.32545444, 0.0360674,  0.33967769, 0.18092504,
                                   0.09479915, 0.52258112, 0.46735646, 0.95689111, 0.51619059, 0.82685718};
  std::vector<float> expect = {0.92227477,  0.40612271,  1.09906762,  1.00176775, 1.19869136,  -0.18535967,
                               -0.7560139,  -1.42843423, -0.62427138, -1.3609569, -0.54381545, -0.92751329,
                               -1.92900686, 0.09479138,  0.17841823,  1.39433545, 0.45465564,  1.0052474};
  std::vector<float> outData(expect.size());
  RunPNNAModel(meta_graph, reinterpret_cast<void *>(input_data.data()), reinterpret_cast<void *>(outData.data()));
  ASSERT_EQ(0, CompareOutputData(outData.data(), expect.data(), expect.size()));
}

TEST_F(TestPNNA_Batchnorm, shape_1_3_3_2_nhwc_int8) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";
  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1, 2, 3, 4};
  node->outputIndex = {5};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_FusedBatchNorm;
  auto primitive = new schema::FusedBatchNormT;
  primitive->epsilon = 0.001;
  node->primitive->value.value = primitive;
  node->name = "Batchnorm";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {5};
  int length = 3 * 3 * 2;
  float out_scale;
  int out_zp;
  float in0[] = {0.59885779, 0.62662862, 0.63011179, 0.82569427, 0.64772359, 0.42895413,
                 0.30216458, 0.01351635, 0.32545444, 0.0360674,  0.33967769, 0.18092504,
                 0.09479915, 0.52258112, 0.46735646, 0.95689111, 0.51619059, 0.82685718};
  std::vector<int8_t> input0_data(length);
  float in1[] = {1.0, 1.0};
  std::vector<int8_t> scale_data(2);
  float in2[] = {0.0, 0.0};
  std::vector<int8_t> offset_data(2);
  float in3[] = {0.43581513, 0.49090168};
  std::vector<int8_t> mean_data(2);
  float in4[] = {0.03025229, 0.11069085};
  std::vector<int8_t> var_data(2);
  float expect[] = {0.92227477,  0.40612271,  1.09906762,  1.00176775, 1.19869136,  -0.18535967,
                    -0.7560139,  -1.42843423, -0.62427138, -1.3609569, -0.54381545, -0.92751329,
                    -1.92900686, 0.09479138,  0.17841823,  1.39433545, 0.45465564,  1.0052474};
  auto input_quant0 = CreateQuantParamAndData(in0, length, 0, 1, &input0_data);
  auto input_quant1 = CreateQuantParamAndData(in1, 2, 0, 1, &scale_data);
  auto input_quant2 = CreateQuantParamAndData(in2, 2, 0, 1, &offset_data);
  auto input_quant3 = CreateQuantParamAndData(in3, 2, 0, 1, &mean_data);
  auto input_quant4 = CreateQuantParamAndData(in4, 2, 0, 1, &var_data);
  auto out_quant = CreateQuantParamAndData(expect, length, -2, 2, nullptr);
  out_scale = out_quant->scale;
  out_zp = out_quant->zeroPoint;
  auto input = std::make_unique<schema::TensorT>();
  input->nodeType = lite::NodeType_Parameter;
  input->dataType = TypeId::kNumberTypeInt8;
  input->format = schema::Format_NHWC;
  input->dims = {1, 3, 3, 2};
  input->quantParams.emplace_back(std::move(input_quant0));
  meta_graph->allTensors.emplace_back(std::move(input));
  auto scale = std::make_unique<schema::TensorT>();
  scale->nodeType = lite::NodeType_ValueNode;
  scale->dataType = TypeId::kNumberTypeInt8;
  scale->format = schema::Format_NHWC;
  scale->dims = {2};
  scale->quantParams.emplace_back(std::move(input_quant1));
  scale->data.resize(sizeof(int8_t) * 2);
  memcpy(scale->data.data(), scale_data.data(), 2 * sizeof(int8_t));
  meta_graph->allTensors.emplace_back(std::move(scale));
  auto offset_tensor = std::make_unique<schema::TensorT>();
  offset_tensor->nodeType = lite::NodeType_ValueNode;
  offset_tensor->dataType = TypeId::kNumberTypeInt8;
  offset_tensor->format = schema::Format_NHWC;
  offset_tensor->dims = {2};
  offset_tensor->quantParams.emplace_back(std::move(input_quant2));
  offset_tensor->data.resize(sizeof(int8_t) * 2);
  memcpy(offset_tensor->data.data(), offset_data.data(), 2 * sizeof(int8_t));
  meta_graph->allTensors.emplace_back(std::move(offset_tensor));
  auto mean = std::make_unique<schema::TensorT>();
  mean->nodeType = lite::NodeType_ValueNode;
  mean->dataType = TypeId::kNumberTypeInt8;
  mean->format = schema::Format_NHWC;
  mean->dims = {2};
  mean->quantParams.emplace_back(std::move(input_quant3));
  mean->data.resize(sizeof(int8_t) * 2);
  memcpy(mean->data.data(), mean_data.data(), 2 * sizeof(int8_t));
  meta_graph->allTensors.emplace_back(std::move(mean));
  auto var = std::make_unique<schema::TensorT>();
  var->nodeType = lite::NodeType_ValueNode;
  var->dataType = TypeId::kNumberTypeInt8;
  var->format = schema::Format_NHWC;
  var->dims = {2};
  var->quantParams.emplace_back(std::move(input_quant4));
  var->data.resize(sizeof(int8_t) * 2);
  memcpy(var->data.data(), var_data.data(), 2 * sizeof(int8_t));
  meta_graph->allTensors.emplace_back(std::move(var));
  auto output = std::make_unique<schema::TensorT>();
  output->nodeType = lite::NodeType_Parameter;
  output->dataType = TypeId::kNumberTypeInt8;
  output->quantParams.emplace_back(std::move(out_quant));
  meta_graph->allTensors.emplace_back(std::move(output));
  std::vector<int8_t> outData(length);
  RunPNNAModel(meta_graph, reinterpret_cast<void *>(input0_data.data()), reinterpret_cast<void *>(outData.data()));
  std::vector<float> out(outData.size());
  Dequantize(outData.data(), outData.size(), out_scale, out_zp, out.data());
  ASSERT_EQ(0, CompareOutputData(out.data(), expect, outData.size(), 1e-1));
}
}  // namespace mindspore::lite::dsp::test
