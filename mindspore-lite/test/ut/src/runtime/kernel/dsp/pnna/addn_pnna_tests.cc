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
#include <utility>
#include "ut/src/runtime/kernel/dsp/pnna/pnna_test.h"
#include "include/api/context.h"
#include "include/api/data_type.h"
#include "include/api/model.h"
#include "src/litert/kernel/dsp/dsp_subgraph.h"
#include "src/litert/kernel_registry.h"
#include "nnacl_c/int8/quantize.h"

namespace mindspore::lite::dsp::test {

class TestPNNA_AddN : public CommonTest {
 public:
  std::shared_ptr<mindspore::Model> BuildAddNModel(int num_inputs, int data_type, const std::vector<int> &shape,
                                                   std::unique_ptr<schema::QuantParamT> input_quant,
                                                   std::unique_ptr<schema::QuantParamT> output_quant) {
    auto meta_graph = std::make_shared<schema::MetaGraphT>();
    meta_graph->name = "graph";
    auto node = std::make_unique<schema::CNodeT>();
    node->primitive = std::make_unique<schema::PrimitiveT>();
    node->primitive->value.type = schema::PrimitiveType_AddN;
    node->primitive->value.value = new schema::AddNT();
    node->name = "AddN";

    std::vector<uint32_t> input_indices(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      input_indices[i] = i;
      auto input = std::make_unique<schema::TensorT>();
      input->nodeType = lite::NodeType_Parameter;
      input->dataType = data_type;
      input->dims = shape;
      input->offset = -1;
      if (data_type == TypeId::kNumberTypeInt8) {
        auto quant_copy = std::make_unique<schema::QuantParamT>();
        quant_copy->scale = input_quant->scale;
        quant_copy->zeroPoint = input_quant->zeroPoint;
        input->quantParams.emplace_back(std::move(quant_copy));
      }
      meta_graph->allTensors.emplace_back(std::move(input));
    }
    node->inputIndex = input_indices;
    node->outputIndex = {static_cast<uint32_t>(num_inputs)};
    meta_graph->nodes.emplace_back(std::move(node));

    auto output = std::make_unique<schema::TensorT>();
    output->nodeType = lite::NodeType_Parameter;
    output->dataType = data_type;
    output->dims = shape;
    output->offset = -1;
    if (data_type == TypeId::kNumberTypeInt8) {
      output->quantParams.emplace_back(std::move(output_quant));
    }
    meta_graph->allTensors.emplace_back(std::move(output));
    meta_graph->inputIndex = input_indices;
    meta_graph->outputIndex = {static_cast<uint32_t>(num_inputs)};

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
    EXPECT_EQ(kSuccess, ret.StatusCode());
    return model;
  }

  void RunAddNTest(int num_inputs, int data_type, const std::vector<int> &shape, std::vector<float> input_data,
                   std::vector<float> &expected) {
    size_t element_num = 1;
    for (int dim : shape) element_num *= dim;
    std::vector<int8_t> quant_data(element_num);
    auto input_quant = std::make_unique<schema::QuantParamT>();
    auto output_quant = std::make_unique<schema::QuantParamT>();
    float out_scale = 0.0f;
    int out_zp = 0;
    if (data_type == TypeId::kNumberTypeInt8) {
      auto input_minmax = std::minmax_element(input_data.begin(), input_data.end());
      float input_min = *input_minmax.first;
      float input_max = *input_minmax.second;
      float scale;
      int zp;
      QuantProcess(input_data.data(), input_data.size(), input_min, input_max, &scale, &zp, quant_data.data());
      input_quant->scale = scale;
      input_quant->zeroPoint = zp;
      auto expected_minmax = std::minmax_element(expected.begin(), expected.end());
      float expected_min = *expected_minmax.first;
      float expected_max = *expected_minmax.second;
      QuantProcess(expected.data(), expected.size(), expected_min, expected_max, &scale, &zp, nullptr);
      output_quant->scale = scale;
      output_quant->zeroPoint = zp;
      out_scale = scale;
      out_zp = zp;
    }
    auto model = BuildAddNModel(num_inputs, data_type, shape, std::move(input_quant), std::move(output_quant));
    ASSERT_NE(model, nullptr);
    auto inputs = model->GetInputs();
    ASSERT_EQ(inputs.size(), num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      if (data_type == TypeId::kNumberTypeInt8) {
        int8_t *in_data = static_cast<int8_t *>(inputs[i].MutableData());
        for (size_t j = 0; j < element_num; ++j) {
          in_data[j] = quant_data[j];
        }
      } else {
        float *in_data = static_cast<float *>(inputs[i].MutableData());
        for (size_t j = 0; j < element_num; ++j) {
          in_data[j] = input_data[j];
        }
      }
    }
    std::vector<mindspore::MSTensor> outputs;
    auto ret = model->Predict(inputs, &outputs);
    ASSERT_EQ(kSuccess, ret.StatusCode());
    ASSERT_EQ(outputs.size(), 1);
    if (data_type == TypeId::kNumberTypeInt8) {
      std::vector<int8_t> outData(outputs[0].ElementNum());
      memcpy(outData.data(), outputs[0].MutableData(), outData.size());
      std::vector<float> output_dequant(outputs.front().ElementNum());
      Dequantize(outData.data(), outputs.front().ElementNum(), out_scale, out_zp, output_dequant.data());
      ASSERT_EQ(0, CompareOutputData(output_dequant.data(), expected.data(), expected.size(), 0.1));
    } else {
      auto *out_data = reinterpret_cast<const float *>(outputs[0].Data().get());
      ASSERT_EQ(0, CompareOutputData(out_data, expected.data(), expected.size()));
    }
  }
};

TEST_F(TestPNNA_AddN, inputN2_shape_4_fp32) {
  int data_type = TypeId::kNumberTypeFloat32;
  std::vector<int> shape = {4};
  size_t element_num = 1;
  for (int dim : shape) {
    element_num *= dim;
  }
  std::vector<float> test_data(element_num);
  for (size_t i = 0; i < element_num; ++i) {
    test_data[i] = static_cast<float>(i + 1);
  }
  std::vector<float> expected(element_num);
  for (size_t i = 0; i < element_num; ++i) {
    expected[i] = test_data[i] * 2;
  }
  RunAddNTest(2, data_type, shape, test_data, expected);
}

TEST_F(TestPNNA_AddN, DISABLED_inputN2_shape_4_int8) {
  int data_type = TypeId::kNumberTypeInt8;
  std::vector<int> shape = {4};
  size_t element_num = 1;
  for (int dim : shape) {
    element_num *= dim;
  }
  std::vector<float> test_data(element_num);
  for (size_t i = 0; i < element_num; ++i) {
    test_data[i] = static_cast<float>(i);
  }
  std::vector<float> expected(element_num);
  for (size_t i = 0; i < element_num; ++i) {
    expected[i] = test_data[i] * 2;
  }
  RunAddNTest(2, data_type, shape, test_data, expected);
}

TEST_F(TestPNNA_AddN, inputN_shape_4_fp32) {
  int data_type = TypeId::kNumberTypeFloat32;
  int i = 2;
  std::vector<int> shape = {4};
  size_t element_num = 1;
  for (int dim : shape) {
    element_num *= dim;
  }
  std::vector<float> test_data(element_num);
  std::vector<float> expected(element_num);
  for (size_t j = 0; j < test_data.size(); ++j) {
    test_data[j] = static_cast<float>(j + 1);
  }
  while (i < 10) {
    for (size_t j = 0; j < test_data.size(); ++j) {
      expected[j] = test_data[j] * i;
    }
    RunAddNTest(i, data_type, shape, test_data, expected);
    ++i;
  }
}

TEST_F(TestPNNA_AddN, DISABLED_inputN_shape_4_int8) {
  int data_type = TypeId::kNumberTypeInt8;
  int i = 2;
  std::vector<int> shape = {4};
  size_t element_num = 1;
  for (int dim : shape) {
    element_num *= dim;
  }
  std::vector<float> test_data(element_num);
  std::vector<float> expected(element_num);
  for (size_t j = 0; j < test_data.size(); ++j) {
    test_data[j] = static_cast<float>(j);
  }
  while (i < 10) {
    for (size_t j = 0; j < test_data.size(); ++j) {
      expected[j] = test_data[j] * i;
    }
    RunAddNTest(i, data_type, shape, test_data, expected);
    ++i;
  }
}
}  // namespace mindspore::lite::dsp::test
