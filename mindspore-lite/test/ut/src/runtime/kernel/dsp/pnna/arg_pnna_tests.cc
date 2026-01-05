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
class TestPNNA_Arg : public CommonTest {
 public:
  std::shared_ptr<mindspore::Model> BuildArgMaxMinModel(schema::PrimitiveType prim_type, int axis, int input_data_type,
                                                        const std::vector<int> &shape,
                                                        std::unique_ptr<schema::QuantParamT> input_quant) {
    auto meta_graph = std::make_shared<schema::MetaGraphT>();
    meta_graph->name = "graph";
    auto node = std::make_unique<schema::CNodeT>();
    node->inputIndex = {0};
    node->outputIndex = {1};
    node->primitive = std::make_unique<schema::PrimitiveT>();
    node->primitive->value.type = prim_type;
    if (prim_type == schema::PrimitiveType_ArgMaxFusion) {
      auto primitive = new schema::ArgMaxFusionT;
      primitive->axis = axis;
      node->primitive->value.value = primitive;
      node->name = "ArgMax";
    } else {
      auto primitive = new schema::ArgMinFusionT;
      primitive->axis = axis;
      node->primitive->value.value = primitive;
      node->name = "ArgMin";
    }
    meta_graph->nodes.emplace_back(std::move(node));
    meta_graph->inputIndex = {0};
    meta_graph->outputIndex = {1};

    auto input0 = std::make_unique<schema::TensorT>();
    input0->nodeType = lite::NodeType_Parameter;
    input0->dataType = input_data_type;
    input0->dims = shape;
    input0->offset = -1;
    meta_graph->allTensors.emplace_back(std::move(input0));

    auto output = std::make_unique<schema::TensorT>();
    output->nodeType = lite::NodeType_Parameter;
    output->dataType = TypeId::kNumberTypeInt32;
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
    EXPECT_EQ(kSuccess, ret.StatusCode());
    return model;
  }

  void RunArgMaxMinTest(schema::PrimitiveType prim_type, int axis, int input_data_type, const std::vector<int> &shape,
                        std::vector<float> input_data, std::vector<int> &expected) {
    size_t element_num = 1;
    for (int dim : shape) {
      element_num *= dim;
    }
    std::vector<int8_t> quant_data(element_num);
    auto input_quant = std::make_unique<schema::QuantParamT>();
    if (input_data_type == TypeId::kNumberTypeInt8) {
      auto input_minmax = std::minmax_element(input_data.begin(), input_data.end());
      float input_min = *input_minmax.first;
      float input_max = *input_minmax.second;
      float scale;
      int zp;
      QuantProcess(input_data.data(), input_data.size(), input_min, input_max, &scale, &zp, quant_data.data());
      input_quant->scale = scale;
      input_quant->zeroPoint = zp;
    }
    auto model = BuildArgMaxMinModel(prim_type, axis, input_data_type, shape, std::move(input_quant));
    ASSERT_NE(model, nullptr);
    auto inputs = model->GetInputs();
    ASSERT_EQ(inputs.size(), 1);
    if (input_data_type == TypeId::kNumberTypeInt8) {
      int8_t *in_data = static_cast<int8_t *>(inputs[0].MutableData());
      for (size_t j = 0; j < element_num; ++j) {
        in_data[j] = quant_data[j];
      }
    } else {
      float *in_data = static_cast<float *>(inputs[0].MutableData());
      for (size_t j = 0; j < element_num; ++j) {
        in_data[j] = input_data[j];
      }
    }
    std::vector<mindspore::MSTensor> outputs;
    auto ret = model->Predict(inputs, &outputs);
    ASSERT_EQ(kSuccess, ret.StatusCode());
    ASSERT_EQ(outputs.size(), 1);
    auto *outData = reinterpret_cast<const int *>(outputs.front().Data().get());
    ASSERT_EQ(0, CompareOutputData(outData, expected.data(), expected.size()));
  }
};

TEST_F(TestPNNA_Arg, argmax_shape_2_2_axis_0_fp32) {
  auto prim_type = schema::PrimitiveType_ArgMaxFusion;
  int axis = 0;
  int data_type = TypeId::kNumberTypeFloat32;
  std::vector<int> shape = {2, 2};
  int element_num = shape[0] * shape[1];
  std::vector<float> test_data(element_num);
  test_data = {2, 1, 3, 10};
  std::vector<int> expect = {1, 1};
  RunArgMaxMinTest(prim_type, axis, data_type, shape, test_data, expect);
}

TEST_F(TestPNNA_Arg, argmax_shape_2_2_axis_0_int8) {
  auto prim_type = schema::PrimitiveType_ArgMaxFusion;
  int axis = 0;
  int data_type = TypeId::kNumberTypeInt8;
  std::vector<int> shape = {2, 2};
  int element_num = shape[0] * shape[1];
  std::vector<float> test_data(element_num);
  test_data = {2, 1, 3, 10};
  std::vector<int> expect = {1, 1};
  RunArgMaxMinTest(prim_type, axis, data_type, shape, test_data, expect);
}

TEST_F(TestPNNA_Arg, argmin_shape_2_2_axis_0_fp32) {
  auto prim_type = schema::PrimitiveType_ArgMinFusion;
  int axis = 0;
  int data_type = TypeId::kNumberTypeFloat32;
  std::vector<int> shape = {2, 2};
  int element_num = shape[0] * shape[1];
  std::vector<float> test_data(element_num);
  test_data = {2, 1, 3, 10};
  std::vector<int> expect = {0, 0};
  RunArgMaxMinTest(prim_type, axis, data_type, shape, test_data, expect);
}

TEST_F(TestPNNA_Arg, argmin_shape_2_2_axis_0_int8) {
  auto prim_type = schema::PrimitiveType_ArgMinFusion;
  int axis = 0;
  int data_type = TypeId::kNumberTypeInt8;
  std::vector<int> shape = {2, 2};
  int element_num = shape[0] * shape[1];
  std::vector<float> test_data(element_num);
  test_data = {2, 1, 3, 10};
  std::vector<int> expect = {0, 0};
  RunArgMaxMinTest(prim_type, axis, data_type, shape, test_data, expect);
}
}  // namespace mindspore::lite::dsp::test
