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

#include <functional>
#include <iostream>
#include <memory>
#include <vector>
#include "ut/src/runtime/kernel/dsp/pnna/pnna_test.h"
#include "include/api/context.h"
#include "include/api/data_type.h"
#include "include/api/model.h"
#include "nnacl_c/arithmetic_parameter.h"
#include "schema/inner/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "ut/src/runtime/kernel/opencl/common.h"

namespace mindspore::lite::dsp::test {
class TestPNNA_Arithmetic : public CommonTest {};

void RunBinaryArithmeticFloat32Test(std::unique_ptr<schema::CNodeT> node, const std::vector<float> &in0_data,
                                    const std::vector<float> &in1_data, const std::vector<float> &out_data) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0, 1};
  meta_graph->outputIndex = {2};

  auto x = std::make_unique<schema::TensorT>();
  x->nodeType = lite::NodeType_Parameter;
  x->dataType = TypeId::kNumberTypeFloat32;
  x->dims = {1, 2, 2, 3};
  meta_graph->allTensors.emplace_back(std::move(x));

  auto y = std::make_unique<schema::TensorT>();
  y->dataType = TypeId::kNumberTypeFloat32;
  y->nodeType = lite::NodeType_Parameter;
  y->dims = {1, 2, 2, 3};
  meta_graph->allTensors.emplace_back(std::move(y));

  auto output = std::make_unique<schema::TensorT>();
  output->nodeType = lite::NodeType_Parameter;
  output->dataType = TypeId::kNumberTypeFloat32;
  output->dims = {1, 2, 2, 3};
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
  ASSERT_EQ(inputs.size(), 2);
  float *x_data = static_cast<float *>(inputs[0].MutableData());
  for (size_t i = 0; i < in0_data.size(); ++i) {
    x_data[i] = in0_data[i];
  }
  float *y_data = static_cast<float *>(inputs[1].MutableData());
  for (size_t i = 0; i < in1_data.size(); ++i) {
    y_data[i] = in1_data[i];
  }

  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);
  ASSERT_EQ(0, mindspore::CommonTest::CompareOutputData(reinterpret_cast<float *>(outputs[0].MutableData()),
                                                        out_data.data(), out_data.size()));
}

void RunBinaryArithmeticInt8Test(std::unique_ptr<schema::CNodeT> node, const TensorInfo &in0_params,
                                 const TensorInfo &in1_params, const TensorInfo &out_params) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0, 1};
  meta_graph->outputIndex = {2};

  int length0 = in0_params.len;
  int length1 = in1_params.len;

  std::vector<int8_t> input0_data(length0);
  float in0_scale;
  int in0_zp;
  QuantProcess(in0_params.data, length0, in0_params.min, in0_params.max, &in0_scale, &in0_zp, input0_data.data());

  std::vector<int8_t> input1_data(length1);
  float in1_scale;
  int in1_zp;
  QuantProcess(in1_params.data, length1, in1_params.min, in1_params.max, &in1_scale, &in1_zp, input1_data.data());

  float out_scale;
  int out_zp;
  QuantProcess(out_params.data, out_params.len, out_params.min, out_params.max, &out_scale, &out_zp, nullptr);
  auto input_quant0 = std::make_unique<schema::QuantParamT>();
  input_quant0->scale = in0_scale;
  input_quant0->zeroPoint = in0_zp;
  auto input_quant1 = std::make_unique<schema::QuantParamT>();
  input_quant1->scale = in1_scale;
  input_quant1->zeroPoint = in1_zp;

  auto out_quant = std::make_unique<schema::QuantParamT>();
  out_quant->scale = out_scale;
  out_quant->zeroPoint = out_zp;
  auto x = std::make_unique<schema::TensorT>();
  x->nodeType = lite::NodeType_Parameter;
  x->dataType = TypeId::kNumberTypeInt8;
  x->dims = {1, 2, 2, 3};
  x->quantParams.emplace_back(std::move(input_quant0));
  meta_graph->allTensors.emplace_back(std::move(x));

  auto y = std::make_unique<schema::TensorT>();
  y->dataType = TypeId::kNumberTypeInt8;
  y->nodeType = lite::NodeType_Parameter;
  y->dims = {1, 2, 2, 3};
  y->quantParams.emplace_back(std::move(input_quant1));
  meta_graph->allTensors.emplace_back(std::move(y));

  auto output = std::make_unique<schema::TensorT>();
  output->nodeType = lite::NodeType_Parameter;
  output->dataType = TypeId::kNumberTypeInt8;
  output->quantParams.emplace_back(std::move(out_quant));
  output->dims = {1, 2, 2, 3};
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
  ASSERT_EQ(inputs.size(), 2);
  int8_t *x_data = static_cast<int8_t *>(inputs[0].MutableData());
  for (int i = 0; i < length0; ++i) {
    x_data[i] = input0_data[i];
  }
  int8_t *y_data = static_cast<int8_t *>(inputs[1].MutableData());
  for (int i = 0; i < length1; ++i) {
    y_data[i] = input1_data[i];
  }

  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);
  std::vector<float> output_dequant(outputs[0].ElementNum());
  Dequantize(static_cast<const int8_t *>(outputs[0].MutableData()), out_params.len, out_scale, out_zp,
             output_dequant.data());
  ASSERT_EQ(0, mindspore::CommonTest::CompareOutputData(reinterpret_cast<float *>(output_dequant.data()),
                                                        out_params.data, out_params.len, 0.6));
}

TEST_F(TestPNNA_Arithmetic, AddFusion_float32) {
  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1};
  node->outputIndex = {2};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_AddFusion;
  auto primitive = new schema::AddFusionT;
  primitive->activation_type = schema::ActivationType_NO_ACTIVATION;
  node->primitive->value.value = primitive;
  node->name = "AddFusion";

  std::vector<float> input0_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<float> input1_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<float> expect = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24};
  RunBinaryArithmeticFloat32Test(std::move(node), input0_data, input1_data, expect);
}

TEST_F(TestPNNA_Arithmetic, SubFusion_float32) {
  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1};
  node->outputIndex = {2};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_SubFusion;
  auto primitive = new schema::SubFusionT;
  primitive->activation_type = schema::ActivationType_NO_ACTIVATION;
  node->primitive->value.value = primitive;
  node->name = "SubFusion";

  std::vector<float> input0_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<float> input1_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<float> expect = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  RunBinaryArithmeticFloat32Test(std::move(node), input0_data, input1_data, expect);
}

TEST_F(TestPNNA_Arithmetic, MulFusion_float32) {
  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1};
  node->outputIndex = {2};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_MulFusion;
  auto primitive = new schema::MulFusionT;
  primitive->activation_type = schema::ActivationType_NO_ACTIVATION;
  node->primitive->value.value = primitive;
  node->name = "MulFusion";
  std::vector<float> input0_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<float> input1_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<float> expect = {1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144};
  RunBinaryArithmeticFloat32Test(std::move(node), input0_data, input1_data, expect);
}

TEST_F(TestPNNA_Arithmetic, DivFusion_float32) {
  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1};
  node->outputIndex = {2};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_DivFusion;
  auto primitive = new schema::DivFusionT;
  primitive->activation_type = schema::ActivationType_NO_ACTIVATION;
  node->primitive->value.value = primitive;
  node->name = "DivFusion";
  std::vector<float> input0_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<float> input1_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<float> expect = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  RunBinaryArithmeticFloat32Test(std::move(node), input0_data, input1_data, expect);
}

TEST_F(TestPNNA_Arithmetic, Maximum_float32) {
  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1};
  node->outputIndex = {2};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_Maximum;
  auto primitive = new schema::MaximumT;
  node->primitive->value.value = primitive;
  node->name = "Maximum";
  std::vector<float> input0_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<float> input1_data = {12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<float> expect = {12, 11, 10, 9, 8, 7, 7, 8, 9, 10, 11, 12};
  RunBinaryArithmeticFloat32Test(std::move(node), input0_data, input1_data, expect);
}

TEST_F(TestPNNA_Arithmetic, Minimum_float32) {
  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1};
  node->outputIndex = {2};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_Minimum;
  auto primitive = new schema::MinimumT;
  node->primitive->value.value = primitive;
  node->name = "Minimum";
  std::vector<float> input0_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<float> input1_data = {12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<float> expect = {1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1};
  RunBinaryArithmeticFloat32Test(std::move(node), input0_data, input1_data, expect);
}

// power_infer.c only supports constant exp_tensor, so the pow test case is temporarily disabled.
TEST_F(TestPNNA_Arithmetic, DISABLED_PowFusion_float32) {
  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1};
  node->outputIndex = {2};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_PowFusion;
  auto primitive = new schema::PowFusionT;
  node->primitive->value.value = primitive;
  node->name = "PowFusion";
  std::vector<float> input0_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<float> input1_data = {1, 2, 3, 1, 2, 2, 1, 2, 2, 3, 1, 1};
  std::vector<float> expect = {1, 4, 27, 4, 25, 36, 7, 64, 81, 1000, 11, 12};
  RunBinaryArithmeticFloat32Test(std::move(node), input0_data, input1_data, expect);
}

TEST_F(TestPNNA_Arithmetic, FloorDiv_float32) {
  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1};
  node->outputIndex = {2};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_FloorDiv;
  auto primitive = new schema::FloorDivT;
  node->primitive->value.value = primitive;
  node->name = "FloorDiv";
  std::vector<float> input0_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<float> input1_data = {1, 2, 3, 1, 2, 2, 1, 2, 2, 3, 1, 1};
  std::vector<float> expect = {1, 1, 1, 4, 2, 3, 7, 4, 4, 3, 11, 12};
  RunBinaryArithmeticFloat32Test(std::move(node), input0_data, input1_data, expect);
}

TEST_F(TestPNNA_Arithmetic, AddFusion_int8) {
  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1};
  node->outputIndex = {2};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_AddFusion;
  auto primitive = new schema::AddFusionT;
  primitive->activation_type = schema::ActivationType_NO_ACTIVATION;
  node->primitive->value.value = primitive;
  node->name = "AddFusion";

  float in0[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  TensorInfo in0_params;
  in0_params.data = in0;
  in0_params.len = 12;
  in0_params.min = -12;
  in0_params.max = 12;

  float in1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  TensorInfo in1_params;
  in1_params.data = in1;
  in1_params.len = 12;
  in1_params.min = -12;
  in1_params.max = 12;

  float expect[] = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24};
  TensorInfo out_params;
  out_params.data = expect;
  out_params.len = 12;
  out_params.min = -24;
  out_params.max = 24;
  RunBinaryArithmeticInt8Test(std::move(node), in0_params, in1_params, out_params);
}

TEST_F(TestPNNA_Arithmetic, SubFusion_int8) {
  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1};
  node->outputIndex = {2};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_SubFusion;
  auto primitive = new schema::SubFusionT;
  primitive->activation_type = schema::ActivationType_NO_ACTIVATION;
  node->primitive->value.value = primitive;
  node->name = "SubFusion";

  float in0[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  TensorInfo in0_params;
  in0_params.data = in0;
  in0_params.len = 12;
  in0_params.min = -12;
  in0_params.max = 12;

  float in1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  TensorInfo in1_params;
  in1_params.data = in1;
  in1_params.len = 12;
  in1_params.min = -12;
  in1_params.max = 12;

  float expect[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  TensorInfo out_params;
  out_params.data = expect;
  out_params.len = 12;
  out_params.min = -0;
  out_params.max = 0;
  RunBinaryArithmeticInt8Test(std::move(node), in0_params, in1_params, out_params);
}

TEST_F(TestPNNA_Arithmetic, MulFusion_int8) {
  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1};
  node->outputIndex = {2};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_MulFusion;
  auto primitive = new schema::MulFusionT;
  primitive->activation_type = schema::ActivationType_NO_ACTIVATION;
  node->primitive->value.value = primitive;
  node->name = "MulFusion";

  float in0[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  TensorInfo in0_params;
  in0_params.data = in0;
  in0_params.len = 12;
  in0_params.min = -12;
  in0_params.max = 12;

  float in1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  TensorInfo in1_params;
  in1_params.data = in1;
  in1_params.len = 12;
  in1_params.min = -12;
  in1_params.max = 12;

  float expect[] = {1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144};
  TensorInfo out_params;
  out_params.data = expect;
  out_params.len = 12;
  out_params.min = -144;
  out_params.max = 144;
  RunBinaryArithmeticInt8Test(std::move(node), in0_params, in1_params, out_params);
}

TEST_F(TestPNNA_Arithmetic, DivFusion_int8) {
  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1};
  node->outputIndex = {2};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_DivFusion;
  auto primitive = new schema::DivFusionT;
  primitive->activation_type = schema::ActivationType_NO_ACTIVATION;
  node->primitive->value.value = primitive;
  node->name = "DivFusion";

  float in0[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  TensorInfo in0_params;
  in0_params.data = in0;
  in0_params.len = 12;
  in0_params.min = -12;
  in0_params.max = 12;

  float in1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  TensorInfo in1_params;
  in1_params.data = in1;
  in1_params.len = 12;
  in1_params.min = -12;
  in1_params.max = 12;

  float expect[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  TensorInfo out_params;
  out_params.data = expect;
  out_params.len = 12;
  out_params.min = -1;
  out_params.max = 1;
  RunBinaryArithmeticInt8Test(std::move(node), in0_params, in1_params, out_params);
}

// int8 kernel is not yet supported on DSP backend.
TEST_F(TestPNNA_Arithmetic, DISABLED_Maximum_int8) {
  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1};
  node->outputIndex = {2};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_Maximum;
  auto primitive = new schema::MaximumT;
  node->primitive->value.value = primitive;
  node->name = "Maximum";

  float in0[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  TensorInfo in0_params;
  in0_params.data = in0;
  in0_params.len = 12;
  in0_params.min = -12;
  in0_params.max = 12;

  float in1[] = {12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  TensorInfo in1_params;
  in1_params.data = in1;
  in1_params.len = 12;
  in1_params.min = -12;
  in1_params.max = 12;

  float expect[] = {12, 11, 10, 9, 8, 7, 7, 8, 9, 10, 11, 12};
  TensorInfo out_params;
  out_params.data = expect;
  out_params.len = 12;
  out_params.min = -12;

  out_params.max = 12;
  RunBinaryArithmeticInt8Test(std::move(node), in0_params, in1_params, out_params);
}

// int8 kernel is not yet supported on DSP backend.
TEST_F(TestPNNA_Arithmetic, DISABLED_Minimum_int8) {
  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1};
  node->outputIndex = {2};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_Minimum;
  auto primitive = new schema::MinimumT;
  node->primitive->value.value = primitive;
  node->name = "Minimum";

  float in0[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  TensorInfo in0_params;
  in0_params.data = in0;
  in0_params.len = 12;
  in0_params.min = -12;
  in0_params.max = 12;

  float in1[] = {12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  TensorInfo in1_params;
  in1_params.data = in1;
  in1_params.len = 12;
  in1_params.min = -12;
  in1_params.max = 12;

  float expect[] = {1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1};
  TensorInfo out_params;
  out_params.data = expect;
  out_params.len = 12;
  out_params.min = -6;
  out_params.max = 6;
  RunBinaryArithmeticInt8Test(std::move(node), in0_params, in1_params, out_params);
}

// int8 kernel is not yet supported on DSP backend.
TEST_F(TestPNNA_Arithmetic, DISABLED_PowFusion_int8) {
  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1};
  node->outputIndex = {2};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_PowFusion;
  auto primitive = new schema::PowFusionT;
  node->primitive->value.value = primitive;
  node->name = "PowFusion";

  float in0[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  TensorInfo in0_params;
  in0_params.data = in0;
  in0_params.len = 12;
  in0_params.min = -12;
  in0_params.max = 12;

  float in1[] = {1, 2, 3, 1, 2, 2, 1, 2, 1, 1, 1, 1};
  TensorInfo in1_params;
  in1_params.data = in1;
  in1_params.len = 12;
  in1_params.min = -3;
  in1_params.max = 3;

  float expect[] = {1, 4, 27, 4, 25, 36, 7, 64, 9, 10, 11, 12};
  TensorInfo out_params;
  out_params.data = expect;
  out_params.len = 12;
  out_params.min = -64;
  out_params.max = 64;
  RunBinaryArithmeticInt8Test(std::move(node), in0_params, in1_params, out_params);
}

// int8 kernel is not yet supported on DSP backend.
TEST_F(TestPNNA_Arithmetic, DISABLED_FloorDiv_int8) {
  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1};
  node->outputIndex = {2};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_FloorDiv;
  auto primitive = new schema::FloorDivT;
  node->primitive->value.value = primitive;
  node->name = "FloorDiv";

  float in0[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  TensorInfo in0_params;
  in0_params.data = in0;
  in0_params.len = 12;
  in0_params.min = -12;
  in0_params.max = 12;

  float in1[] = {1, 2, 3, 1, 2, 2, 1, 2, 2, 3, 1, 1};
  TensorInfo in1_params;
  in1_params.data = in1;
  in1_params.len = 12;
  in1_params.min = -12;
  in1_params.max = 12;

  float expect[] = {1, 1, 1, 4, 2, 3, 7, 4, 4, 3, 11, 12};
  TensorInfo out_params;
  out_params.data = expect;
  out_params.len = 12;
  out_params.min = -12;
  out_params.max = 12;
  RunBinaryArithmeticInt8Test(std::move(node), in0_params, in1_params, out_params);
}
}  // namespace mindspore::lite::dsp::test
