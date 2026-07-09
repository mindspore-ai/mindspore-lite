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

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"
#include "schema/model_generated.h"
#include "src/common/log_adapter.h"
#include "src/litert/lite_model.h"
#include "include/errorcode.h"

namespace mindspore {
namespace lite {

class ModelObfuscationDeprecatedTest : public testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

// Build a minimal MetaGraph buffer with obfuscate=true and non-empty obfMetaData
static std::vector<uint8_t> BuildModelWithObfuscate() {
  flatbuffers::FlatBufferBuilder fbb;

  auto name = fbb.CreateString("test_model");
  auto version = fbb.CreateString("2.10.0");

  std::vector<int32_t> shape = {1};
  auto tensor_shape = fbb.CreateVector(shape);
  schema::TensorBuilder tensor_builder(fbb);
  tensor_builder.add_name(fbb.CreateString("input"));
  tensor_builder.add_dataType(43);  // kNumberTypeFloat32
  tensor_builder.add_dims(tensor_shape);
  tensor_builder.add_format(schema::Format_NCHW);
  auto input_tensor = tensor_builder.Finish();

  auto tensors = fbb.CreateVector<flatbuffers::Offset<schema::Tensor>>({input_tensor});
  auto inputs = fbb.CreateVector<uint32_t>({0});
  auto outputs = fbb.CreateVector<uint32_t>({0});
  auto nodes = fbb.CreateVector<flatbuffers::Offset<schema::CNode>>({});

  std::vector<uint8_t> obf_data = {0x01, 0x02, 0x03, 0x04};

  schema::MetaGraphBuilder graph_builder(fbb);
  graph_builder.add_name(name);
  graph_builder.add_version(version);
  graph_builder.add_inputIndex(inputs);
  graph_builder.add_outputIndex(outputs);
  graph_builder.add_allTensors(tensors);
  graph_builder.add_nodes(nodes);
  graph_builder.add_obfuscate(true);
  graph_builder.add_obfMetaData(fbb.CreateVector(obf_data));
  fbb.Finish(graph_builder.Finish());

  return std::vector<uint8_t>(fbb.GetBufferPointer(), fbb.GetBufferPointer() + fbb.GetSize());
}

// Build a minimal MetaGraph buffer with obfuscate=false and no obfMetaData
static std::vector<uint8_t> BuildNormalModel() {
  flatbuffers::FlatBufferBuilder fbb;

  auto name = fbb.CreateString("normal_model");
  auto version = fbb.CreateString("2.10.0");

  std::vector<int32_t> shape = {1};
  auto tensor_shape = fbb.CreateVector(shape);
  schema::TensorBuilder tensor_builder(fbb);
  tensor_builder.add_name(fbb.CreateString("input"));
  tensor_builder.add_dataType(43);  // kNumberTypeFloat32
  tensor_builder.add_dims(tensor_shape);
  tensor_builder.add_format(schema::Format_NCHW);
  auto input_tensor = tensor_builder.Finish();

  auto tensors = fbb.CreateVector<flatbuffers::Offset<schema::Tensor>>({input_tensor});
  auto inputs = fbb.CreateVector<uint32_t>({0});
  auto outputs = fbb.CreateVector<uint32_t>({0});
  auto nodes = fbb.CreateVector<flatbuffers::Offset<schema::CNode>>({});

  schema::MetaGraphBuilder graph_builder(fbb);
  graph_builder.add_name(name);
  graph_builder.add_version(version);
  graph_builder.add_inputIndex(inputs);
  graph_builder.add_outputIndex(outputs);
  graph_builder.add_allTensors(tensors);
  graph_builder.add_nodes(nodes);
  graph_builder.add_obfuscate(false);
  fbb.Finish(graph_builder.Finish());

  return std::vector<uint8_t>(fbb.GetBufferPointer(), fbb.GetBufferPointer() + fbb.GetSize());
}

// Test 1: Model with obfuscate=true should fail
TEST_F(ModelObfuscationDeprecatedTest, ObfuscatedModelReturnsError) {
  auto model_buf = BuildModelWithObfuscate();
  auto *model = new (std::nothrow) LiteModel();
  ASSERT_NE(model, nullptr);

  auto ret = model->ConstructModel(reinterpret_cast<const char *>(model_buf.data()), model_buf.size(), false);
  EXPECT_NE(ret, RET_OK);

  delete model;
}

// Test 2: Normal model should not be affected by the obfuscation check
TEST_F(ModelObfuscationDeprecatedTest, NormalModelUnaffected) {
  auto model_buf = BuildNormalModel();
  auto *model = new (std::nothrow) LiteModel();
  ASSERT_NE(model, nullptr);

  auto ret = model->ConstructModel(reinterpret_cast<const char *>(model_buf.data()), model_buf.size(), false);
  // The minimal normal model will fail at ModelVerify (same input/output indices), but
  // it should NOT fail at the obfuscation check stage.
  EXPECT_NE(ret, RET_OK);

  delete model;
}

}  // namespace lite
}  // namespace mindspore
