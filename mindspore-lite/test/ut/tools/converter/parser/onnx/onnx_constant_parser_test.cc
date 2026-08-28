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

#define USE_DEPRECATED_API
#include <memory>
#include <vector>
#include "common/common_test.h"
#include "include/registry/converter_context.h"
#include "include/registry/node_parser_registry.h"
#include "ops_utils/op_utils.h"
#include "tools/converter/parser/onnx/onnx_node_parser_registry.h"
#include "ir/tensor.h"

namespace mindspore {
namespace {

onnx::AttributeProto *AddConstantAttr(onnx::NodeProto *node, const std::string &name,
                                      onnx::AttributeProto_AttributeType type) {
  auto attr = node->add_attribute();
  attr->set_name(name);
  attr->set_type(type);
  return attr;
}

ops::PrimitiveCPtr ParseConstantNode(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto parser = lite::OnnxNodeParserRegistry::GetInstance().GetNodeParser(onnx_node.op_type());
  if (parser == nullptr) {
    MS_LOG(ERROR) << "parser not registered for " << onnx_node.op_type();
    return nullptr;
  }
  return parser->Parse(onnx_graph, onnx_node);
}

tensor::TensorPtr GetConstData(const ops::PrimitiveCPtr &prim) {
  auto attr = prim->GetAttr("const_data");
  if (attr == nullptr) {
    return nullptr;
  }
  return attr->cast<tensor::TensorPtr>();
}
}  // namespace

class OnnxConstantParserTest : public mindspore::CommonTest {
 public:
  OnnxConstantParserTest() = default;
};

// The tensor `value` attribute (ONNX spec) must be stored as an int32 scalar const_data tensor.
TEST_F(OnnxConstantParserTest, Constant_value_attr_parses_scalar_tensor) {
  onnx::GraphProto onnx_graph;
  onnx_graph.set_name("onnx_constant_graph");
  onnx::NodeProto *node = onnx_graph.add_node();
  node->set_name("const_value");
  node->set_op_type("Constant");

  onnx::TensorProto value_proto;
  value_proto.set_data_type(onnx::TensorProto_DataType_INT32);
  int32_t value = 42;
  value_proto.set_raw_data(&value, sizeof(value));
  auto attr = AddConstantAttr(node, "value", onnx::AttributeProto_AttributeType_TENSOR);
  *attr->mutable_t() = value_proto;

  auto prim = ParseConstantNode(onnx_graph, *node);
  ASSERT_NE(prim, nullptr);
  auto tensor_info = GetConstData(prim);
  ASSERT_NE(tensor_info, nullptr);
  ASSERT_EQ(tensor_info->data_type_c(), kNumberTypeInt32);
  ASSERT_TRUE(tensor_info->shape_c().empty());
  ASSERT_EQ(tensor_info->DataSize(), 1u);
  ASSERT_EQ(*static_cast<const int32_t *>(tensor_info->data_c()), 42);
}

// The scalar `value_float` attribute (fp32) must be stored as a float scalar const_data tensor.
TEST_F(OnnxConstantParserTest, Constant_value_float_attr_parses_scalar) {
  onnx::GraphProto onnx_graph;
  onnx::NodeProto *node = onnx_graph.add_node();
  node->set_name("const_value_float");
  node->set_op_type("Constant");
  auto attr = AddConstantAttr(node, "value_float", onnx::AttributeProto_AttributeType_FLOAT);
  attr->set_f(3.75f);

  auto prim = ParseConstantNode(onnx_graph, *node);
  ASSERT_NE(prim, nullptr);
  auto tensor_info = GetConstData(prim);
  ASSERT_NE(tensor_info, nullptr);
  ASSERT_EQ(tensor_info->data_type_c(), kNumberTypeFloat32);
  ASSERT_TRUE(tensor_info->shape_c().empty());
  ASSERT_EQ(tensor_info->DataSize(), 1u);
  ASSERT_FLOAT_EQ(*static_cast<const float *>(tensor_info->data_c()), 3.75f);
}

// The 1D `value_floats` attribute must be stored as a float 1D const_data tensor.
TEST_F(OnnxConstantParserTest, Constant_value_floats_attr_parses_1d) {
  onnx::GraphProto onnx_graph;
  onnx::NodeProto *node = onnx_graph.add_node();
  node->set_name("const_value_floats");
  node->set_op_type("Constant");
  auto attr = AddConstantAttr(node, "value_floats", onnx::AttributeProto_AttributeType_FLOATS);
  attr->add_floats(1.0f);
  attr->add_floats(-2.5f);
  attr->add_floats(3.25f);

  auto prim = ParseConstantNode(onnx_graph, *node);
  ASSERT_NE(prim, nullptr);
  auto tensor_info = GetConstData(prim);
  ASSERT_NE(tensor_info, nullptr);
  ASSERT_EQ(tensor_info->data_type_c(), kNumberTypeFloat32);
  ASSERT_EQ(tensor_info->shape_c(), (ShapeVector{3}));
  ASSERT_EQ(tensor_info->DataSize(), 3u);
  auto data = static_cast<const float *>(tensor_info->data_c());
  ASSERT_FLOAT_EQ(data[0], 1.0f);
  ASSERT_FLOAT_EQ(data[1], -2.5f);
  ASSERT_FLOAT_EQ(data[2], 3.25f);
}

// The scalar `value_int` attribute is an int64 in ONNX and must be stored as an int64 scalar const_data tensor.
TEST_F(OnnxConstantParserTest, Constant_value_int_attr_parses_scalar) {
  onnx::GraphProto onnx_graph;
  onnx::NodeProto *node = onnx_graph.add_node();
  node->set_name("const_value_int");
  node->set_op_type("Constant");
  auto attr = AddConstantAttr(node, "value_int", onnx::AttributeProto_AttributeType_INT);
  attr->set_i(7);

  auto prim = ParseConstantNode(onnx_graph, *node);
  ASSERT_NE(prim, nullptr);
  auto tensor_info = GetConstData(prim);
  ASSERT_NE(tensor_info, nullptr);
  ASSERT_EQ(tensor_info->data_type_c(), kNumberTypeInt64);
  ASSERT_TRUE(tensor_info->shape_c().empty());
  ASSERT_EQ(tensor_info->DataSize(), 1u);
  ASSERT_EQ(*static_cast<const int64_t *>(tensor_info->data_c()), 7);
}

// The 1D `value_ints` attribute must be stored as an int64 1D const_data tensor.
TEST_F(OnnxConstantParserTest, Constant_value_ints_attr_parses_1d) {
  onnx::GraphProto onnx_graph;
  onnx::NodeProto *node = onnx_graph.add_node();
  node->set_name("const_value_ints");
  node->set_op_type("Constant");
  auto attr = AddConstantAttr(node, "value_ints", onnx::AttributeProto_AttributeType_INTS);
  attr->add_ints(10);
  attr->add_ints(20);
  attr->add_ints(30);

  auto prim = ParseConstantNode(onnx_graph, *node);
  ASSERT_NE(prim, nullptr);
  auto tensor_info = GetConstData(prim);
  ASSERT_NE(tensor_info, nullptr);
  ASSERT_EQ(tensor_info->data_type_c(), kNumberTypeInt64);
  ASSERT_EQ(tensor_info->shape_c(), (ShapeVector{3}));
  ASSERT_EQ(tensor_info->DataSize(), 3u);
  auto data = static_cast<const int64_t *>(tensor_info->data_c());
  ASSERT_EQ(data[0], 10);
  ASSERT_EQ(data[1], 20);
  ASSERT_EQ(data[2], 30);
}

// Unsupported attributes (e.g. value_string) must make the parser bail out with nullptr.
TEST_F(OnnxConstantParserTest, Constant_unsupported_attr_returns_null) {
  onnx::GraphProto onnx_graph;
  onnx::NodeProto *node = onnx_graph.add_node();
  node->set_name("const_value_string");
  node->set_op_type("Constant");
  auto attr = AddConstantAttr(node, "value_string", onnx::AttributeProto_AttributeType_STRING);
  attr->set_s("not supported");

  auto prim = ParseConstantNode(onnx_graph, *node);
  ASSERT_EQ(prim, nullptr);
}
}  // namespace mindspore
