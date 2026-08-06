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

namespace mindspore {
namespace {
constexpr int64_t kRoundModeFloor = 0;
constexpr int64_t kRoundModeCeil = 1;

ops::PrimitiveCPtr ParseMaxPoolNode(const std::vector<int64_t> &kernel_shape, const std::vector<int64_t> &strides,
                                    const std::vector<int64_t> &pads, int ceil_mode) {
  auto onnx_graph = std::make_shared<onnx::GraphProto>();
  onnx_graph->set_name("onnx_maxpool_graph");
  onnx::NodeProto *node = onnx_graph->add_node();
  node->set_name("maxpool_node");
  node->set_op_type("MaxPool");
  node->add_input("x");

  auto ks_attr = node->add_attribute();
  ks_attr->set_name("kernel_shape");
  ks_attr->set_type(onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);
  for (auto v : kernel_shape) {
    ks_attr->add_ints(v);
  }

  auto st_attr = node->add_attribute();
  st_attr->set_name("strides");
  st_attr->set_type(onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);
  for (auto v : strides) {
    st_attr->add_ints(v);
  }

  auto pads_attr = node->add_attribute();
  pads_attr->set_name("pads");
  pads_attr->set_type(onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);
  for (auto v : pads) {
    pads_attr->add_ints(v);
  }

  auto ceil_attr = node->add_attribute();
  ceil_attr->set_name("ceil_mode");
  ceil_attr->set_i(ceil_mode);

  auto parser = lite::OnnxNodeParserRegistry::GetInstance().GetNodeParser(node->op_type());
  if (parser == nullptr) {
    MS_LOG(ERROR) << "parser not registered for " << node->op_type();
    return nullptr;
  }
  return parser->Parse(*onnx_graph, *node);
}

std::vector<int64_t> GetIntVecAttr(const ops::PrimitiveCPtr &prim, const std::string &name) {
  auto ptr = prim->GetAttr(name);
  if (ptr == nullptr) {
    return {};
  }
  return GetValue<std::vector<int64_t>>(ptr);
}
}  // namespace

class OnnxPoolParserTest : public mindspore::CommonTest {
 public:
  OnnxPoolParserTest() = default;
};

// Reproduces the originally reported MaxPool1D failure: kernel_shape=[3], strides=[3], pads=[0,0].
// Before the fix, parser left kernel_size/strides stuck at {1,1} — pooling was a no-op.
TEST_F(OnnxPoolParserTest, MaxPool1D_expands_kernel_and_strides_to_2d) {
  auto prim = ParseMaxPoolNode({3}, {3}, {0, 0}, 0);
  ASSERT_NE(prim, nullptr);
  ASSERT_EQ(GetIntVecAttr(prim, ops::kKernelSize), (std::vector<int64_t>{1, 3}));
  ASSERT_EQ(GetIntVecAttr(prim, ops::kStrides), (std::vector<int64_t>{1, 3}));
  ASSERT_EQ(GetIntVecAttr(prim, ops::kPad), (std::vector<int64_t>{0, 0, 0, 0}));
}

// Asymmetric pads [begin, end] must round-trip to {0, 0, begin, end} — guards against the
// earlier bug where end (ints(1)) was discarded and begin was duplicated into both slots.
TEST_F(OnnxPoolParserTest, MaxPool1D_preserves_asymmetric_pads_begin_and_end) {
  auto prim = ParseMaxPoolNode({5}, {2}, {1, 2}, 0);
  ASSERT_NE(prim, nullptr);
  ASSERT_EQ(GetIntVecAttr(prim, ops::kKernelSize), (std::vector<int64_t>{1, 5}));
  ASSERT_EQ(GetIntVecAttr(prim, ops::kStrides), (std::vector<int64_t>{1, 2}));
  ASSERT_EQ(GetIntVecAttr(prim, ops::kPad), (std::vector<int64_t>{0, 0, 1, 2}));
}

// ceil_mode=1 should map to RoundMode::CEIL; ceil_mode=0 to FLOOR.
TEST_F(OnnxPoolParserTest, MaxPool1D_maps_ceil_mode_to_round_mode) {
  auto prim_floor = ParseMaxPoolNode({3}, {1}, {0, 0}, 0);
  ASSERT_NE(prim_floor, nullptr);
  ASSERT_EQ(GetValue<int64_t>(prim_floor->GetAttr(ops::kRoundMode)), kRoundModeFloor);

  auto prim_ceil = ParseMaxPoolNode({3}, {1}, {0, 0}, 1);
  ASSERT_NE(prim_ceil, nullptr);
  ASSERT_EQ(GetValue<int64_t>(prim_ceil->GetAttr(ops::kRoundMode)), kRoundModeCeil);
}

// 2D path is unchanged by this fix — keep as regression guard.
TEST_F(OnnxPoolParserTest, MaxPool2D_keeps_two_element_kernel_size) {
  auto prim = ParseMaxPoolNode({2, 2}, {2, 2}, {0, 0, 0, 0}, 0);
  ASSERT_NE(prim, nullptr);
  ASSERT_EQ(GetIntVecAttr(prim, ops::kKernelSize), (std::vector<int64_t>{2, 2}));
  ASSERT_EQ(GetIntVecAttr(prim, ops::kStrides), (std::vector<int64_t>{2, 2}));
}
}  // namespace mindspore
