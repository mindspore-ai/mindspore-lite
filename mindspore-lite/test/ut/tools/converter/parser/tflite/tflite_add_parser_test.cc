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
// Expand the core logging headers (complete MS_LOG family) first: later converter
// headers pull in lite's src/common/log.h whose narrower MS_LOG macro would
// otherwise break the inline functions of already-open core headers.
#include <memory>
#include "src/common/log_adapter.h"
#include "ir/func_graph.h"
#include "common/common_test.h"
#include "tools/converter/parser/tflite/tflite_arithmetic_parser.h"
#include "src/common/ops/primitive/add_fusion.h"

namespace mindspore {
namespace {
// Some exporters emit ADD without an AddOptions table (BuiltinOptions_NONE).
// The parser must tolerate the missing table instead of failing the model.
lite::PrimitiveCPtr ParseAdd(bool with_options, tflite::ActivationFunctionType activation) {
  auto tflite_op = std::make_unique<tflite::OperatorT>();
  auto subgraph = std::make_unique<tflite::SubGraphT>();
  auto model = std::make_unique<tflite::ModelT>();
  if (with_options) {
    auto options = new (std::nothrow) tflite::AddOptionsT();
    options->fused_activation_function = activation;
    tflite_op->builtin_options.Set(options);
    tflite_op->builtin_options.type = tflite::BuiltinOptions_AddOptions;
  }
  lite::TfliteAddParser parser;
  return parser.Parse(tflite_op, subgraph, model);
}
}  // namespace

class TfliteAddParserTest : public mindspore::CommonTest {
 public:
  TfliteAddParserTest() = default;
};

// Guards the fix for "get AddFusion attr failed": a tflite ADD without the
// AddOptions union entry must still parse and carry the activation attribute
// (defaulting to NO_ACTIVATION).
TEST_F(TfliteAddParserTest, Add_without_options_parses_with_none_activation) {
  auto prim = ParseAdd(false, tflite::ActivationFunctionType_NONE);
  ASSERT_NE(prim, nullptr);
  auto attr = prim->GetAttr(ops::kActivationType);
  ASSERT_NE(attr, nullptr);
  ASSERT_EQ(GetValue<int64_t>(attr), static_cast<int64_t>(mindspore::NO_ACTIVATION));
}

// Regression: a regular AddOptions table keeps parsing into an AddFusion
// primitive that carries the activation attribute.
TEST_F(TfliteAddParserTest, Add_with_options_maps_activation) {
  auto prim = ParseAdd(true, tflite::ActivationFunctionType_RELU);
  ASSERT_NE(prim, nullptr);
  auto attr = prim->GetAttr(ops::kActivationType);
  ASSERT_NE(attr, nullptr);
}
}  // namespace mindspore
