/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "ut/tools/converter/parser/tflite/tflite_parsers_test_utils.h"
#include "common/common_test.h"

namespace mindspore {
class TestTfliteParserSoftmax : public TestTfliteParser {
 public:
  TestTfliteParserSoftmax() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./softmax.tflite"); }
};

TEST_F(TestTfliteParserSoftmax, OpType) {
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_Softmax) << "wrong Op Type";
}

TEST_F(TestTfliteParserSoftmax, AttrValue) {
  ASSERT_NE(meta_graph->nodes.front()->primitive->value.AsSoftmax(), nullptr);
  auto val = meta_graph->nodes.front()->primitive->value.AsSoftmax();
  ASSERT_EQ(val->axis[0], -1);
}

}  // namespace mindspore
