/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "tools/converter/parser/tf/tf_logical_parser.h"
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"
#include "mindspore/ops/op_def/auto_generate/gen_lite_ops.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr TFLogicalAndParser::Parse(const tensorflow::NodeDef &tf_op,
                                        const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                        std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::LogicalAnd>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  *output_size = 1;
  for (int i = 0; i < tf_op.input_size(); i++) {
    inputs->emplace_back(tf_op.input(i));
  }

  return prim->GetPrim();
}

PrimitiveCPtr TFLogicalOrParser::Parse(const tensorflow::NodeDef &tf_op,
                                       const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                       std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::LogicalOr>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  *output_size = 1;
  for (int i = 0; i < tf_op.input_size(); i++) {
    inputs->emplace_back(tf_op.input(i));
  }

  return prim->GetPrim();
}

PrimitiveCPtr TFLogicalNotParser::Parse(const tensorflow::NodeDef &tf_op,
                                        const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                        std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::LogicalNot>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  *output_size = 1;
  for (int i = 0; i < tf_op.input_size(); i++) {
    inputs->emplace_back(tf_op.input(i));
  }

  return prim->GetPrim();
}

TFNodeRegistrar g_tfLogicalAndParser("LogicalAnd", new TFLogicalAndParser());
TFNodeRegistrar g_tfLogicalOrParser("LogicalOr", new TFLogicalOrParser());
TFNodeRegistrar g_tfLogicalNotParser("LogicalNot", new TFLogicalNotParser());
}  // namespace lite
}  // namespace mindspore
