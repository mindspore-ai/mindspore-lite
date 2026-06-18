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
#include "tools/converter/parser/tf/tf_range_parser.h"
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"
#include "mindspore/ops/op_def/auto_generate/gen_lite_ops.h"

namespace mindspore {
namespace lite {
bool TFRangeParser::ParseRangeAttrFromNodeOrInput(const tensorflow::NodeDef &tf_op,
                                                  const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                                  const RangeAttrMapping &mapping,
                                                  const std::unique_ptr<ops::Range> &prim) {
  tensorflow::AttrValue attr_value;
  if (TensorFlowUtils::FindAttrValue(tf_op, mapping.tf_attr_name, &attr_value)) {
    prim->AddAttr(mapping.prim_attr_name, api::MakeValue(attr_value.i()));
  } else {
    auto input_name = TensorFlowUtils::GetFlattenNodeName(tf_op.input(mapping.input_index));
    if (tf_node_map.find(input_name) != tf_node_map.end()) {
      auto node = tf_node_map.at(input_name);
      MS_CHECK_TRUE_RET(node != nullptr, false);
      if (TensorFlowUtils::FindAttrValue(*node, "value", &attr_value)) {
        MS_LOG(INFO) << "Found raggedrange " << mapping.prim_attr_name
                     << " node value attr, means it has default value";
        prim->AddAttr(mapping.prim_attr_name, api::MakeValue(attr_value.i()));
      }
    }
  }
  return true;
}

bool TFRangeParser::ParseRangeDeltaAttr(const tensorflow::NodeDef &tf_op,
                                        const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                        const std::unique_ptr<ops::Range> &prim, tensorflow::AttrValue *attr_value) {
  if (TensorFlowUtils::FindAttrValue(tf_op, "deltas", attr_value)) {
    prim->AddAttr("delta", api::MakeValue(attr_value->i()));
  } else {
    auto input_2_name = TensorFlowUtils::GetFlattenNodeName(tf_op.input(THIRD_INPUT));
    if (tf_node_map.find(input_2_name) == tf_node_map.end()) {
      MS_LOG(ERROR) << "not find delta node.";
      return false;
    }
    auto delta_node = tf_node_map.at(input_2_name);
    MS_CHECK_TRUE_RET(delta_node != nullptr, false);
    if (TensorFlowUtils::FindAttrValue(*delta_node, "value", attr_value)) {
      MS_LOG(INFO) << "Found raggedrange delta node value attr, means it has default value";
    }
    prim->AddAttr("delta", api::MakeValue(attr_value->i()));
  }
  return true;
}

PrimitiveCPtr TFRangeParser::Parse(const tensorflow::NodeDef &tf_op,
                                   const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                   std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::Range>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  tensorflow::AttrValue attr_value;

  if (!ParseRangeAttrFromNodeOrInput(tf_op, tf_node_map, RangeAttrMapping{"starts", FIRST_INPUT, "start"}, prim)) {
    return nullptr;
  }
  if (!ParseRangeAttrFromNodeOrInput(tf_op, tf_node_map, RangeAttrMapping{"limits", SECOND_INPUT, "limit"}, prim)) {
    return nullptr;
  }
  if (!ParseRangeDeltaAttr(tf_op, tf_node_map, prim, &attr_value)) {
    return nullptr;
  }

  *output_size = 1;
  for (int i = 0; i < 3; i++) {
    if (AddOpInput(tf_op, i, inputs) != RET_OK) {
      MS_LOG(ERROR) << "add op input " << i << " failed!";
      return nullptr;
    }
  }

  return prim->GetPrim();
}

TFNodeRegistrar g_tfRangeParser("Range", new TFRangeParser());
}  // namespace lite
}  // namespace mindspore
