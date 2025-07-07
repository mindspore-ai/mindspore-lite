/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "tools/converter/parser/onnx/onnx_custom_parser.h"
#include <memory>
#include <vector>
#include <string>
#include "infer/custom.h"
#include "mindspore/ccsrc/include/common/utils/utils.h"

namespace mindspore {
namespace lite {
namespace {
bool CheckAttrs(const std::unique_ptr<ops::Custom> &prim, const std::vector<std::string> &input_names,
                const std::vector<std::string> &output_names, const std::vector<std::string> &optional_input_names,
                const std::string &type) {
  if (type.empty()) {
    MS_LOG(ERROR) << "For custom, attr of type must be not empty, please set it!";
    return false;
  }
  if (input_names.empty()) {
    MS_LOG(ERROR) << "For custom, attr of input_names must be not empty, please set it!";
    return false;
  }
  if (output_names.empty()) {
    MS_LOG(ERROR) << "For custom, attr of output_names must be not empty, please set it!";
    return false;
  }
  if (optional_input_names.empty()) {
    MS_LOG(WARNING) << "optional_input_names is empty, please check whether the result meets the expectation!";
  }
  auto optional_name_value = prim->GetAttr(kAttrOptionalInputNames);
  if (optional_name_value != nullptr) {
    auto optional_names = GetValue<const std::vector<std::string>>(optional_name_value);
    for (size_t i = 0; i < optional_names.size(); i++) {
      if (find(input_names.begin(), input_names.end(), optional_names[i]) == input_names.end()) {
        MS_LOG(ERROR) << "optional_input_name: " << optional_names[i] << " is not in input_names, please check!";
        return false;
      }
    }
  }
  return true;
}

void SetAttrString(const std::unique_ptr<ops::Custom> &prim, const std::string &attribute_name,
                   const std::string &attr_value) {
  if (attr_value == kValueTrueToupper || attr_value == kValueTrue) {
    prim->AddAttr(attribute_name, api::MakeValue(true));
  } else if (attr_value == kValueFalseToupper || attr_value == kValueFalse) {
    prim->AddAttr(attribute_name, api::MakeValue(false));
  } else {
    prim->AddAttr(attribute_name, api::MakeValue(attr_value));
  }
  MS_LOG(INFO) << "attribute_name: " << attribute_name << ", value is: " << attr_value;
}

void SetAttrInts(const std::unique_ptr<ops::Custom> &prim, const ::onnx::AttributeProto onnx_node_attr) {
  const auto &attribute_name = onnx_node_attr.name();
  std::vector<int64_t> ints = {};
  for (auto i = 0; i < onnx_node_attr.ints().size(); i++) {
    ints.push_back(onnx_node_attr.ints(i));
  }
  prim->AddAttr(attribute_name, api::MakeValue(ints));
  MS_LOG(INFO) << "attribute_name: " << attribute_name << ", value is: " << ints;
}

void SetAttrFloats(const std::unique_ptr<ops::Custom> &prim, const ::onnx::AttributeProto onnx_node_attr) {
  const auto &attribute_name = onnx_node_attr.name();
  std::vector<float> floats = {};
  for (auto i = 0; i < onnx_node_attr.floats().size(); i++) {
    floats.push_back(onnx_node_attr.floats(i));
  }
  prim->AddAttr(attribute_name, api::MakeValue(floats));
  MS_LOG(INFO) << "attribute_name: " << attribute_name << ", value is: " << floats;
}

void SetAttrStrings(const std::unique_ptr<ops::Custom> &prim, const ::onnx::AttributeProto onnx_node_attr) {
  const auto &attribute_name = onnx_node_attr.name();
  std::vector<std::string> strings = {};
  for (auto i = 0; i < onnx_node_attr.strings().size(); i++) {
    strings.push_back(onnx_node_attr.strings(i));
  }
  prim->AddAttr(attribute_name, api::MakeValue(strings));
  MS_LOG(INFO) << "attribute_name: " << attribute_name << ", value is: " << strings;
}

std::vector<std::string> CheckAndSetStrings(const std::unique_ptr<ops::Custom> &prim,
                                            const ::onnx::AttributeProto onnx_node_attr) {
  const auto &attribute_name = onnx_node_attr.name();
  std::vector<std::string> strings = {};
  if (onnx_node_attr.type() != onnx::AttributeProto_AttributeType_STRINGS) {
    MS_LOG(ERROR) << "For attribute name: " << attribute_name << ", AttributeType should be String[], please check!";
    return {};
  }
  for (auto i = 0; i < onnx_node_attr.strings().size(); i++) {
    strings.push_back(onnx_node_attr.strings(i));
  }
  prim->AddAttr(attribute_name, api::MakeValue(strings));
  MS_LOG(INFO) << "attribute_name: " << attribute_name << ", value is: " << strings;
  return strings;
}
}  // namespace
PrimitiveCPtr OnnxCustomParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  MS_LOG(INFO) << "Start to parse ONNX Custom node: " << onnx_node.name();
  auto prim = std::make_unique<ops::Custom>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  // Initialize required attributes.
  std::vector<std::string> input_names = {};
  std::vector<std::string> output_names = {};
  std::vector<std::string> optional_input_names = {};
  std::string type = "";

  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    MS_LOG(INFO) << "attribute_name: " << attribute_name << ", type is: " << onnx_node_attr.type();
    if (attribute_name == kAttrType) {
      if (onnx_node_attr.type() != onnx::AttributeProto_AttributeType_STRING) {
        MS_LOG(ERROR) << "For attribute name: " << kAttrType << ", AttributeType should be String, please check!";
        return nullptr;
      }
      type = onnx_node_attr.s();
      prim->set_type(type);
      prim->AddAttr(kAttrRegOpName, api::MakeValue(type));
      MS_LOG(INFO) << "attribute_name: " << attribute_name << ", value is: " << type;
    } else if (attribute_name == kAttrInputNames) {
      input_names = CheckAndSetStrings(prim, onnx_node_attr);
    } else if (attribute_name == kAttrOptionalInputNames) {
      optional_input_names = CheckAndSetStrings(prim, onnx_node_attr);
    } else if (attribute_name == kAttrOutputNames) {
      output_names = CheckAndSetStrings(prim, onnx_node_attr);
    } else if (attribute_name == kAttrOutputNum) {
      if (onnx_node_attr.type() != onnx::AttributeProto_AttributeType_INT) {
        MS_LOG(ERROR) << "For attribute name: " << kAttrOutputNum << ", AttributeType should be int, please check!";
        return nullptr;
      }
      prim->AddAttr(kAttrOutputNum, api::MakeValue(onnx_node_attr.i()));
      MS_LOG(INFO) << "attribute_name: " << attribute_name << ", value is: " << onnx_node_attr.i();
    } else if (attribute_name == "dtype") {
      auto dst_type = GetDataTypeFromOnnx(static_cast<onnx::TensorProto_DataType>(onnx_node_attr.i()));
      auto prim_c = prim->GetPrim();
      MS_CHECK_TRUE_RET(prim_c != nullptr, nullptr);
      (void)prim_c->AddAttr(kAttrDType, MakeValue(TypeIdToType(dst_type)));
      MS_LOG(INFO) << "attribute_name: " << attribute_name << ", value is: " << onnx_node_attr.i();
    } else if (onnx_node_attr.type() == onnx::AttributeProto_AttributeType_FLOAT) {
      prim->AddAttr(attribute_name, api::MakeValue(onnx_node_attr.f()));
      MS_LOG(INFO) << "attribute_name: " << attribute_name << ", value is: " << onnx_node_attr.f();
    } else if (onnx_node_attr.type() == onnx::AttributeProto_AttributeType_INT) {
      MS_LOG(INFO) << "attribute_name: " << attribute_name << ", value is: " << onnx_node_attr.i();
      prim->AddAttr(attribute_name, api::MakeValue(onnx_node_attr.i()));
    } else if (onnx_node_attr.type() == onnx::AttributeProto_AttributeType_STRING) {
      SetAttrString(prim, attribute_name, onnx_node_attr.s());
    } else if (onnx_node_attr.type() == onnx::AttributeProto_AttributeType_INTS) {
      SetAttrInts(prim, onnx_node_attr);
    } else if (onnx_node_attr.type() == onnx::AttributeProto_AttributeType_FLOATS) {
      SetAttrFloats(prim, onnx_node_attr);
    } else if (onnx_node_attr.type() == onnx::AttributeProto_AttributeType_STRINGS) {
      SetAttrStrings(prim, onnx_node_attr);
    } else {
      MS_LOG(ERROR) << "ONNX Datatype: " << onnx_node_attr.type() << " is not supported!";
      return nullptr;
    }
  }
  if (!CheckAttrs(prim, input_names, output_names, optional_input_names, type)) {
    return nullptr;
  }
  return prim->GetPrim();
}
OnnxNodeRegistrar g_onnxCustomParser("Custom", new OnnxCustomParser());
}  // namespace lite
}  // namespace mindspore
