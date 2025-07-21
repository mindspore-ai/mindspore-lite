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

#include "tools/converter/parser/onnx/onnx_col2im_parser.h"
#include <memory>
#include <vector>
#include "tools/converter/parser/onnx/onnx_model_parser.h"
#include "nnacl_c/op_base.h"
#include "infer/col2im.h"

namespace mindspore {
namespace lite {
namespace {
constexpr int kMaxPadSize = 2;
}
PrimitiveCPtr OnnxCol2lmParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_shared<ops::Col2Im>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto prim_c = prim->GetPrim();
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "dilations") {
      std::vector<int64_t> dilations;
      for (int i = 0; i < onnx_node_attr.ints().size(); i++) {
        dilations.push_back(onnx_node_attr.ints(i));
      }
      prim_c->AddAttr("dilation", MakeValue<std::vector<int64_t>>(dilations));
    } else if (attribute_name == "pads") {
      std::vector<int64_t> pads;
      for (int i = 0; i < onnx_node_attr.ints().size() && i < kMaxPadSize; i++) {
        pads.push_back(onnx_node_attr.ints(i));
      }
      prim_c->AddAttr("padding", MakeValue<std::vector<int64_t>>(pads));
    } else if (attribute_name == "strides") {
      std::vector<int64_t> strides;
      for (int i = 0; i < onnx_node_attr.ints().size(); i++) {
        strides.push_back(onnx_node_attr.ints(i));
      }
      prim_c->AddAttr("stride", MakeValue<std::vector<int64_t>>(strides));
    }
  }
  return prim->GetPrim();
}

OnnxNodeRegistrar g_onnxCol2imParser("Col2Im", new OnnxCol2lmParser());
}  // namespace lite
}  // namespace mindspore
