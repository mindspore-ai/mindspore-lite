/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "parser/onnx/onnx_bitshift_parser.h"
#include <memory>
#include <string>
#include "common/op_attr.h"
#include "infer/custom.h"
#include "./onnx.pb.h"
#include "include/registry/node_parser_registry.h"
#include "mindapi/base/logging.h"

namespace mindspore {
namespace lite {
ops::BaseOperatorPtr OnnxBitShiftParser::Parse(const onnx::GraphProto &onnx_proto, const onnx::NodeProto &onnx_node) {
  auto prim = api::MakeShared<ops::Custom>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "new Custom prim failed.";
    return nullptr;
  }
  prim->set_type("BitShift");
  std::string direction = "RIGHT";
  auto iter =
    std::find_if(onnx_node.attribute().begin(), onnx_node.attribute().end(),
                 [](const onnx::AttributeProto onnx_node_attr) { return onnx_node_attr.name() == "direction"; });
  if (iter != onnx_node.attribute().end()) {
    direction = (*iter).s();
  }
  // set attr for mapper
  (void)prim->AddAttr("direction", api::MakeValue(direction));

  return prim;
}
}  // namespace lite
}  // namespace mindspore
using mindspore::converter::kFmkTypeOnnx;
namespace mindspore::registry {
REG_NODE_PARSER(kFmkTypeOnnx, BitShift, std::make_shared<lite::OnnxBitShiftParser>())
}  // namespace mindspore::registry
