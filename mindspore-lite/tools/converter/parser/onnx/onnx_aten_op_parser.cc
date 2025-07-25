/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "tools/converter/parser/onnx/onnx_aten_op_parser.h"
#include <memory>
#include <vector>
#include "mindspore/ops/op_def/auto_generate/gen_lite_ops.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr OnnxAtenOpBroadCastToParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::BroadcastTo>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

OnnxNodeRegistrar g_onnxAtenOpParser("broadcast_to", new OnnxAtenOpBroadCastToParser());
}  // namespace lite
}  // namespace mindspore
