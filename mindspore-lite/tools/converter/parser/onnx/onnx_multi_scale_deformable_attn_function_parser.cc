/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "tools/converter/parser/onnx/onnx_multi_scale_deformable_attn_function_parser.h"
#include <memory>
#include <string>
#include <vector>
#include "infer/custom.h"
#include "nnacl_c/op_base.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr OnnxMultiScaleDeformableAttnFunctionParser::Parse(const onnx::GraphProto &onnx_graph,
                                                                const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Custom>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  std::vector<std::string> input_name = {"value", "value_spatial_shapes", "value_level_start_index",
                                         "sampling_locations", "attention_weights"};
  std::vector<std::string> output_name = {"output"};
  prim->AddAttr("input_names", api::MakeValue(input_name));
  prim->AddAttr("output_names", api::MakeValue(output_name));
  prim->set_type("MultiScaleDeformableAttnFunction");
  prim->AddAttr("reg_op_name", api::MakeValue("MultiScaleDeformableAttnFunction"));
  return prim->GetPrim();
}

OnnxNodeRegistrar g_onnxMultiScaleDeformableAttnFunctionParser("MultiScaleDeformableAttnFunction",
                                                               new OnnxMultiScaleDeformableAttnFunctionParser());
}  // namespace lite
}  // namespace mindspore
