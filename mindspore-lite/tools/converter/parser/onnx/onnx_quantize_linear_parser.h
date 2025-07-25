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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_ONNX_ONNX_QUANTIZE_LINEAR_PARSER_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_ONNX_ONNX_QUANTIZE_LINEAR_PARSER_H

#include "tools/converter/parser/onnx/onnx_node_parser.h"
#include <string>
#include <memory>
#include <vector>
#include <utility>
#include "tools/converter/ops/ops_def.h"
#include "tools/converter/parser/onnx/onnx_node_parser_registry.h"

namespace mindspore {
namespace lite {
class OnnxQuantizeLinearParser : public OnnxNodeParser {
 public:
  OnnxQuantizeLinearParser() : OnnxNodeParser("QuantizeLinear") {}
  ~OnnxQuantizeLinearParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;

 private:
  tensor::TensorPtr GetConstData(const onnx::GraphProto &onnx_graph, const std::string &input_name);

  template <typename T>
  std::vector<T> GetConstTData(const onnx::GraphProto &onnx_graph, const std::string &input_name);

  bool SetScaleAttr(const onnx::GraphProto &onnx_graph, const string &onnx_quantize_scale,
                    const std::unique_ptr<QuantizeLinear> &prim);

  bool SetZeroPointAttr(const onnx::GraphProto &onnx_graph, const string &onnx_quantize_zero_point,
                        const std::unique_ptr<QuantizeLinear> &prim);
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_ONNX_ONNX_QUANTIZE_LINEAR_PARSER_H
