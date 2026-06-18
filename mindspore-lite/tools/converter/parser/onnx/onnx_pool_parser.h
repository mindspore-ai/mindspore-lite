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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_ONNX_ONNX_POOL_PARSER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_ONNX_ONNX_POOL_PARSER_H_

#include "tools/converter/parser/onnx/onnx_node_parser.h"
#include "tools/converter/parser/onnx/onnx_node_parser_registry.h"
#include "src/common/ops/primitive/max_pool_fusion.h"

namespace mindspore {
namespace lite {
class OnnxAvgPoolParser : public OnnxNodeParser {
 public:
  OnnxAvgPoolParser() : OnnxNodeParser("AvgPool") {}
  ~OnnxAvgPoolParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

class OnnxMaxPoolParser : public OnnxNodeParser {
 public:
  OnnxMaxPoolParser() : OnnxNodeParser("MaxPool") {}
  ~OnnxMaxPoolParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;

 private:
  int GetKernelSize(const onnx::NodeProto &onnx_node) const;
  PrimitiveCPtr ParseMaxPool1D(const onnx::NodeProto &onnx_node, std::unique_ptr<ops::MaxPoolFusion> &prim);
  bool ParseMaxPool2DAttrs(const onnx::NodeProto &onnx_node, std::unique_ptr<ops::MaxPoolFusion> &prim,
                           std::vector<int64_t> *strides, std::vector<int64_t> *pads, mindspore::RoundMode *round_mode);
  PrimitiveCPtr ParseMaxPool2D(const onnx::NodeProto &onnx_node, std::unique_ptr<ops::MaxPoolFusion> &prim);
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_ONNX_ONNX_POOL_PARSER_H_
