/**
 * Copyright 2020-2026 Huawei Technologies Co., Ltd
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

#include "tools/converter/parser/onnx/onnx_reduce_parser.h"
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "src/common/ops/primitive/reduce_fusion.h"
#include "nnacl_c/op_base.h"

namespace mindspore {
namespace lite {
namespace {
int GetTensorRankByName(const onnx::GraphProto &onnx_graph, const std::string &name) {
  const auto match_by_name = [&name](const auto &tensor) { return tensor.name() == name; };
  auto input_iter = std::find_if(onnx_graph.input().begin(), onnx_graph.input().end(), match_by_name);
  if (input_iter != onnx_graph.input().end() && input_iter->type().has_tensor_type()) {
    return input_iter->type().tensor_type().shape().dim_size();
  }
  auto info_iter = std::find_if(onnx_graph.value_info().begin(), onnx_graph.value_info().end(), match_by_name);
  if (info_iter != onnx_graph.value_info().end() && info_iter->type().has_tensor_type()) {
    return info_iter->type().tensor_type().shape().dim_size();
  }
  return 0;
}

// ONNX semantics: empty axes with noop_with_empty_axes=false reduces ALL axes. Normalize it
// to an explicit full axis list here: a 0-element axes constant would later hang the MindRT
// actor dispatch (the op never receives enough input notifications) and fail the micro coder
// tensor parsing. Identity cases (skip_mode=true) are rewritten by AdjustReduceSumPass.
void FillFullAxesIfEmpty(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node,
                         std::vector<int32_t> *axes, bool skip_mode) {
  if (!axes->empty() || skip_mode || onnx_node.input().empty()) {
    return;
  }
  int32_t rank = GetTensorRankByName(onnx_graph, onnx_node.input(0));
  for (int32_t i = 0; i < rank; ++i) {
    axes->push_back(i);
  }
  if (rank > 0) {
    MS_LOG(INFO) << "Fill full axis list for empty-axes reduce node: " << onnx_node.name() << ", rank: " << rank;
  }
}
}  // namespace

PrimitiveCPtr OnnxReduceParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::ReduceFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  prim->set_keep_dims(true);
  auto prim_c = prim->GetPrim();
  MS_CHECK_TRUE_RET(prim_c != nullptr, nullptr);
  std::vector<int32_t> axes = {};
  bool skip_mode = false;
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "axes") {
      const int &size = onnx_node_attr.ints_size();
      for (int i = 0; i < size; ++i) {
        axes.push_back(onnx_node_attr.ints(i));
      }
    } else if (attribute_name == "keepdims") {
      prim->set_keep_dims(static_cast<bool>(onnx_node_attr.i()));
    } else if (attribute_name == "noop_with_empty_axes") {
      MS_LOG(INFO) << "The noop_with_empty_axes attribute for reduction‑type operators in the latest ONNX release "
                      "controls behavior when axes is omitted or empty (default: false).";
      skip_mode = static_cast<bool>(onnx_node_attr.i());
      prim->set_skip_mode(skip_mode);
    }
  }
  // ONNX semantics: empty axes with noop_with_empty_axes=false reduces ALL axes. Normalize it
  // to an explicit full axis list here: a 0-element axes constant would later hang the MindRT
  // actor dispatch (the op never receives enough input notifications) and fail the micro coder
  // tensor parsing. Identity cases (skip_mode=true) are rewritten by AdjustReduceSumPass.
  FillFullAxesIfEmpty(onnx_graph, onnx_node, &axes, skip_mode);
  // An empty axis means that for all axes, the axis attributes will be adjusted to input in inputs_adjust.cc
  (void)prim_c->AddAttr("axes", MakeValue(axes));

  const auto &type = onnx_node.op_type();
  if (type == "ReduceMean") {
    prim->set_mode(mindspore::ReduceMode::Reduce_Mean);
  } else if (type == "ReduceMax") {
    prim->set_mode(mindspore::ReduceMode::Reduce_Max);
  } else if (type == "ReduceMin") {
    prim->set_mode(mindspore::ReduceMode::Reduce_Min);
  } else if (type == "ReduceSum") {
    prim->set_mode(mindspore::ReduceMode::Reduce_Sum);
  } else if (type == "ReduceProd") {
    prim->set_mode(mindspore::ReduceMode::Reduce_Prod);
  } else if (type == "ReduceSumSquare") {
    prim->set_mode(mindspore::ReduceMode::Reduce_Sum_Square);
  } else if (type == "ReduceL2") {
    prim->set_mode(mindspore::ReduceMode::Reduce_L2);
  } else if (type == "ReduceL1") {
    prim->set_mode(mindspore::ReduceMode::Reduce_L1);
  } else if (type == "ReduceLogSum") {
    prim->set_mode(mindspore::ReduceMode::Reduce_Log_Sum);
  } else if (type == "ReduceLogSumExp") {
    prim->set_mode(mindspore::ReduceMode::Reduce_Log_Sum_Exp);
  } else {
    MS_LOG(ERROR) << "unsupported reduce type: " << type;
    return nullptr;
  }

  return prim->GetPrim();
}

OnnxNodeRegistrar g_onnxReduceMeanParser("ReduceMean", new OnnxReduceParser());
OnnxNodeRegistrar g_onnxReduceMaxParser("ReduceMax", new OnnxReduceParser());
OnnxNodeRegistrar g_onnxReduceMinParser("ReduceMin", new OnnxReduceParser());
OnnxNodeRegistrar g_onnxReduceProdParser("ReduceProd", new OnnxReduceParser());
OnnxNodeRegistrar g_onnxReduceSumParser("ReduceSum", new OnnxReduceParser());
OnnxNodeRegistrar g_onnxReduceSumSquareParser("ReduceSumSquare", new OnnxReduceParser());
OnnxNodeRegistrar g_onnxReduceL2Parser("ReduceL2", new OnnxReduceParser());
OnnxNodeRegistrar g_onnxReduceL1Parser("ReduceL1", new OnnxReduceParser());
OnnxNodeRegistrar g_onnxReduceLogSumParser("ReduceLogSum", new OnnxReduceParser());
OnnxNodeRegistrar g_onnxReduceLogSumExpParser("ReduceLogSumExp", new OnnxReduceParser());
}  // namespace lite
}  // namespace mindspore
