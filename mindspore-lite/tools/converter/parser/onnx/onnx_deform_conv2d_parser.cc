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

#include "tools/converter/parser/onnx/onnx_deform_conv2d_parser.h"
#include <vector>
#include <string>
#include "infer/deformable_conv2d.h"
#include "nnacl/op_base.h"

namespace mindspore::lite {
namespace {
constexpr int kWeightIdx3 = 3;
}
STATUS ParseKernelSize(std::vector<int64_t> *kernel_size, const onnx::GraphProto &onnx_graph,
                       const onnx::NodeProto &onnx_node, const int &weight_index) {
  MS_CHECK_TRUE_RET(kernel_size != nullptr, RET_ERROR);
  // output_channel, input_channel, kH, kW
  const int kKernelSizeHeightIndex = 2;
  const int kKernelSizeWidthIndex = 3;
  const size_t kIndex0 = 0;
  const size_t kIndex1 = 1;

  const auto &onnx_weight_name = onnx_node.input(weight_index);
  auto node_iter =
    std::find_if(onnx_graph.initializer().begin(), onnx_graph.initializer().end(),
                 [onnx_weight_name](const onnx::TensorProto &proto) { return proto.name() == onnx_weight_name; });
  if (node_iter == onnx_graph.initializer().end()) {
    MS_LOG(ERROR) << "Can not parse kernel_size.";
    return RET_ERROR;
  } else {
    auto size = (*node_iter).dims_size();
    if (size < DIMENSION_4D) {
      MS_LOG(ERROR) << "weight_shape.size() should not be less than 4, but is " << size;
      return RET_ERROR;
    }
    (*kernel_size)[kIndex0] = (*node_iter).dims(kKernelSizeHeightIndex);
    (*kernel_size)[kIndex1] = (*node_iter).dims(kKernelSizeWidthIndex);
  }
  return RET_OK;
}

STATUS ParseVecAttr(const onnx::NodeProto &onnx_node, std::vector<int64_t> *strides, std::vector<int64_t> *dilation,
                    std::vector<int64_t> *padding) {
  MS_CHECK_TRUE_RET(strides != nullptr && dilation != nullptr && padding != nullptr, RET_NULL_PTR);
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    if (onnx_node_attr.name() == "dilations") {
      if (onnx_node_attr.ints().size() < DIMENSION_2D) {
        MS_LOG(ERROR) << "Parse dilations failed!";
        return RET_ERROR;
      }
      dilation->push_back(onnx_node_attr.ints(0));
      dilation->push_back(onnx_node_attr.ints(1));
    } else if (onnx_node_attr.name() == "dilation") {
      if (onnx_node_attr.ints().size() < DIMENSION_2D) {
        MS_LOG(ERROR) << "Parse dilation failed!";
        return RET_ERROR;
      }
      dilation->push_back(onnx_node_attr.ints(0));
      dilation->push_back(onnx_node_attr.ints(1));
    } else if (onnx_node_attr.name() == "padding") {
      if (onnx_node_attr.ints().size() < DIMENSION_2D) {
        MS_LOG(ERROR) << "Parse padding failed!";
        return RET_ERROR;
      }
      padding->push_back(onnx_node_attr.ints(0));
      padding->push_back(onnx_node_attr.ints(0));
      padding->push_back(onnx_node_attr.ints(1));
      padding->push_back(onnx_node_attr.ints(1));
    } else if (onnx_node_attr.name() == "paddings") {
      if (onnx_node_attr.ints().size() < DIMENSION_2D) {
        MS_LOG(ERROR) << "Parse paddings failed!";
        return RET_ERROR;
      }
      padding->push_back(onnx_node_attr.ints(0));
      padding->push_back(onnx_node_attr.ints(0));
      padding->push_back(onnx_node_attr.ints(1));
      padding->push_back(onnx_node_attr.ints(1));
    } else if (onnx_node_attr.name() == "stride") {
      if (onnx_node_attr.ints().size() < DIMENSION_2D) {
        MS_LOG(ERROR) << "Parse stride failed!";
        return RET_ERROR;
      }
      strides->push_back(onnx_node_attr.ints(0));
      strides->push_back(onnx_node_attr.ints(1));
    } else if (onnx_node_attr.name() == "strides") {
      if (onnx_node_attr.ints().size() < DIMENSION_2D) {
        MS_LOG(ERROR) << "Parse strides failed!";
        return RET_ERROR;
      }
      strides->push_back(onnx_node_attr.ints(0));
      strides->push_back(onnx_node_attr.ints(1));
    }
  }
  return RET_OK;
}

PrimitiveCPtr OnnxDeformConv2dParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::DeformableConv2d>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto prim_c = prim->GetPrim();
  MS_CHECK_TRUE_RET(prim_c != nullptr, nullptr);

  std::vector<int64_t> strides = {1, 1};
  std::vector<int64_t> dilation = {1, 1};
  std::vector<int64_t> pads = {};

  (void)prim_c->AddAttr(mindspore::ops::kOriginalFormat, MakeValue<int64_t>(mindspore::Format::NCHW));

  std::vector<int64_t> kernel_size(DIMENSION_2D);
  int weight_index = 0;
  if (onnx_node.op_type() == "NPUDeformableConv2d") {
    prim_c->AddAttr("keep_origin", MakeValue<bool>(true));
    weight_index = 1;
  } else if (onnx_node.op_type() == "MMCVModulatedDeformConv2d") {
    weight_index = kWeightIdx3;
  } else {
    MS_LOG(ERROR) << "Only support NPUDeformableConv2d and MMCVModulatedDeformConv2d now!";
    return nullptr;
  }
  if (::mindspore::lite::ParseVecAttr(onnx_node, &strides, &dilation, &pads) != RET_OK) {
    MS_LOG(ERROR) << "Parse vector attr failed for " << onnx_node.name();
    return nullptr;
  }
  if (ParseKernelSize(&kernel_size, onnx_graph, onnx_node, weight_index) != RET_OK) {
    MS_LOG(ERROR) << "Parse kernel_size failed for " << onnx_node.name();
    return nullptr;
  }
  int64_t deformable_groups = 1;
  bool modulated = true;
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "deform_groups") {
      deformable_groups = onnx_node_attr.i();
    } else if (attribute_name == "deformable_groups") {
      deformable_groups = onnx_node_attr.i();
    } else if (attribute_name == "modulated") {
      modulated = static_cast<bool>(onnx_node_attr.i());
    }
  }
  prim->set_strides(strides);
  prim->set_dilations(dilation);
  prim->set_pads(pads);
  prim->set_deformable_groups(deformable_groups);
  prim->set_modulated(modulated);  // True for v2, false for v1

  return prim->GetPrim();
}

OnnxNodeRegistrar g_onnx_DeformConv2dParser("MMCVModulatedDeformConv2d", new OnnxDeformConv2dParser());
OnnxNodeRegistrar g_onnx_NPUDeformConv2dParser("NPUDeformableConv2d", new OnnxDeformConv2dParser());
}  // namespace mindspore::lite
