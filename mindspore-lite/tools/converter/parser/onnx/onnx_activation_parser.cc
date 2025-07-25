/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include "tools/converter/parser/onnx/onnx_activation_parser.h"
#include <memory>
#include <vector>
#include "include/securec.h"
#include "infer/cxx_api/prelu_fusion.h"
#include "mindspore/ops/op_def/auto_generate/gen_lite_ops.h"
#include "infer/cxx_api/activation.h"
#include "nnacl/op_base.h"
#include "infer/softplus.h"
#include "infer/selu.h"
#include "infer/ops_func_impl/hswish.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr OnnxReluParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Activation>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  prim->set_activation_type(mindspore::ActivationType::RELU);
  prim->set_min_val(0);
  prim->set_max_val(FLT_MAX);

  return prim->GetPrim();
}

PrimitiveCPtr OnnxLeakyReluParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Activation>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == kAttrAlpha) {
      prim->set_alpha(onnx_node_attr.f());
    }
  }

  prim->set_activation_type(mindspore::ActivationType::LEAKY_RELU);

  return prim->GetPrim();
}

PrimitiveCPtr OnnxPReluParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::PReLUFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  MS_CHECK_GE(onnx_node.input_size(), kInputSize1, nullptr);
  std::vector<onnx::TensorProto> params;
  const auto &input_name = onnx_node.input(1);
  auto node_iter = std::find_if(onnx_graph.initializer().begin(), onnx_graph.initializer().end(),
                                [input_name](const onnx::TensorProto &proto) { return proto.name() == input_name; });
  if (node_iter == onnx_graph.initializer().end()) {
    MS_LOG(WARNING) << "cannot find node: " << input_name.c_str();
  } else {
    params.push_back(*node_iter);
  }

  if (!params.empty()) {
    const onnx::TensorProto *slope_data = &params[0];
    if (slope_data == nullptr) {
      MS_LOG(ERROR) << "input error: params[0] is null";
      return nullptr;
    }
    std::vector<float> slope;
    if (slope_data->float_data_size() > 0) {
      const int64_t slope_size = slope_data->float_data_size();
      for (int64_t i = 0; i < slope_size; i++) {
        slope.emplace_back(slope_data->float_data(i));
      }
      prim->set_slope(slope);
      prim->set_channel_shared(slope_size == 1);
    } else {
      const auto slope_raw_data = reinterpret_cast<const float *>(slope_data->raw_data().data());
      MS_CHECK_TRUE_RET(slope_raw_data != nullptr, nullptr);
      const int64_t slope_size = slope_data->raw_data().size() / sizeof(float);
      slope.resize(slope_size);
      bool channel_shared = false;
      if (slope_size == 1) {
        slope.push_back(*slope_raw_data);
        channel_shared = true;
      } else {
        if (INT_MUL_OVERFLOW_THRESHOLD(slope_size, sizeof(float), SIZE_MAX)) {
          MS_LOG(ERROR) << "data_size overflow";
          return nullptr;
        }
        if (memcpy_s(slope.data(), slope_size * sizeof(float), slope_raw_data, slope_data->raw_data().size()) != EOK) {
          MS_LOG(ERROR) << "memcpy_s failed";
          return nullptr;
        }
      }
      prim->set_slope(slope);
      prim->set_channel_shared(channel_shared);
    }
  } else {
    MS_LOG(WARNING) << "The slope pf prelu is null, which may cause errors.";
  }

  return prim->GetPrim();
}

PrimitiveCPtr OnnxEluParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Activation>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  prim->set_activation_type(mindspore::ActivationType::ELU);
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == kAttrAlpha) {
      prim->set_alpha(onnx_node_attr.f());
    }
  }

  return prim->GetPrim();
}

PrimitiveCPtr OnnxGeluParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Activation>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  prim->set_activation_type(mindspore::ActivationType::GELU);
  return prim->GetPrim();
}

PrimitiveCPtr OnnxTanhParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Activation>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  prim->set_activation_type(mindspore::ActivationType::TANH);
  return prim->GetPrim();
}

PrimitiveCPtr OnnxSigmoidParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Activation>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  prim->set_activation_type(mindspore::ActivationType::SIGMOID);
  return prim->GetPrim();
}

PrimitiveCPtr OnnxHardSigmoidParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Activation>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  prim->set_activation_type(mindspore::ActivationType::HSIGMOID);

  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    if (onnx_node_attr.name() == kAttrAlpha) {
      prim->set_alpha(onnx_node_attr.f());
    }
  }
  return prim->GetPrim();
}

PrimitiveCPtr OnnxSoftPlusParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Activation>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  prim->set_activation_type(mindspore::ActivationType::SOFTPLUS);

  return prim->GetPrim();
}

PrimitiveCPtr OnnxSeluParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::SeLU>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr OnnxCeluParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::CeLU>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "alpha") {
      prim->set_alpha(onnx_node_attr.f());
    }
  }

  return prim->GetPrim();
}

OnnxNodeRegistrar g_onnxReluParser("Relu", new OnnxReluParser());
OnnxNodeRegistrar g_onnxLeakyReluParser("LeakyRelu", new OnnxLeakyReluParser());
OnnxNodeRegistrar g_onnxPReluParser("PRelu", new OnnxPReluParser());
OnnxNodeRegistrar g_onnxEluParser("Elu", new OnnxEluParser());
OnnxNodeRegistrar g_onnxTanhParser("Tanh", new OnnxTanhParser());
OnnxNodeRegistrar g_onnxSigmoidParser("Sigmoid", new OnnxSigmoidParser());
OnnxNodeRegistrar g_onnxHardSigmoidParser("HardSigmoid", new OnnxHardSigmoidParser());
OnnxNodeRegistrar g_onnxSoftPlusParser("Softplus", new OnnxSoftPlusParser());
OnnxNodeRegistrar g_onnxGeluParser("Gelu", new OnnxGeluParser());
OnnxNodeRegistrar g_onnxMegatronGeluParser("GeLUFunction", new OnnxGeluParser());
OnnxNodeRegistrar g_onnxSeluParser("Selu", new OnnxSeluParser());
OnnxNodeRegistrar g_onnxCeluParser("Celu", new OnnxCeluParser());
}  // namespace lite
}  // namespace mindspore
