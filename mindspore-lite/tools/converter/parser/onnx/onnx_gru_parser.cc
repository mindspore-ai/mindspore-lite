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

#include "tools/converter/parser/onnx/onnx_gru_parser.h"
#include <memory>
#include <string>
#include <vector>
#include "infer/gru.h"
#include "nnacl_c/op_base.h"
#include "include/registry/converter_context.h"
#include "mindspore/ops/infer/grad/gru_v2_grad.h"

namespace mindspore {
namespace {
constexpr auto kGruAttrDirection = "direction";
constexpr auto kGruAttrDirectionValueForward = "forward";
constexpr auto kGruAttrDirectionValueReverse = "reverse";
constexpr auto kGruAttrDirectionValueBidirectional = "bidirectional";
constexpr auto kGruAttrActivationAlpha = "activation_alpha";
constexpr auto kGruAttrActivationBeta = "activation_beta";
constexpr auto kGruAttrActivations = "activations";
constexpr auto kGruAttrClip = "clip";
constexpr auto kGruAttrHiddenSize = "hidden_size";
constexpr auto kGruAttrLinearBeforeReset = "linear_before_reset";
}  // namespace
namespace lite {
PrimitiveCPtr OnnxGruParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::GRU>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  (void)prim->AddAttr(kGruAttrDirection, api::MakeValue(kGruAttrDirectionValueForward));
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    if (onnx_node_attr.name() == kGruAttrDirection) {
      const auto &direction = onnx_node_attr.s();
      bool bidirectional = direction == kGruAttrDirectionValueBidirectional;
      prim->set_bidirectional(bidirectional);
      if (direction == kGruAttrDirectionValueBidirectional) {
        (void)prim->AddAttr(kGruAttrDirection, api::MakeValue(kGruAttrDirectionValueBidirectional));
      } else if (direction == kGruAttrDirectionValueReverse) {
        (void)prim->AddAttr(kGruAttrDirection, api::MakeValue(kGruAttrDirectionValueReverse));
      } else if (direction == kGruAttrDirectionValueForward) {
        (void)prim->AddAttr(kGruAttrDirection, api::MakeValue(kGruAttrDirectionValueForward));
      } else {
        MS_LOG(ERROR) << " not support direction value : " << direction;
        return nullptr;
      }
    } else if (onnx_node_attr.name() == kGruAttrActivationAlpha) {
      std::vector<float> activation_alpha;
      for (int i = 0; i < onnx_node_attr.ints_size(); ++i) {
        activation_alpha.push_back(onnx_node_attr.floats(i));
      }
      (void)prim->AddAttr(kGruAttrActivationAlpha, api::MakeValue(activation_alpha));
    } else if (onnx_node_attr.name() == kGruAttrActivationBeta) {
      std::vector<float> activation_beta;
      for (int i = 0; i < onnx_node_attr.ints_size(); ++i) {
        activation_beta.push_back(onnx_node_attr.floats(i));
      }
      (void)prim->AddAttr(kGruAttrActivationBeta, api::MakeValue(activation_beta));
    } else if (onnx_node_attr.name() == kGruAttrActivations) {
      std::vector<std::string> activations;
      for (int i = 0; i < onnx_node_attr.ints_size(); ++i) {
        activations.push_back(onnx_node_attr.strings(i));
      }
      (void)prim->AddAttr(kGruAttrActivations, api::MakeValue(activations));
    } else if (onnx_node_attr.name() == kGruAttrClip) {
      (void)prim->AddAttr(kGruAttrClip, api::MakeValue(onnx_node_attr.f()));
    } else if (onnx_node_attr.name() == kGruAttrHiddenSize) {
      (void)prim->AddAttr(kGruAttrHiddenSize, api::MakeValue(onnx_node_attr.i()));
    } else if (onnx_node_attr.name() == kGruAttrLinearBeforeReset) {
      (void)prim->AddAttr(kGruAttrLinearBeforeReset, api::MakeValue(onnx_node_attr.i()));
    }
  }

  int fmk_type = mindspore::converter::FmkType::kFmkTypeOnnx;
  (void)prim->AddAttr(ops::kFmkType, api::MakeValue(fmk_type));
  return prim->GetPrim();
}

OnnxNodeRegistrar g_onnxGruParser("GRU", new OnnxGruParser());
}  // namespace lite
}  // namespace mindspore
