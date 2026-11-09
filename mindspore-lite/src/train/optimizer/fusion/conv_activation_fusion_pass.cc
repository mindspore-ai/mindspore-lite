/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "src/train/optimizer/fusion/conv_activation_fusion_pass.h"
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include "schema/inner/model_generated.h"
#include "tools/common/meta_graph_utils.h"
#include "src/train/optimizer/common/fusion_utils.h"

namespace {
constexpr std::string_view ConvName = "CONV";
constexpr std::string_view ActName = "ACTIVATION";

// Conv2DFusion populate (SetPadAndAct) maps only RELU and RELU6; fusing any other activation
// type would leave a node the runtime cannot load.
bool IsConvFusableActivation(mindspore::schema::ActivationType act_type) {
  return act_type == mindspore::schema::ActivationType_RELU || act_type == mindspore::schema::ActivationType_RELU6;
}
}  // namespace

namespace mindspore {
namespace lite {
STATUS ConvActivationFusionPass::DefinePattern() {
  auto conv_op = std::make_shared<PatternOp>();
  MS_CHECK_TRUE_RET(conv_op != nullptr, RET_NULL_PTR);
  conv_op->id = ConvName;
  conv_op->types = {schema::PrimitiveType_Conv2DFusion};
  auto act_op = std::make_shared<PatternOp>();
  MS_CHECK_TRUE_RET(act_op != nullptr, RET_NULL_PTR);
  act_op->id = ActName;
  act_op->types = {schema::PrimitiveType_Activation};
  act_op->left = conv_op;
  auto fusion_pattern = std::make_unique<FusionPattern>("ConvActivationFusion");
  MS_CHECK_TRUE_MSG(fusion_pattern != nullptr, RET_NULL_PTR, "new fusion_pattern failed");
  fusion_pattern->AddPatternOp(conv_op);
  fusion_pattern->AddPatternOp(act_op);
  fusion_pattern->Finish();
  this->patterns.emplace_back(fusion_pattern.release());
  return RET_OK;
}

STATUS ConvActivationFusionPass::DoFusion(MetaGraphT *graph, const std::string &pattern_name,
                                          const std::unordered_map<std::string, std::shared_ptr<Path>> &matched_path) {
  MS_CHECK_TRUE_RET(graph != nullptr, RET_NULL_PTR);
  if (matched_path.size() != opt::kMatchPathLenTwo) {
    MS_LOG(ERROR) << "Conv-Activation-Fusion should have two NodeIndex in matchedPair";
    return RET_PARAM_INVALID;
  }

  size_t conv_index = 0;
  auto ret = opt::GetMatchNodeIndex(graph, matched_path, std::string(ConvName), &conv_index);
  MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "cannot get conv_index");
  auto &conv_node = graph->nodes.at(conv_index);
  MS_CHECK_TRUE_MSG(conv_node != nullptr, RET_NULL_PTR, "conv_node is nullptr");
  size_t act_index = 0;
  ret = opt::GetMatchNodeIndex(graph, matched_path, std::string(ActName), &act_index);
  MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "cannot get act_index");
  auto &act_node = graph->nodes.at(act_index);
  MS_CHECK_TRUE_MSG(act_node != nullptr, RET_NULL_PTR, "act_node is nullptr");

  if (conv_node->quantType != schema::QuantType_QUANT_NONE) {
    MS_LOG(DEBUG) << "cannot fusion.";
    return RET_NO_CHANGE;
  }
  MS_CHECK_TRUE_RET(conv_node->primitive != nullptr, RET_NULL_PTR);
  auto conv_value = conv_node->primitive->value.AsConv2DFusion();
  MS_CHECK_TRUE_RET(conv_value != nullptr, RET_NULL_PTR);
  MS_CHECK_TRUE_RET(conv_value->activation_type == ActivationType::ActivationType_NO_ACTIVATION, RET_NO_CHANGE);
  MS_CHECK_TRUE_RET(act_node->primitive != nullptr, RET_NULL_PTR);
  auto act_value = act_node->primitive->value.AsActivation();
  MS_CHECK_TRUE_RET(act_value != nullptr, RET_NULL_PTR);
  auto act_type = act_value->activation_type;
  // Keep activations the conv path cannot execute as separate nodes.
  MS_CHECK_TRUE_RET(IsConvFusableActivation(act_type), RET_NO_CHANGE);
  // The fusion hands conv the activation's output and drops conv's own output. Both nodes must
  // have exactly one output, and conv's output must not be a model output, or it would be left
  // without a producer.
  if (conv_node->outputIndex.size() != opt::kOutputSizeOne || act_node->outputIndex.size() != opt::kOutputSizeOne) {
    MS_LOG(DEBUG) << "Conv/Activation output count is not 1, skip fusion.";
    return RET_NO_CHANGE;
  }
  if (opt::IsTensorGraphOutput(graph, conv_node->outputIndex.at(opt::kOutputIndexZero))) {
    MS_LOG(DEBUG) << "Conv output is a graph output, skip fusion.";
    return RET_NO_CHANGE;
  }
  conv_value->activation_type = act_type;
  conv_node->outputIndex = {act_node->outputIndex.at(opt::kOutputIndexZero)};
  act_node->inputIndex.clear();
  act_node->outputIndex.clear();
  return RET_OK;
}

ConvActivationFusionPass::~ConvActivationFusionPass() = default;
}  // namespace lite
}  // namespace mindspore
