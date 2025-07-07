/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "tools/optimizer/graph/attr_to_args_pass.h"
#include <utility>
#include "tools/common/node_util.h"
#include "nnacl/op_base.h"
#include "src/common/log_util.h"
#include "ops/primitive_c.h"
#include "ops/base_operator.h"
#include "utils/anf_utils.h"

namespace mindspore {
namespace opt {
namespace {

static const std::map<std::string, std::vector<std::pair<std::string, size_t>>> kAttrMapNeedAdjust = {
  {"LogSoftmax", {{"axis", 2}}},
  {"ArgMin", {{"axis", 2}, {"output_type", 3}}},
  {"PromptFlashAttention",
   {{"num_heads", 13},
    {"scale_value", 14},
    {"pre_tokens", 15},
    {"next_tokens", 16},
    {"input_layout", 17},
    {"num_key_value_heads", 18},
    {"sparse_mode", 19},
    {"inner_precise", 20}}},
  {"BroadcastTo", {{"shape", 2}}},
  {"ArgMaxV2", {{"axis", 2}, {"output_type", 3}}},
  {"ArgMaxWithValue", {{"axis", 2}, {"keep_dims", 3}}},
  {"AvgPool", {{"kernel_size", 2}, {"strides", 3}, {"pad_mode", 4}, {"data_format", 5}}},
  {"ApplyRotaryPosEmb", {{"layout", 6}}},
  {"StridedSlice",
   {{"begin_mask", 5}, {"end_mask", 6}, {"ellipsis_mask", 7}, {"new_axis_mask", 8}, {"shrink_axis_mask", 9}}},
  {"BatchNorm", {{"is_training", 6}, {"epsilon", 7}, {"momentum", 8}, {"data_format", 9}}},
  {"FusedBatchNorm", {{"is_training", 6}, {"epsilon", 7}, {"momentum", 8}, {"data_format", 9}}},
  {"Elu", {{"alpha", 2}}},
  {"Gather", {{"batch_dims", 4}}},
  {"LayerNorm", {{"begin_norm_axis", 4}, {"begin_params_axis", 5}, {"epsilon", 6}}},
  {"LayerNormV3", {{"begin_norm_axis", 4}, {"begin_params_axis", 5}, {"epsilon", 6}}},
  {"Range", {{"maxlen", 4}}},
  {"Concat", {{"axis", 2}}},
  {"ConcatV2", {{"axis", 2}}},
  {"CumSum", {{"exclusive", 3}, {"reverse", 4}}},
  {"ReduceAll", {{"keep_dims", 3}}},
  {"ReduceMax", {{"keep_dims", 3}}},
  {"ReduceMin", {{"keep_dims", 3}}},
  {"ReduceMean", {{"keep_dims", 3}}},
  {"ReduceSum", {{"keep_dims", 3}, {"skip_mode", 4}}},
  {"Split", {{"axis", 2}, {"output_num", 3}}},
  {"SplitD", {{"split_dim", 2}, {"num_split", 3}}},
  {"ResizeBicubic", {{"align_corners", 3}, {"half_pixel_centers", 4}}},
  {"ResizeBilinear", {{"size", 2}, {"align_corners", 3}, {"half_pixel_centers", 4}}},
  {"ResizeNearestNeighbor", {{"size", 2}, {"align_corners", 3}, {"half_pixel_centers", 4}}},
  {"ResizeBilinearV2", {{"align_corners", 3}, {"half_pixel_centers", 4}}},
  {"ResizeNearestNeighborV2", {{"align_corners", 3}, {"half_pixel_centers", 4}}},
  {"ReverseV2", {{"axis", 2}}},
  {"MatMul", {{"transpose_a", 3}, {"transpose_b", 4}}},    // special
  {"MatMulV2", {{"transpose_a", 3}, {"transpose_b", 4}}},  // special
  {"Meshgrid", {{"indexing", 2}}},
  {"NanToNum", {{"nan", 2}, {"posinf", 3}, {"neginf", 4}}},
  {"BatchMatMul", {{"transpose_a", 3}, {"transpose_b", 4}}},
  {"Softmax", {{"axis", 2}}},
  {"Softshrink", {{"lambd", 2}}},
  {"Squeeze", {{"axis", 2}}},
  {"FusedInferAttentionScore",
   {{"num_heads", 25},
    {"scale", 26},
    {"pre_tokens", 27},
    {"next_tokens", 28},
    {"input_layout", 29},
    {"num_key_value_heads", 30},
    {"sparse_mode", 31},
    {"inner_precise", 32},
    {"block_size", 33},
    {"antiquant_mode", 34},
    {"softmax_lse_flag", 35},
    {"key_antiquant_mode", 36},
    {"value_antiquant_mode", 37}}},
  {"IncreFlashAttention",
   {{"num_heads", 16},
    {"input_layout", 17},
    {"scale_value", 18},
    {"num_key_value_heads", 19},
    {"block_size", 20},
    {"inner_precise", 21}}},
  {"GridSampler3D", {{"interpolation_mode", 3}, {"padding_mode", 4}, {"align_corners", 5}}},
  {"GridSampler2D", {{"interpolation_mode", 3}, {"padding_mode", 4}, {"align_corners", 5}}},
  {"WeightQuantBatchMatmul", {{"transpose_x", 8}, {"transpose_weight", 9}, {"antiquant_group_size", 10}}},
  {"QuantBatchMatmul", {{"transpose_x1", 7}, {"transpose_x2", 8}, {"dtype", 9}}},
  {"QuantBatchMatmulV3", {{"transpose_x1", 7}, {"transpose_x2", 8}, {"dtype", 9}}},
  {"GroupedMatmul", {{"split_item", 9}, {"group_type", 10}, {"transpose_a", 11}, {"transpose_b", 12}}},
  {"AdaptiveMaxPool2D", {{"output_size", 2}}},
  {"BinaryCrossEntropy", {{"reduction", 4}}},
  {"Cross", {{"dim", 3}}},
  {"Triu", {{"diagonal", 2}}},
  {"SoftMarginLoss", {{"reduction", 3}}},
  {"SmoothL1Loss", {{"beta", 3}, {"reduction", 4}}},
  {"TensorScatterElements", {{"axis", 4}, {"reduction", 5}, {"reduce", 5}}},  // reduce OR reduction is passed
  {"ResizeD", {{"sizes", 2}, {"scales", 3}, {"coordinate_transformation_mode", 4}}}};

constexpr size_t kMatMulInputSizeWithBias = 6;  // primitive, x1, x2, bias, transpose_a, transpose_b
constexpr size_t kInputSizeTwo = 2;
constexpr size_t kInputSizeThree = 3;
constexpr auto kMatMulOpName = "MatMul";
constexpr auto kMatMulV2OpName = "MatMulV2";
constexpr auto kSqueezeOpName = "Squeeze";
constexpr auto kStridedSliceOpName = "StridedSlice";
constexpr auto kCustomOpName = "Custom";
constexpr auto kPromptFlashAttentionOpName = "PromptFlashAttention";

void RearrangeBiasForMatMul(const FuncGraphManagerPtr &manager, const CNodePtr &cnode) {
  auto node_inputs = cnode->inputs();
  auto bias_add_node_it = node_inputs.begin() + kIndexThree;
  std::rotate(bias_add_node_it, bias_add_node_it + 1, node_inputs.end());
  cnode->set_inputs(node_inputs);
}

int AdjustInputsAndAttrsForSqueeze(const FuncGraphManagerPtr &manager, const CNodePtr &cnode,
                                   const mindspore::PrimitivePtr &origin_prim) {
  auto node_inputs = cnode->inputs();
  auto actual_input_num = node_inputs.size();
  const auto &attrs_adjust = kAttrMapNeedAdjust.at(kSqueezeOpName);
  const auto &origin_attrs = origin_prim->attrs();
  auto attrs_name = attrs_adjust.begin()->first;
  // Create new primitive and inherit the origin attributes.
  if (origin_attrs.count(attrs_name) != 0) {
    // Convert the specific attr to input and erase the specific attr.
    auto attr_value = origin_prim->GetAttr(attrs_name);
    MS_CHECK_TRUE_MSG(attr_value != nullptr, RET_ERROR, "attr_value is nullptr");
    auto new_value_node = std::make_shared<ValueNode>(attr_value);
    MS_CHECK_TRUE_MSG(new_value_node != nullptr, RET_ERROR, "new_value_node is nullptr");
    new_value_node->set_abstract(attr_value->ToAbstract());
    if (actual_input_num == kInputSizeThree) {
      auto axis_input_node = cnode->input(kIndexTwo);
      if (axis_input_node->isa<Parameter>() && axis_input_node->cast<ParameterPtr>()->has_default()) {
        MS_LOG(INFO) << "Origin primitive: Squeeze already has a const input, replacing it with the attribute.";
        manager->Replace(axis_input_node, new_value_node);
      }
      return RET_OK;
    }
    MS_CHECK_TRUE_MSG(actual_input_num == kInputSizeTwo, RET_ERROR,
                      "Origin primitive: Squeeze must has only one or two inputs");
    manager->AddEdge(cnode, new_value_node);
    return RET_OK;
  }
  MS_LOG(INFO) << "Origin primitive: Squeeze has no attribute : " << attrs_name;
  return RET_OK;
}

bool IsValidInputIndex(const std::vector<int64_t> &input_index) {
  if (input_index.empty()) {
    return true;
  }
  if (input_index[0] < 0) {
    return false;
  }
  for (size_t i = 1; i < input_index.size(); ++i) {
    if (input_index[i] <= input_index[i - 1]) {
      return false;
    }
  }
  return true;
}

int ConvertAttrToArgsForNode(const AnfNodePtr &node, const FuncGraphManagerPtr &manager) {
  auto cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_MSG(cnode != nullptr, RET_ERROR, "cnode is nullptr");
  const auto &origin_prim = GetCNodePrimitive(node);
  MS_CHECK_TRUE_MSG(origin_prim != nullptr, RET_ERROR, "origin_prim is nullptr");
  auto prim_name = origin_prim->name();
  if (prim_name == kSqueezeOpName) {
    return AdjustInputsAndAttrsForSqueeze(manager, cnode, origin_prim);
  }
  if (prim_name == kCustomOpName) {
    auto attr_type = origin_prim->GetAttr("type");
    auto attr_reg_op_name = origin_prim->GetAttr("reg_op_name");
    if (attr_type != nullptr) {
      prim_name = GetValue<std::string>(attr_type);
    } else if (attr_reg_op_name != nullptr) {
      prim_name = GetValue<std::string>(attr_reg_op_name);
    } else {
      MS_LOG(ERROR) << "Custom op has no attribute type or reg_op_name!";
      return RET_ERROR;
    }
    auto attr_input_index = origin_prim->GetAttr("input_index");
    if (attr_input_index != nullptr) {
      AnfNodePtrList new_inputs = {};
      new_inputs.emplace_back(cnode->input(0));
      auto attr_input_names = origin_prim->GetAttr("input_names");
      MS_CHECK_TRUE_MSG(attr_input_names != nullptr, RET_ERROR, "Custom op has no attribute input_names!");
      auto input_names = GetValue<std::vector<std::string>>(attr_input_names);
      auto input_index = GetValue<std::vector<int64_t>>(attr_input_index);
      if (!IsValidInputIndex(input_index)) {
        MS_LOG(ERROR) << "Custom op attribute input_index is invalid!";
        return RET_ERROR;
      }
      for (size_t i = 0; i < input_names.size(); i++) {
        auto index_it = std::find(input_index.begin(), input_index.end(), i);
        if (index_it == input_index.end()) {
          auto none_input = NewValueNode(std::make_shared<None>());
          none_input->set_abstract(std::make_shared<abstract::AbstractNone>());
          new_inputs.emplace_back(none_input);
        } else {
          auto index = index_it - input_index.begin() + 1;
          auto input = cnode->input(index);
          if (input == nullptr) {
            MS_LOG(ERROR) << "Failed to get cnode input at index " << index << ", input name is " << input_names[i]
                          << "!";
            return RET_ERROR;
          }
          new_inputs.emplace_back(input);
        }
      }
      cnode->set_inputs(new_inputs);
    }
    if (kAttrMapNeedAdjust.find(prim_name) == kAttrMapNeedAdjust.end()) {
      MS_LOG(INFO) << "Custom with type: '" << prim_name << "' does not need to do attr_to_args conversion.";
      return RET_OK;
    }
  }
  const auto &attrs_adjust = kAttrMapNeedAdjust.at(prim_name);
  const auto &origin_attrs = origin_prim->attrs();
  auto node_inputs = cnode->inputs();
  auto actual_input_num = node_inputs.size();
  // skip when attr to arg conversion has been completed.
  if ((prim_name == kStridedSliceOpName) && ((attrs_adjust.back().second + 1) == actual_input_num)) {
    return RET_OK;
  }
  // Pad none for optional input, first input of cnode is Primitive, so an extra none is padded.
  if (attrs_adjust.begin()->second > actual_input_num) {
    auto pad_none_size = attrs_adjust.begin()->second - actual_input_num;
    auto none_input = NewValueNode(std::make_shared<None>());
    none_input->set_abstract(std::make_shared<abstract::AbstractNone>());
    node_inputs.insert(node_inputs.end(), pad_none_size, none_input);
    cnode->set_inputs(node_inputs);
  }

  // Create new primitive and inherit the origin attributes.
  MS_LOG(INFO) << "Begin to convert Primitive to Primitive_Func for node: " << node->DebugString()
               << "new name: " << prim_name;
  for (const auto &attr_pair : attrs_adjust) {
    auto attr = attr_pair.first;
    if (origin_attrs.count(attr) != 0) {
      // Convert the specific attr to input and erase the specific attr.
      auto attr_value = origin_prim->GetAttr(attr);
      MS_CHECK_TRUE_MSG(attr_value != nullptr, RET_ERROR, "attr_value is nullptr");
      auto new_value_node = std::make_shared<ValueNode>(attr_value);
      MS_CHECK_TRUE_MSG(new_value_node != nullptr, RET_ERROR, "new_value_node is nullptr");
      new_value_node->set_abstract(attr_value->ToAbstract());
      manager->AddEdge(cnode, new_value_node);
    } else {
      MS_LOG(INFO) << "Origin primitive " << prim_name << " has no attribute : " << attr << ", pad none for "
                   << prim_name << ".";
      auto none_attribute = NewValueNode(std::make_shared<None>());
      none_attribute->set_abstract(std::make_shared<abstract::AbstractNone>());
      manager->AddEdge(cnode, none_attribute);
    }
  }

  if ((prim_name == kMatMulOpName || prim_name == kMatMulV2OpName) &&
      cnode->inputs().size() == kMatMulInputSizeWithBias) {
    RearrangeBiasForMatMul(manager, cnode);
  }
  MS_LOG(INFO) << "End, new node: " << node->DebugString();
  return RET_OK;
}
}  // namespace

bool AttrToArgsPass::Run(const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "func_graph is nullptr.";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return false;
  }

  auto manager = Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "get func graph manager is nullptr";
    return false;
  }

  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (prim == nullptr) {
      continue;
    }
    auto prim_name = prim->name();
    if (kAttrMapNeedAdjust.find(prim->name()) == kAttrMapNeedAdjust.end() && !(prim_name == kCustomOpName)) {
      continue;
    }
    if (ConvertAttrToArgsForNode(node, manager) != RET_OK) {
      MS_LOG(ERROR) << "Convert attr to args for node " << node->fullname_with_scope() << "failed.";
      return false;
    }
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
