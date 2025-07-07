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

#define USE_DEPRECATED_API
#include "tools/optimizer/fusion/leaky_relu_fusion.h"
#include <memory>
#include <vector>
#include "mindspore/ops/op_def/lite_ops.h"
#include "infer/cxx_api/activation.h"
#include "infer/cxx_api/mul_fusion.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "nnacl/op_base.h"
#include "ops_utils/op_utils.h"
#include "infer/leaky_relu.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"

namespace mindspore::opt {
namespace {
constexpr size_t kNumIndex1 = 1;
constexpr size_t kNumIndex2 = 2;

STATUS GetNegativeSlopeNum(const AnfNodePtr &negative_slope_node, float *negative_slope_num) {
  MS_CHECK_TRUE_RET(negative_slope_node != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(negative_slope_num != nullptr, RET_ERROR);
  tensor::TensorPtr tensor_info = nullptr;
  if (utils::isa<ValueNodePtr>(negative_slope_node)) {
    auto value_node = negative_slope_node->cast<ValueNodePtr>();
    if (value_node == nullptr) {
      MS_LOG(WARNING) << "value node is nullptr.";
      return RET_ERROR;
    }
    auto value = value_node->value();
    if (value == nullptr) {
      MS_LOG(WARNING) << "value is nullptr.";
      return RET_ERROR;
    }
    tensor_info = value->cast<tensor::TensorPtr>();
  } else if (utils::isa<ParameterPtr>(negative_slope_node)) {
    auto negative_slope_param = negative_slope_node->cast<ParameterPtr>()->default_param();
    if (negative_slope_param == nullptr) {
      MS_LOG(WARNING) << "negative_slope_param is nullptr.";
      return RET_ERROR;
    }
    tensor_info = negative_slope_param->cast<tensor::TensorPtr>();
  } else {
    MS_LOG(WARNING) << "negative_slope_node is not ValueNode or ParamNode, now not support other node type.";
    return RET_ERROR;
  }
  if (tensor_info == nullptr) {
    MS_LOG(WARNING) << "tensor_info is nullptr.";
    return RET_ERROR;
  }
  if (tensor_info->ElementsNum() != 1) {
    MS_LOG(WARNING) << "negative slope value elements num is not 1, ElementsNum is: " << tensor_info->ElementsNum();
    return RET_ERROR;
  }
  if (tensor_info->data_type() == kNumberTypeFloat32) {
    auto negative_slope_data = static_cast<float *>(tensor_info->data_c());
    if (negative_slope_data == nullptr) {
      MS_LOG(WARNING) << "negative_slope_data is nullptr.";
      return RET_ERROR;
    }
    *negative_slope_num = static_cast<float>(negative_slope_data[0]);
    MS_LOG(INFO) << "negative_slope_num: " << *negative_slope_num;
    return RET_OK;
  } else {
    MS_LOG(WARNING) << "negative slope is not float32, now not support other data type.";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace
const BaseRef LeakyReluFusion::DefinePattern() const {
  auto is_mul_1 = std::make_shared<Var>();
  auto is_mul_2 = std::make_shared<Var>();
  auto is_mul = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMul>);
  VectorRef mul = VectorRef({is_mul, is_mul_1, is_mul_2});
  auto is_maximum = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMaximum>);
  VectorRef pattern_ref = VectorRef({is_maximum, mul, is_mul_2});
  return pattern_ref;
}

const AnfNodePtr LeakyReluFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                          const EquivPtr &) const {
  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
  MS_CHECK_TRUE_RET(node != nullptr, nullptr);

  MS_LOG(DEBUG) << "node name: " << node->fullname_with_scope();
  auto maximum_cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(maximum_cnode != nullptr, nullptr);
  auto mul_cnode = maximum_cnode->input(1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(mul_cnode != nullptr, nullptr);
  auto negative_slope_node = mul_cnode->input(1);
  float negative_slope_num = -1;
  MS_LOG(INFO) << "negative_slope_node name : " << negative_slope_node->fullname_with_scope();
  auto status = GetNegativeSlopeNum(negative_slope_node, &negative_slope_num);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "GetNegativeSlopeNum failed!";
    return nullptr;
  }
  MS_LOG(INFO) << "negative_slope_num: " << negative_slope_num;

  auto activate_prim = std::make_shared<ops::LeakyRelu>();
  MS_CHECK_TRUE_RET(activate_prim != nullptr, nullptr);

  activate_prim->set_negative_slope(negative_slope_num);
  activate_prim->AddAttr("alpha", api::MakeValue(negative_slope_num));
  auto activate_prim_c = activate_prim->GetPrim();
  MS_CHECK_TRUE_RET(activate_prim_c != nullptr, nullptr);

  auto cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(cnode != nullptr, nullptr);
  MS_CHECK_TRUE_RET(cnode->input(kNumIndex1)->cast<CNodePtr>() != nullptr, nullptr);
  auto leaky_relu_input = cnode->input(kNumIndex1)->cast<CNodePtr>()->input(kNumIndex2);
  MS_CHECK_TRUE_RET(leaky_relu_input != nullptr, nullptr);
  MS_LOG(INFO) << leaky_relu_input->fullname_with_scope();
  std::vector<AnfNodePtr> op_inputs = {leaky_relu_input};
  auto activate_node = func_graph->NewCNode(activate_prim_c, op_inputs);
  if (activate_node == nullptr) {
    MS_LOG(ERROR) << "new node failed!";
    return nullptr;
  }
  activate_node->set_fullname_with_scope(node->fullname_with_scope() + "_leaky_relu");
  if (node->abstract() != nullptr) {
    activate_node->set_abstract(node->abstract()->Clone());
  }

  return activate_node;
}
}  // namespace mindspore::opt
