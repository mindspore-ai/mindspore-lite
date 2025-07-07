/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "pass/pass_registry_tutorial.h"
#include <map>
#include <memory>
#include <string>
#include <cmath>
#include <vector>
#include <iostream>
#include "include/registry/pass_registry.h"
#include "infer/custom.h"
#include "infer/cxx_api/add_fusion.h"
#include "include/op_def/auto_generate/gen_lite_ops.h"
#include "include/mindapi/ir/anf.h"
#include "mindspore/ops/op_def/op_enum.h"

namespace mindspore {
namespace opt {
namespace {
constexpr int kInputIndexZ2 = 2;
constexpr int64_t num_heads = 8;
constexpr int64_t next_tokens = 65535;
constexpr int64_t pre_tokens = 2147483647;
constexpr int64_t d_value = 64;
constexpr float sqrt_root = 0.5;
constexpr int64_t sparse_mode = 0;
constexpr int64_t inner_precise = 0;
// check a certain node is designated node's type.
bool CheckPrimitiveTypeTutorial(const api::AnfNodePtr &node, const api::PrimitivePtr &primitive_type) {
  if (node == nullptr) {
    return false;
  }
  if (node->isa<api::CNode>()) {
    auto cnode = node->cast<api::CNodePtr>();
    return IsPrimitive(cnode->input(0), primitive_type);
  } else if (node->isa<api::ValueNode>()) {
    return IsPrimitive(node, primitive_type);
  }
  return false;
}
}  // namespace

bool PassTutorial::CreateCustomOp(const api::FuncGraphPtr func_graph, const api::AnfNodePtr &node) {
  if (node == nullptr) {
    return false;
  }
  auto manager = api::FuncGraphManager::Manage(func_graph, true);
  if (manager == nullptr) {
    return false;
  }
  auto cnode = node->cast<api::CNodePtr>();
  if (cnode == nullptr) {
    return false;
  }

  // You have learned the connection relationships between nodes in an ONNX graph.
  // Note: Check whether all nodes are nullptr.
  // cnode is softmax, q_BNSD->mul_q->matmul->softmax->matmul_1.
  // k_BNSD->reshape->transpose->mul_k->matmul->softmax->matmul_1.
  // v_BNSD->matmul_1.
  auto matmul = cnode->input(1)->cast<api::CNodePtr>();
  if (matmul == nullptr) {
    std::cout << "matmul is nullptr, please check!";
    return false;
  }
  auto mul_q = matmul->input(1)->cast<api::CNodePtr>();
  if (mul_q == nullptr) {
    std::cout << "mul_q is nullptr, please check!";
    return false;
  }
  auto mul_k = matmul->input(kInputIndexZ2)->cast<api::CNodePtr>();
  if (mul_k == nullptr) {
    std::cout << "mul_k is nullptr, please check!";
    return false;
  }
  auto transpose_q_BNSD = mul_q->input(1)->cast<api::CNodePtr>();
  if (transpose_q_BNSD == nullptr) {
    std::cout << "transpose_q_BNSD is nullptr, please check!";
    return false;
  }
  auto transpose_k_BNDS = mul_k->input(1)->cast<api::CNodePtr>();
  if (transpose_k_BNDS == nullptr) {
    std::cout << "transpose_k_BNDS is nullptr, please check!";
    return false;
  }
  auto reshape_k_BNDS = transpose_k_BNDS->input(1)->cast<api::CNodePtr>();
  if (reshape_k_BNDS == nullptr) {
    std::cout << "reshape_k_BNDS is nullptr, please check!";
    return false;
  }
  auto transpose_k_BNSD = reshape_k_BNDS->input(1)->cast<api::CNodePtr>();
  if (transpose_k_BNSD == nullptr) {
    std::cout << "transpose_k_BNSD is nullptr, please check!";
    return false;
  }
  auto softtmax_node_users = manager->GetUsers(node);
  api::CNodePtr matmul_1;
  for (auto node_user : softtmax_node_users) {
    matmul_1 = node_user.first->cast<api::CNodePtr>();
  }
  auto transpose_v_BNSD = matmul_1->input(kInputIndexZ2)->cast<api::CNodePtr>();

  auto primc = api::MakeShared<ops::Custom>();
  if (primc == nullptr) {
    return false;
  }
  std::vector<std::string> input_names = {"query", "key", "value"};
  std::vector<std::string> output_names = {"attention_out"};
  primc->AddAttr("input_names", api::MakeValue(input_names));
  primc->AddAttr("output_names", api::MakeValue(output_names));
  primc->set_type("PromptFlashAttention");
  primc->AddAttr("reg_op_name", api::MakeValue("PromptFlashAttention"));
  float scale_value = 1 / (pow(d_value, sqrt_root));
  int64_t num_key_value_heads = num_heads;
  primc->AddAttr("num_heads", api::MakeValue(num_heads));
  primc->AddAttr("pre_tokens", api::MakeValue(pre_tokens));
  primc->AddAttr("next_tokens", api::MakeValue(next_tokens));
  primc->AddAttr(
    "input_layout",
    api::MakeValue(static_cast<mindspore::ops::FASInputLayoutMode>(mindspore::ops::FASInputLayoutMode::BNSD)));
  primc->AddAttr("num_key_value_heads", api::MakeValue(num_key_value_heads));
  primc->AddAttr("scale_value", api::MakeValue(scale_value));
  primc->AddAttr("sparse_mode", api::MakeValue(sparse_mode));
  primc->AddAttr("inner_precise", api::MakeValue(inner_precise));
  auto custom_cnode = func_graph->NewCNode(primc, {transpose_q_BNSD, transpose_k_BNSD, transpose_v_BNSD});
  if (custom_cnode == nullptr) {
    return false;
  }
  custom_cnode->set_fullname_with_scope(cnode->fullname_with_scope() + "_FA");
  custom_cnode->set_abstract(cnode->abstract()->Clone());
  manager->Replace(matmul_1, custom_cnode);
  return true;
}

bool PassTutorial::Execute(const api::FuncGraphPtr &func_graph) {
  std::cout << "Start to execute pass." << std::endl;
  if (func_graph == nullptr) {
    std::cout << "func_graph is nullptr!" << std::endl;
    return false;
  }
  auto node_list = api::FuncGraph::TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (!api::utils::isa<api::CNode>(node)) {
      continue;
    }
    if (!CheckPrimitiveTypeTutorial(node, mindspore::api::MakeShared<mindspore::ops::Softmax>())) {
      continue;
    }
    if (!CreateCustomOp(func_graph, node)) {
      std::cout << "Create custome_cnode failed!" << std::endl;
      return false;
    }
  }
  return true;
}
}  // namespace opt

namespace lite {
// register customed Pass
using mindspore::registry::POSITION_ASCEND;
REG_PASS(PassTutorial, opt::PassTutorial)
REG_SCHEDULED_PASS(POSITION_ASCEND, {"PassTutorial"})
}  // namespace lite
}  // namespace mindspore
