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

#define USE_DEPRECATED_API
#include "tools/optimizer/fusion/flash_attention_tik_fusion.h"
#include "op_def/auto_generate/gen_lite_ops.h"
#include "mindspore/ops/op_def/nn_ops.h"
#include "infer/custom.h"
#include "nnacl_c/base/cast_base.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_p.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kInputIndex1 = 1;
constexpr size_t kInputIndex2 = 2;
constexpr size_t kInputIndex3 = 3;
constexpr size_t kPromptFlashAttentionInputSize = 13;

CNodePtr CreateMulNode(const FuncGraphPtr &func_graph, const CNodePtr &input_cnode, const float mul_scale) {
  MS_LOG(INFO) << "create mul node start.";
  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
  MS_CHECK_TRUE_RET(input_cnode != nullptr, nullptr);
  auto mul_op = std::make_unique<ops::Mul>();
  MS_CHECK_TRUE_RET(mul_op != nullptr, nullptr);
  auto mul_prim_c = mul_op->GetPrim();
  MS_CHECK_TRUE_RET(mul_prim_c != nullptr, nullptr);
  auto mul_scale_fp16 = Float32ToFloat16_(mul_scale);
  auto scale_node =
    BuildFloat16ValueParameterNode(func_graph, mul_scale_fp16, input_cnode->fullname_with_scope() + "_scale", False);
  if (scale_node == nullptr) {
    MS_LOG(ERROR) << "scale_node is nullptr!";
    return nullptr;
  }
  auto cnode = func_graph->NewCNode(mul_prim_c, {input_cnode, scale_node});
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "cnode is nullptr!";
    return nullptr;
  }
  cnode->set_fullname_with_scope(cnode->fullname_with_scope() + "_mul");
  if (input_cnode->abstract() != nullptr) {
    cnode->set_abstract(input_cnode->abstract()->Clone());
  }
  MS_LOG(INFO) << "create mul node end.";
  return cnode;
}
}  // namespace

bool FlashAttentionTikPass::Run(const FuncGraphPtr &func_graph) {
  MS_LOG(INFO) << "FlashAttentionTikPass run.";
  MS_CHECK_TRUE_RET(func_graph != nullptr, false);
  auto node_list = TopoSort(func_graph->get_return());
  auto manager = Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr!";
    return false;
  }
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    if (!CheckPrimitiveType(node, prim::kPrimPromptFlashAttention)) {
      MS_LOG(DEBUG) << "node is not PromptFlashAttention.";
      continue;
    }
    auto fa_cnode = node->cast<CNodePtr>();
    if (fa_cnode == nullptr) {
      MS_LOG(WARNING) << "fa_cnode is nullptr";
      continue;
    }
    if (fa_cnode->inputs().size() != kPromptFlashAttentionInputSize) {
      MS_LOG(WARNING) << "fa_cnode input size must be " << kPromptFlashAttentionInputSize << ", but get "
                      << fa_cnode->inputs().size();
      continue;
    }
    auto prim = std::make_unique<ops::Custom>();
    if (prim == nullptr) {
      MS_LOG(WARNING) << "prim is nullptr";
      continue;
    }
    std::vector<std::string> input_name = {"q", "k", "v"};  // the input of q here is (q * scale_value)
    std::vector<std::string> output_name = {"y"};
    prim->AddAttr("input_names", api::MakeValue(input_name));
    prim->AddAttr("output_names", api::MakeValue(output_name));
    prim->set_type("FlashAttentionTik");
    prim->AddAttr("reg_op_name", api::MakeValue("FlashAttentionTik"));

    auto pfa_prim = GetValueNode<PrimitivePtr>(fa_cnode->input(0));
    if (pfa_prim == nullptr) {
      MS_LOG(WARNING) << "pfa_prim is nullptr";
      continue;
    }
    auto scale_value_ptr = pfa_prim->GetAttr("scale_value");
    if (scale_value_ptr == nullptr) {
      MS_LOG(WARNING) << "scale_value_ptr is nullptr";
      continue;
    }
    if (!scale_value_ptr->isa<FP32Imm>()) {
      MS_LOG(WARNING) << "scale_value_ptr value dtype is not float";
      continue;
    }
    float scale_value = -1;
    scale_value = GetValue<float>(scale_value_ptr);
    if (scale_value < 0) {
      MS_LOG(WARNING) << "get scale_value failed";
      continue;
    }
    MS_LOG(INFO) << "scale value is " << scale_value;
    auto fa_tik_prim_c = prim->GetPrim();
    if (fa_tik_prim_c == nullptr) {
      MS_LOG(WARNING) << "fa_tik_prim_c is nullptr";
      continue;
    }
    auto q_cnode = fa_cnode->inputs()[kInputIndex1]->cast<CNodePtr>();
    auto q_adjust = CreateMulNode(func_graph, q_cnode, scale_value);
    if (q_adjust == nullptr) {
      MS_LOG(WARNING) << "q_adjust is nullptr";
      continue;
    }
    auto fa_tik_cnode = func_graph->NewCNode(
      fa_tik_prim_c, {q_adjust, fa_cnode->inputs()[kInputIndex2], fa_cnode->inputs()[kInputIndex3]});
    if (fa_tik_cnode == nullptr) {
      MS_LOG(WARNING) << "new fa_tik_cnode failed, cnode is nullptr";
      continue;
    }
    fa_tik_cnode->set_fullname_with_scope(fa_cnode->fullname_with_scope() + "_tik");
    if (fa_cnode->abstract() != nullptr) {
      fa_tik_cnode->set_abstract(fa_cnode->abstract()->Clone());
    }
    (void)manager->Replace(fa_cnode, fa_tik_cnode);
    MS_LOG(INFO) << "create FlashAttentionTik node end.";
  }
  return true;
}

}  // namespace opt
}  // namespace mindspore
