/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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
#include "tools/optimizer/graph/adjust_ascend_quant_pass.h"
#include <memory>
#include <vector>
#include "mindspore/ops/op_def/lite_ops.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/converter/quantizer/quant_param_holder.h"
#include "tools/converter/quantizer/insert_quant_node_manager.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"

namespace mindspore {
namespace opt {
namespace {
bool AdjustAscendQuant(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, false);
  auto graph_manager = func_graph->manager();
  MS_CHECK_TRUE_RET(graph_manager != nullptr, false);
  MS_CHECK_TRUE_RET(cnode != nullptr, false);
  MS_CHECK_TRUE_RET(cnode->abstract() != nullptr, false);
  // for MatMul/Conv size(prim, input1, input2) = 3 or size(prim, input1, input2, bias) = 4
  if (cnode->size() < kInputSizeThree || cnode->size() > kInputSizeFour) {
    MS_LOG(ERROR) << "The number of inputs of cnode can only be 3 or 4!, but get " << cnode->size();
    return false;
  }
  auto quant_param_holder = mindspore::lite::GetCNodeQuantHolder(cnode);
  MS_CHECK_TRUE_RET(quant_param_holder != nullptr, false);
  if (!quant_param_holder->IsInputQuantParamsInited()) {
    MS_LOG(INFO) << "InputQuantParamsInited is false, this node is " << cnode->fullname_with_scope();
    return true;
  }
  auto quant_params_vec = quant_param_holder->get_input_quant_params();
  lite::quant::InsertQuantNodeManager insert_node_manager;
  auto input_2_node = cnode->input(kInputIndexTwo);
  MS_CHECK_TRUE_RET(input_2_node != nullptr, false);
  if (quant_params_vec.empty()) {
    MS_LOG(INFO) << "This node has no quantization parameter. Skip it. node name: " << cnode->fullname_with_scope();
    return true;
  } else if (quant_params_vec.size() == kInputSizeTwo && input_2_node->isa<mindspore::CNode>()) {
    MS_LOG(INFO) << "Start do double per_tensor(A&W) pass.";
    auto ret = insert_node_manager.InsertAscendQuantNode(func_graph, cnode);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Insert AscendQuant node failed! cnode name: " << cnode->fullname_with_scope();
      return false;
    }
    return true;
  } else if (quant_params_vec.size() == kInputSizeTwo && input_2_node->isa<mindspore::Parameter>()) {
    MS_LOG(INFO) << "Start do per_tensor(A) + per_channel(W) pass. The weight is already of the int8 type, don't need "
                    "to insert AscendQuant node.";
    auto ret = insert_node_manager.InsertAscendQuantNode(func_graph, cnode, kInputIndexOne);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Insert AscendQuant node failed, cnode name: " << cnode->fullname_with_scope();
      return false;
    }
    return true;
  } else {
    MS_LOG(ERROR) << "Dont support! The number of quantization parameters is " << quant_params_vec.size();
    return False;
  }
}
}  // namespace

bool AdjustAscendQunatPass::Run(const FuncGraphPtr &func_graph) {
  if (lite::ConverterInnerContext::GetInstance()->GetTargetDevice() != "Ascend") {
    MS_LOG(INFO) << "Adjust ascend quant pass is only supported on the Ascend backend.";
    return true;
  }
  MS_CHECK_TRUE_RET(func_graph != nullptr, false);
  MS_LOG(INFO) << "AdjustAscendQunatPass start.";
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    if (!opt::CheckPrimitiveType(node, prim::kPrimMatMulFusion) &&
        !opt::CheckPrimitiveType(node, prim::kPrimConv2DFusion)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_CHECK_TRUE_RET(cnode != nullptr, false);
    if (!AdjustAscendQuant(func_graph, cnode)) {
      MS_LOG(ERROR) << "This node run AdjustAscendQunatPass failed! node name is: " << cnode->fullname_with_scope();
      return false;
    }
    MS_LOG(INFO) << "This node run AdjustAscendQunatPass success, node name is: " << cnode->fullname_with_scope();
  }
  MS_LOG(INFO) << "AdjustAscendQunatPass end.";
  return true;
}
}  // namespace opt
}  // namespace mindspore
