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
#include "tools/converter/parser/onnx/onnx_shape_adjust.h"
#include <vector>
#include "tools/optimizer/common/gllo_utils.h"
#include "nnacl_c/op_base.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"

namespace mindspore::lite {
bool OnnxShapeAdjust::Adjust(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, false);
  auto manager = Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr.";
    return false;
  }
  auto cnodes = func_graph->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    if (!opt::CheckPrimitiveType(cnode, prim::kPrimShape)) {
      continue;
    }
    auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    MS_CHECK_TRUE_RET(prim != nullptr, false);
    auto start_attr = prim->GetAttr("start");
    auto end_attr = prim->GetAttr("end");
    if (start_attr == nullptr || end_attr == nullptr) {
      continue;
    }
    auto start_val = GetValue<int64_t>(start_attr);
    auto end_val = GetValue<int64_t>(end_attr);
    std::vector<int> indices;
    for (int64_t i = start_val; i < end_val; ++i) {
      indices.push_back(static_cast<int>(i));
    }
    if (indices.empty()) {
      MS_LOG(WARNING) << "Shape node " << cnode->fullname_with_scope() << " has start=" << start_val
                      << ", end=" << end_val << " with empty range, skipping.";
      continue;
    }
    auto node_users = manager->node_users()[cnode];
    auto gather_node = opt::GenGatherNode(func_graph, cnode, indices, cnode->fullname_with_scope() + "_gather", {0});
    if (gather_node == nullptr) {
      MS_LOG(ERROR) << "create gather node failed for Shape node " << cnode->fullname_with_scope();
      return false;
    }
    if (cnode->abstract() != nullptr && gather_node->abstract() == nullptr) {
      gather_node->set_abstract(cnode->abstract()->Clone());
    }
    for (auto &node_user : node_users) {
      manager->SetEdge(node_user.first, node_user.second, gather_node);
    }
    prim->EraseAttr("start");
    prim->EraseAttr("end");
    MS_LOG(INFO) << "Replace non-standard Shape(start/end) with Gather for node " << cnode->fullname_with_scope()
                 << ", start=" << start_val << ", end=" << end_val << ".";
  }
  return true;
}
}  // namespace mindspore::lite
