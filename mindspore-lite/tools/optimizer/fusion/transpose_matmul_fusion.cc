/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "tools/optimizer/fusion/transpose_matmul_fusion.h"
#include <memory>
#include <vector>
#include "mindspore/ops/op_def/lite_ops.h"
#include "mindspore/ops/op_def/array_ops.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/optimizer/common/format_utils.h"
#include "nnacl_c/op_base.h"
#include "mindspore/ops/op_def/auto_generate/gen_lite_ops.h"
#include "ops_utils/op_utils.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_t.h"
#include "mindspore/core/include/ir/graph_utils.h"

namespace mindspore {
namespace opt {
namespace {
inline const std::vector<int> kMatMulTransPerm1 = {0, 1, 3, 2};
inline const std::vector<int> kMatMulTransPerm2 = {0, 2, 1};
inline const std::vector<int> kMatMulTransPerm3 = {1, 0};

bool CheckInputTransposeNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode, bool *indices, size_t indices_len) {
  MS_ASSERT(cnode != nullptr && indices != nullptr);
  MS_ASSERT(indices_len / sizeof(bool) == DIMENSION_2D);
  if (cnode->size() != kInputSizeThree) {
    return false;
  }
  for (size_t i = 1; i < cnode->size(); ++i) {
    indices[i - 1] = false;
    if (CheckPrimitiveType(cnode->input(i), prim::kPrimTranspose)) {
      std::vector<int> perm;
      auto trans_cnode = cnode->input(i)->cast<CNodePtr>();
      MS_CHECK_TRUE_RET(trans_cnode != nullptr, false);
      if (GetTransposePerm(trans_cnode, &perm) != lite::RET_OK) {
        MS_LOG(ERROR) << "get transpose perm failed.";
        return false;
      }
      if (perm == kMatMulTransPerm1 || perm == kMatMulTransPerm2 || perm == kMatMulTransPerm3) {
        indices[i - 1] = true;
      }
    }
  }
  if (!indices[FIRST_INPUT] && !indices[SECOND_INPUT]) {
    return false;
  }
  return true;
}
}  // namespace

bool TransposeMatMulFusion::FuseTransposeInput(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                               size_t input_index, size_t cnode_input_index) {
  MS_CHECK_TRUE_RET(func_graph != nullptr && cnode != nullptr, false);
  auto matmul_prim = ops::GetOperator<mindspore::ops::MatMul>(cnode->input(0));
  MS_CHECK_TRUE_RET(matmul_prim != nullptr, false);
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_RET(manager != nullptr, false);

  if (input_index == FIRST_INPUT) {
    if (matmul_prim->GetAttr(ops::kTransposeA) == nullptr) {
      matmul_prim->set_transpose_a(true);
    } else {
      auto org_transpose_a = matmul_prim->get_transpose_a();
      matmul_prim->set_transpose_a(!org_transpose_a);
    }
  } else {
    if (matmul_prim->GetAttr(ops::kTransposeB) == nullptr) {
      matmul_prim->set_transpose_b(true);
    } else {
      auto org_transpose_b = matmul_prim->get_transpose_b();
      matmul_prim->set_transpose_b(!org_transpose_b);
    }
  }
  auto trans_cnode = cnode->input(cnode_input_index)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(trans_cnode != nullptr, false);
  MS_CHECK_TRUE_RET(trans_cnode->size() == kInputSizeThree, false);
  auto pre_node = trans_cnode->input(SECOND_INPUT);
  (void)manager->SetEdge(cnode, cnode_input_index, pre_node);
  return true;
}

bool TransposeMatMulFusion::Run(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    MS_CHECK_TRUE_RET(node != nullptr, false);
    if (!utils::isa<CNode>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (!CheckPrimitiveType(node, prim::kPrimMatMulFusion)) {
      continue;
    }
    if (IsMarkedTrainOp(cnode)) {
      return false;
    }

    bool indices_need_fuse[DIMENSION_2D] = {false, false};
    if (!CheckInputTransposeNode(func_graph, cnode, indices_need_fuse, sizeof(bool) * DIMENSION_2D)) {
      continue;
    }

    if (indices_need_fuse[FIRST_INPUT]) {
      MS_CHECK_TRUE_RET(FuseTransposeInput(func_graph, cnode, FIRST_INPUT, SECOND_INPUT), false);
    }
    if (indices_need_fuse[SECOND_INPUT]) {
      MS_CHECK_TRUE_RET(FuseTransposeInput(func_graph, cnode, SECOND_INPUT, THIRD_INPUT), false);
    }
  }
  return false;
}
}  // namespace opt
}  // namespace mindspore
