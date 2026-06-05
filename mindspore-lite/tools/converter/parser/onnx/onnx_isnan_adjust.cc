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
#include "tools/converter/parser/onnx/onnx_isnan_adjust.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "nnacl_c/op_base.h"
#include "primitive/math_ops.h"

namespace mindspore::lite {
bool OnnxIsNanAdjust::Adjust(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, false);
  auto manager = Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr.";
    return false;
  }
  auto cnodes = func_graph->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    if (!opt::CheckPrimitiveType(cnode, prim::kPrimWhere)) {
      continue;
    }
    if (cnode->size() != opt::kInputSizeFour) {
      continue;
    }
    auto condition_input = cnode->input(1);
    if (!opt::CheckPrimitiveType(condition_input, prim::kPrimIsNan)) {
      continue;
    }
    auto value_if_false = cnode->input(opt::kInputIndexThree);
    if (!manager->Replace(cnode, value_if_false)) {
      MS_LOG(ERROR) << "replace Where(IsNaN) with value_if_false failed for node " << cnode->fullname_with_scope();
      return false;
    }
  }
  return true;
}
}  // namespace mindspore::lite
