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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_CONST_FOLD_CONSTANT_TAG_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_CONST_FOLD_CONSTANT_TAG_H_

#include "tools/optimizer/const_fold/fold_along_infershape.h"
#include "tools/optimizer/const_fold/constant_tag_node_infer.h"

namespace mindspore {
namespace opt {
class ConstantTag : public ConstFoldAlongInferShape {
 public:
  explicit ConstantTag(FmkType fmk_type = converter::kFmkTypeMs, bool train_flag = false)
      : ConstFoldAlongInferShape(fmk_type, train_flag, "ConstantTag") {}
  ~ConstantTag() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 protected:
  STATUS PostProcess(const FuncGraphPtr &func_graph, const CNodePtr &cnode) override;
  std::shared_ptr<NodeInferShape> CreateNodeInferShape() override;
  bool CheckAllConstInput(const FuncGraphPtr &func_graph, const CNodePtr &cnode) override;

  STATUS UpdateDynamicShapeAttr(const CNodePtr &cnode);
  STATUS ConvertAbstractToValue(const std::shared_ptr<abstract::AbstractBase> &abstract, ValuePtr *ret);
  STATUS UpdateShapeValueAttr(const CNodePtr &cnode);
  bool CheckCanFold(const FuncGraphPtr &func_graph, const CNodePtr &cnode);
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_CONST_FOLD_CONSTANT_TAG_H_
