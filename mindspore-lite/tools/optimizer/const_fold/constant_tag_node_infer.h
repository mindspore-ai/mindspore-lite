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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_CONST_FOLD_CONSTANT_TAG_NODE_INFER_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_CONST_FOLD_CONSTANT_TAG_NODE_INFER_H_

#include <vector>
#include "src/tensor.h"
#include "tools/optimizer/graph/node_infershape.h"

using mindspore::converter::FmkType;
namespace mindspore {
namespace opt {
class ConstantTagNodeInfer : public NodeInferShape {
 public:
  explicit ConstantTagNodeInfer(FmkType fmk_type = converter::kFmkTypeMs, bool train_flag = false)
      : NodeInferShape(fmk_type, train_flag) {}
  ~ConstantTagNodeInfer() override = default;

 protected:
  int GetCNodeInputTensors(const CNodePtr &cnode, std::vector<TensorPtr> *inputs, converter::FmkType fmk_type,
                           bool train_flag, bool copy_data) override;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_CONST_FOLD_CONSTANT_TAG_NODE_INFER_H_
