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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_FFN_CUSTOM_PASS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_FFN_CUSTOM_PASS_H_

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <unordered_map>
#include "tools/optimizer/common/multiple_pattern_process_pass.h"

namespace mindspore {
namespace opt {
class FFNCustomPass : public MultiplePatternProcessPass {
 public:
  typedef std::map<std::string, std::map<std::string, std::string>> op_attrs_type;
  explicit FFNCustomPass(const op_attrs_type &op_attrs_map, const std::string &name = "FFNCustomPass",
                         bool multigraph = true)
      : MultiplePatternProcessPass(name, multigraph) {
    op_attrs_map_ = op_attrs_map;
  }

  ~FFNCustomPass() override = default;

  std::unordered_map<std::string, VectorRef> DefinePatterns() const override;

  AnfNodePtr Process(const std::string &, const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  bool CheckInputShpae(const CNodePtr &input_x, const AnfNodePtr &weight1, const AnfNodePtr &weight2) const;
  CNodePtr CreateFFNFusionNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &equiv) const;

  op_attrs_type op_attrs_map_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_FFN_CUSTOM_PASS_H_
