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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_ADD_STREAM_LABEL_PASS_H
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_ADD_STREAM_LABEL_PASS_H

#include <memory>
#include <set>
#include <string>
#include <map>
#include "include/backend/optimizer/pass.h"
#include "tools/converter/cxx_api/converter_para.h"

namespace mindspore {
namespace opt {
class AddStreamLabelPass : public Pass {
 public:
  explicit AddStreamLabelPass(const std::shared_ptr<ConverterPara> &param) : Pass("AddStreamLabelPass") {
    param_ = param;
  }
  ~AddStreamLabelPass() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  Status AddStreamLabel(const FuncGraphPtr &func_graph);
  Status ParseStreamLable();

  // node name , stream label
  std::map<std::string, std::string> node_to_label_map_;
  Status GetNodeNames(const FuncGraphPtr &func_graph);
  std::shared_ptr<ConverterPara> param_ = nullptr;
  std::set<std::string> all_node_names_ = {};
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_ADJUST_MATMUL_PASS_H
