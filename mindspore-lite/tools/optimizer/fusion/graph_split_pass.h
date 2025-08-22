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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_GRAPH_SPLIT_PASS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_GRAPH_SPLIT_PASS_H_
#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include "tools/optimizer/common/multiple_pattern_process_pass.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/converter/cxx_api/converter_para.h"

namespace mindspore {
namespace opt {
class GraphSplitPass : public Pass {
 public:
  explicit GraphSplitPass(const std::shared_ptr<ConverterPara> &param) : Pass("GraphSplitPass"), param_(param) {}
  ~GraphSplitPass() override = default;
  bool Run(const FuncGraphPtr &graph);

 private:
  std::shared_ptr<ConverterPara> param_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_GRAPH_SPLIT_PASS_H_
