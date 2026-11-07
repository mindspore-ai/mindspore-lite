/**
 * Copyright 2022-2026 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_COMMON_NODE_PASS_EXTENDS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_COMMON_NODE_PASS_EXTENDS_H_

#include <memory>
#include <string>
#include <vector>
#include "tools/converter/ms_depend/node_pass.h"

namespace mindspore {
namespace opt {
class LiteNodePass : public NodePass {
 public:
  explicit LiteNodePass(const std::string &name) : NodePass(name) {}
  ~LiteNodePass() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;
  virtual AnfNodePtr Run(const FuncGraphPtr &func_graph, const AnfNodePtr &node) = 0;

 protected:
  // Fusion tracing policy, chosen once at construction:
  //   kAutoRoot   - the Run() exit carries the matched root's origin records onto the returned
  //                 survivor automatically (default);
  //   kAutoRegion - additionally, equiv-matched parts that lost their last user inside Process
  //                 were consumed by the fusion, so their records move to the survivor too;
  //   kManual     - multi-survivor / fan-out fusions trace explicitly inside Process.
  enum class TracePolicy { kAutoRoot, kAutoRegion, kManual };
  void SetTracePolicy(TracePolicy policy) { trace_policy_ = policy; }
  // Shared Run() exit for both pattern-pass families (LitePatternProcessPass /
  // MultiplePatternProcessPass): `ret` is the replacement the caller replaces `root` with, so
  // under kAutoRoot/kAutoRegion it absorbs the root's origin identity; under kAutoRegion the
  // vanished `matched_parts` (the equiv region) are absorbed as well. Does not touch the
  // Process contract or the replacement timing - the caller still performs the Replace.
  AnfNodePtr FinishFusionProcess(const FuncGraphPtr &func_graph, const AnfNodePtr &root, const AnfNodePtr &ret,
                                 const std::vector<AnfNodePtr> &matched_parts) const;

 private:
  TracePolicy trace_policy_ = TracePolicy::kAutoRoot;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_COMMON_NODE_PASS_EXTENDS_H_
