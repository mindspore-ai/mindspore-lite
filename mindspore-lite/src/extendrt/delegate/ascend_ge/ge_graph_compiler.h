/**
 * Copyright 2025-2026 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_GE_GRAPH_COMPILER_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_GE_GRAPH_COMPILER_H_

#include <memory>
#include <string>
#include "tools/converter/adapter/acl/backend/ge_backend/graph_ir/types.h"
#include "tools/converter/adapter/acl/backend/ge_backend/graph_ir/utils.h"
#include "extendrt/delegate/ascend_ge/ge_options_container.h"

namespace mindspore {
using GeSessionPtr = std::shared_ptr<ge::Session>;

struct GeSessionInfo {
  GeSessionPtr session_{nullptr};
  int64_t session_id_{-1};
  std::vector<uint32_t> graph_ids_;
  backend::ge_backend::DfGraphPtr df_ptr_;
};

class GeGraphCompiler {
 public:
  GeGraphCompiler() = default;
  ~GeGraphCompiler() = default;

  bool CompileGraph(const FuncGraphPtr &graph, GeSessionInfo *ge_session_info,
                    const GeOptionsContainer &ge_options_container);
  bool ReCompileGraph(GeSessionInfo *ge_session_info, const GeOptionsContainer &ge_options_container,
                      uint32_t *graph_id);

 private:
  backend::ge_backend::DfGraphPtr ToGeGraph(const FuncGraphPtr &graph, GeSessionInfo *ge_session_info,
                                            const std::string &graph_key);
  bool ProcessGeInitGraph(GeSessionInfo *ge_session_info, backend::ge_backend::DfGraphPtr init_graph,
                          const backend::ge_backend::TensorOrderMap &params,
                          const std::vector<std::string> &init_data_names);
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_GE_GRAPH_COMPILER_H_
