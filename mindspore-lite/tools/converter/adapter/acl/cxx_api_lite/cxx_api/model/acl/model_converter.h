/**
 * Copyright 2020-2026 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_CXXAPI_SESSION_ACL_MODEL_CONVERTER_H
#define MINDSPORE_CCSRC_CXXAPI_SESSION_ACL_MODEL_CONVERTER_H
#include <string>
#include <map>
#include <memory>
#include <vector>
#include "include/api/types.h"
#include "include/api/status.h"
#include "ir/func_graph.h"
#include "tools/converter/adapter/acl/backend/ge_backend/graph_ir/types.h"
#include "ge/ge_ir_build.h"
#include "cxx_api/model/acl/acl_model_options.h"

namespace mindspore {
class MS_API ModelConverter {
 public:
  ModelConverter() : options_() {}
  ~ModelConverter() = default;

  Buffer LoadMindIR(const FuncGraphPtr &func_graph);

  void set_options(const std::weak_ptr<AclModelOptions> &options) { options_ = options; }
  void set_update_graph(const FuncGraphPtr &update_graph) { update_func_graph_ = update_graph; }
  void set_variable_node_names(const std::vector<string> variable_node_names) {
    variable_node_names_ = variable_node_names;
  }

 private:
  backend::ge_backend::DfGraphPtr ConvertFuncGraphToAIR(const FuncGraphPtr &anf_graph) const;
  Buffer BuildAirModel(const backend::ge_backend::DfGraphPtr &graph,
                       const std::map<std::string, std::string> &init_options,
                       const std::map<std::string, std::string> &build_options) const;
  Buffer LoadAscendIRInner(const Buffer &model_data);
  bool CompileBundleModel(const backend::ge_backend::DfGraphPtr &graph,
                          const std::map<std::string, std::string> &build_options, ge::ModelBufferData *model) const;
  Buffer LoadBufferFromSavedOm(const std::string &saved_om_path, const ge::ModelBufferData &model) const;
  Status SaveModel(const ge::ModelBufferData &model, bool is_bundle, std::string *saved_om_path = nullptr) const;
  Status CreateUpdateGraph(const std::vector<std::string> &const_names, const std::vector<AbstractBasePtr> &abstarcts,
                           backend::ge_backend::DfGraphPtr *df_graph) const;

  std::weak_ptr<AclModelOptions> options_;
  std::vector<std::string> variable_node_names_;
  FuncGraphPtr update_func_graph_ = nullptr;
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_CXXAPI_SESSION_ACL_MODEL_CONVERTER_H
