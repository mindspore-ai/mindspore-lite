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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_ADD_VARIABLE_NODE_PASS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_ADD_VARIABLE_NODE_PASS_H_
#include <string>
#include <map>
#include <unordered_map>
#include <vector>
#include <memory>
#include <set>
#include "include/backend/optimizer/pass.h"
#include "tools/converter/cxx_api/converter_para.h"
#include "include/errorcode.h"

namespace mindspore {
namespace opt {
class InsertVariableNodePass : public Pass {
 public:
  explicit InsertVariableNodePass(const std::shared_ptr<ConverterPara> &param) : Pass("InsertVariableNodePass") {
    param_ = param;
  }
  ~InsertVariableNodePass() override = default;
  bool Run(const FuncGraphPtr &graph) override;

 private:
  lite::STATUS BuildVariableNode(FuncGraphPtr func_graph);
  template <typename T>
  ParameterPtr BuildZeroVecNDParameterNode(const FuncGraphPtr &anf_graph, ShapeVector weight_shape,
                                           const std::string &node_name, T value, TypeId dtype);
  void InitWeightParam(std::string *variable_weights_file, int32_t *max_weight_batch);
  FuncGraphPtr CreateUpdateGraph(const std::vector<std::string> &const_names,
                                 const std::vector<AbstractBasePtr> &abstarcts);
  lite::STATUS RecordParameterVariableName(const FuncGraphPtr &func_graph, const ParameterPtr &para_node,
                                           const string &search_key,
                                           std::unordered_map<std::string, std::string> *node_name_map,
                                           std::unordered_map<std::string, AbstractBasePtr> *node_abstract_map);
  std::shared_ptr<ConverterPara> param_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_ADD_VARIABLE_NODE_PASS_H_
