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
  lite::STATUS BuildVariableNode(const std::shared_ptr<ConverterPara> &param, FuncGraphPtr func_graph,
                                 std::vector<std::string> *const_names);
  lite::STATUS InsertVariableNodeForMatmul(const AnfNodePtr &node, const CNodePtr &cnode,
                                           const FuncGraphPtr &func_graph, const std::vector<int> &up_shape,
                                           std::unordered_map<std::string, std::string> *node_name_map, bool has_alpha,
                                           int max_weight_batch);
  lite::STATUS InsertVariableNodeForConv(const AnfNodePtr &node, const CNodePtr &cnode, const FuncGraphPtr &func_graph,
                                         const std::vector<int> &up_shape,
                                         std::unordered_map<std::string, std::string> *node_name_map, bool has_alpha,
                                         int max_weight_batch);
  lite::STATUS ParseInsertNode(std::string file_path, std::map<std::string, std::vector<int>> *variable_nodes,
                               std::unordered_map<std::string, std::string> *node_name_map,
                               std::vector<std::string> *node_name_list, bool *has_alpha);
  lite::STATUS ParseShapeStr(std::string shape_str, std::vector<int> *shape);
  lite::STATUS InsertVariableAddNode(const CNodePtr &cnode, const FuncGraphPtr &func_graph, const bool &is_matmul,
                                     std::unordered_map<std::string, std::string> *node_name_map);
  lite::STATUS CheckOnlyReplace(CNodePtr cnode, const std::vector<int> &para_shape, const bool &is_matmul,
                                bool *compare_res);
  lite::STATUS RecordVariableName(const FuncGraphPtr &func_graph, const CNodePtr &cnode, const string &search_key,
                                  bool is_matmul, std::unordered_map<std::string, std::string> *node_name_map);
  template <typename T>
  ParameterPtr BuildFloat16ZeroVecNDParameterNode(const FuncGraphPtr &anf_graph, ShapeVector weight_shape,
                                                  const std::string &node_name, T value, TypeId dtype);
  void InitWeightParam(const std::shared_ptr<ConverterPara> &param, std::string *variable_weights_file,
                       int32_t *max_weight_batch);

  std::shared_ptr<ConverterPara> param_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_ADD_VARIABLE_NODE_PASS_H_
