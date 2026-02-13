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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_MIXED_ACLNN_PASS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_MIXED_ACLNN_PASS_H_

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "common/common.h"
#include "mindspore/core/include/ir/graph_utils.h"
#include "tools/converter/cxx_api/converter_para.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/optimizer/common/multiple_pattern_process_pass.h"

namespace mindspore {
namespace opt {

struct Edge {
  AnfNodePtr source;
  CNodePtr target;
  int64_t target_pos;
};

struct SubgraphInfo {
  FuncGraphPtr graph;               // new graph
  std::vector<AnfNodePtr> outputs;  // nodes in original graph
  std::unordered_map<AnfNodePtr, AnfNodePtr> old_to_new_map;
  std::unordered_map<AnfNodePtr, AnfNodePtr> new_to_old_map;
};

class SubgraphInfoBuilder {
 public:
  SubgraphInfoBuilder() = default;
  ~SubgraphInfoBuilder() = default;

  STATUS Build(const FuncGraphPtr &original_graph, const std::set<std::string> &split_op_names,
               std::vector<SubgraphInfo> *out_subgraphs, std::unordered_map<AnfNodePtr, size_t> *out_node_to_graph_id);

 private:
  STATUS SplitGraph(const FuncGraphPtr &original_graph, const std::set<std::string> &split_op_names,
                    std::vector<std::vector<AnfNodePtr>> *out_subgraphs,
                    std::unordered_map<AnfNodePtr, size_t> *out_node_to_graph_id);
  STATUS BuildAllSubgraphInfo(const FuncGraphPtr &original_graph,
                              const std::unordered_map<AnfNodePtr, size_t> &node_to_graph_id,
                              const std::vector<std::vector<AnfNodePtr>> &subgraph_nodes,
                              std::vector<SubgraphInfo> *out);
  STATUS BuildSubGraphInfo(const FuncGraphPtr &original_graph,
                           const std::unordered_map<AnfNodePtr, size_t> &node_graph_id_map,
                           const std::vector<AnfNodePtr> &subgraph_node, size_t graph_id, SubgraphInfo *graph_info,
                           std::map<int64_t, std::unordered_set<CNodePtr>> *extra_outputs);
  STATUS BuildSubGraphOutputs(const FuncGraphPtr &original_graph,
                              const std::unordered_map<AnfNodePtr, size_t> &node_graph_id_map,
                              const std::vector<AnfNodePtr> &subgraph_node,
                              const std::unordered_set<CNodePtr> &required_outputs, SubgraphInfo *graph_info);
  STATUS GetSuccessors(const FuncGraphPtr &graph, const AnfNodePtr &predecessor, std::vector<Edge> *successor);
  STATUS ProcessInputs(const CNodePtr &old_cnode, const size_t &input_id, const FuncGraphPtr &subgraph,
                       const std::unordered_map<AnfNodePtr, AnfNodePtr> &node_map, std::vector<AnfNodePtr> *new_inputs);
  STATUS BFSShapeValueCNode(const CNodePtr &cnode, std::vector<CNodePtr> *out);
  STATUS CloneCNode(const CNodePtr &old_cnode, const FuncGraphPtr &graph,
                    std::unordered_map<AnfNodePtr, AnfNodePtr> *old_to_new_map,
                    std::unordered_map<AnfNodePtr, AnfNodePtr> *new_to_old_map);
  STATUS HandleShapeValueInputs(const FuncGraphPtr &original_graph, const FuncGraphPtr &subgraph,
                                std::unordered_map<AnfNodePtr, AnfNodePtr> *old_to_new_map,
                                std::unordered_map<AnfNodePtr, AnfNodePtr> *new_to_old_map, size_t graph_id);
  STATUS ProcessExtraOutputs(const std::unordered_map<AnfNodePtr, size_t> &node_graph_id_map,
                             const FuncGraphPtr &subgraph,
                             const std::unordered_map<AnfNodePtr, AnfNodePtr> &new_to_old_map,
                             std::map<int64_t, std::unordered_set<CNodePtr>> *extra_outputs);
};

class MixedAclnnPass : public Pass {
 public:
  explicit MixedAclnnPass(const std::shared_ptr<ConverterPara> &param) : Pass("MixedAclnnPass"), param_(param) {}
  ~MixedAclnnPass() override = default;
  bool Run(const FuncGraphPtr &graph) override;

 private:
  const std::shared_ptr<ConverterPara> param_;

  STATUS TransposeValueVec(const std::vector<std::vector<ValuePtr>> &matrix, std::vector<std::vector<ValuePtr>> *ret);
  STATUS CreateCustomAclnnSubgraph(const FuncGraphPtr &original_graph,
                                   const std::unordered_set<AnfNodePtr> &begin_nodes,
                                   std::vector<SubgraphInfo> &subgraphs,
                                   std::unordered_map<AnfNodePtr, size_t> &node_to_graph_id);

  STATUS CollectSubgraphInputs(const SubgraphInfo &subgraph, const std::unordered_set<AnfNodePtr> &begin_nodes,
                               const std::unordered_map<AnfNodePtr, size_t> &node_to_graph_id,
                               const std::vector<SubgraphInfo> &subgraphs,
                               const std::unordered_map<size_t, std::vector<AnfNodePtr>> &custom_outputs,
                               std::vector<AnfNodePtr> *new_inputs);
  STATUS ReplaceWithSingleOutput(const FuncGraphPtr &original_graph, const CNodePtr &output,
                                 const AnfNodePtr &new_custom_node, const FuncGraphManagerPtr &manager);
  STATUS ReplaceWithMultipleOutputs(const FuncGraphPtr &original_graph, const SubgraphInfo &subgraph,
                                    const AnfNodePtr &new_custom_node, const FuncGraphManagerPtr &manager,
                                    AnfNodePtrList *outputs);
};

}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_MIXED_ACLNN_PASS_H_
