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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_SRC_ACLNN_ACLNN_SUBGRAPH_PROCESSOR_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_SRC_ACLNN_ACLNN_SUBGRAPH_PROCESSOR_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "tools/converter/adapter/acl/src/acl_pass_impl.h"
#include "tools/converter/adapter/acl/src/aclnn/aclnn_shape_infer.h"

namespace mindspore {
namespace opt {

class AclnnSubgraphProcessor {
 public:
  explicit AclnnSubgraphProcessor(const std::shared_ptr<ConverterPara> &param,
                                  const lite::acl::AclModelOptionCfg &user_options_cfg);
  ~AclnnSubgraphProcessor() = default;

  STATUS ProcessCustomAclnnSubgraph(const CNodePtr &cnode, const FuncGraphPtr &func_graph);
  STATUS RunAclPassForSubgraph(const CNodePtr &cnode, const FuncGraphPtr &func_graph);
  STATUS ValidateSubgraphInputs(const CNodePtr &cnode, const FuncGraphPtr &func_graph);
  STATUS CollectSubgraphInputShapes(const CNodePtr &cnode, const std::vector<std::vector<int64_t>> &global_dim_groups,
                                    OrderShapes *shapes, std::vector<std::vector<int64_t>> *dim_groups);
  STATUS PreparePass(const FuncGraphPtr &func_graph, const OrderShapes &shapes,
                     const std::vector<std::vector<int64_t>> &dim_groups, std::unique_ptr<AclPassImpl> &acl_pass);
  STATUS GetGlobalInputDynShape(const std::string &input_name, std::vector<int64_t> *shapes,
                                std::vector<std::vector<int64_t>> *dim_groups);
  STATUS GetCNodeInputDynShape(const CNodePtr &cnode, std::vector<int64_t> *shape,
                               std::vector<std::vector<int64_t>> *dim_groups);
  STATUS ExtractSubgraphOM(const CNodePtr &custom_node, const CNodePtr &cnode, const FuncGraphPtr &func_graph);
  STATUS FindCustomAscend(const FuncGraphPtr &graph, std::vector<CNodePtr> *out);

 private:
  std::shared_ptr<ConverterPara> param_;
  lite::acl::AclModelOptionCfg user_options_cfg_;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_SRC_ACLNN_ACLNN_SUBGRAPH_PROCESSOR_H_
