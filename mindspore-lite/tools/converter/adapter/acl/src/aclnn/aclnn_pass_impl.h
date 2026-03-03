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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_SRC_ACLNN_ACLNN_PASS_IMPL_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_SRC_ACLNN_ACLNN_PASS_IMPL_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "tools/converter/adapter/acl/src/acl_pass_impl.h"
#include "tools/converter/adapter/acl/src/aclnn/aclnn_shape_infer.h"
#include "tools/converter/adapter/acl/src/aclnn/aclnn_graph_builder.h"

namespace mindspore {
namespace opt {

class AclnnPassImpl : public AclPassImpl {
 public:
  explicit AclnnPassImpl(const std::shared_ptr<ConverterPara> &param, bool is_subgraph = false);
  ~AclnnPassImpl();

  bool Run(const FuncGraphPtr &func_graph) override;

 protected:
  STATUS BuildGraph(const FuncGraphPtr &func_graph) override;

 private:
  bool is_subgraph_;
  std::unique_ptr<AclnnShapeInfer> shape_infer_;
  std::unique_ptr<AclnnGraphBuilder> graph_builder_;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_SRC_ACLNN_ACLNN_PASS_IMPL_H_
