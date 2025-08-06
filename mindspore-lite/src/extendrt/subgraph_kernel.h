/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_EXTENDRT_SUBGRAPH_KERNEL_H
#define MINDSPORE_LITE_EXTENDRT_SUBGRAPH_KERNEL_H
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "common/kernel.h"
#include "ir/func_graph.h"
#include "runtime/hardware/device_context.h"
#include "common/common_utils.h"
#include "src/extendrt/session/lite_graph_executor.h"

namespace mindspore::kernel {
class SubgraphKernel {
 public:
  SubgraphKernel(FuncGraphPtr subgraph, std::shared_ptr<LiteGraphExecutor> executor)
      : subgraph_(subgraph), executor_(executor) {}
  virtual ~SubgraphKernel() = default;
  bool Init(const std::vector<MSTensor *> & /* inputs */, const std::vector<MSTensor *> & /* outputs */);

  int Resize(const std::vector<MSTensor *> &inputs, const std::vector<MSTensor *> &outputs);

  bool Launch(const std::vector<MSTensor *> &inputs, const std::vector<MSTensor *> &workspace,
              const std::vector<MSTensor *> &outputs, void *stream_ptr);
  std::vector<KernelAttr> GetOpSupport() { return {}; }

 protected:
  FuncGraphPtr subgraph_;
  std::shared_ptr<LiteGraphExecutor> executor_;
};
}  // namespace mindspore::kernel
#endif
