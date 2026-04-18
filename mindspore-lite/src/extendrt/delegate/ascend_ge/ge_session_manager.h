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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_GE_SESSION_MANAGER_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_GE_SESSION_MANAGER_H_

#include <memory>
#include <vector>
#include <string>
#include "extendrt/delegate/ascend_ge/ge_memory_manager.h"
#include "extendrt/delegate/ascend_ge/ge_context_manager.h"
#include "tools/converter/adapter/acl/backend/ge_backend/graph_ir/types.h"
#include "tools/converter/adapter/acl/backend/ge_backend/graph_ir/utils.h"

namespace mindspore {
struct RefDataInfo {
  std::string name;
  ShapeVector shape;
  ShapeVector dyn_shape;
  TypeId dtype = kTypeUnknown;
  tensor::TensorPtr host_data = nullptr;  // will be released after device tensor allocated
  size_t offset = 0;
  size_t size = 0;
  GeTensor ge_tensor;
};

struct GeSessionContext {
  std::weak_ptr<ge::Session> ge_session;
  std::map<std::string, std::string> session_options;
  std::set<std::string> session_variables;
  std::map<std::string, RefDataInfo> ref_data_map_;
  std::weak_ptr<GeMemoryManager> memory_manager;
  std::weak_ptr<GeContextManager> context_manager;
  std::vector<void *> ref_data_device_memories;
  void *feature_memory = nullptr;
  size_t feature_size = 0;
  std::map<uint32_t, size_t> feature_graph_ids;
};

class GeSessionManager {
 public:
  static std::shared_ptr<ge::Session> CreateGeSession(int64_t session_id,
                                                      const std::map<std::string, std::string> &session_options);
  // return new Variables not in session
  static std::set<std::string> UpdateSessionVariables(int64_t session_id,
                                                      const std::vector<std::string> &graph_variables);
  static void TryReleaseGeSessionContext(int64_t session_id);

  static std::shared_ptr<GeSessionContext> GetGeSessionContext(int64_t session_id);

 private:
  static std::map<int64_t, std::shared_ptr<GeSessionContext>> ge_session_map_;
  static std::mutex session_mutex_;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_GE_SESSION_MANAGER_H_
