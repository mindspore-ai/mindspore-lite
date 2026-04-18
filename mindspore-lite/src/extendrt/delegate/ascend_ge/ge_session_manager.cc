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

#include "extendrt/delegate/ascend_ge/ge_session_manager.h"
#include "src/common/common.h"

namespace mindspore {
std::map<int64_t, std::shared_ptr<GeSessionContext>> GeSessionManager::ge_session_map_;
std::mutex GeSessionManager::session_mutex_;

std::shared_ptr<ge::Session> GeSessionManager::CreateGeSession(
  int64_t session_id, const std::map<std::string, std::string> &session_options) {
  std::shared_ptr<ge::Session> ge_session = nullptr;
  if (session_id == lite::kUnkonwnSessionId) {
    ge_session = std::make_shared<ge::Session>(session_options);
    if (ge_session == nullptr) {
      MS_LOG(ERROR) << "Failed to create ge session";
      return nullptr;
    }
    MS_LOG(INFO) << "Create ge session successfully, which will not be shared with other graph";
    return ge_session;
  }
  std::lock_guard<std::mutex> lock(session_mutex_);
  auto session_item = ge_session_map_.find(session_id);
  if (session_item != ge_session_map_.end() && session_item->second != nullptr) {
    ge_session = session_item->second->ge_session.lock();
  }
  if (ge_session == nullptr) {
    ge_session = std::make_shared<ge::Session>(session_options);
    if (ge_session == nullptr) {
      MS_LOG(ERROR) << "Failed to create ge session";
      return nullptr;
    }
    auto session_context = std::make_shared<GeSessionContext>();
    if (session_context == nullptr) {
      MS_LOG(ERROR) << "Failed to create GeSessionContext";
      return nullptr;
    }
    session_context->ge_session = ge_session;
    session_context->session_options = session_options;
    ge_session_map_[session_id] = session_context;
    MS_LOG(INFO) << "Create ge session successfully, lite session id: " << session_id;
  } else {
    auto old_options = session_item->second->session_options;
    if (old_options != session_options) {
      MS_LOG(ERROR) << "Session options is not equal in diff config infos when models' weights are shared, last "
                       "session options != current session options, please check your session options.";
      return nullptr;
    }
    MS_LOG(INFO) << "Get ge session from session map, lite session id: " << session_id;
  }
  return ge_session;
}

std::shared_ptr<GeSessionContext> GeSessionManager::GetGeSessionContext(int64_t session_id) {
  std::lock_guard<std::mutex> lock(session_mutex_);
  auto session_item = ge_session_map_.find(session_id);
  if (session_item != ge_session_map_.end()) {
    return session_item->second;
  }
  MS_LOG(INFO) << "can not find session id.";
  return nullptr;
}

std::set<std::string> GeSessionManager::UpdateSessionVariables(int64_t session_id,
                                                               const std::vector<std::string> &graph_variables) {
  std::set<std::string> new_variables;
  if (session_id == lite::kUnkonwnSessionId) {
    std::transform(graph_variables.begin(), graph_variables.end(), std::inserter(new_variables, new_variables.begin()),
                   [](const auto &item) { return item; });
    return new_variables;
  }
  std::lock_guard<std::mutex> lock(session_mutex_);
  std::shared_ptr<ge::Session> ge_session = nullptr;
  auto session_item = ge_session_map_.find(session_id);
  if (session_item != ge_session_map_.end() && session_item->second != nullptr) {
    ge_session = session_item->second->ge_session.lock();
  }
  if (ge_session == nullptr) {
    std::transform(graph_variables.begin(), graph_variables.end(), std::inserter(new_variables, new_variables.begin()),
                   [](const auto &item) { return item; });
    return new_variables;
  }
  auto &current_session_variables = session_item->second->session_variables;
  for (auto &item : graph_variables) {
    if (current_session_variables.find(item) == current_session_variables.end()) {
      new_variables.insert(item);
      current_session_variables.insert(item);
    }
  }
  return new_variables;
}

void GeSessionManager::TryReleaseGeSessionContext(int64_t session_id) {
  std::lock_guard<std::mutex> lock(session_mutex_);
  auto session_item = ge_session_map_.find(session_id);
  if (session_item != ge_session_map_.end()) {
    if (session_item->second != nullptr) {
      auto ge_session = session_item->second->ge_session.lock();
      if (ge_session == nullptr) {
        ge_session_map_.erase(session_item);
      }
    } else {
      ge_session_map_.erase(session_item);
    }
  }
}
}  // namespace mindspore
