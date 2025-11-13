/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_GE_OPTIONS_CONTAINER_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_GE_OPTIONS_CONTAINER_H_

#include <atomic>
#include "common/config_infos.h"
#include "include/api/context.h"
#include "ir/func_graph.h"

namespace mindspore {
class GeOptionsContainer {
 public:
  GeOptionsContainer() = default;
  ~GeOptionsContainer() = default;
  bool InitGeOptions(const FuncGraphPtr &graph, const ConfigInfos &config_info,
                     const std::shared_ptr<mindspore::Context> &context);
  const std::map<std::string, std::string> &GeSessionOptions() const { return ge_session_options_; }
  const std::map<std::string, std::string> &GeGraphOptions() const { return ge_graph_options_; }

 private:
  bool InitGeSessionOptions(const ConfigInfos &config_info, const std::shared_ptr<mindspore::Context> &context);
  bool GetGeSessionOptionsFromAscendSection(const ConfigInfos &config_info,
                                            const std::shared_ptr<AscendDeviceInfo> &ascend_device_info);
  bool InitGeGraphOptions(const ConfigInfos &config_info, const std::shared_ptr<mindspore::Context> &context,
                          const std::string &graph_key_suffix);
  std::map<std::string, std::string> ge_session_options_;
  std::map<std::string, std::string> ge_graph_options_;
  static std::atomic_int64_t unique_identification_;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_GE_OPTIONS_CONTAINER_H_
