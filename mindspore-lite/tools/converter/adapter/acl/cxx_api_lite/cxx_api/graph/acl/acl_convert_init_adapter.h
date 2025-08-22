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
#ifndef MINDSPORE_CCSRC_CXX_API_GRAPH_ACL_ACL_ACL_CONVERT_INIT_ADAPTER_H
#define MINDSPORE_CCSRC_CXX_API_GRAPH_ACL_ACL_ACL_CONVERT_INIT_ADAPTER_H

#include <memory>
#include <mutex>
#include <map>
#include <string>
#include "acl/acl_base.h"
#include "ge/ge_ir_build.h"

namespace mindspore {
class __attribute__((visibility("default"))) AclConvertInitAdapter {
 public:
  static AclConvertInitAdapter &GetInstance();
  aclError AclInit(const char *config_file);
  aclError AclFinalize();
  aclError ForceFinalize();
  ge::graphStatus AclBuildInit(const std::map<std::string, std::string> &init_options);

 private:
  AclConvertInitAdapter() : init_flag_(false) {}
  ~AclConvertInitAdapter() = default;

  bool init_flag_;
  bool init_build_flag_;
  std::mutex flag_mutex_;
  std::mutex build_flag_mutex_;
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_CXX_API_GRAPH_ACL_ACL_ACL_CONVERT_INIT_ADAPTER_H
