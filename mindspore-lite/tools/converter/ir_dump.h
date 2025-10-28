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
#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_IR_DUMP_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_IR_DUMP_H

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <unordered_map>
#include "include/api/status.h"
#include "ir/dtype/type.h"
#include "ir/anf.h"

namespace mindspore::lite {
enum DumpGraphLevel : int { kLevel0 = 0, kLevel1 = 1 };

struct SubGraphIRInfo {
  int32_t node_index;
  std::ostringstream buffer;
  std::map<AnfNodePtr, int32_t> local_var_map;
  int32_t cnode_num = 0;
};

Status DumpGraph(const std::string &pass_name, const FuncGraphPtr &graph);

}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_IR_DUMP_H
