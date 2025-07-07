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

#define USE_DEPRECATED_API
#include "tools/optimizer/fusion/add_stream_label_pass.h"
#include <memory>
#include <vector>
#include "tools/common/string_util.h"
#include "tools/common/parse_config_utils.h"
#include "common/common.h"
namespace mindspore {
namespace opt {
namespace {
constexpr size_t kMaxLineCount = 100;
constexpr size_t kMaxLineLen = 9999;
constexpr size_t kNumOfVetSize = 2;
constexpr size_t kStreamLabelIndex = 0;
constexpr size_t kNodeNameIndex = 1;
constexpr size_t kSize0 = 0;
}  // namespace

Status AddStreamLabelPass::GetNodeNames(const FuncGraphPtr &func_graph) {
  auto node_list = TopoSort(func_graph->get_return());
  for (auto node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    all_node_names_.insert(node->fullname_with_scope());
  }
  return kSuccess;
}

Status AddStreamLabelPass::ParseStreamLable() {
  std::string stream_label_file = "";
  if (param_->config_infos.find(lite::kAscendContextSection) != param_->config_infos.end()) {
    auto ascend_context = param_->config_infos.at(lite::kAscendContextSection);
    if (ascend_context.find(lite::kStreamLabelFile) != ascend_context.end()) {
      stream_label_file = ascend_context.at(lite::kStreamLabelFile);
      MS_LOG(INFO) << "stream_label_file: " << stream_label_file;
    }
  }
  if (stream_label_file.size() == kSize0) {
    MS_LOG(INFO) << "Unspecified stream_label_file.";
    return kLiteNoChange;
  }
  std::ifstream ifs;
  auto ret = lite::ReadFileToIfstream(stream_label_file, &ifs);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "read file to ifstream failed! stream_label_file:" << stream_label_file;
    return kLiteError;
  }
  std::string raw_line;
  size_t num_of_line = 0;
  while (std::getline(ifs, raw_line)) {
    if (num_of_line > kMaxLineCount) {
      MS_LOG(ERROR) << "the line count is exceeds the maximum range 9999!";
      ifs.close();
      return kLiteError;
    }
    if (raw_line.size() > kMaxLineLen) {
      MS_LOG(ERROR) << "Length of line is exceeds the maximum range 9999! Current length:" << raw_line.size();
      ifs.close();
      return kLiteError;
    }
    if (raw_line.empty() || raw_line.at(0) == '#') {
      continue;
    }
    num_of_line++;
    if (!lite::EraseBlankSpaceAndLineBreak(&raw_line)) {
      MS_LOG(ERROR) << "Erase Blank Space failed!";
      ifs.close();
      return kLiteError;
    }
    // remove value quotes eg: "/mnt/image" -> /mnt/image
    if (!lite::EraseQuotes(&raw_line)) {
      MS_LOG(ERROR) << "Erase Quotes failed!";
      ifs.close();
      return kLiteError;
    }
    if (raw_line.empty()) {
      continue;
    }
    auto split_vector = lite::SplitStringToVector(raw_line, ':');
    if (split_vector.size() != kNumOfVetSize) {
      MS_LOG(ERROR) << "split vector size != 2, " << raw_line << " is illegal!";
      ifs.close();
      return kLiteError;
    }
    std::string stream_label = split_vector.at(kStreamLabelIndex);
    if (!lite::EraseBlankSpaceAndLineBreak(&stream_label)) {
      MS_LOG(ERROR) << "Erase Blank Space for key failed!";
      ifs.close();
      return kLiteError;
    }
    std::string node_names = split_vector.at(kNodeNameIndex);
    if (!lite::EraseBlankSpaceAndLineBreak(&node_names)) {
      MS_LOG(ERROR) << "Erase Blank Space for node_names failed!";
      ifs.close();
      return kLiteError;
    }
    auto node_name_vec = lite::SplitStringToVector(node_names, ',');
    MS_LOG(INFO) << "node_name_vec: " << node_name_vec;
    for (auto node_name : node_name_vec) {
      if (all_node_names_.find(node_name) == all_node_names_.end()) {
        MS_LOG(ERROR) << "The specified node name is not in the graph, node name:" << node_name;
        ifs.close();
        return kLiteError;
      }
      node_to_label_map_[node_name] = stream_label;
    }
  }
  ifs.close();
  return kSuccess;
}

Status AddStreamLabelPass::AddStreamLabel(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, kLiteError);
  auto node_list = TopoSort(func_graph->get_return());
  auto manager = Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "Manager is nullptr!";
    return kLiteError;
  }
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_CHECK_TRUE_RET(cnode != nullptr, kLiteError);
    auto it = node_to_label_map_.find(cnode->fullname_with_scope());
    if (it != node_to_label_map_.end()) {
      auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
      auto stream_label = it->second;
      prim->AddAttr("_stream_label", MakeValue<std::string>(stream_label));
      MS_LOG(INFO) << "node name: " << cnode->fullname_with_scope()
                   << ", stream label: " << MakeValue<std::string>(it->second);
    } else {
      MS_LOG(INFO) << "node name: " << cnode->fullname_with_scope() << ", no need to add stream label.";
    }
  }
  return kSuccess;
}

bool AddStreamLabelPass::Run(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, false);
  MS_LOG(INFO) << "AddStreamLabelPass start.";
  auto status = GetNodeNames(func_graph);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "GetNodeNames fialed!";
    return false;
  }
  status = ParseStreamLable();
  if (status == kLiteNoChange) {
    MS_LOG(INFO) << "No change";
    return true;
  }
  if (status != kSuccess && status != kLiteNoChange) {
    MS_LOG(ERROR) << "ParseStreamLable failed!";
    return false;
  }
  status = AddStreamLabel(func_graph);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "AddStreamLabel failed!";
    return false;
  }
  MS_LOG(INFO) << "AddStreamLabelPass end.";
  return true;
}
}  // namespace opt
}  // namespace mindspore
