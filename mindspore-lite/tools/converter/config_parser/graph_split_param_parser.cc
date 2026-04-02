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
#include "tools/converter/config_parser/graph_split_param_parser.h"
#include <vector>
#include <string>
#include <set>
#include "tools/converter/cxx_api/converter_para.h"
#include "src/common/common.h"

namespace mindspore {
namespace lite {
std::vector<std::string> ParseInnerList(const std::string &s) {
  std::vector<std::string> result;
  constexpr size_t kMinBracketPairLength = 2;  // Minimum length for a pair of brackets "[]"
  if (s.length() < kMinBracketPairLength) {
    return result;
  }
  std::string content = s.substr(1, s.length() - 2);
  if (content.empty()) {
    return result;
  }
  std::stringstream ss(content);
  std::string item;

  while (std::getline(ss, item, ',')) {
    if (!item.empty()) {
      result.push_back(item);
    }
  }
  return result;
}

static STATUS ProcessBlock(const std::string &block, std::set<std::string> *op_set,
                           std::vector<std::pair<std::vector<std::string>, std::vector<std::string>>> *output_cfg) {
  int inner_bracket_level = 0;
  size_t split_pos = std::string::npos;
  for (size_t j = 1; j < block.length() - 1; j++) {
    if (block[j] == '[') {
      inner_bracket_level++;
    } else if (block[j] == ']') {
      inner_bracket_level--;
    }
    if (block[j] == ',' && inner_bracket_level == 0) {
      split_pos = j;
      break;
    }
  }
  if (split_pos == std::string::npos) {
    return RET_OK;
  }
  std::string first_part = block.substr(1, split_pos - 1);
  std::string second_part = block.substr(split_pos + 1, block.length() - (split_pos + 1) - 1);
  std::vector<std::string> first_vector = ParseInnerList(first_part);
  std::vector<std::string> second_vector = ParseInnerList(second_part);
  for (const auto &s : first_vector) {
    op_set->insert(s);
  }
  for (const auto &s : second_vector) {
    op_set->insert(s);
  }
  MS_CHECK_TRUE_MSG(!second_vector.empty(), lite::RET_ERROR, "Current subgraph output name is empty!");
  output_cfg->emplace_back(first_vector, second_vector);
  return RET_OK;
}

void GetSplitNode(const std::shared_ptr<ConverterPara> &param, std::string *split_node_str) {
  auto config_infos = param->config_infos;
  if (config_infos.find(kSplitGraph) != config_infos.end()) {
    auto split_graph_section = config_infos.at(kSplitGraph);
    if (split_graph_section.find("split_node_name") != split_graph_section.end()) {
      *split_node_str = split_graph_section.at("split_node_name");
    }
  }
}

// Helper function to find the next block start position
static size_t FindNextBlockStart(const std::string &str, size_t pos) {
  while (pos < str.length() && str[pos] != '[') {
    pos++;
  }
  return pos;
}

// Helper function to extract a complete block (from [ to matching ])
static bool ExtractBlock(const std::string &str, size_t start_pos, std::string *block, size_t *next_pos) {
  int bracket_level = 0;
  for (size_t i = start_pos; i < str.length(); i++) {
    if (str[i] == '[') {
      bracket_level++;
    } else if (str[i] == ']') {
      bracket_level--;
    }
    if (bracket_level == 0 && i > start_pos) {
      *block = str.substr(start_pos, i - start_pos + 1);
      *next_pos = i + 1;
      return true;
    }
  }
  return false;
}

// Helper function to process all blocks in the split node string
static STATUS ProcessAllBlocks(const std::string &split_node_str, std::set<std::string> *op_set,
                               std::vector<std::pair<std::vector<std::string>, std::vector<std::string>>> *output_cfg) {
  size_t pos = 0;
  while (pos < split_node_str.length()) {
    pos = FindNextBlockStart(split_node_str, pos);
    if (pos >= split_node_str.length()) {
      break;
    }

    std::string block;
    size_t next_pos;
    if (!ExtractBlock(split_node_str, pos, &block, &next_pos)) {
      break;
    }

    auto ret = ProcessBlock(block, op_set, output_cfg);
    if (ret != RET_OK) {
      return ret;
    }
    pos = next_pos;
  }
  return RET_OK;
}

STATUS GraphSplitParamParser::ParseGraphSplitCfg(const std::shared_ptr<ConverterPara> &param) {
  MS_CHECK_TRUE_MSG(param != nullptr, RET_ERROR, "param is nullptr!");
  std::string split_node_str = "";
  GetSplitNode(param, &split_node_str);
  MS_CHECK_TRUE_RET(!split_node_str.empty(), RET_OK);

  std::set<std::string> op_set;
  auto ret = ProcessAllBlocks(split_node_str, &op_set, &param->splitGraphCfg.subgraph_input_output);
  if (ret != RET_OK) {
    return ret;
  }

  for (const auto &s : op_set) {
    param->splitGraphCfg.split_node_names.push_back(s);
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
