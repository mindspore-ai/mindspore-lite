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

#include "extendrt/delegate/ascend_ge/ge_options_container.h"
#include <algorithm>
#include <memory>
#include <string>
#include <nlohmann/json.hpp>
#include "extendrt/delegate/ascend_ge/ge_utils.h"
#include "src/common/common.h"
#include "src/common/file_utils.h"
#include "tools/converter/adapter/acl/cxx_api_lite/cxx_api/acl_utils.h"

namespace mindspore {
namespace {
constexpr auto kDump = "dump";
constexpr auto kDumpMode = "dump_mode";
constexpr auto kProfiling = "profiler";
}  // namespace
std::atomic_int64_t GeOptionsContainer::unique_identification_ = 0;
bool GeOptionsContainer::InitGeOptions(const FuncGraphPtr &graph, const ConfigInfos &config_info,
                                       const std::shared_ptr<mindspore::Context> &context) {
  MS_CHECK_TRUE_MSG(graph != nullptr, false, "graph is NULL.");
  MS_CHECK_TRUE_MSG(context != nullptr, false, "context is NULL.");
  ge_session_options_.clear();
  ge_graph_options_.clear();
  if (!InitGeSessionOptions(config_info, context)) {
    MS_LOG(ERROR) << "Init ge_session_options failed.";
    return false;
  }
  if (!InitGeGraphOptions(config_info, context, graph->ToString())) {
    MS_LOG(ERROR) << "Init ge_graph_options failed.";
    return false;
  }
  return true;
}

bool GeOptionsContainer::InitGeSessionOptions(const ConfigInfos &config_info,
                                              const std::shared_ptr<mindspore::Context> &context) {
  auto ascend_device_info = GeUtils::GetAscendDeviceInfo(context);
  if (ascend_device_info == nullptr) {
    MS_LOG(ERROR) << "Failed to get graph session options, can not find ascend device context.";
    return false;
  }

  ge_session_options_["ge.trainFlag"] = "0";
  ge_session_options_["ge.enablePrintOpPass"] = "0";
  ge_session_options_["ge.exec.device_id"] = std::to_string(ascend_device_info->GetDeviceID());
  ge_session_options_["ge.exec.staticMemoryPolicy"] = "2";
  auto config_it = config_info.find(lite::kGeSessionOptionsSection);
  if (config_it != config_info.end()) {
    for (auto &item : config_it->second) {
      ge_session_options_[item.first] = item.second;
    }
  }
  return GetGeSessionOptionsFromAscendSection(config_info, ascend_device_info);
}

bool GeOptionsContainer::GetGeSessionOptionsFromAscendSection(
  const ConfigInfos &config_info, const std::shared_ptr<AscendDeviceInfo> &ascend_device_info) {
  auto config_it = config_info.find(lite::kAscendContextSection);
  if (config_it == config_info.end()) {
    return true;
  }

  auto config = config_it->second;
  auto option_it = config.find(lite::kModelCacheMode);
  if (option_it != config.end()) {
    if (!option_it->second.empty()) {
      auto build_cache_dir = "model_build_cache_" + std::to_string(ascend_device_info->GetRankID());
      if (lite::CreateDir(build_cache_dir) != lite::RET_OK) {
        MS_LOG(ERROR) << "Failed to create build cache dir " << build_cache_dir;
        return false;
      }
      ge_session_options_[lite::kGeGraphCompilerCacheDir] = build_cache_dir;
      MS_LOG(INFO) << "Update session attr " << lite::kGeGraphCompilerCacheDir << " to " << build_cache_dir;
    }
  }
  option_it = config.find(lite::kDumpPathKey);
  if (option_it != config.end()) {
    auto dump_path = option_it->second;
    auto real_path = lite::RealPath(dump_path.c_str());
    std::ifstream ifs(real_path);
    if (!ifs.good() || !ifs.is_open()) {
      MS_LOG(ERROR) << "The dump config file is not exit or open failed.";
      return false;
    }
    nlohmann::json dump_cfg_json;
    try {
      dump_cfg_json = nlohmann::json::parse(ifs);
    } catch (const nlohmann::json::parse_error &error) {
      MS_LOG(EXCEPTION) << "parse json failed, please check the file.";
    }
    if (dump_cfg_json[kDump] != nullptr && dump_cfg_json[kDump][kDumpMode] != nullptr) {
      ge_session_options_["ge.exec.enableDump"] = "1";
      ge_session_options_["ge.exec.dumpMode"] = dump_cfg_json[kDump][kDumpMode].get<std::string>();
    }
  }
  option_it = config.find(lite::kProfilingPathKey);
  if (option_it != config.end()) {
    auto profiling_path = option_it->second;
    auto real_path = lite::RealPath(profiling_path.c_str());
    std::ifstream ifs(real_path);
    if (!ifs.good() || !ifs.is_open()) {
      MS_LOG(EXCEPTION) << "The profiling_path config file is not exit or open failed.";
    }
    nlohmann::json profiling_cfg_json;
    try {
      profiling_cfg_json = nlohmann::json::parse(ifs);
    } catch (const nlohmann::json::parse_error &error) {
      MS_LOG(EXCEPTION) << "parse json failed, please check the file.";
    }
    if (profiling_cfg_json[kProfiling] != nullptr) {
      ge_session_options_["ge.exec.profilingMode"] = "1";
      ge_session_options_["ge.exec.profilingOptions"] = profiling_cfg_json[kProfiling].dump();
    }
  }
  option_it = config.find(lite::kGeVariableMemoryMaxSize);
  if (option_it != config.end()) {
    ge_session_options_["ge.variableMemoryMaxSize"] = option_it->second;
  }
  option_it = config.find(lite::kGeGraphMemoryMaxSize);
  if (option_it != config.end()) {
    ge_session_options_["ge.graphMemoryMaxSize"] = option_it->second;
  }
  option_it = config.find(lite::kGraphCompilerCacheDirKey);
  if (option_it != config.end()) {
    ge_session_options_[lite::kGeGraphCompilerCacheDir] = option_it->second;
  }
  return true;
}

bool GeOptionsContainer::InitGeGraphOptions(const ConfigInfos &config_info,
                                            const std::shared_ptr<mindspore::Context> &context,
                                            const std::string &graph_key_suffix) {
  auto ascend_device_info = GeUtils::GetAscendDeviceInfo(context);
  if (ascend_device_info == nullptr) {
    MS_LOG(ERROR) << "Failed to get graph session options, can not find ascend device context.";
    return false;
  }
  auto graph_key = std::to_string(ascend_device_info->GetRankID()) + "_" + std::to_string(unique_identification_++) +
                   "_" + graph_key_suffix;
  std::replace_if(graph_key.begin(), graph_key.end(), [](char c) { return c == '.'; }, '_');
  ge_graph_options_[lite::kGeGraphKey] = graph_key;
  auto config_it = config_info.find(lite::kGeGraphOptionsSection);
  if (config_it != config_info.end()) {
    for (auto &item : config_it->second) {
      ge_graph_options_[item.first] = item.second;
    }
  }
  auto precision_mode = ascend_device_info->GetPrecisionMode();
  if (!precision_mode.empty()) {
    ge_graph_options_["ge.exec.precision_mode"] = TransforPrecisionToAcl(precision_mode);
  }
  config_it = config_info.find(lite::kAscendContextSection);
  if (config_it == config_info.end()) {
    return true;
  }
  auto config = config_it->second;
  auto option_it = config.find(lite::kModifyMixList);
  if (option_it != config.end()) {
    ge_graph_options_["ge.exec.modify_mixlist"] = option_it->second;
  }
  return true;
}
}  // namespace mindspore
