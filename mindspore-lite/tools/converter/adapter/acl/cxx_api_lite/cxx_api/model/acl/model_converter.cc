/**
 * Copyright 2020-2026 Huawei Technologies Co., Ltd
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

#include "cxx_api/model/acl/model_converter.h"
#include <dirent.h>
#include <memory>
#include <algorithm>
#include <utility>
#include <vector>
#include <set>
#include <map>
#include <string>
#include <ctime>
#include "tools/converter/adapter/acl/backend/ge_backend/graph_ir/utils.h"
#include "tools/converter/adapter/acl/backend/ge_backend/graph_ir/convert.h"
#include "graph/graph_buffer.h"
#include "graph/graph.h"
#include "cxx_api/model/aoe/auto_tune_process.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "src/common/file_utils.h"
#include "cxx_api/graph/acl/acl_convert_init_adapter.h"
#include "mindspore/core/include/ir/func_graph.h"
#include "ir/graph_utils.h"

namespace mindspore {
namespace {
constexpr size_t kOmSuffixLen = 3;
const char kOmSuffix[] = ".om";
std::string FindSavedOmFile(const std::string &base_path) {
  // Try the standard path first (no platform suffix)
  auto candidate = base_path + kOmSuffix;
  if (access(candidate.c_str(), F_OK) == 0) {
    return candidate;
  }
  // aclgrphSaveModel may append platform suffix like _linux_x86_64
  auto pos = base_path.find_last_of('/');
  std::string dir = (pos != std::string::npos) ? base_path.substr(0, pos + 1) : "./";
  std::string prefix = (pos != std::string::npos) ? base_path.substr(pos + 1) : base_path;
  candidate = "";
  DIR *dp = opendir(dir.c_str());
  if (dp == nullptr) {
    MS_LOG(ERROR) << "Failed to open directory: " << dir;
    return "";
  }
  struct dirent *entry = nullptr;
  while ((entry = readdir(dp)) != nullptr) {
    std::string name = entry->d_name;
    if (name.size() > prefix.size() && name.compare(0, prefix.size(), prefix) == 0 && name.size() >= kOmSuffixLen &&
        name.compare(name.size() - kOmSuffixLen, kOmSuffixLen, kOmSuffix) == 0) {
      candidate = dir + name;
      break;
    }
  }
  (void)closedir(dp);
  return candidate;
}

// some config is not supported in the update subgraph, do not add to the update_options. e.g. lora weight update.
const std::set<std::string> update_options_blacklist = {
  "ge.dynamicDims",
};
std::string GetAscendPath() {
  Dl_info info;
  if (dladdr(reinterpret_cast<void *>(aclrtMalloc), &info) == 0) {
    MS_LOG(ERROR) << "Get dladdr failed.";
    return "";
  }
  auto path_tmp = std::string(info.dli_fname);
  const std::string kCann = "cann";
  const std::string kLatest = "latest";
  auto posCann = path_tmp.rfind(kCann);
  auto posLatest = path_tmp.rfind(kLatest);
  if (posCann != std::string::npos && posLatest != std::string::npos) {
    // Both found, choose the rightmost one
    if (posCann > posLatest) {
      return path_tmp.substr(0, posCann) + kCann + "/";
    } else {
      return path_tmp.substr(0, posLatest) + kLatest + "/";
    }
  } else if (posCann != std::string::npos) {
    return path_tmp.substr(0, posCann) + kCann + "/";
  } else if (posLatest != std::string::npos) {
    return path_tmp.substr(0, posLatest) + kLatest + "/";
  } else {
    MS_EXCEPTION(ValueError)
      << "Get ascend path from aclrtMalloc path " << path_tmp
      << " failed, please check whether CANN packages are \n"
         "installed correctly, and environment variables are set by source ${LOCAL_ASCEND}/cann/set_env.sh.";
  }
  return "";
}

// todo: acl doesn't support to clear current context
void ClearCurrentRtCtx() {
  aclrtContext tmp_ctx = nullptr;
  auto ret = CALL_ASCEND_API(aclrtCreateContext, &tmp_ctx, 0);
  if (ret != ACL_RT_SUCCESS) {
    MS_LOG(WARNING) << "Call aclrtCreateContext failed, ret = " << ret;
    return;
  }
  ret = CALL_ASCEND_API(aclrtDestroyContext, tmp_ctx);
  if (ret != ACL_RT_SUCCESS) {
    MS_LOG(WARNING) << "Call aclrtDestroyContext failed, ret = " << ret;
    return;
  }
}

backend::ge_backend::TensorOrderMap GetParams(const FuncGraphPtr &anf_graph) {
  backend::ge_backend::TensorOrderMap res;
  for (auto &anf_node : anf_graph->parameters()) {
    MS_EXCEPTION_IF_NULL(anf_node);
    auto para = anf_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(para);
    if (para->has_default()) {
      auto value = para->default_param();
      MS_EXCEPTION_IF_NULL(value);
      auto tensor = value->cast<std::shared_ptr<tensor::Tensor>>();
      res.emplace(para->name(), tensor);
      MS_LOG(INFO) << "Parameter " << para->name() << " has default value.";
    }
  }
  return res;
}

bool HasSubgraph(const FuncGraphPtr &func_graph) {
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    for (auto &input : cnode->inputs()) {
      if (GetValueNode<FuncGraphPtr>(input) != nullptr) {
        return true;
      }
    }
  }
  return false;
}
}  // namespace

backend::ge_backend::DfGraphPtr ModelConverter::ConvertFuncGraphToAIR(const FuncGraphPtr &anf_graph) const {
  MS_EXCEPTION_IF_NULL(anf_graph);
#ifndef BUILD_LITE
  opt::ReduceOptimization(anf_graph);
#endif
  std::string compute_graph_name = anf_graph->ToString();
  auto option = options_.lock();
  if (option != nullptr && !option->GetDumpModelName().empty()) {
    compute_graph_name = option->GetDumpModelName();
  }
  bool is_control_model = HasSubgraph(anf_graph);
  if (!is_control_model) {
    auto converter = std::make_shared<backend::ge_backend::DfGraphConvertor>(
      anf_graph, "", backend::ge_backend::RefModeFlag::kRefModeNone, std::vector<std::string>{}, nullptr, true);
    MS_CHECK_TRUE_MSG(converter != nullptr, nullptr, "Create converter failed.");
    converter->ConvertAllNode().InitParam(GetParams(anf_graph)).BuildGraph(compute_graph_name);
    return converter->GetComputeGraph();
  }
  auto converter =
    backend::ge_backend::NewConverter(anf_graph, "", backend::ge_backend::RefModeFlag::kRefModeNone, true);
  backend::ge_backend::SetTraining(converter, false);
  backend::ge_backend::BuildGraph(compute_graph_name, converter, GetParams(anf_graph));
  return backend::ge_backend::GetComputeGraph(converter);
}

bool ModelConverter::CompileBundleModel(const backend::ge_backend::DfGraphPtr &graph,
                                        const std::map<std::string, std::string> &build_options,
                                        ge::ModelBufferData *model) const {
  MS_CHECK_TRUE_MSG(model != nullptr, false, "Model buffer is nullptr.");
  ge::WeightRefreshableGraphs split_graphs;
  std::vector<ge::AscendString> ascend_variable_names(variable_node_names_.size());
  std::transform(variable_node_names_.begin(), variable_node_names_.end(), ascend_variable_names.begin(),
                 [](std::string s) { return ge::AscendString(s.c_str()); });
  auto ret = ge::aclgrphConvertToWeightRefreshableGraphs(*graph, ascend_variable_names, split_graphs);
  if (ret != 0) {
    MS_LOG(ERROR) << "aclgraphConvertToWeightRefreshableGraphs failed! ret:" << ret;
    return false;
  }
  std::map<ge::AscendString, ge::AscendString> bund_bundle_options;
  std::map<ge::AscendString, ge::AscendString> update_options;
  for (auto it : build_options) {
    bund_bundle_options.insert(std::make_pair(ge::AscendString(it.first.c_str()), ge::AscendString(it.second.c_str())));
    if (update_options_blacklist.find(it.first) == update_options_blacklist.end()) {
      update_options.insert(std::make_pair(ge::AscendString(it.first.c_str()), ge::AscendString(it.second.c_str())));
    }
  }
  auto update_graph = ConvertFuncGraphToAIR(update_func_graph_);
  if (update_graph == nullptr) {
    MS_LOG(ERROR) << "Convert FuncGraph to AscendIR failed.";
    return false;
  }
  std::vector<ge::GraphWithOptions> graph_and_options;
  graph_and_options.push_back(ge::GraphWithOptions{split_graphs.infer_graph, bund_bundle_options});
  graph_and_options.push_back(ge::GraphWithOptions{*update_graph, update_options});
  ret = ge::aclgrphBundleBuildModel(graph_and_options, *model);
  if (ret != ge::SUCCESS) {
    MS_LOG(ERROR) << "Call aclgrphBundleBuildModel fail: " << CALL_ASCEND_API(aclGetRecentErrMsg);
    return false;
  }
  return true;
}

Buffer ModelConverter::LoadBufferFromSavedOm(const std::string &saved_om_path, const ge::ModelBufferData &model) const {
  if (saved_om_path.empty()) {
    MS_LOG(ERROR) << "Saved om file path is empty.";
    return Buffer();
  }
  size_t om_size = 0;
  auto om_data = lite::ReadFile(saved_om_path.c_str(), &om_size);
  std::unique_ptr<char[]> om_guard(om_data);
  if (om_data != nullptr && om_size > 0) {
    Buffer om_buffer(om_data, om_size);
    om_guard.reset();
    if (remove(saved_om_path.c_str()) != 0) {
      MS_LOG(WARNING) << "Remove temp om file " << saved_om_path << " failed.";
    }
    return om_buffer;
  }
  MS_LOG(ERROR) << "Read om file " << saved_om_path << " failed.";
  (void)remove(saved_om_path.c_str());
  return Buffer();
}

Buffer ModelConverter::BuildAirModel(const backend::ge_backend::DfGraphPtr &graph,
                                     const std::map<std::string, std::string> &init_options,
                                     const std::map<std::string, std::string> &build_options) const {
  ge::ModelBufferData model;
  if (AclConvertInitAdapter::GetInstance().AclBuildInit(init_options) != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "AclBuildInit failed!";
    return Buffer();
  }
  auto option = options_.lock();
  bool is_bundle = variable_node_names_.size() > 0 && update_func_graph_ != nullptr;
  if (is_bundle) {
    if (!CompileBundleModel(graph, build_options, &model)) {
      AclConvertInitAdapter::GetInstance().AclBuildFinalize();
      return Buffer();
    }
  } else {
    auto ret = ge::aclgrphBuildModel(*graph, build_options, model);
    if (ret != ge::SUCCESS) {
      MS_LOG(ERROR) << "Call aclgrphBuildModel fail: " << CALL_ASCEND_API(aclGetRecentErrMsg);
      AclConvertInitAdapter::GetInstance().AclBuildFinalize();
      return Buffer();
    }
  }
  Buffer result;
  bool use_file_buffer = option != nullptr && option->GetTilingGeneration() && !option->GetOmFilePath().empty();
  if (use_file_buffer) {
    std::string saved_om_path;
    auto save_ret = SaveModel(model, is_bundle, &saved_om_path);
    if (save_ret != kSuccess) {
      MS_LOG(ERROR) << "SaveModel failed, model build succeeded but save/reload step failed.";
      AclConvertInitAdapter::GetInstance().AclBuildFinalize();
      return Buffer();
    }
    result = LoadBufferFromSavedOm(saved_om_path, model);
  } else {
    result = Buffer(model.data.get(), model.length);
  }
  if (option != nullptr && option->IsLastModel()) {
    AclConvertInitAdapter::GetInstance().AclBuildFinalize();
  }
  return result;
}

Status ModelConverter::SaveModel(const ge::ModelBufferData &model, bool is_bundle, std::string *saved_om_path) const {
  std::string file_path;
  auto option = options_.lock();
  if (option != nullptr) {
    file_path = option->GetOmFilePath();
  }
  if (file_path.empty()) {
    MS_LOG(INFO) << "File path is empty, there is no need to save model";
    return kSuccess;
  }
  auto save_path = file_path + "_tmp_" + std::to_string(static_cast<uint64_t>(time(nullptr)));
  MS_LOG(INFO) << "Om save path: " << save_path;
  if (is_bundle) {
    auto ret = ge::aclgrphBundleSaveModel(save_path.c_str(), model);
    if (ret != ge::SUCCESS) {
      MS_LOG(ERROR) << "Call aclgrphBundleSaveModel fail.";
      return kLiteError;
    }
  } else {
    auto ret = ge::aclgrphSaveModel(save_path, model);
    if (ret != ge::SUCCESS) {
      MS_LOG(ERROR) << "Call aclgrphSaveModel fail.";
      return kLiteError;
    }
  }
  auto om_path = FindSavedOmFile(save_path);
  if (om_path.empty()) {
    MS_LOG(ERROR) << "Saved om file not found for base path: " << save_path;
    return kLiteError;
  }
  if (saved_om_path != nullptr) {
    *saved_om_path = om_path;
  }
  return kSuccess;
}

Buffer ModelConverter::LoadMindIR(const FuncGraphPtr &func_graph) {
  Buffer buffer_ret;
  ClearCurrentRtCtx();
  auto ascend_path = GetAscendPath();
#ifdef MACHINE_LINUX_ARM64
  std::string lib_opsproto_file = ascend_path + "opp/built-in/op_proto/lib/linux/aarch64/libopsproto.so";
#else
  std::string lib_opsproto_file = ascend_path + "opp/built-in/op_proto/lib/linux/x86_64/libopsproto.so";
#endif
  static void *handler = dlopen(lib_opsproto_file.c_str(), RTLD_LAZY);
  if (handler == nullptr) {
    MS_LOG(ERROR) << "dlopen opsproto library failed: " << lib_opsproto_file;
    return buffer_ret;
  }
  auto df_graph = ConvertFuncGraphToAIR(func_graph);
  if (df_graph == nullptr) {
    MS_LOG(ERROR) << "Convert FuncGraph to AscendIR failed.";
    return buffer_ret;
  }
  ge::GraphBuffer model_data;
  auto ge_ret = df_graph->SaveToMem(model_data);
  if (ge_ret != ge::SUCCESS) {
    MS_LOG(ERROR) << "Save ge model to buffer failed.";
    return buffer_ret;
  }
  buffer_ret.SetData(model_data.GetData(), model_data.GetSize());
  Buffer model_result = LoadAscendIRInner(buffer_ret);
  if (model_result.DataSize() == 0) {
    MS_LOG(ERROR) << "Convert model from MindIR to OM failed";
    return {};
  }
  return model_result;
}

Buffer ModelConverter::LoadAscendIRInner(const Buffer &model_data) {
  backend::ge_backend::DfGraphPtr df_graph = std::make_shared<backend::ge_backend::DfGraph>("tmp");
  if (df_graph == nullptr) {
    MS_LOG(ERROR) << "Convert FuncGraph to AscendIR failed.";
    return {};
  }
  auto ret = df_graph->LoadFromMem(static_cast<const uint8_t *>(model_data.Data()), model_data.DataSize());
  if (ret != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "Convert FuncGraph to AscendIR failed.";
  }

  std::map<std::string, std::string> init_options;
  std::map<std::string, std::string> build_options;
  auto option = options_.lock();
  if (option != nullptr) {
    std::tie(init_options, build_options) = option->GenAclOptions();
  }
#ifdef BUILD_LITE
  if (AutoTuneProcess::AoeOfflineTurningGraph(options_, df_graph) != kSuccess) {
    MS_LOG(ERROR) << "Aoe tune graph failed.";
    return Buffer();
  }
#endif
  return BuildAirModel(df_graph, init_options, build_options);
}
}  // namespace mindspore
