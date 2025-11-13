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
#include "extendrt/delegate/ascend_ge/ge_graph_compiler.h"
#include <memory>
#include "src/common/log_util.h"
#include "tools/common/graph_util.h"
#include "tools/optimizer/graph/remove_load_pass.h"
#include "tools/optimizer/graph/attr_to_args_pass.h"

namespace mindspore {
namespace {
constexpr auto kDump = "dump";
constexpr auto kDumpMode = "dump_mode";
constexpr auto kProfiling = "profiler";
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
}  // namespace

bool GeGraphCompiler::CompileGraph(const FuncGraphPtr &graph, GeSessionInfo *ge_session_info,
                                   const GeOptionsContainer &ge_options_container) {
  MS_CHECK_TRUE_MSG(graph != nullptr, false, "graph is NULL.");
  MS_CHECK_TRUE_MSG(ge_session_info != nullptr, false, "ge_session_info is NULL.");
  (void)setenv("GE_TRAIN", "0", 1);
  if (ge_session_info->session_ == nullptr) {
    MS_LOG(ERROR) << "ge_session hasn't been created.";
    return false;
  }
  auto df_graph = ToGeGraph(graph, ge_session_info, ge_options_container.GeGraphOptions().at(kGeGraphKey));
  if (df_graph == nullptr) {
    MS_LOG(ERROR) << "Convert FuncGraph to ge::graph failed.";
    return false;
  }
  uint32_t graph_id = ge_session_info->graph_ids_.size();
  auto ge_status = ge_session_info->session_->AddGraph(graph_id, *(df_graph), ge_options_container.GeGraphOptions());
  if (ge_status != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "Call GE AddGraph Failed: " << ge::GEGetErrorMsg();
    return false;
  }
  ge_status = ge_session_info->session_->CompileGraph(graph_id);
  if (ge_status != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "Call GE CompileGraph Failed: " << ge::GEGetErrorMsg();
    return false;
  }
  ge_session_info->graph_ids_.push_back(graph_id);
  return true;
}

backend::ge_backend::DfGraphPtr GeGraphCompiler::ToGeGraph(const FuncGraphPtr &graph, GeSessionInfo *ge_session_info,
                                                           const std::string &graph_key) {
  MS_EXCEPTION_IF_NULL(graph);
  lite::AdjustDuplicateNodeName(graph);
  auto remove_load_pass = std::make_shared<opt::RemoveLoadPass>();
  remove_load_pass->Run(graph);
  opt::UpdateManager(graph);

  // Convert mindir attributes to inputs because of dynamic_shape operator.
  // For the transformed operators, the GE adapter only supports inputs but not attributes.
  auto args_to_attr_pass = std::make_shared<opt::AttrToArgsPass>();
  if (args_to_attr_pass == nullptr) {
    MS_LOG(ERROR) << "create AttrToArgsPass failed";
    return nullptr;
  }
  if (!args_to_attr_pass->Run(graph)) {
    MS_LOG(ERROR) << "convert args to attr pass failed";
    return nullptr;
  }

  // Convert to ge::graph
  backend::ge_backend::TensorOrderMap params_vals = GetParams(graph);
  auto converter =
    std::make_shared<backend::ge_backend::DfGraphConvertor>(graph, "", backend::ge_backend::RefModeFlag::kRefModeNone);
  backend::ge_backend::BuildGraph(graph_key, converter, params_vals);
  if (backend::ge_backend::ErrCode(converter) != 0) {
    backend::ge_backend::ClearGraph();
    MS_LOG(ERROR) << "Convert df graph failed";
    return nullptr;
  }
  if (!ProcessGeInitGraph(ge_session_info, backend::ge_backend::GetInitGraph(converter), params_vals,
                          converter->GetInitDataNames())) {
    MS_LOG(ERROR) << "Process GE's Init-Graph failed.";
    return nullptr;
  }
  auto df_graph = backend::ge_backend::GetComputeGraph(converter);
  return df_graph;
}

bool GeGraphCompiler::ProcessGeInitGraph(GeSessionInfo *ge_session_info, backend::ge_backend::DfGraphPtr init_graph,
                                         const backend::ge_backend::TensorOrderMap &params,
                                         const std::vector<std::string> &init_data_names) {
  if (init_graph == nullptr) {
    return true;
  }
  MS_LOG(DEBUG) << "ExecInitGraph start.";
  std::vector<::ge::Tensor> ge_inputs;
  for (auto &item : init_data_names) {
    auto it = params.find(item);
    if (it == params.end()) {
      MS_LOG(ERROR) << "Cannot find parameter " << item << " in parameter map";
      return false;
    }
    auto ge_tensor = device::ascend::TransformUtil::ConvertTensor(it->second, kOpFormat_NCHW, false);
    if (ge_tensor == nullptr) {
      MS_LOG(ERROR) << "Failed to converter MS Tensor to GE Tensor";
      return false;
    }
    ge_inputs.emplace_back(*ge_tensor);
  }
  std::vector<::ge::Tensor> ge_outputs;
  uint32_t graph_id = ge_session_info->graph_ids_.size();
  auto ge_status = ge_session_info->session_->AddGraph(graph_id, *(init_graph), std::map<std::string, std::string>{});
  if (ge_status != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "Call GE AddGraph Failed: " << ge::GEGetErrorMsg();
    return false;
  }
  ge_status = ge_session_info->session_->RunGraph(graph_id, ge_inputs, ge_outputs);
  if (ge_status != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "Exec init graph failed, graph id " << graph_id;
    return false;
  }
  MS_LOG(INFO) << "Exec init graph success, graph id " << graph_id;
  ge_session_info->session_->RemoveGraph(graph_id);
  return true;
}
}  // namespace mindspore
