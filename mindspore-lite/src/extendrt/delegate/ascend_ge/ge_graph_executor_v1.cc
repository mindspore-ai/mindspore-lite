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

#include "extendrt/delegate/ascend_ge/ge_graph_executor_v1.h"
#include <algorithm>
#include <map>
#include <string>
#include "extendrt/delegate/ascend_ge/ge_utils.h"
#include "extendrt/utils/func_graph_utils.h"
#include "extendrt/delegate/factory.h"
#include "src/common/common.h"
#include "src/common/log_util.h"

namespace mindspore {
namespace {
constexpr auto kProviderGeV1 = "ge-v1";
constexpr auto kIsAdapted = "is_adapted";
std::mutex g_compile_graph_mutex;
constexpr size_t kAlignRefData = 32;
std::atomic_int64_t global_session_idx = 0;
}  // namespace

bool GeGraphExecutorV1::Init() {
  ge_global_context_ = GeDeviceContext::InitGlobalContext(context_, config_infos_);
  if (ge_global_context_ == nullptr) {
    MS_LOG(ERROR) << "Failed to Init global context";
    return false;
  }
  return true;
}
GeGraphExecutorV1::~GeGraphExecutorV1() {
  if (ge_session_info_.session_) {
    for (auto graph_id : ge_session_info_.graph_ids_) {
      ge_session_info_.session_->RemoveGraph(graph_id);
    }
  }
  ge_session_info_.session_ = nullptr;
}

bool GeGraphExecutorV1::CompileGraph(const FuncGraphPtr &graph, const std::map<string, string> &, uint32_t *graph_id) {
  MS_CHECK_TRUE_MSG(graph != nullptr, false, "graph is NULL.");
  MS_CHECK_TRUE_MSG(graph_id != nullptr, false, "graph_id is NULL.");
  if (!ge_options_container_.InitGeOptions(graph, config_infos_, context_)) {
    MS_LOG(ERROR) << "Init Ge options failed.";
    return false;
  }
  if (ge_session_info_.session_ == nullptr) {
    ge_session_info_.session_ = std::make_shared<ge::Session>(ge_options_container_.GeSessionOptions());
    if (ge_session_info_.session_ == nullptr) {
      MS_LOG(ERROR) << "Failed to create ge session";
      return false;
    }
    ge_session_info_.session_id_ = global_session_idx++;
  }
  if (CheckParallelCompile()) {
    MS_LOG(WARNING) << lite::kCompileGraphParallel << " does not support ge provider";
  }

  std::lock_guard lock(g_compile_graph_mutex);
  bool is_adapted = graph->has_attr(kIsAdapted);
  if (!is_adapted) {
    auto ret = GeUtils::AdaptGraph(graph);
    MS_CHECK_TRUE_MSG(ret == kSuccess, false, "Adapt graph failed");
    graph->set_attr(kIsAdapted, MakeValue(true));
  }
  if (!ge_graph_compiler_.CompileGraph(graph, &ge_session_info_, ge_options_container_)) {
    MS_LOG(ERROR) << "GE compile graph failed.";
    return false;
  }
  *graph_id = ge_session_info_.graph_ids_.back();
  if (!InitGEResource()) {
    MS_LOG(ERROR) << "Init resource for GE failed.";
    return false;
  }
  if (!InitMsTensor(graph, *graph_id)) {
    MS_LOG(ERROR) << "Init MSTensor for inputs/outputs failed.";
    return false;
  }
  if (!InitGeTensor(*graph_id)) {
    MS_LOG(ERROR) << "Init ge::Tensor for inputs/outputs failed.";
    return false;
  }
  return true;
}

bool GeGraphExecutorV1::CheckParallelCompile() {
  auto config_it = config_infos_.find(lite::kCommonContextSection);
  if (config_it == config_infos_.end()) {
    return false;
  }
  auto &options = config_it->second;
  auto option_it = options.find(lite::kCompileGraphParallel);
  if (option_it == options.end()) {
    return false;
  }
  return option_it->second == lite::kEnableValue;
}

bool GeGraphExecutorV1::InitGEResource() {
  if (memory_manager_ == nullptr) {
    memory_manager_ = std::make_shared<GeMemoryManager>();
    if (memory_manager_ == nullptr) {
      MS_LOG(ERROR) << "Failed to create memory manager";
      return false;
    }
  }
  if (context_manager_ == nullptr) {
    context_manager_ = std::make_shared<GeContextManager>();
    if (context_manager_ == nullptr) {
      MS_LOG(ERROR) << "Failed to create context manager";
      return false;
    }
    auto ascend_info = GeUtils::GetAscendDeviceInfo(context_);
    if (ascend_info == nullptr) {
      MS_LOG(ERROR) << "Can not find ascend device context.";
      return false;
    }
    if (!context_manager_->InitContext(ascend_info->GetDeviceID())) {
      MS_LOG(ERROR) << "Failed to init device";
      return false;
    }
  }
  if (!context_manager_->SetContext()) {
    MS_LOG(ERROR) << "Failed to set ge context";
    return false;
  }
  return true;
}

bool GeGraphExecutorV1::InitMsTensor(const FuncGraphPtr &graph, uint32_t graph_id) {
  auto create_func = [](const std::vector<AnfWithOutIndex> &nodes, std::vector<mindspore::MSTensor> *ms_tensors,
                        bool is_input) {
    std::string prefix = is_input ? "Input_" : "Output_";
    for (size_t i = 0; i < nodes.size(); ++i) {
      auto shape = FuncGraphUtils::GetTensorShape(nodes[i]);
      auto dtype = static_cast<DataType>(FuncGraphUtils::GetTensorDataType(nodes[i]));
      auto ms_tensor_ptr = mindspore::MSTensor::CreateTensor(prefix + std::to_string(i), dtype, {}, nullptr, 0);
      if (ms_tensor_ptr == nullptr) {
        MS_LOG(ERROR) << "Create " << prefix + std::to_string(i) << " MSTensor failed.";
        return false;
      }
      ms_tensor_ptr->SetShape(shape);
      ms_tensors->push_back(*ms_tensor_ptr);
      delete ms_tensor_ptr;
    }
    return true;
  };
  auto inputs = graph->get_inputs();
  std::vector<AnfWithOutIndex> in_nodes;
  (void)std::transform(inputs.begin(), inputs.end(), std::back_inserter(in_nodes),
                       [](const AnfNodePtr &input) { return std::make_pair(input, 0); });
  ms_inputs_[graph_id] = {};
  if (!create_func(in_nodes, &ms_inputs_[graph_id], true)) {
    MS_LOG(ERROR) << "Create MSTensor for inputs failed.";
    return false;
  }
  std::vector<AnfWithOutIndex> out_nodes;
  if (!FuncGraphUtils::GetFuncGraphOutputs(graph, &out_nodes)) {
    MS_LOG(ERROR) << "Failed to get func graph outputs";
    return false;
  }
  ms_outputs_[graph_id] = {};
  if (!create_func(out_nodes, &ms_outputs_[graph_id], false)) {
    MS_LOG(ERROR) << "Create MSTensor for outputs failed.";
    return false;
  }
  return true;
}

bool GeGraphExecutorV1::InitGeTensor(uint32_t graph_id) {
  // Delayed HBM memory allocation.
  auto create_func = [](const std::vector<mindspore::MSTensor> &ms_tensors, std::vector<GeTensor> *ge_tensors) {
    ge_tensors->resize(ms_tensors.size());
    for (size_t i = 0; i < ms_tensors.size(); ++i) {
      auto dtype = static_cast<TypeId>(ms_tensors[i].DataType());
      auto desc = device::ascend::TransformUtil::GetGeTensorDesc({}, dtype, kOpFormat_NCHW);
      if (desc == nullptr) {
        MS_LOG(ERROR) << "Failed to create Tensor Desc";
        return false;
      }
      desc->SetPlacement(::ge::kPlacementDevice);
      auto ret = ge_tensors->at(i).SetTensorDesc(*desc);
      if (ret != ACL_SUCCESS) {
        MS_LOG(ERROR) << "Failed to call ge::Tensor::SetTensorDesc, ret " << ret;
        return false;
      }
    }
    return true;
  };
  ge_inputs_[graph_id] = {};
  if (!create_func(ms_inputs_[graph_id], &ge_inputs_[graph_id])) {
    MS_LOG(ERROR) << "Create ge::Tensor for inputs failed.";
    return false;
  }
  ge_outputs_[graph_id] = {};
  if (!create_func(ms_outputs_[graph_id], &ge_outputs_[graph_id])) {
    MS_LOG(ERROR) << "Create ge::Tensor for outputs failed.";
    return false;
  }
  return true;
}

std::vector<mindspore::MSTensor> GeGraphExecutorV1::GetInputInfos(uint32_t graph_id) {
  return ms_inputs_.find(graph_id) != ms_inputs_.end() ? ms_inputs_.at(graph_id) : std::vector<mindspore::MSTensor>();
}
std::vector<mindspore::MSTensor> GeGraphExecutorV1::GetOutputInfos(uint32_t graph_id) {
  return ms_outputs_.find(graph_id) != ms_outputs_.end() ? ms_outputs_.at(graph_id)
                                                         : std::vector<mindspore::MSTensor>();
}

bool GeGraphExecutorV1::RunGraph(uint32_t graph_id, const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                                 const std::map<string, string> & /* compile_options */) {
  if (outputs == nullptr) {
    MS_LOG(ERROR) << " outputs param is nullptr.";
    return false;
  }

  MS_LOG(INFO) << "Run ge graph [" << graph_id << "] with " << inputs.size() << " ms_tensor_inputs";
  if (!PrepareGeInputs(inputs, graph_id)) {
    MS_LOG(ERROR) << "Prepare ge inputs failed.";
    return false;
  }
  if (!PrepareGeOutputs(outputs, graph_id)) {
    MS_LOG(ERROR) << "Prepare ge outputs failed.";
    return false;
  }
  auto stream = context_manager_->GetDefaultStream();
  auto time_start = std::chrono::system_clock::now();
  auto ret =
    ge_session_info_.session_->RunGraphWithStreamAsync(graph_id, stream, ge_inputs_[graph_id], ge_outputs_[graph_id]);
  if (ret != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "Call GE RunGraphWithStreamAsync Failed, ret is: " << ret;
    return false;
  }
  if (!context_manager_->SyncStream(stream)) {
    MS_LOG(ERROR) << "Sync stream for RunGraphWithStreamAsync failed";
    return false;
  }
  auto time_cost =
    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - time_start).count();
  MS_LOG(INFO) << "Call GE RunGraph Success in " << time_cost << " us, graph id " << graph_id;
  if (!PostProcessGeOutputs(outputs, graph_id)) {
    MS_LOG(ERROR) << "PostPrecess ge outputs failed.";
    return false;
  }
  return true;
}

// Todo
bool GeGraphExecutorV1::PrepareGeInputs(const std::vector<MSTensor> &inputs, uint32_t graph_id) { return true; }

// Todo
bool GeGraphExecutorV1::PrepareGeOutputs(std::vector<MSTensor> *outputs, uint32_t graph_id) { return true; }

// Todo
bool GeGraphExecutorV1::PostProcessGeOutputs(std::vector<MSTensor> *outputs, uint32_t graph_id) { return true; }

static std::shared_ptr<LiteGraphExecutor> GeGraphExecutorCreatorV1(const std::shared_ptr<Context> &ctx,
                                                                   const ConfigInfos &config_infos) {
  auto ge_executor = std::make_shared<GeGraphExecutorV1>(ctx, config_infos);
  if (ge_executor == nullptr || !ge_executor->Init()) {
    MS_LOG(ERROR) << "Failed to init GeGraphExecutor";
    return nullptr;
  }
  return ge_executor;
}

REG_DELEGATE(kAscend, kProviderGeV1, GeGraphExecutorCreatorV1)
}  // namespace mindspore
