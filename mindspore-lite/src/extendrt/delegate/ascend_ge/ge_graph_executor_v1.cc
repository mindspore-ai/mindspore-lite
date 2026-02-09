/**
 * Copyright 2025-2026 Huawei Technologies Co., Ltd
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
#include "external/ge_common/ge_api_error_codes.h"
#include "tools/common/graph_util.h"
#include "utils/ms_utils_secure.h"

namespace mindspore {
namespace {
constexpr auto kProviderGeV1 = "ge-v1";
constexpr auto kIsAdapted = "is_adapted";
std::mutex g_compile_graph_mutex;
constexpr size_t kAlignRefData = 32;
std::atomic_int64_t global_session_idx = 0;
// provide an empty deleter
static void EmptyFree(uint8_t *) {}
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
  for (auto &it1 : ge_inputs_) {
    for (auto &it2 : it1.second) {
      if (it2.second.first) {
        memory_manager_->FreeDeviceMemory(it2.second.first);
      }
    }
  }
  for (auto &it1 : ge_outputs_) {
    for (auto &it2 : it1.second) {
      void *device_ptr = it2.second.first;
      if (device_ptr != nullptr) {
        memory_manager_->FreeDeviceMemory(device_ptr);
      }
    }
  }
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
  auto graph_summary = ge_session_info_.session_->GetCompiledGraphSummary(*graph_id);
  if (graph_summary == nullptr) {
    MS_LOG(ERROR) << "GetCompiledGraphSummary failed for graph " << graph_id;
    return false;
  }
  graph_id_group_.emplace(*graph_id, std::make_pair(*graph_id, UINT32_MAX));
  if (!graph_summary->IsStatic()) {
    if (!ge_graph_compiler_.ReCompileGraph(&ge_session_info_, ge_options_container_,
                                           &graph_id_group_.at(*graph_id).second)) {
      MS_LOG(ERROR) << "GE compile  graph  secondly failed.";
      return false;
    }
    ge_session_info_.df_ptr_ = nullptr;
  }
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
  auto create_func = [this, graph_id](const std::vector<mindspore::MSTensor> &ms_tensors,
                                      std::vector<std::pair<GeTensor, std::pair<void *, size_t>>> *ge_tensors,
                                      bool graph_input) {
    std::vector<std::vector<int64_t>> shapes;
    if (!graph_input) {
      if (graph_id_group_[graph_id].second != UINT32_MAX) {
        shapes.resize(ms_tensors.size());
      } else {
        auto summary = this->ge_session_info_.session_->GetCompiledGraphSummary(graph_id);
        std::vector<ge::Shape> ge_shapes;
        auto res = summary->GetOutputShapes(ge_shapes);
        if (res != ge::GRAPH_SUCCESS) {
          MS_LOG(ERROR) << "GetOutputShapes failed!";
          return false;
        }
        (void)std::transform(ge_shapes.begin(), ge_shapes.end(), std::back_inserter(shapes),
                             [](const ge::Shape &ge_shape) { return ge_shape.GetDims(); });
      }
    } else {
      shapes.resize(ms_tensors.size());
    }
    if (shapes.size() != ms_tensors.size()) {
      MS_LOG(ERROR) << "The number of shape is different with that of tensor when initializing graph "
                    << (graph_input ? "input" : "output") << ", which is " << shapes.size() << " VS "
                    << ms_tensors.size();
    }
    ge_tensors->resize(ms_tensors.size());
    for (size_t i = 0; i < ms_tensors.size(); ++i) {
      auto dtype = static_cast<TypeId>(ms_tensors[i].DataType());
      auto desc = device::ascend::TransformUtil::GetGeTensorDesc(shapes[i], dtype, kOpFormat_NCHW);
      if (desc == nullptr) {
        MS_LOG(ERROR) << "Failed to create Tensor Desc";
        return false;
      }
      desc->SetPlacement(::ge::kPlacementDevice);
      auto ret = ge_tensors->at(i).first.SetTensorDesc(*desc);
      if (ret != ACL_SUCCESS) {
        MS_LOG(ERROR) << "Failed to call ge::Tensor::SetTensorDesc, ret " << ret;
        return false;
      }
      ge_tensors->at(i).second = {nullptr, 0};
    }
    return true;
  };
  ge_inputs_[graph_id] = {};
  if (!create_func(ms_inputs_[graph_id], &ge_inputs_[graph_id], true)) {
    MS_LOG(ERROR) << "Create ge::Tensor for inputs failed.";
    return false;
  }
  ge_outputs_[graph_id] = {};
  if (!create_func(ms_outputs_[graph_id], &ge_outputs_[graph_id], false)) {
    MS_LOG(ERROR) << "Create ge::Tensor for outputs failed.";
    return false;
  }
  return true;
}

bool GeGraphExecutorV1::IsDynamical(const std::vector<MSTensor> &outputs, uint32_t graph_id) {
  if (graph_id_group_[graph_id].second != UINT32_MAX) {
    if (outputs.empty()) {
      return true;
    }
    for (size_t i = 0; i < outputs.size(); i++) {
      auto output = outputs[i];
      if (output.GetDeviceData() == nullptr) {
        return true;
      }
    }
    return false;
  }
  return false;
}

std::vector<mindspore::MSTensor> GeGraphExecutorV1::GetInputInfos(uint32_t graph_id) {
  return ms_inputs_.find(graph_id) != ms_inputs_.end() ? ms_inputs_.at(graph_id) : std::vector<mindspore::MSTensor>();
}
std::vector<mindspore::MSTensor> GeGraphExecutorV1::GetOutputInfos(uint32_t graph_id) {
  return ms_outputs_.find(graph_id) != ms_outputs_.end() ? ms_outputs_.at(graph_id)
                                                         : std::vector<mindspore::MSTensor>();
}

bool GeGraphExecutorV1::RunStaticGraph(uint32_t graph_id, const std::vector<GeTensor> &ge_inputs,
                                       std::vector<MSTensor> *outputs) {
  std::vector<GeTensor> ge_outputs;
  if (!PrepareGeOutputsForStatic(outputs, &ge_outputs, graph_id)) {
    MS_LOG(ERROR) << "Prepare ge outputs for static graph  failed.";
    return false;
  }
  auto time_start = lite::GetTimeUs();
  auto stream = context_manager_->GetDefaultStream();
  auto ret =
    ge_session_info_.session_->RunGraphWithStreamAsync(graph_id_group_[graph_id].first, stream, ge_inputs, ge_outputs);
  if (ret != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "Call GE RunGraphWithStreamAsync Failed, ret is: " << ret;
    return false;
  }
  if (!context_manager_->SyncStream(stream)) {
    MS_LOG(ERROR) << "Sync stream for RunGraphWithStreamAsync failed";
    return false;
  }
  auto time_cost = lite::GetTimeUs() - time_start;
  MS_LOG(INFO) << "Call GE RunGraph Success in " << static_cast<float>(time_cost) / 1000.0f << " ms, graph id "
               << graph_id;
  auto status = PostProcessOutputsForStatic(outputs, graph_id);
  if (!status) {
    MS_LOG(ERROR) << "PostProcess ge outputs failed.";
    return false;
  }
  return true;
}

bool GeGraphExecutorV1::RunDynamicGraph(uint32_t graph_id, const std::vector<GeTensor> &ge_inputs,
                                        std::vector<MSTensor> *outputs) {
  std::vector<GeTensor> ge_outputs;
  bool is_finished = false;
  bool end_of_sequence = false;
  std::promise<void> promise;
  auto time_start = lite::GetTimeUs();
  auto call_back = [&ge_outputs, &is_finished, &end_of_sequence, &promise](ge::Status ge_status,
                                                                           const std::vector<ge::Tensor> &outputs) {
    if (ge_status == ge::GRAPH_SUCCESS) {
      ge_outputs = outputs;
      is_finished = true;
    } else if (ge_status == ge::END_OF_SEQUENCE) {
      end_of_sequence = true;
    } else {
      MS_LOG(ERROR) << "RunAsync failed." << ge::GEGetErrorMsg();
    }
    promise.set_value();
  };
  if (ge_session_info_.session_ == nullptr) {
    MS_LOG(ERROR) << "The GE session is null, can't run the graph!";
    return false;
  }
  ge::Status ret = ge_session_info_.session_->RunGraphAsync(graph_id_group_[graph_id].second, ge_inputs, call_back);
  if (ret != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "Call GE RunGraphAsync Failed: " << ge::GEGetErrorMsg();
    return false;
  }
  auto future = promise.get_future();
  future.wait();
  if (end_of_sequence) {
    MS_LOG(ERROR) << "Failed to call GE RunGraphAsync: End of sequence";
    return false;
  }
  if (!is_finished) {
    MS_LOG(ERROR) << "Failed to call GE RunGraphAsync";
    return false;
  }
  auto time_cost = lite::GetTimeUs() - time_start;
  MS_LOG(INFO) << "Call GE RunGraph Success in " << static_cast<float>(time_cost) / 1000.0f << " ms, graph id "
               << graph_id;
  auto status = PostProcessOutputsForDynamic(outputs, graph_id, ge_outputs);
  if (!status) {
    MS_LOG(ERROR) << "PostProcess ge outputs failed.";
    return false;
  }
  return true;
}

bool GeGraphExecutorV1::RunGraph(uint32_t graph_id, const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                                 const std::map<string, string> & /* compile_options */) {
  if (outputs == nullptr) {
    MS_LOG(ERROR) << " outputs param is nullptr.";
    return false;
  }
  MS_LOG(INFO) << "Run ge graph [" << graph_id << "] with " << inputs.size() << " ms_tensor_inputs";
  std::vector<GeTensor> ge_inputs;
  if (!PrepareGeInputs(inputs, &ge_inputs, graph_id)) {
    MS_LOG(ERROR) << "Prepare ge inputs failed.";
    return false;
  }
  auto run_dynamic_graph = IsDynamical(*outputs, graph_id);
  if (run_dynamic_graph) {
    if (!RunDynamicGraph(graph_id, ge_inputs, outputs)) {
      MS_LOG(ERROR) << "RunDynamicGraph failed.";
      return false;
    }
  } else {
    if (!RunStaticGraph(graph_id, ge_inputs, outputs)) {
      MS_LOG(ERROR) << "RunStaticGraph failed.";
      return false;
    }
  }
  return true;
}

bool GeGraphExecutorV1::MallocDeviceMem(std::pair<void *, size_t> &tensor_mem_info, void *&device_addr, size_t size) {
  if (size > tensor_mem_info.second) {
    if (tensor_mem_info.first) {
      memory_manager_->FreeDeviceMemory(tensor_mem_info.first);
      tensor_mem_info = {nullptr, 0};
    }
    device_addr = memory_manager_->MallocDeviceMemory("Data", size);
    if (device_addr == nullptr) {
      MS_LOG(ERROR) << "Malloc device memory failed.";
      return false;
    }
    tensor_mem_info = {device_addr, size};
  } else {
    device_addr = tensor_mem_info.first;
  }
  return true;
}

bool GeGraphExecutorV1::PrepareGeInputs(const std::vector<MSTensor> &inputs, std::vector<GeTensor> *ge_inputs,
                                        uint32_t graph_id) {
  if (ge_inputs == nullptr) {
    MS_LOG(ERROR) << "ge_inputs pointer is null.";
    return false;
  }
  if (ge_inputs_[graph_id].size() != inputs.size()) {
    MS_LOG(ERROR) << "ge_inputs_[graph_id].size()!=inputs.size()!!!";
    return false;
  }
  for (size_t i = 0; i < inputs.size(); i++) {
    auto input = inputs[i];
    auto &it = ge_inputs_[graph_id][i];
    auto desc = it.first.GetTensorDesc();
    desc.SetShape(::ge::Shape(input.Shape()));
    desc.SetOriginShape(::ge::Shape(input.Shape()));
    it.first.SetTensorDesc(desc);
    auto size = input.DataSize();
    void *device_addr = nullptr;
    if (input.GetDeviceData() != nullptr) {
      device_addr = input.GetDeviceData();
    } else if (input.Data() != nullptr) {
      if (!MallocDeviceMem(it.second, device_addr, size)) {
        MS_LOG(ERROR) << "malloc input ge_tensor device memory failed.";
      }
      auto mem_ret = memory_manager_->MemcpyHost2Device(device_addr, size, input.MutableData(), size);
      if (!mem_ret) {
        MS_LOG(ERROR) << "Failed to H2D, input " << i;
        return false;
      }
    } else {
      MS_LOG(ERROR) << "Provided graph-input " << i << " has no data.";
      return false;
    }
    // cppcheck-suppress internalAstError
    auto ret = it.first.SetData(static_cast<uint8_t *>(device_addr), size, EmptyFree);
    if (ret != ge::GRAPH_SUCCESS) {
      MS_LOG(ERROR) << "Failed to call ge::Tensor SetData for graph-input " << i;
      return false;
    }
    ge_inputs->push_back(it.first);
  }
  return true;
}

bool GeGraphExecutorV1::PrepareGeOutputsForStatic(std::vector<MSTensor> *outputs, std::vector<GeTensor> *ge_outputs,
                                                  uint32_t graph_id) {
  auto fill_addr_func = [this](std::pair<GeTensor, std::pair<void *, size_t>> *it, const std::vector<int64_t> &shape) {
    size_t size = GetSizeByDataType(it->first.GetDataType());
    size = std::accumulate(shape.begin(), shape.end(), size, std::multiplies<>());
    void *device_addr = nullptr;
    if (!MallocDeviceMem(it->second, device_addr, size)) {
      MS_LOG(ERROR) << "malloc output ge_tensor device memory failed.";
    }
    it->first.SetData(static_cast<uint8_t *>(device_addr), size, EmptyFree);
    return true;
  };
  if (outputs->empty()) {
    for (auto &it : ge_outputs_[graph_id]) {
      auto ge_tensor = it.first;
      auto desc = ge_tensor.GetTensorDesc();
      auto ge_shape = desc.GetOriginShape();
      desc.SetShape(ge_shape);
      ge_tensor.SetTensorDesc(desc);
      auto shape = ge_shape.GetDims();
      if (!fill_addr_func(&it, shape)) {
        MS_LOG(ERROR) << "Fill device-addr to graph-output failed.";
        return false;
      }
      ge_outputs->push_back(it.first);
    }
    return true;
  }
  if (outputs->size() != ge_outputs_[graph_id].size()) {
    MS_LOG(ERROR) << "Provided graph-output's number is not equal ge::outputs, which is " << outputs->size() << " VS "
                  << ge_outputs_[graph_id].size();
    return false;
  }
  for (size_t i = 0; i < outputs->size(); ++i) {
    auto output = outputs->at(i);
    auto &it = ge_outputs_[graph_id][i];
    auto ms_shape = output.Shape();
    auto shape = ms_shape;
    auto desc = it.first.GetTensorDesc();
    auto ge_shape = desc.GetOriginShape();
    auto is_determined = std::all_of(ms_shape.begin(), ms_shape.end(), [](int64_t dim) { return dim > 0; });
    if (!is_determined) {
      shape = ge_shape.GetDims();
    }
    desc.SetShape(::ge::Shape(shape));
    it.first.SetTensorDesc(desc);
    if (output.GetDeviceData() != nullptr) {
      auto size = output.DataSize();
      auto ret = it.first.SetData(static_cast<uint8_t *>(output.GetDeviceData()), size, EmptyFree);
      if (ret != ge::GRAPH_SUCCESS) {
        MS_LOG(ERROR) << "Failed to call ge::Tensor SetData(uint8_t*, size, DeleteFunc) for graph-output " << i;
        return false;
      }
    } else {
      if (!fill_addr_func(&it, shape)) {
        MS_LOG(ERROR) << "Fill device-addr to graph-output failed.";
        return false;
      }
    }
    ge_outputs->push_back(it.first);
  }
  return true;
}

bool GeGraphExecutorV1::PostProcessOutputsForDynamic(std::vector<MSTensor> *outputs, uint32_t graph_id,
                                                     const std::vector<GeTensor> &outputs_ge_tensors) {
  if (outputs->empty()) {
    for (size_t i = 0; i < outputs_ge_tensors.size(); ++i) {
      auto name = ms_outputs_[graph_id][i].Name();
      auto ge_tensor = outputs_ge_tensors[i];
      auto data_type = ms_outputs_[graph_id][i].DataType();
      auto ms_tensor = MSTensor(name, data_type, {}, nullptr, 0);
      ms_tensor.SetShape(ge_tensor.GetTensorDesc().GetShape().GetDims());
      outputs->push_back(ms_tensor);
    }
  }
  for (size_t i = 0; i < outputs_ge_tensors.size(); ++i) {
    auto ge_tensor = outputs_ge_tensors[i];
    auto ms_tensor = outputs->at(i);
    auto size = ge_tensor.GetSize();
    auto data_addr = ge_tensor.GetData();
    if (ms_tensor.DataSize() < size) {
      MS_LOG(ERROR) << "Output[" << i << "] DataSize notice: "
                    << "Allocated: " << ms_tensor.DataSize() << ", Actual data: " << size;
      return false;
    }
    if (ms_tensor.GetDeviceData() != nullptr) {
      auto mem_ret = memory_manager_->MemcpyHost2Device(ms_tensor.GetDeviceData(), size, data_addr, size);
      if (!mem_ret) {
        MS_LOG(ERROR) << "Failed to H2D, output " << i;
        return false;
      }
    } else {
      if (ms_tensor.MutableData() == nullptr) {
        MS_LOG(ERROR) << "ms_tensor.MutableData() = nullptr!" << i;
        return false;
      }
      auto mem_ret = common::huge_memcpy(static_cast<uint8_t *>(ms_tensor.MutableData()), size, data_addr, size);
      if (mem_ret != EOK) {
        MS_LOG(ERROR) << "Failed to copy output data, dst size: " << ms_tensor.DataSize()
                      << ", src size: " << ge_tensor.GetSize();
        return false;
      }
    }
  }
  return true;
}

bool GeGraphExecutorV1::PostProcessOutputsForStatic(std::vector<MSTensor> *outputs, uint32_t graph_id) {
  if (outputs->empty()) {
    for (size_t i = 0; i < ge_outputs_[graph_id].size(); ++i) {
      auto name = ms_outputs_[graph_id][i].Name();
      auto ms_tensor = MSTensor(name, ms_outputs_[graph_id][i].DataType(), {}, nullptr, 0);
      outputs->push_back(ms_tensor);
    }
  }
  for (size_t i = 0; i < outputs->size(); ++i) {
    auto ms_tensor = outputs->at(i);
    auto &it = ge_outputs_[graph_id][i];
    if (ms_tensor.GetDeviceData() != nullptr) {
      it.first.ResetData(nullptr, 0, EmptyFree);
      continue;
    }
    if (ms_tensor.Data() == nullptr) {
      ms_tensor.SetDataType(ms_outputs_[graph_id][i].DataType());
      ms_tensor.SetShape(it.first.GetTensorDesc().GetShape().GetDims());
    }
    if (ms_tensor.DataSize() > it.second.second) {
      MS_LOG(ERROR) << "The data-size of MSTensor is more than that of GETensor. which is " << ms_tensor.DataSize()
                    << " VS " << it.second.second;
      return false;
    }
    auto mem_ret = memory_manager_->MemcpyDevice2Host(ms_tensor.MutableData(), ms_tensor.DataSize(), it.second.first,
                                                      ms_tensor.DataSize());
    if (!mem_ret) {
      MS_LOG(ERROR) << "Failed to D2H, output " << i;
      return false;
    }
  }
  return true;
}

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
