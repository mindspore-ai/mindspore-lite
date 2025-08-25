/**
 * Copyright 2022-2024 Huawei Technologies Co., Ltd
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

#include "extendrt/delegate/ascend_acl/acl_graph_executor.h"
#include "extendrt/delegate/ascend_acl/ascend_allocator_plugin.h"
#include "extendrt/session/lite_graph_executor.h"
#include "extendrt/delegate/factory.h"
#include "extendrt/utils/func_graph_utils.h"

#include "plugin/ascend/res_manager/symbol_interface/acl_base_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_mdl_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "src/common/common.h"
#include "src/common/utils.h"
namespace mindspore {
namespace {
constexpr auto kProviderAcl = "litert";
constexpr size_t kSupportedWeightNum = 1;
}  // namespace

bool AclGraphExecutor::GetDeviceID(int32_t *device_id) {
  if (context_ != nullptr && !context_->MutableDeviceInfo().empty()) {
    auto device_info = context_->MutableDeviceInfo()[0];
    if (device_info == nullptr) {
      MS_LOG(ERROR) << "device info is nullptr!";
      return false;
    }
    if (device_info->GetDeviceType() == DeviceType::kAscend) {
      auto ascend_device = device_info->Cast<AscendDeviceInfo>();
      if (ascend_device == nullptr) {
        MS_LOG(ERROR) << "not ascend device!";
        return false;
      }
      *device_id = ascend_device->GetDeviceID();
    }
  }
  return true;
}

std::string AclGraphExecutor::GetConfigOption(const std::string &section_name, const std::string &option_name) {
  auto config_it = config_info_.find(section_name);
  if (config_it == config_info_.end()) {
    return "";
  }
  auto &options = config_it->second;
  auto option_it = options.find(option_name);
  if (option_it == options.end()) {
    return "";
  }
  return option_it->second;
}

Status AclGraphExecutor::Init() { return kSuccess; }

void AclGraphExecutor::GetShareMemInfos(std::shared_ptr<AclModelOptions> acl_options_ptr) {
  std::string multi_model_sharing_mem_prepare_value =
    GetConfigOption(lite::kInnerCommon, lite::kInnerCalcWorkspaceSize);
  if (multi_model_sharing_mem_prepare_value != "") {
    bool is_multi_model_sharing_mem_prepare = multi_model_sharing_mem_prepare_value == "true" ? true : false;
    acl_options_ptr->multi_model_sharing_mem_prepare = is_multi_model_sharing_mem_prepare;
  }

  std::string multi_model_sharing_mem_value = GetConfigOption(lite::kInnerCommon, lite::kInnerSharingWorkspace);
  if (multi_model_sharing_mem_value != "") {
    bool is_inner_sharing_workspace = multi_model_sharing_mem_value == "true" ? true : false;
    acl_options_ptr->multi_model_sharing_mem = is_inner_sharing_workspace;
  }

  std::string model_path = GetConfigOption(lite::kInnerCommon, lite::kInnerModelPath);
  if (model_path != "") {
    acl_options_ptr->model_path = model_path;
  }

  std::string share_workspace_value = GetConfigOption(lite::kInnerCommon, lite::kInnerWorkspace);
  if (share_workspace_value != "") {
    bool is_workspace = share_workspace_value == "true" ? true : false;
    acl_options_ptr->share_workspace = is_workspace;
  }

  std::string share_weightspace_value = GetConfigOption(lite::kInnerCommon, lite::kInnerWeightspace);
  if (share_weightspace_value != "") {
    bool is_weightspace = share_weightspace_value == "true" ? true : false;
    acl_options_ptr->share_weightspace = is_weightspace;
  }

  std::string weightspace_workspace_value = GetConfigOption(lite::kInnerCommon, lite::kInnerWeightspaceWorkspace);
  if (weightspace_workspace_value != "") {
    bool is_weightspace_workspace = weightspace_workspace_value == "true" ? true : false;
    acl_options_ptr->share_weightspace_workspace = is_weightspace_workspace;
  }

  std::string pids_str = GetConfigOption(lite::kInnerCommon, lite::kInnerPids);
  if (pids_str != "") {
    acl_options_ptr->pids = pids_str;
  }

  std::string shareable_handle_str = GetConfigOption(lite::kInnerCommon, lite::kInnerSharableHandle);
  if (shareable_handle_str != "") {
    acl_options_ptr->sharable_handle = std::stoull(shareable_handle_str.c_str());
  }
}

std::shared_ptr<AclModelOptions> AclGraphExecutor::GenAclOptions() {
  auto acl_options_ptr = std::make_shared<AclModelOptions>();
  MS_CHECK_TRUE_MSG(acl_options_ptr != nullptr, nullptr, "Acl options make shared failed.");
  std::string profiling_path = GetConfigOption(lite::kAscendContextSection, lite::kProfilingPathKey);
  if (profiling_path != "") {
    acl_options_ptr->profiling_path = profiling_path;
  }

  std::string dump_path = GetConfigOption(lite::kAscendContextSection, lite::kDumpPathKey);
  if (dump_path != "") {
    acl_options_ptr->dump_path = dump_path;
  }
  GetShareMemInfos(acl_options_ptr);
  std::string bundle_model = GetConfigOption(lite::kInnerCommon, lite::kBundleModel);
  if (bundle_model != "") {
    bool is_bundle_model = bundle_model == "true" ? true : false;
    acl_options_ptr->is_bundle_model = is_bundle_model;
  }
  int32_t device_id = 0;
  if (!GetDeviceID(&device_id)) {
    MS_LOG(ERROR) << "GetDeviceID failed!";
    return nullptr;
  }
  acl_options_ptr->device_id = device_id;
  auto input_name_str = GetConfigOption(lite::kInnerGraphSplit, lite::kInnerInputNames);
  if (input_name_str != "") {
    acl_options_ptr->input_names = lite::StrSplit(input_name_str, ",");
  }

  auto output_name_str = GetConfigOption(lite::kInnerGraphSplit, lite::kInnerOutputNames);
  if (output_name_str != "") {
    acl_options_ptr->output_names = lite::StrSplit(output_name_str, ",");
  }

  return acl_options_ptr;
}

bool AclGraphExecutor::UpdateWeights(const std::vector<std::vector<MSTensor>> &inputs) {
  MS_CHECK_TRUE_MSG(model_infer_ != nullptr, false, "model_infer_ is nullptr!");
  MS_CHECK_TRUE_MSG(inputs.size() == kSupportedWeightNum, false, "Only support single weight now!");
  return model_infer_->UpdateWeights(inputs[0]);
}

bool AclGraphExecutor::CompileGraph(const FuncGraphPtr &graph, const std::map<string, string> &compile_options,
                                    uint32_t *graph_id) {
  for (const auto &input : graph->get_inputs()) {
    MS_CHECK_TRUE_MSG(input != nullptr, false, "graph's inputs[i] is nullptr.");
    MS_LOG(INFO) << "input name: " << input->fullname_with_scope();
    input_names_.push_back(input->fullname_with_scope());
  }
  // Get whether the current model is a bundle model for LORA.
  if (graph->get_attr(lite::kBundleModel) != nullptr) {
    config_info_["inner_common"][lite::kBundleModel] = "true";
  }
  auto nodes = graph->TopoSort(graph->get_return());
  if (nodes.empty()) {
    MS_LOG(ERROR) << "There are no nodes in the graph";
    return false;
  }
  void *om_data = nullptr;
  size_t om_data_size = 0;
  size_t cnode_count = 0;
  BaseOperatorPtr op;
  for (const auto &node : nodes) {
    auto cnode = node->cast<CNodePtr>();
    if (!cnode || !AnfUtils::IsRealKernel(cnode)) {
      continue;
    }
    std::string kernel_name = common::AnfAlgo::GetCNodeName(cnode);
    if (kernel_name != lite::kNameCustomAscend) {
      MS_LOG(ERROR) << "Only support " << lite::kNameCustomAscend << ", but got " << kernel_name << ", node "
                    << cnode->fullname_with_scope();
      return false;
    }
    cnode_count += 1;
    if (cnode_count > 1) {
      MS_LOG(ERROR) << "Only support one " << lite::kNameCustomAscend << " node, but got " << kernel_name << ", node "
                    << cnode->fullname_with_scope();
      return false;
    }
    std::vector<AnfWithOutIndex> inputs;
    std::vector<AnfWithOutIndex> outputs;
    FuncGraphUtils::GetCNodeInputsOutputs(cnode, &inputs, &outputs);
    auto &input = inputs[inputs.size() - 1];
    auto tensor_data = FuncGraphUtils::GetConstNodeValue(input.first);
    om_data_size = tensor_data->Size();
    om_data = tensor_data->data_c();
    (void)FuncGraphUtils::GetCNodeOperator(cnode, &op);
  }
  if (om_data == nullptr || op == nullptr) {
    MS_LOG(ERROR) << "om data is nullptr.";
    return false;
  }
  primitive_ = op->GetPrim();
  auto acl_options = GenAclOptions();
  if (acl_options == nullptr) {
    MS_LOG(ERROR) << "Generate acl options failed.";
    return false;
  }

  model_infer_ = std::make_shared<ModelInfer>(acl_options);
  if (model_infer_ == nullptr) {
    MS_LOG(ERROR) << "Create ModelInfer failed.";
    return false;
  }
  if (!model_infer_->Init()) {
    MS_LOG(ERROR) << "Model infer init failed.";
    return false;
  }
  if (!model_infer_->Load(om_data, om_data_size)) {
    MS_LOG(ERROR) << "Load om data failed.";
    return false;
  }
  sharable_handle_ = model_infer_->GetSharableHandle();
  AclEnvGuard::AddModel(model_infer_);
  return true;
}

bool AclGraphExecutor::CompileGraph(const void *model_data, size_t data_size,
                                    const std::map<std::string, std::string> &compile_options, uint32_t *graph_id) {
  auto acl_options = GenAclOptions();
  if (acl_options == nullptr) {
    MS_LOG(ERROR) << "Generate acl options failed!";
    return false;
  }
  input_names_.insert(input_names_.end(), acl_options->input_names.begin(), acl_options->input_names.end());
  output_names_.insert(output_names_.end(), acl_options->output_names.begin(), acl_options->output_names.end());
  model_infer_ = std::make_shared<ModelInfer>(acl_options);
  if (model_infer_ == nullptr) {
    MS_LOG(ERROR) << "Create ModelInfer failed.";
    return false;
  }
  if (!model_infer_->Init()) {
    MS_LOG(ERROR) << "Model infer init failed.";
    return false;
  }
  if (!model_infer_->Load(model_data, data_size)) {
    MS_LOG(ERROR) << "Load om data failed.";
    return false;
  }
  AclEnvGuard::AddModel(model_infer_);
  return true;
}

bool AclGraphExecutor::Resize(uint32_t graph_id, const std::vector<mindspore::MSTensor> &inputs,
                              const std::vector<std::vector<int64_t>> &dims) {
  return model_infer_->Resize(dims);
}

std::vector<mindspore::MSTensor> AclGraphExecutor::GetInputInfos(uint32_t graph_id) {
  auto inputs_shape = model_infer_->GetInputShape();
  auto inputs_dtype = model_infer_->GetInputDataType();
  std::vector<mindspore::MSTensor> inputs;
  for (size_t i = 0; i < input_names_.size(); ++i) {
    // Create a fake tensor that includes input_names, dtype, and shape for input info.
    auto tensor = mindspore::MSTensor(input_names_[i], static_cast<enum DataType>(inputs_dtype[i]), {}, nullptr, 0);
    // To avoid internal checking for empty shape and data during creation,
    // we use `SetShape` after MSTensor is created.
    tensor.SetShape(inputs_shape[i]);
    inputs.push_back(tensor);
  }
  return inputs;
}

std::vector<mindspore::MSTensor> AclGraphExecutor::GetOutputInfos(uint32_t graph_id) {
  std::vector<mindspore::MSTensor> output_infos = {};
  if (output_names_.empty()) {
    output_infos = graph_outputs_.find(graph_id) != graph_outputs_.end() ? graph_outputs_.at(graph_id)
                                                                         : std::vector<mindspore::MSTensor>();
  } else {
    auto outputs_shape = model_infer_->GetOutputShape();
    auto outputs_dtype = model_infer_->GetOutputDataType();
    for (size_t i = 0; i < output_names_.size(); i++) {
      auto tensor = mindspore::MSTensor(output_names_[i], static_cast<enum DataType>(outputs_dtype[i]), {}, nullptr, 0);
      tensor.SetShape(outputs_shape[i]);
      output_infos.push_back(tensor);
    }
  }

  return output_infos;
}

bool AclGraphExecutor::RunGraph(uint32_t graph_id, const std::vector<mindspore::MSTensor> &inputs,
                                std::vector<mindspore::MSTensor> *output,
                                const std::map<string, string> &compile_options) {
  std::vector<std::vector<int64_t>> inputs_shape_new;
  for (auto &tensor : inputs) {
    MS_CHECK_TRUE_MSG(tensor != nullptr, false, "Input tensor is null.");
    auto tensor_shape = tensor.Shape();
    inputs_shape_new.push_back(tensor_shape);
  }
  auto inputs_shape_model = model_infer_->GetInputShape();

  if (inputs_shape_model != inputs_shape_new)
    MS_CHECK_TRUE_MSG(model_infer_->Resize(inputs_shape_new), false, "Resize input shape failed.");

  auto ret = model_infer_->Inference(inputs, output);
  if (!ret) {
    MS_LOG(ERROR) << "Model infer failed.";
    return false;
  }
  graph_outputs_[graph_id] = *output;
  return true;
}

static std::shared_ptr<LiteGraphExecutor> AclGraphExecutorCreator(const std::shared_ptr<Context> &ctx,
                                                                  const ConfigInfos &config_infos) {
  auto acl_executor = std::make_shared<mindspore::AclGraphExecutor>(ctx, config_infos);
  if (acl_executor == nullptr || acl_executor->Init() != kSuccess) {
    MS_LOG(ERROR) << "Failed to init GeGraphExecutor";
    return nullptr;
  }
  return acl_executor;
}

REG_DELEGATE(kAscend, kProviderAcl, AclGraphExecutorCreator)
}  // namespace mindspore
