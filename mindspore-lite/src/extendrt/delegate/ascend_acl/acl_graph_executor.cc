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

#include "plugin/res_manager/ascend/symbol_interface/acl_base_symbol.h"
#include "plugin/res_manager/ascend/symbol_interface/acl_mdl_symbol.h"
#include "plugin/res_manager/ascend/symbol_interface/acl_rt_symbol.h"
#include "plugin/res_manager/ascend/symbol_interface/acl_symbol.h"
#include "plugin/res_manager/ascend/symbol_interface/symbol_utils.h"
namespace mindspore {
namespace {
constexpr auto kProviderAcl = "litert";
}  // namespace

Status AclGraphExecutor::Init() {
  auto device_list = context_->MutableDeviceInfo();
  for (const auto &device_info : device_list) {
    if (device_info == nullptr) {
      MS_LOG(ERROR) << "Device info get from Context cannot be nullptr";
      return kLiteError;
    }
    if (device_info->GetDeviceType() == DeviceType::kAscend) {
      bool is_registered = AscendAllocatorPlugin::GetInstance().Register();
      if (!is_registered) {
        MS_LOG(ERROR) << "AscendAllocatorPlugin failed to register, cannot do acl memory operations";
        return kLiteError;
      }
      auto ascend_device_info = device_info->Cast<mindspore::AscendDeviceInfo>();
      if (ascend_device_info == nullptr) {
        MS_LOG(ERROR) << "Failed to cast device info to AscendDeviceInfo";
        return kLiteError;
      }
      return kSuccess;
    }
  }
  return kSuccess;
}

std::shared_ptr<AclModelOptions> AclGraphExecutor::GenAclOptions() {
  auto acl_options_ptr = std::make_shared<AclModelOptions>();
  if (acl_options_ptr == nullptr) {
    MS_LOG(ERROR) << "Acl options make shared failed.";
    return nullptr;
  }
  auto profiling_path_val = primitive_->GetAttr(lite::kProfilingPathKey);
  if (profiling_path_val != nullptr) {
    auto val = GetValue<std::string>(profiling_path_val);
    acl_options_ptr->profiling_path = val;
  }
  auto dump_path_val = primitive_->GetAttr(lite::kDumpPathKey);
  if (dump_path_val != nullptr) {
    auto val = GetValue<std::string>(dump_path_val);
    acl_options_ptr->dump_path = val;
  }
  auto inner_calc_workspace_size = primitive_->GetAttr(lite::kInnerCalcWorkspaceSize);
  if (inner_calc_workspace_size != nullptr) {
    auto val = GetValue<bool>(inner_calc_workspace_size);
    acl_options_ptr->multi_model_sharing_mem_prepare = val;
    // is_multi_model_sharing_mem_prepare_ = true;
  }
  auto inner_sharing_workspace = primitive_->GetAttr(lite::kInnerSharingWorkspace);
  if (inner_sharing_workspace != nullptr) {
    auto val = GetValue<bool>(inner_sharing_workspace);
    acl_options_ptr->multi_model_sharing_mem = val;
  }
  auto inner_model_path = primitive_->GetAttr(lite::kInnerModelPath);
  if (inner_model_path != nullptr) {
    auto val = GetValue<std::string>(inner_model_path);
    acl_options_ptr->model_path = val;
  }
  auto workspace_key = primitive_->GetAttr(lite::kInnerWorkspace);
  if (workspace_key != nullptr) {
    auto val = GetValue<bool>(workspace_key);
    acl_options_ptr->share_workspace = val;
  }
  auto weightspace_key = primitive_->GetAttr(lite::kInnerWeightspace);
  if (weightspace_key != nullptr) {
    auto val = GetValue<bool>(weightspace_key);
    acl_options_ptr->share_weightspace = val;
  }
  auto weightspace_workspace_key = primitive_->GetAttr(lite::kInnerWeightspaceWorkspace);
  if (weightspace_workspace_key != nullptr) {
    auto val = GetValue<bool>(weightspace_workspace_key);
    acl_options_ptr->share_weightspace_workspace = val;
  }
  auto bundle_model = primitive_->GetAttr(lite::kBundleModel);
  if (bundle_model != nullptr) {
    auto val = GetValue<bool>(bundle_model);
    acl_options_ptr->is_bundle_model = val;
  }
  acl_options_ptr->device_id = static_cast<int32_t>(0);
  return acl_options_ptr;
}

bool AclGraphExecutor::CompileGraph(const FuncGraphPtr &graph, const std::map<string, string> &compile_options,
                                    uint32_t *graph_id) {
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
    // for (size_t i = 0; i < inputs.size(); i++) {
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
  // todo
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
  AclEnvGuard::AddModel(model_infer_);
  return true;
}

bool AclGraphExecutor::Resize(uint32_t graph_id, const std::vector<mindspore::MSTensor> &inputs,
                              const std::vector<std::vector<int64_t>> &dims) {
  (void)model_infer_->Resize(dims);
  return true;
}

std::vector<mindspore::MSTensor> AclGraphExecutor::GetOutputInfos(uint32_t graph_id) {
  auto output_infos = graph_outputs_.find(graph_id) != graph_outputs_.end() ? graph_outputs_.at(graph_id)
                                                                            : std::vector<mindspore::MSTensor>();
  return output_infos;
}

bool AclGraphExecutor::RunGraph(uint32_t graph_id, const std::vector<mindspore::MSTensor> &inputs,
                                std::vector<mindspore::MSTensor> *output,
                                const std::map<string, string> &compile_options) {
  auto ret = model_infer_->Inference(inputs, output);
  if (!ret) {
    MS_LOG(ERROR) << "model infer failed.";
    return false;
  }
  graph_outputs_[graph_id] = *output;
  return true;
}

static std::shared_ptr<LiteGraphExecutor> AclGraphExecutorCreator(const std::shared_ptr<Context> &ctx,
                                                                  const ConfigInfos &config_infos) {
  auto acl_executor = std::make_shared<mindspore::AclGraphExecutor>(ctx, config_infos);
  if (acl_executor == nullptr && acl_executor->Init() != kSuccess) {
    MS_LOG(ERROR) << "Failed to init GeGraphExecutor";
    return nullptr;
  }
  return acl_executor;
}

REG_DELEGATE(kAscend, kProviderAcl, AclGraphExecutorCreator)
}  // namespace mindspore
