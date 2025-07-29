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
constexpr size_t kSupportedWeightNum = 1;
}  // namespace

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

std::shared_ptr<AclModelOptions> AclGraphExecutor::GenAclOptions() {
  auto device_id = 0;
  auto acl_options_ptr = std::make_shared<AclModelOptions>();
  if (acl_options_ptr == nullptr) {
    MS_LOG(ERROR) << "Acl options make shared failed.";
    return nullptr;
  }
  std::string profiling_path = GetConfigOption(lite::kAscendContextSection, lite::kProfilingPathKey);
  if (profiling_path != "") {
    acl_options_ptr->profiling_path = profiling_path;
  }

  std::string dump_path = GetConfigOption(lite::kAscendContextSection, lite::kDumpPathKey);
  if (dump_path != "") {
    acl_options_ptr->dump_path = dump_path;
  }

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

  std::string bundle_model = GetConfigOption(lite::kInnerCommon, lite::kBundleModel);
  if (bundle_model != "") {
    bool is_bundle_model = bundle_model == "true" ? true : false;
    acl_options_ptr->is_bundle_model = is_bundle_model;
  }
  acl_options_ptr->device_id = static_cast<int32_t>(device_id);
  return acl_options_ptr;
}

bool AclGraphExecutor::UpdateWeights(const std::vector<std::vector<MSTensor>> &inputs) {
  MS_CHECK_TRUE_MSG(model_infer_ != nullptr, false, "model_infer_ is nullptr!");
  MS_CHECK_TRUE_MSG(inputs.size() == kSupportedWeightNum, false, "Only support single weight now!");
  return model_infer_->UpdateWeights(inputs[0]);
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
  if (acl_executor == nullptr || acl_executor->Init() != kSuccess) {
    MS_LOG(ERROR) << "Failed to init GeGraphExecutor";
    return nullptr;
  }
  return acl_executor;
}

REG_DELEGATE(kAscend, kProviderAcl, AclGraphExecutorCreator)
}  // namespace mindspore
