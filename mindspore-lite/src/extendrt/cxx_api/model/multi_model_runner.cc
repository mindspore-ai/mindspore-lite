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
#include "include/api/multi_model_runner.h"
#include "include/api/context.h"
#include "extendrt/cxx_api/model/model_impl.h"
#include "src/common/config_file.h"
#include "src/common/common.h"
#include "load_mindir/load_model.h"
#include "extendrt/cxx_api/file_utils.h"
#include "extendrt/utils/func_graph_utils.h"
#include "include/api/dual_abi_helper.h"
#include "mindspore/core/include/ir/graph_utils.h"
#include "include/api/types.h"
#include "src/extendrt/delegate/ascend_acl/ascend_allocator_plugin.h"
namespace mindspore {
namespace {
std::mutex g_load_mindir_lock;
std::mutex g_config_lock;
constexpr size_t kMaxSectionNum = 100;
constexpr size_t kMaxConfigNumPerSection = 1000;
constexpr size_t kInferPathSize = 2;
constexpr size_t kInferPathBeginIndex = 0;
constexpr size_t kInferPathEndIndex = 1;

FuncGraphPtr LoadGraphByBufferImpl(const void *model_buff, const size_t &model_size, const ModelType &model_type,
                                   const std::shared_ptr<Context> &model_context, const std::string &model_path) {
  if (model_type != kMindIR) {
    MS_LOG(ERROR) << "Invalid model type!";
    return nullptr;
  }
  std::string weight_path = "./";
  if (model_path.find("/") != std::string::npos) {
    weight_path = model_path.substr(0, model_path.rfind("/"));
  }
  FuncGraphPtr func_graph;
  std::string user_info_string;
  {
    std::unique_lock<std::mutex> l(g_load_mindir_lock);
    MindIRLoader mindir_loader(true, nullptr, 0, kDecModeAesGcm, false);
    auto ret = mindir_loader.LoadMindIR(model_buff, model_size, weight_path, &func_graph, &user_info_string);
    if (!ret || func_graph == nullptr) {
      MS_LOG(ERROR) << "Failed to load MindIR model, please check the validity of the model.";
      return nullptr;
    }
  }
  return func_graph;
}

void SetInputOutputNames(const size_t &cnode_count, const std::vector<std::vector<std::string>> &subgraph_input_names,
                         const std::vector<std::vector<std::string>> &subgraph_output_names,
                         const std::shared_ptr<ModelImpl> &model_impl_ptr) {
  std::string input_name_str = "";
  for (size_t index = 0; index < subgraph_input_names[cnode_count - 1].size(); index++) {
    input_name_str += subgraph_input_names[cnode_count - 1][index];
    if (index != subgraph_input_names[cnode_count - 1].size() - 1) {
      input_name_str += ",";
    }
  }
  std::string output_name_str = "";
  for (size_t index = 0; index < subgraph_output_names[cnode_count - 1].size(); index++) {
    output_name_str += subgraph_output_names[cnode_count - 1][index];
    if (index != subgraph_output_names[cnode_count - 1].size() - 1) {
      output_name_str += ",";
    }
  }
  model_impl_ptr->UpdateConfig(lite::kInnerGraphSplit, std::make_pair(lite::kInnerInputNames, input_name_str));
  model_impl_ptr->UpdateConfig(lite::kInnerGraphSplit, std::make_pair(lite::kInnerOutputNames, output_name_str));
}

Status GetDeviceIdFromContext(const std::shared_ptr<Context> &model_context, int32_t *device_id) {
  MS_CHECK_TRUE_MSG(model_context != nullptr, kLiteNullptr, "model_context is nullptr!");
  if (model_context->MutableDeviceInfo().empty()) {
    MS_LOG(ERROR) << "deviceinfo of context is empty!";
    return kLiteError;
  }
  auto device_info = model_context->MutableDeviceInfo()[0];
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "device info is nullptr!";
    return kLiteError;
  }
  if (device_info->GetDeviceType() != DeviceType::kAscend) {
    MS_LOG(ERROR) << "ModelExecutor only support ascend backend!";
    return kLiteError;
  }
  auto ascend_device = device_info->Cast<AscendDeviceInfo>();
  if (ascend_device == nullptr) {
    MS_LOG(ERROR) << "not ascend device!";
    return kLiteError;
  }
  *device_id = ascend_device->GetDeviceID();
  return kSuccess;
}

Status GetOmInfoFromCnode(const CNodePtr &cnode, void **om_data, size_t *om_size) {
  MS_CHECK_TRUE_MSG(om_data != nullptr, kLiteNullptr, "om_data is nullptr!");
  MS_CHECK_TRUE_MSG(om_size != nullptr, kLiteNullptr, "om_size is nullptr!");
  std::vector<mindspore::AnfWithOutIndex> inputs;
  std::vector<mindspore::AnfWithOutIndex> outputs;
  auto ret = mindspore::FuncGraphUtils::GetCNodeInputsOutputs(cnode, &inputs, &outputs);
  if (ret == false) {
    MS_LOG(ERROR) << "GetCnodeInputsOutputs failed!";
    return kLiteError;
  }
  auto om_inputs = inputs.back();
  auto tensor_data = mindspore::FuncGraphUtils::GetConstNodeValue(om_inputs.first);
  if (tensor_data == nullptr) {
    MS_LOG(ERROR) << "tensor_data is nullptr!";
    return kLiteError;
  }
  *om_data = tensor_data->data_c();
  *om_size = tensor_data->Size();
  return kSuccess;
}

Status UpdateModelConfig(const std::string &config_file, const ConfigInfos &config_info,
                         const std::shared_ptr<ModelImpl> &model_impl_ptr) {
  if (!config_file.empty()) {
    auto ret = model_impl_ptr->LoadConfig(config_file);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "Model LoadConfig failed!";
      return ret;
    }
  }
  for (auto key_value : config_info) {
    for (auto pair : key_value.second) {
      auto ret = model_impl_ptr->UpdateConfig(key_value.first, pair);
      if (ret != kSuccess) {
        MS_LOG(ERROR) << "Model updateconfig failed!";
        return ret;
      }
    }
  }
  return kSuccess;
}

Status BuildModels(const FuncGraphPtr &func_graph, const std::vector<std::vector<std::string>> &subgraph_input_names,
                   const std::vector<std::vector<std::string>> &subgraph_output_names,
                   const std::shared_ptr<Context> &model_context, const std::string &config_file,
                   const ConfigInfos &config_info, std::vector<std::shared_ptr<ModelImpl>> *models,
                   std::vector<std::vector<MSTensor>> *model_output_tensors) {
  MS_CHECK_TRUE_MSG(func_graph != nullptr, kLiteError, "func_graph is nullptr!");
  MS_CHECK_TRUE_MSG(model_context != nullptr, kLiteError, "model_context is nullptr!");
  MS_CHECK_TRUE_MSG(models != nullptr, kLiteError, "modes is nullptr!");
  auto nodes = func_graph->TopoSort(func_graph->get_return());
  MS_CHECK_TRUE_MSG(!nodes.empty(), kLiteError, "There are no nodes in the func_graph");
  size_t cnode_count = 0;
  std::vector<void *> om_datas;
  std::vector<size_t> om_sizes;
  for (const auto &node : nodes) {
    auto cnode = node->cast<CNodePtr>();
    if (!cnode || !mindspore::AnfUtils::IsRealKernel(cnode)) {
      continue;
    }
    std::string kernel_name = mindspore::common::AnfAlgo::GetCNodeName(cnode);
    if (kernel_name != lite::kNameCustomAscend) {
      MS_LOG(ERROR) << "Only support " << lite::kNameCustomAscend << ", but got " << kernel_name << ", node "
                    << cnode->fullname_with_scope();
      return kLiteError;
    }
    cnode_count += 1;
    if (cnode_count > 1) {
      auto inputs = cnode->inputs();
      inputs.pop_back();
      cnode->set_inputs(inputs);
    }
    void *om_data = nullptr;
    size_t om_size = 0;
    auto ret = GetOmInfoFromCnode(cnode, &om_data, &om_size);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "GetOmInfo From Cnode failed!";
      return ret;
    }
    om_datas.push_back(om_data);
    om_sizes.push_back(om_size);
    auto model_impl_ptr = std::make_shared<ModelImpl>();
    MS_CHECK_TRUE_MSG(model_impl_ptr != nullptr, kLiteError, "model_impl_ptr is nullptr");
    // share work space
    model_impl_ptr->UpdateConfig(lite::kInnerCommon, std::make_pair(lite::kInnerCalcWorkspaceSize, "true"));
    ret = model_impl_ptr->Build(om_data, om_size, kOM, model_context);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "Model build failed!";
      return ret;
    }
  }
  for (size_t i = 0; i < om_datas.size(); i++) {
    auto om_data = om_datas[i];
    auto om_size = om_sizes[i];
    auto model_impl_ptr = std::make_shared<ModelImpl>();
    MS_CHECK_TRUE_MSG(model_impl_ptr != nullptr, kLiteError, "model_impl_ptr is nullptr");
    auto ret = model_impl_ptr->UpdateConfig(lite::kInnerCommon, std::make_pair(lite::kInnerSharingWorkspace, "true"));
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "UpdateConfig failed!";
      return ret;
    }
    ret = model_impl_ptr->UpdateConfig(lite::kInnerCommon, std::make_pair(lite::kInnerWorkspace, "true"));
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "UpdateConfig failed!";
      return ret;
    }
    ret = UpdateModelConfig(config_file, config_info, model_impl_ptr);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "update model config failed!";
      return ret;
    }
    SetInputOutputNames(i + 1, subgraph_input_names, subgraph_output_names, model_impl_ptr);
    ret = model_impl_ptr->Build(om_data, om_size, kOM, model_context);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "Model build failed!";
      return ret;
    }
    int32_t device_id = 0;
    ret = GetDeviceIdFromContext(model_context, &device_id);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "GetDeviceIdFromContext failed!";
      return ret;
    }
    auto model_outputs = model_impl_ptr->GetOutputs();
    std::vector<MSTensor> output_tensors = {};
    for (auto output : model_outputs) {
      auto tensor = MSTensor::CreateTensor(output.Name(), output, "ascend", device_id);
      MS_CHECK_TRUE_MSG(tensor != nullptr, kLiteError, "tensor is nullptr!");
      output_tensors.push_back(*tensor);
      delete tensor;
    }
    model_output_tensors->push_back(output_tensors);
    models->emplace_back(model_impl_ptr);
  }
  return kSuccess;
}
}  // namespace

Status MultiModelRunner::Build(const std::vector<char> &model_path, const ModelType &model_type,
                               const std::shared_ptr<Context> &model_context) {
  MS_CHECK_TRUE_MSG(!model_path.empty(), kLiteError, "Model path cannot be empty!");
  MS_CHECK_TRUE_MSG(model_context != nullptr, kLiteError, "model_context is nullptr!");
  auto buffer = ReadFile(CharToString(model_path));
  MS_CHECK_TRUE_MSG(buffer.DataSize() != 0, kLiteError, "Failed to read buffer from model file.");
  auto func_graph =
    LoadGraphByBufferImpl(buffer.Data(), buffer.DataSize(), model_type, model_context, CharToString(model_path));
  MS_CHECK_TRUE_MSG(func_graph != nullptr, kLiteError, "LoadGraphByBufferImpl failed!");
  auto subgraph_infer_path_val = func_graph->get_attr(lite::kSubgraphInferPath);
  MS_CHECK_TRUE_MSG(subgraph_infer_path_val != nullptr, kLiteError, "subgraph_infer_path is nullptr!");
  auto subgraph_infer_path = GetValue<std::vector<std::vector<int32_t>>>(subgraph_infer_path_val);
  auto subgraph_inputs_name_val = func_graph->get_attr(lite::kSubgraphInputNames);
  MS_CHECK_TRUE_MSG(subgraph_inputs_name_val != nullptr, kLiteError, "subgraph_inputs_name_val is nullptr!");
  auto subgraph_input_names = GetValue<std::vector<std::vector<std::string>>>(subgraph_inputs_name_val);
  auto subgraph_output_names_val = func_graph->get_attr(lite::kSubgraphOutputNames);
  MS_CHECK_TRUE_MSG(subgraph_output_names_val != nullptr, kLiteError, "subgraph_output_names_val is nullptr!");
  auto subgraph_output_names = GetValue<std::vector<std::vector<std::string>>>(subgraph_output_names_val);
  auto main_graph_inputs_names_val = func_graph->get_attr(lite::kGraphInputNames);
  MS_CHECK_TRUE_MSG(main_graph_inputs_names_val != nullptr, kLiteError, "main_graph_inputs_nmes_val is nullptr!");
  auto main_graph_input_names = GetValue<std::vector<std::string>>(main_graph_inputs_names_val);
  auto extended_subgraph_input_output_val = func_graph->get_attr(lite::kExtendedSubgraphInputOutput);
  MS_CHECK_TRUE_MSG(extended_subgraph_input_output_val != nullptr, kLiteError,
                    "extended_subgraph_input_output_val is nullptr");
  auto extended_subgraph_input_output =
    GetValue<std::vector<std::vector<std::vector<std::string>>>>(extended_subgraph_input_output_val);
  auto ret = BuildModels(func_graph, subgraph_input_names, subgraph_output_names, model_context, config_file_,
                         config_info_, &models_, &model_output_tensors_);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "BuildModels failed!";
    return ret;
  }
  for (size_t executor_id = 0; executor_id < subgraph_infer_path.size(); executor_id++) {
    std::vector<std::shared_ptr<ModelImpl>> curr_executor_models;
    std::vector<std::vector<MSTensor>> curr_executor_model_output_tensors;
    MS_CHECK_TRUE_MSG(extended_subgraph_input_output.size() > executor_id, kLiteError,
                      "size of extended_subgraph_input_output should larger than executor_id!");
    auto curr_executor_input_output_names = extended_subgraph_input_output[executor_id];
    std::vector<std::vector<std::string>> curr_subgraph_input_names;
    MS_CHECK_TRUE_MSG(subgraph_infer_path[executor_id].size() == kInferPathSize, kLiteError,
                      "size of elsments of subgraph_infer_path should be 2!");
    auto min_index = subgraph_infer_path[executor_id][kInferPathBeginIndex];
    auto max_index = subgraph_infer_path[executor_id][kInferPathEndIndex];
    for (auto subgraph_id = min_index; subgraph_id <= max_index; subgraph_id++) {
      MS_CHECK_TRUE_MSG(model_output_tensors_.size() > static_cast<size_t>(subgraph_id), kLiteError,
                        "subgraph id out of range!");
      curr_executor_model_output_tensors.emplace_back(model_output_tensors_[subgraph_id]);
      curr_executor_models.emplace_back(models_[subgraph_id]);
      curr_subgraph_input_names.push_back(subgraph_input_names[subgraph_id]);
    }
    std::vector<std::string> curr_executor_input_names;
    MS_CHECK_TRUE_MSG(!curr_executor_input_output_names.empty(), kLiteError,
                      "curr_executor_input_output_names should not be empty!");
    if (curr_executor_input_output_names[0].empty()) {
      for (auto single_subgraph_input_names : curr_subgraph_input_names) {
        for (auto input_name : single_subgraph_input_names) {
          if (std::find(main_graph_input_names.begin(), main_graph_input_names.end(), input_name) !=
              main_graph_input_names.end()) {
            curr_executor_input_names.push_back(input_name);
          }
        }
      }
    } else {
      curr_executor_input_names.insert(curr_executor_input_names.end(), curr_executor_input_output_names[0].begin(),
                                       curr_executor_input_output_names[0].end());
    }
    auto executor = ModelExecutor(curr_executor_models, curr_executor_input_names, curr_executor_input_output_names[1],
                                  curr_subgraph_input_names, curr_executor_model_output_tensors);
    executors_.push_back(executor);
  }
  return kSuccess;
}

std::vector<ModelExecutor> MultiModelRunner::GetModelExecutor() const { return executors_; }

Status MultiModelRunner::LoadConfig(const std::vector<char> &config_path) {
  config_file_ = CharToString(config_path);
  return kSuccess;
}

Status MultiModelRunner::UpdateConfig(const std::vector<char> &section,
                                      const std::pair<std::vector<char>, std::vector<char>> &config) {
  std::unique_lock<std::mutex> lock(g_config_lock);
  auto section_str = CharToString(section);
  auto config_str = std::make_pair(CharToString(config.first), CharToString(config.second));
  auto iter = config_info_.find(section_str);
  if (iter == config_info_.end()) {
    if (config_info_.size() >= kMaxSectionNum) {
      MS_LOG(ERROR) << "config too many sections!";
      return kLiteError;
    }
    config_info_[section_str][config_str.first] = config_str.second;
    return kSuccess;
  }
  if (iter->second.size() >= kMaxConfigNumPerSection) {
    MS_LOG(ERROR) << "config too many items!";
    return kLiteError;
  }
  iter->second[config_str.first] = config_str.second;
  return kSuccess;
}

Status ModelExecutor::Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) {
  MS_CHECK_TRUE_MSG(outputs != nullptr, kLiteError, "outputs is nullptr!");
  MS_CHECK_TRUE_MSG(models_.size() == model_output_tensors_.size(), kLiteError,
                    "size of models must equal model_output_tensors! models size:"
                      << models_.size() << " model_output_tensors size:" << model_output_tensors_.size());
  std::map<std::string, MSTensor> output_tensor_map;
  if (!outputs->empty()) {
    MS_CHECK_TRUE_MSG(
      outputs->size() == executor_output_names_.size(), kLiteError,
      "Outputs size should squal to " << executor_output_names_.size() << " but given " << outputs->size());
    std::set<std::string> output_names;
    std::vector<std::string> output_names_vec;
    std::set<std::string> executor_output_names_set(executor_output_names_.begin(), executor_output_names_.end());
    for (auto output : (*outputs)) {
      output_names.insert(output.Name());
      output_tensor_map[output.Name()] = output;
      output_names_vec.push_back(output.Name());
    }
    MS_CHECK_TRUE_MSG(output_names == executor_output_names_set, kLiteError,
                      "output name should be " << executor_output_names_ << " but given " << output_names_vec);
  }
  std::map<std::string, MSTensor> sub_model_output_map;
  std::map<std::string, MSTensor> input_map;
  for (auto tensor : inputs) {
    input_map[tensor.Name()] = tensor;
  }
  for (size_t model_id = 0; model_id < models_.size(); model_id++) {
    std::vector<MSTensor> curr_inputs;
    auto model_inputs = models_[model_id]->GetInputs();
    for (auto tensor : model_inputs) {
      if (sub_model_output_map.find(tensor.Name()) != sub_model_output_map.end() &&
          input_map.find(tensor.Name()) != input_map.end()) {
        curr_inputs.push_back(input_map[tensor.Name()]);
      } else if (sub_model_output_map.find(tensor.Name()) != sub_model_output_map.end()) {
        curr_inputs.push_back(sub_model_output_map[tensor.Name()]);
      } else if (input_map.find(tensor.Name()) != input_map.end()) {
        curr_inputs.push_back(input_map[tensor.Name()]);
      } else {
        MS_LOG(ERROR) << "Can not find current input, tensor name:" << tensor.Name();
        return kLiteError;
      }
    }
    std::vector<MSTensor> model_outputs;
    if (!model_output_tensors_[model_id].empty()) {
      for (auto output_tensor : model_output_tensors_[model_id]) {
        if (output_tensor_map.find(output_tensor.Name()) != output_tensor_map.end()) {
          model_outputs.push_back(output_tensor_map[output_tensor.Name()]);
        } else {
          model_outputs.push_back(output_tensor);
        }
      }
    }
    if (models_[model_id]->Predict(curr_inputs, &model_outputs) != kSuccess) {
      MS_LOG(ERROR) << "Predict failed!";
      return kLiteError;
    }
    for (auto output : model_outputs) {
      sub_model_output_map[output.Name()] = output;
    }
  }
  if (outputs->empty()) {
    for (auto out_name : executor_output_names_) {
      if (sub_model_output_map.find(out_name) != sub_model_output_map.end()) {
        auto tensor = sub_model_output_map[out_name];
        auto device_data = tensor.GetDeviceData();
        if (device_data != nullptr) {
          auto ret = AscendAllocatorPlugin::GetInstance().CopyDeviceDataToHost(device_data, tensor.MutableData(),
                                                                               tensor.DataSize(), tensor.GetDeviceId());
          MS_CHECK_TRUE_MSG(ret == kSuccess, ret, "copy device data to host failed!");
        }
        outputs->push_back(tensor);
      } else {
        MS_LOG(ERROR) << "Can not find output:" << out_name;
        return kLiteError;
      }
    }
  }
  return kSuccess;
}

std::vector<MSTensor> ModelExecutor::GetInputs() const {
  std::vector<MSTensor> exec_inputs;
  std::map<std::string, MSTensor> model_input_map;
  for (auto model : models_) {
    auto model_inputs = model->GetInputs();
    for (auto input : model_inputs) {
      model_input_map[input.Name()] = input;
    }
  }
  for (auto input_name : executor_input_names_) {
    if (model_input_map.find(input_name) != model_input_map.end()) {
      exec_inputs.push_back(model_input_map[input_name]);
    } else {
      MS_LOG(ERROR) << "can not find output:" << input_name;
      return {};
    }
  }
  return exec_inputs;
}

std::vector<MSTensor> ModelExecutor::GetOutputs() const {
  std::vector<MSTensor> exec_outputs;
  std::map<std::string, MSTensor> model_output_map;
  for (auto model : models_) {
    auto model_outputs = model->GetOutputs();
    for (auto output : model_outputs) {
      model_output_map[output.Name()] = output;
    }
  }
  for (auto output_name : executor_output_names_) {
    if (model_output_map.find(output_name) != model_output_map.end()) {
      exec_outputs.push_back(model_output_map[output_name]);
    } else {
      MS_LOG(ERROR) << "Can not find output:" << output_name;
      return {};
    }
  }
  return exec_outputs;
}
}  // namespace mindspore
