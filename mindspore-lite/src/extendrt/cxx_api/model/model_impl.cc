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

#include <algorithm>
#include <set>
#include <shared_mutex>
#include <cstring>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <unordered_map>
#include "ops/primitive_c.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "nnacl_c/op_base.h"
#include "extendrt/cxx_api/model/model_impl.h"
#include "extendrt/cxx_api/dlutils.h"
#include "extendrt/cxx_api/file_utils.h"
#include "utils/ms_context.h"
#include "extendrt/mindir_loader/mindir_model/mindir_model_util.h"
#include "src/extendrt/convert/runtime_convert.h"
#include "src/common/config_file.h"
#include "mindapi/ir/func_graph.h"
#include "mindapi/base/base.h"
#include "src/extendrt/delegate/graph_executor/litert/func_graph_reuse_manager.h"
#include "load_mindir/load_model.h"
#include "src/extendrt/delegate/ascend_acl/ascend_allocator_plugin.h"
#include "utils/ms_utils_secure.h"
#include "src/extendrt/model_manager.h"
#include "include/api/model_group.h"
#include "src/common/common.h"
#include "mindspore/core/include/ir/graph_utils.h"
#if defined(ENABLE_PRE_INFERENCE) && defined(__linux__) && !defined(Debug)
#include "src/common/random_data_generator.h"
#include "src/common/thread_utils.h"
#endif

namespace mindspore {
namespace {
const char *const kExecutionPlan = "execution_plan";
constexpr size_t kMaxSectionNum = 100;
constexpr size_t kMaxConfigNumPerSection = 1000;
constexpr auto kCommonSection = "common";  // support external user configuration
constexpr auto kEnablePreInferenceKey = "enable_pre_inference";
constexpr auto kEnablePreInferenceValue = "true";
std::shared_mutex g_model_converter_lock;
std::mutex g_load_mindir_lock;
std::mutex g_ms_context_lock;

std::unordered_map<std::string, mindspore::Format> kStr2FormatMap{{"DEFAULT_FORMAT", mindspore::Format::DEFAULT_FORMAT},
                                                                  {"NCHW", mindspore::Format::NCHW},
                                                                  {"NHWC", mindspore::Format::NHWC},
                                                                  {"NHWC4", mindspore::Format::NHWC4},
                                                                  {"HWKC", mindspore::Format::HWKC},
                                                                  {"HWCK", mindspore::Format::HWCK},
                                                                  {"KCHW", mindspore::Format::KCHW},
                                                                  {"CKHW", mindspore::Format::CKHW},
                                                                  {"KHWC", mindspore::Format::KHWC},
                                                                  {"CHWK", mindspore::Format::CHWK},
                                                                  {"HW", mindspore::Format::HW},
                                                                  {"HW4", mindspore::Format::HW4},
                                                                  {"NC", mindspore::Format::NC},
                                                                  {"NC4", mindspore::Format::NC4},
                                                                  {"NC4HW4", mindspore::Format::NC4HW4},
                                                                  {"NUM_OF_FORMAT", mindspore::Format::NUM_OF_FORMAT},
                                                                  {"NCDHW", mindspore::Format::NCDHW},
                                                                  {"NWC", mindspore::Format::NWC},
                                                                  {"NCW", mindspore::Format::NCW},
                                                                  {"NDHWC", mindspore::Format::NDHWC},
                                                                  {"NC8HW8", mindspore::Format::NC8HW8}};

std::string WeightBufferParamsDisplayStr(const void *weight_data, size_t weight_size) {
  std::stringstream ss;
  ss << (weight_data == nullptr ? " weight_data is nullptr." : " weight_data is not nullptr.")
     << " weight_size: " << weight_size;
  return ss.str();
}

}  // namespace

Status ModelImpl::BuildAndRunCore(const std::function<Status()> &build_fn) {
#if defined(ENABLE_PRE_INFERENCE) && defined(__linux__) && !defined(Debug)
  Status ret = build_fn();
  if (ret != kSuccess) {
    return ret;
  }
  ret = InferWithRandomData();
  if (ret != kSuccess) {
    return ret;
  }
#else
  (void)build_fn;
#endif
  return kSuccess;
}

Status ModelImpl::BuildAndRun(const void *model_data, size_t data_size, ModelType model_type,
                              const std::shared_ptr<Context> &model_context) {
  return BuildAndRunCore([this, model_data, data_size, model_type, &model_context]() {
    return this->Build(model_data, data_size, model_type, model_context);
  });
}

Status ModelImpl::InferWithRandomData() {
#if defined(ENABLE_PRE_INFERENCE) && defined(__linux__) && !defined(Debug)
  for (auto &tensor : this->GetInputs()) {
    if (tensor.Shape().empty() || tensor.DataSize() == 0 ||
        std::find(tensor.Shape().begin(), tensor.Shape().end(), -1) != tensor.Shape().end()) {
      return kSuccess;
    }
    auto status = lite::GenRandomData(&tensor);
    if (status != RET_OK) {
      return Status(kLiteError, "generate random data failed for input tensor.");
    }
  }
  auto ret = this->Predict();
  if (ret != kSuccess) {
    return ret;
  }
#endif
  return kSuccess;
}

Status ModelImpl::BuildAndRun(const std::string &model_path, ModelType model_type,
                              const std::shared_ptr<Context> &model_context) {
  return BuildAndRunCore(
    [this, &model_path, model_type, &model_context]() { return this->Build(model_path, model_type, model_context); });
}

Status ModelImpl::PreInferCore(const std::function<Status()> &build_and_run_fn) {
#if defined(ENABLE_PRE_INFERENCE) && defined(__linux__) && !defined(Debug)
  if (lite::GetNumThreads() == lite::kSingleThread && IsEnablePreInference()) {
    pid_t pid = fork();
    if (pid < 0) {
      return kLiteError;
    } else if (pid == 0) {
      auto ret = build_and_run_fn();
      int ret_code = ret == kSuccess ? lite::kProcessSuccess : lite::kProcessFailed;
      exit(ret_code);
    }
    auto ret = lite::CheckPidStatus(pid);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "PreBuild or PreInference failed!";
      return ret;
    }
  }
#else
  (void)build_and_run_fn;
#endif
  return kSuccess;
}

Status ModelImpl::PreInfer(const std::string &model_path, ModelType model_type,
                           const std::shared_ptr<Context> &model_context) {
  return PreInferCore([this, &model_path, model_type, &model_context]() {
    return this->BuildAndRun(model_path, model_type, model_context);
  });
}

Status ModelImpl::PreInfer(const void *model_data, size_t data_size, ModelType model_type,
                           const std::shared_ptr<Context> &model_context) {
  return PreInferCore([this, model_data, data_size, model_type, &model_context]() {
    return this->BuildAndRun(model_data, data_size, model_type, model_context);
  });
}

bool ModelImpl::IsEnablePreInference() {
  if (config_info_.find(kCommonSection) == config_info_.end()) {
    return false;
  }
  auto common_config = config_info_.at(kCommonSection);
  if (common_config.find(kEnablePreInferenceKey) == common_config.end()) {
    return false;
  }
  return common_config.at(kEnablePreInferenceKey) == kEnablePreInferenceValue;
}

void ModelImpl::SetMsContext() {
  std::lock_guard lock(g_ms_context_lock);
  if (MsContext::GetInstance() != nullptr) {
    auto back_policy_env = std::getenv("ASCEND_BACK_POLICY");
    if (back_policy_env != nullptr) {
      (void)MsContext::GetInstance()->set_backend_policy(std::string(back_policy_env));
    }
  }
}

std::mutex ConverterPlugin::mutex_;
ConverterPlugin::ConverterPlugin() = default;

ConverterPlugin::~ConverterPlugin() {
#ifndef _WIN32
  if (handle_ != nullptr) {
    (void)dlclose(handle_);
    handle_ = nullptr;
  }
#endif
}

ConverterPlugin::ConverterFunc ConverterPlugin::GetConverterFunc() {
  std::lock_guard<std::mutex> lock(mutex_);
  static ConverterPlugin instance;
  return instance.GetConverterFuncInner();
}

ConverterPlugin::ConverterFunc ConverterPlugin::GetConverterFuncInner() {
#ifndef _WIN32
  if (converter_func_ == nullptr) {
    std::string plugin_path;
    auto ret = DLSoPath({"libmindspore-lite.so", "_c_lite"}, "libruntime_convert_plugin.so", &plugin_path);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "Get path of libruntime_convert_plugin.so failed. error: " << ret;
      return nullptr;
    }
    void *function = nullptr;
    ret = DLSoOpen(plugin_path, "RuntimeConvert", &handle_, &function, true);
    if (ret != kSuccess) {
      MS_LOG(WARNING) << "DLSoOpen RuntimeConvert failed, so path: " << plugin_path;
      return nullptr;
    }
    converter_func_ = reinterpret_cast<ConverterPlugin::ConverterFunc>(function);
  }
  return converter_func_;
#else
  MS_LOG(ERROR) << "Not support libruntime_convert_plugin.so in Windows";
  return nullptr;
#endif
}

ModelImpl::ModelImpl() : graph_(nullptr), session_(nullptr), context_(nullptr) {}

FuncGraphPtr ModelImpl::DispatchLoadGraph(const void *model_buff, size_t model_size, const void *weight_data,
                                          size_t weight_size, const std::string &model_path) {
  std::string weight_path = "./";
  auto mindir_path = GetConfig(lite::kConfigModelFileSection, lite::kConfigMindIRPathKey);
  std::string base_path = "";
  if (!mindir_path.empty()) {
    base_path = mindir_path;
  } else {
    // user does not set mindir_path, convert from model_path
    base_path = model_path;
  }
  FuncGraphPtr func_graph;
  std::string user_info_string;
  bool build_from_file = weight_data == nullptr && weight_size == 0 && !base_path.empty();
  bool build_from_buffer_model = weight_data == nullptr && weight_size == 0 && base_path.empty();
  bool build_from_buffer_model_weight = weight_data != nullptr && weight_size != 0 && base_path.empty();
  std::unique_lock<std::mutex> l(g_load_mindir_lock);
  MindIRLoader mindir_loader;
  bool ret = false;
  if (build_from_file) {
    if (base_path.find("/") != std::string::npos) {
      weight_path = base_path.substr(0, base_path.rfind("/"));
    }
    MS_LOG(INFO) << "model will build from file.";
    ret = mindir_loader.LoadMindIR(model_buff, model_size, weight_path, &func_graph, &user_info_string);
  } else if (build_from_buffer_model || build_from_buffer_model_weight) {
    MS_LOG(INFO) << "model will build from buffer.";
    ret = mindir_loader.LoadMindIR(model_buff, model_size, weight_data, weight_size, &func_graph, &user_info_string);
  } else {
    MS_LOG(ERROR) << "cannot determine how to build model."
                  << " got:" << WeightBufferParamsDisplayStr(weight_data, weight_size) << " model_path: \""
                  << model_path << "\"";
  }
  if (!ret || func_graph == nullptr) {
    MS_LOG(ERROR) << "Failed to load MindIR model, please check the validity of the model.";
    return nullptr;
  }
  if (!user_info_string.empty()) {
    SetModelInfo(lite::KModelUserInfo, user_info_string);
  }
  return func_graph;
}

FuncGraphPtr ModelImpl::LoadGraphByBufferImpl(const void *model_buff, size_t model_size, const void *weight_data,
                                              size_t weight_size, ModelType model_type,
                                              const std::shared_ptr<Context> &model_context,
                                              const std::string &model_path) {
  if (model_type != kMindIR) {
    MS_LOG(ERROR) << "Invalid model type!";
    return nullptr;
  }

  auto status = UpdateSharingWorkspaceConfig(model_buff, model_size, model_path);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "UpdateSharingWorkspaceConfig failed!";
    return nullptr;
  }

  auto dump_path = GetConfig(lite::kAscendContextSection, lite::kDumpPathKey);
  if (!dump_path.empty()) {
    auto dir_pos = model_path.find_last_of('/');
    auto mindir_name = dir_pos != std::string::npos ? model_path.substr(dir_pos + 1) : model_path;
    auto dot_pos = mindir_name.find_last_of('.');
    auto model_name = mindir_name.substr(0, dot_pos);
    (void)UpdateConfig(lite::kAscendContextSection,
                       std::pair<std::string, std::string>(lite::kDumpModelNameKey, model_name));
  }
  FuncGraphPtr func_graph = DispatchLoadGraph(model_buff, model_size, weight_data, weight_size, model_path);
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Failed to load MindIR model, please check the validity of the model.";
    return nullptr;
  }

  if (func_graph->get_attr(lite::kDynamicDimsKey) != nullptr) {
    auto dynamic_dims = GetValue<std::string>(func_graph->get_attr(lite::kDynamicDimsKey));
    SetModelInfo(lite::kDynamicDimsKey, dynamic_dims);
  }
  if (func_graph->get_attr(lite::KModelInputShape) != nullptr) {
    auto input_shape = GetValue<std::string>(func_graph->get_attr(lite::KModelInputShape));
    SetModelInfo(lite::KModelInputShape, input_shape);
  }
  return func_graph;
}

bool ModelImpl::IsEnableModelSharing(const std::string &model_path, ModelGroupFlag *model_group_flag) {
  const std::map<std::string, ModelGroupFlag> &model_path_set = ModelManager::GetInstance().GetModelPath();
  auto it = model_path_set.find(model_path);
  if (it == model_path_set.end()) {
    return false;
  } else {
    *model_group_flag = it->second;
    return true;
  }
}

std::map<std::string, std::string> ModelImpl::GetModelInfo() const {
  auto current_info = model_info_;
  if (model_info_.find(lite::KCurrentPid) == model_info_.end()) {
    int32_t pid;
    if (!AscendAllocatorPlugin::GetInstance().Register()) {
      MS_LOG(WARNING) << "Register ascendallocatorplugin failed!";
      return current_info;
    }
    if (!AscendAllocatorPlugin::GetInstance().GetPid(&pid)) {
      MS_LOG(WARNING) << "GetPid failed!";
    } else {
      current_info[lite::KCurrentPid] = std::to_string(pid);
    }
  }
  return current_info;
}

bool ModelImpl::IsEnableModelSharing(const std::pair<const void *, size_t> &model_buff) {
  const std::set<std::pair<const void *, size_t>> &model_buff_set = ModelManager::GetInstance().GetModelBuff();
  return (model_buff_set.find(model_buff) != model_buff_set.end());
}

Status ModelImpl::SetSharingMemoryConfig(const std::string &model_path, ModelGroupFlag model_group_flag) {
  MS_LOG(INFO) << "model_sharing_flag: true";
  auto ret = UpdateConfig(lite::kInnerCommon, std::make_pair(lite::kInnerSharingWorkspace, "true"));
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "UpdateConfig failed.";
    return ret;
  }
  ret = UpdateConfig(lite::kInnerCommon, std::make_pair(lite::kInnerModelPath, model_path));
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "UpdateConfig failed.";
    return ret;
  }
  if (model_group_flag == ModelGroupFlag::kShareWeight) {
    ret = UpdateConfig(lite::kInnerCommon, std::make_pair(lite::kInnerWeightspace, "true"));
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "UpdateConfig " << lite::kInnerCommon << " " << lite::kInnerWeightspace << " failed!";
      return ret;
    }
  } else if (model_group_flag == ModelGroupFlag::kShareWorkspace) {
    ret = UpdateConfig(lite::kInnerCommon, std::make_pair(lite::kInnerWorkspace, "true"));
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "UpdateConfig " << lite::kInnerCommon << " " << lite::kInnerWorkspace << " failed!";
      return ret;
    }
  } else if (model_group_flag == ModelGroupFlag::kShareWeightAndWorkspace) {
    ret = UpdateConfig(lite::kInnerCommon, std::make_pair(lite::kInnerWeightspaceWorkspace, "true"));
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "UpdateConfig " << lite::kInnerCommon << " " << lite::kInnerWeightspaceWorkspace << " failed!";
      return ret;
    }
  }
  return kSuccess;
}

Status ModelImpl::UpdateSharingWorkspaceConfig(const void *model_buff, size_t model_size,
                                               const std::string &model_path) {
  bool model_sharing_flag = false;
  ModelGroupFlag model_group_flag = ModelGroupFlag::kUnknown;
  if (!model_path.empty()) {
    model_sharing_flag = IsEnableModelSharing(model_path, &model_group_flag);
  } else {
    model_sharing_flag = IsEnableModelSharing(std::make_pair(model_buff, model_size));
  }
  if (model_sharing_flag) {
    auto ret = SetSharingMemoryConfig(model_path, model_group_flag);
    if (ret != kSuccess) {
      return ret;
    }
  }
  auto pids = GetConfig(lite::kAscendContextSection, lite::kShareableWeightPidList);
  auto sharable_handle = GetConfig(lite::kAscendContextSection, lite::kSharableWeightMemHandle);
  if (pids != "" && sharable_handle != "") {
    MS_LOG(ERROR) << "You can only set pids or sharable_handle, but not set both of them!";
    return Status(kLiteParamInvalid, "You can only set pids or sharable_handle, but not set both of them!");
  }
  if (pids != "") {
    auto ret = UpdateConfig(lite::kInnerCommon, std::make_pair(lite::kInnerPids, pids));
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "UpdateConfig " << lite::kInnerCommon << " " << lite::kInnerPids << " failed!";
      return ret;
    }
  }
  if (sharable_handle != "") {
    auto ret = UpdateConfig(lite::kInnerCommon, std::make_pair(lite::kInnerSharableHandle, sharable_handle));
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "UpdateConfig " << lite::kInnerCommon << " " << lite::kInnerSharableHandle << " failed!";
      return ret;
    }
  }
  return kSuccess;
}

void ModelImpl::UpdateProvider() {
  if (context_ == nullptr) {
    return;
  }
  auto provider = GetConfig(lite::kAscendContextSection, lite::kProvider);
  if (!provider.empty()) {
    for (auto &device_info : context_->MutableDeviceInfo()) {
      if (device_info && device_info->GetDeviceType() == DeviceType::kAscend && device_info->GetProvider().empty()) {
        device_info->SetProvider(provider);
      }
    }
  }
}
Status ModelImpl::CheckBuildFromBuffer(ModelType model_type, const void *weight_data, size_t weight_size) {
  if (model_type != kMindIR && (weight_data != nullptr || weight_size != 0)) {
    MS_LOG(ERROR) << "Build from weight buffer is not support model_type:" << model_type
                  << ". got: " << WeightBufferParamsDisplayStr(weight_data, weight_size);
    return kLiteParamInvalid;
  }
  return kSuccess;
}
Status ModelImpl::InitBuildSession(const std::shared_ptr<Context> &model_context) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (session_) {
    MS_LOG(ERROR) << "Model has been called Build!";
    return Status(kLiteModelRebuild, "Model has been called build!");
  }
  if (model_context == nullptr) {
    MS_LOG(ERROR) << "Invalid context pointers!";
    return Status(kLiteNullptr, "context is nullptr!");
  }
  for (auto &device_info : model_context->MutableDeviceInfo()) {
    if (device_info == nullptr) {
      MS_LOG(ERROR) << "There is nullptr device info in context!";
      return Status(kLiteNullptr, "device_info is nullptr!");
    }
  }
  SetMsContext();
  auto thread_num = model_context->GetThreadNum();
  if (thread_num < 0) {
    MS_LOG(ERROR) << "Invalid thread num " << thread_num;
    return Status(kLiteParamInvalid, "Invalid thread num!");
  }
  UpdateProvider();
  session_ = InferSession::CreateSession(model_context, config_info_);
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Create session failed!";
    return Status(kLiteNullptr, "session is nullptr, Create session failed!");
  }
  return kSuccess;
}

Status ModelImpl::BuildByBufferImpl(const void *model_buff, size_t model_size, const void *weight_data,
                                    size_t weight_size, ModelType model_type,
                                    const std::shared_ptr<Context> &model_context, const std::string &model_path) {
  MS_CHECK_TRUE_MSG(model_buff != nullptr, Status(kLiteFileError, "The input model buffer is nullptr!"),
                    "The input model buffer is nullptr!");
  MS_CHECK_TRUE_MSG(model_size != 0, Status(kLiteFileError, "The input model buffer size is 0!"),
                    "The input model buffer size is 0!");
  MS_CHECK_TRUE_MSG(model_context != nullptr, Status(kLiteNullptr, "context is nullptr!"), "Invalid context pointers!");
  auto ret = CheckBuildFromBuffer(model_type, weight_data, weight_size);
  if (ret != kSuccess) {
    return ret;
  }
  auto status = UpdateSharingWorkspaceConfig(model_buff, model_size, model_path);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "UpdateSharingWorkspaceConfig failed!";
    return status;
  }
  ret = InitBuildSession(model_context);
  if (ret != kSuccess) {
    return ret;
  }
  auto mindir_path = GetConfig(lite::kConfigModelFileSection, lite::kConfigMindIRPathKey);
  if (mindir_path.empty()) {
    (void)UpdateConfig(lite::kConfigModelFileSection,
                       std::pair<std::string, std::string>(lite::kConfigMindIRPathKey, model_path));
  }

  if (model_type == kMindIR_Lite) {
    ret = session_->CompileGraph(model_buff, model_size, &graph_id_);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "compile graph failed!ret = " << ret;
      return ret;
    }
    return kSuccess;
  }
  // for model pool
  FuncGraphPtr func_graph = FuncGraphReuseManager::GetInstance()->GetSharedFuncGraph(config_info_);
  if (func_graph != nullptr) {
    MS_LOG(INFO) << "the model buffer is the same as the last time. we can directly use the cached function graph.";
    std::unique_lock<std::shared_mutex> build_lock(g_model_converter_lock);
    return session_->CompileGraph(func_graph, nullptr, 0, &graph_id_);
  }

  if (model_type != kOM) {
    func_graph =
      LoadGraphByBufferImpl(model_buff, model_size, weight_data, weight_size, model_type, model_context, model_path);
    if (func_graph == nullptr) {
      MS_LOG(ERROR) << "Failed to load MindIR model, please check the validity of the model.";
      if (weight_data == nullptr && weight_size == 0 && model_path.empty()) {
        return kLiteError;
      }
      return Status(kLiteNullptr, "func_graph is nullptr, failed to load MindIR model!");
    }
    // convert and optimize func graph to infer
    ret = ConvertGraphOnline(func_graph, model_context);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "convert graph failed!ret = " << ret;
      return ret;
    }
    ret = session_->CompileGraph(func_graph, nullptr, 0, &graph_id_);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "compile graph failed!ret = " << ret;
      return ret;
    }
    auto sharable_handle = session_->GetSharableHandle();
    if (sharable_handle != 0) {
      SetModelInfo(lite::kSharableWeightMemHandle, std::to_string(sharable_handle));
    }
    std::shared_lock<std::shared_mutex> build_lock(g_model_converter_lock);
    return FuncGraphReuseManager::GetInstance()->StoreFuncGraph(func_graph, config_info_);
  } else {
    ret = session_->CompileGraph(model_buff, model_size, &graph_id_);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "compile graph failed!ret = " << ret;
      return ret;
    }
  }
  return kSuccess;
}

Status ModelImpl::Build(const FuncGraphPtr &func_graph, const std::shared_ptr<Context> &model_context) {
  auto ret = InitBuildSession(model_context);
  if (ret != kSuccess) {
    return ret;
  }
  // get func_graph
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Input func graph is nullptr!";
    return kLiteError;
  }
  ret = ConvertGraphOnline(func_graph, model_context);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "convert graph failed!ret = " << ret;
    return ret;
  }
  ret = session_->CompileGraph(func_graph, nullptr, 0, &graph_id_);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "compile graph failed!ret = " << ret;
    return ret;
  }
  std::shared_lock<std::shared_mutex> build_lock(g_model_converter_lock);
  return FuncGraphReuseManager::GetInstance()->StoreFuncGraph(func_graph, config_info_);
}

Status ModelImpl::Build(const void *model_data, size_t data_size, ModelType model_type,
                        const std::shared_ptr<Context> &model_context) {
  return BuildByBufferImpl(model_data, data_size, nullptr, 0, model_type, model_context);
}

Status ModelImpl::Build(const void *model_data, size_t data_size, const void *weight_data, size_t weight_size,
                        ModelType model_type, const std::shared_ptr<Context> &model_context) {
  return BuildByBufferImpl(model_data, data_size, weight_data, weight_size, model_type, model_context);
}

Status ModelImpl::Build(const std::string &model_path, ModelType model_type,
                        const std::shared_ptr<Context> &model_context) {
  if (model_path.empty()) {
    MS_LOG(ERROR) << "Model path cannot be empty!";
    return Status(kLiteFileError, "Model path is empty!");
  }
  auto buffer = ReadFile(model_path);
  if (buffer.DataSize() == 0) {
    MS_LOG(ERROR) << "Failed to read buffer from model file.";
    return Status(kLiteFileError, "Failed to read buffer from model file!");
  }
  return BuildByBufferImpl(buffer.Data(), buffer.DataSize(), nullptr, 0, model_type, model_context, model_path);
}

Status ModelImpl::ConvertGraphOnline(const FuncGraphPtr &func_graph, const std::shared_ptr<Context> &model_context) {
  MS_CHECK_TRUE_MSG(func_graph != nullptr, Status(kLiteNullptr, "func_graph is nullptr, failed to load MindIR model!"),
                    "func_graph is nullptr!");
  bool is_device_ascend = false;
  auto device_list = model_context->MutableDeviceInfo();
  for (const auto &device_info : device_list) {
    if (device_info == nullptr) {
      continue;
    }
    if (device_info->GetDeviceType() == kAscend) {
      is_device_ascend = true;
    }
  }
  auto value = func_graph->get_attr(lite::kIsOptimized);
  if (value != nullptr) {
    if (GetValue<bool>(value)) {
      // it does not need to convert, if funcgraph is optimized.
      return kSuccess;
    } else if (config_info_.find(lite::kInnerModelParallelRunnerSection) != config_info_.end() && is_device_ascend) {
      MS_LOG(ERROR) << "Model Parallel Runner is not supported on Ascend, due to the func_graph is unoptimized!";
      return Status(kLiteError,
                    "Model Parallel Runner is not supported on Ascend, due to the func_graph is unoptimized!");
    }
  }

  auto convert = ConverterPlugin::GetConverterFunc();
  if (convert == nullptr) {
    MS_LOG(ERROR) << "get Converter func failed";
    return Status(kLiteNullptr, "Converter is nullptr, get Converter func failed!");
  }
  auto api_graph = mindspore::api::MakeShared<mindspore::api::FuncGraph>(func_graph);
  std::unique_lock<std::shared_mutex> build_lock(g_model_converter_lock);
  auto status = convert(api_graph, model_context, config_info_);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "Failed to converter graph";
    return Status(kLiteError, "Failed to converter graph");
  }

  return kSuccess;
}  // namespace mindspore

Status ModelImpl::Resize(const std::vector<MSTensor> &inputs, const std::vector<std::vector<int64_t>> &dims) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (MS_UNLIKELY(session_ == nullptr)) {
    MS_LOG(ERROR) << "Model has not been called Build, or Model Build has failed!";
    return Status(kLiteUninitializedObj, "Model has not been called Build, or Model Build has failed!");
  }
  if (inputs.empty()) {
    MS_LOG(ERROR) << "Inputs is empty!";
    return Status(kLiteInputParamInvalid, "Inputs is empty!");
  }
  if (dims.empty()) {
    MS_LOG(ERROR) << "Dims is empty!";
    return Status(kLiteInputParamInvalid, "dims is empty!");
  }
  for (size_t j = 0; j < dims.size(); j++) {
    auto dims_v = dims[j];
    for (size_t i = 0; i < dims_v.size(); i++) {
      auto dim = dims_v[i];
      if (dim <= 0 || dim > INT_MAX) {
        MS_LOG(ERROR) << "Invalid shape! dim: " << dim;
        return Status(kLiteInputParamInvalid, "Invalid shape!");
      }
    }
  }
  if (inputs.size() != dims.size()) {
    MS_LOG(ERROR) << "The size of inputs does not match the size of dims.";
    return Status(kLiteInputParamInvalid, "The size of inputs does not match the size of dims!");
  }
  auto model_inputs = session_->GetInputs(graph_id_);
  if (model_inputs.empty()) {
    MS_LOG(ERROR) << "The inputs of model is empty.";
    return Status(kLiteInputParamInvalid, "The inputs of model is empty.");
  }
  if (inputs.size() != model_inputs.size()) {
    MS_LOG(ERROR) << "The size of inputs is incorrect.";
    return Status(kLiteInputParamInvalid, "The given input size is inconsistent with the input size of the model.");
  }
  return session_->Resize(graph_id_, inputs, dims);
}

std::vector<MSTensor> ModelImpl::GetInputs() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Model has not been called Build, or Model Build has failed";
    return {};
  }
  auto graph_inputs = session_->GetInputs(graph_id_);
  std::vector<MSTensor> inputs;
  std::transform(graph_inputs.begin(), graph_inputs.end(), std::back_inserter(inputs),
                 [](auto &impl) { return MSTensor(impl); });
  return inputs;
}

std::vector<MSTensor> ModelImpl::GetOutputs() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Model has not been called Build, or Model Build has failed!";
    return {};
  }
  auto graph_outputs = session_->GetOutputs(graph_id_);
  std::vector<MSTensor> outputs;
  std::transform(graph_outputs.begin(), graph_outputs.end(), std::back_inserter(outputs),
                 [](auto &impl) { return MSTensor(impl); });
  return outputs;
}

MSTensor ModelImpl::GetInputByTensorName(const std::string &name) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (MS_UNLIKELY(session_ == nullptr)) {
    MS_LOG(ERROR) << "Model has not been called Build, or Model Build failed!";
    return MSTensor(nullptr);
  }
  auto tensor_impl = session_->GetInputByTensorName(graph_id_, name);
  if (tensor_impl == nullptr) {
    MS_LOG(ERROR) << "Model does not contains tensor " << name << " .";
    return MSTensor(nullptr);
  }
  return MSTensor(tensor_impl);
}

std::vector<std::string> ModelImpl::GetOutputTensorNames() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Model has not been called Build, or Model Build has failed";
    return {};
  }
  return session_->GetOutputNames(graph_id_);
}

MSTensor ModelImpl::GetOutputByTensorName(const std::string &name) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (MS_UNLIKELY(session_ == nullptr)) {
    MS_LOG(ERROR) << "Model has not been called Build, or Model Build failed!";
    return MSTensor(nullptr);
  }
  auto tensor_impl = session_->GetOutputByTensorName(graph_id_, name);
  if (tensor_impl == nullptr) {
    MS_LOG(ERROR) << "Model does not contains tensor " << name << " .";
    return MSTensor(nullptr);
  }
  return MSTensor(tensor_impl);
}

Status ModelImpl::UpdateWeights(const std::vector<std::vector<MSTensor>> &weights) {
  MS_CHECK_TRUE_MSG(session_, kLiteError, "session is nullptr!");
  return session_->UpdateWeights(weights);
}

Status ModelImpl::Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                          const MSKernelCallBack &before, const MSKernelCallBack &after) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (MS_UNLIKELY(session_ == nullptr)) {
    MS_LOG(ERROR) << "Model has not been called Build, or Model Build has failed!";
    return Status(kLiteUninitializedObj, "Model has not been called Build, or Model Build has failed");
  }
  if (outputs == nullptr) {
    MS_LOG(ERROR) << "outputs pointer is nullptr!";
    return Status(kLiteOutputParamInvalid, "outputs pointer is nullptr");
  }
  if (inputs.empty()) {
    MS_LOG(ERROR) << "user input tensor is empty!";
    return Status(kLiteInputParamInvalid, "user input tensor is empty!");
  }
  auto ret = session_->RunGraph(graph_id_, inputs, outputs, before, after);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "ModelImpl::Predict RunGraph failed!ret = " << ret;
    return ret;
  }
  auto session_outputs = session_->GetOutputs(graph_id_);
  if (outputs->size() != session_outputs.size()) {
    MS_LOG(ERROR) << "Outputs count get from session " << session_outputs.size() << " != outputs count of RunGraph "
                  << outputs->size();
    return Status(kLiteOutputParamInvalid, "output size is wrong!");
  }
  for (size_t i = 0; i < session_outputs.size(); i++) {
    MSTensor session_output(session_outputs[i]);
    auto &execute_output = outputs->at(i);
    session_output.SetShape(execute_output.Shape());
    if (session_output.GetDeviceData() != execute_output.GetDeviceData()) {
      session_output.SetDeviceData(execute_output.GetDeviceData());
    }
    if (execute_output.GetDeviceData() == nullptr && session_output.Data().get() != execute_output.Data().get()) {
      session_output.SetData(execute_output.MutableData(), false);
    }
  }
  return kSuccess;
}

Status ModelImpl::Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) {
  return Predict(inputs, outputs, nullptr, nullptr);
}

Status ModelImpl::Predict() {
  auto inputs = GetInputs();
  auto outputs = GetOutputs();
  return Predict(inputs, &outputs);
}

bool ModelImpl::HasPreprocess() {
  if (!graph_ || !graph_->graph_data_) {
    MS_LOG(ERROR) << "Model has not been called Build, or Model Build has failed";
    return false;
  }
  return false;
}

Status ModelImpl::Preprocess(const std::vector<std::vector<MSTensor>> &inputs, std::vector<MSTensor> *outputs) {
  return kLiteError;
}

Status ModelImpl::PredictWithPreprocess(const std::vector<std::vector<MSTensor>> &inputs,
                                        std::vector<MSTensor> *outputs) {
#if !defined(_WIN32) && !defined(_WIN64)
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Model has not been called Build, or Model Build failed!";
    return kLiteError;
  }
  // Run preprocess
  std::vector<MSTensor> preprocess_outputs;
  Status ret = Preprocess(inputs, &preprocess_outputs);
  if (ret != kSuccess) {
    return ret;
  }

  // Run prediction
  ret = Predict(preprocess_outputs, outputs);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Run predict failed: " << ret.GetErrDescription();
    return ret;
  }
  return kSuccess;
#else
  MS_LOG(ERROR) << "Predict with data preprocess is not supported on Windows yet.";
  return Status(kMEFailed, "Predict with data preprocess is not supported on Windows yet.");
#endif
}

Status ModelImpl::LoadConfig(const std::string &config_path) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (session_ != nullptr) {
    MS_LOG(ERROR) << "Model has been called Build, please call LoadConfig before build.";
    return Status(kLiteError, "Model has been called build, please call LoadConfig before build.");
  }
  ConfigInfos all_config_info;
  int ret = lite::GetAllSectionInfoFromConfigFile(config_path, &all_config_info);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "GetAllSectionInfoFromConfigFile fail!ret = " << ret;
    return Status(kLiteFileError, "GetAllSectionInfoFromConfigFile failed, please check your config file.");
  }
  for (auto &section : all_config_info) {
    const auto &section_name = section.first;
    auto sec_it = config_info_.find(section_name);
    if (sec_it == config_info_.end()) {
      config_info_.emplace(section.first, section.second);
    } else {
      auto &cur_sec = sec_it->second;
      for (auto &config_item : section.second) {
        cur_sec[config_item.first] = config_item.second;
      }
    }
  }
  return kSuccess;
}

Status ModelImpl::UpdateConfig(const std::string &section, const std::pair<std::string, std::string> &config) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto iter = config_info_.find(section);
  if (iter == config_info_.end()) {
    if (config_info_.size() >= kMaxSectionNum) {
      MS_LOG(ERROR) << "The config has too many sections!";
      return Status(kLiteParamInvalid, "The config has too many sections!");
    }
    config_info_[section][config.first] = config.second;
    return kSuccess;
  }
  if (iter->second.size() >= kMaxConfigNumPerSection) {
    MS_LOG(ERROR) << "The config has too mant items!";
    return Status(kLiteParamInvalid, "The config has too mant items!");
  }
  iter->second[config.first] = config.second;
  return kSuccess;
}

std::string ModelImpl::GetConfig(const std::string &section, const std::string &key) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto iter = config_info_.find(section);
  if (iter == config_info_.end()) {
    return "";
  }
  auto elem_iter = iter->second.find(key);
  if (elem_iter == iter->second.end()) {
    return "";
  }
  return elem_iter->second;
}

ModelImpl::~ModelImpl() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  FuncGraphReuseManager::GetInstance()->ReleaseSharedFuncGraph(config_info_);
  session_ = nullptr;
}

bool ModelImpl::CheckModelSupport(DeviceType device_type, ModelType model_type) {
  if (device_type == kCPU) {
    return true;
  }
  if (model_type != kMindIR) {
    return false;
  }
  return false;
}

Status ModelImpl::Finalize() {
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "session_ is nullptr, please build model first!";
    return Status(kLiteUninitializedObj, "session_ is nullptr, please build model first!");
  }
  return session_->Finalize();
}
}  // namespace mindspore
