/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
#include "cxx_api/graph/ascend/ascend_graph_impl.h"
#include <algorithm>
#include "include/api/context.h"
#include "cxx_api/factory.h"
#include "cxx_api/akg_kernel_register.h"
#include "cxx_api/utils.h"
#include "cxx_api/acl_utils.h"
#include "src/common/log_adapter.h"
#include "base/base_ref_utils.h"
#include "backend/common/session/executor_manager.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "include/common/utils/python_adapter.h"
#include "backend/common/session/session_basic.h"
#include "runtime/hardware/device_context_manager.h"
#include "include/backend/distributed/init.h"
#include "plugin/res_manager/ascend/symbol_interface/acl_rt_symbol.h"
#include "plugin/res_manager/ascend/symbol_interface/symbol_utils.h"

namespace mindspore {
API_GRAPH_REG(kAscendDevice, AscendGraphImpl);
namespace {
constexpr auto kHcclEnable = "MS_ENABLE_HCCL";
constexpr auto kHcclGroupFile = "PARA_GROUP_FILE";

void InitHccl() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  mindspore::python_adapter::set_python_env_flag(true);
  // init hccl from distributed
  if (!mindspore::distributed::Initialize()) {
    MS_LOG(EXCEPTION) << "InitHccl failed.";
  }
  uint32_t device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  if (ms_context->backend_policy() == "ms") {
    auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id);
    MS_EXCEPTION_IF_NULL(runtime_instance);
#ifndef ENABLE_SECURITY
    runtime_instance->PreInit();
#endif
    const auto &device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
      {kAscendDevice, ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
    MS_EXCEPTION_IF_NULL(device_context);
    MS_EXCEPTION_IF_NULL(device_context->GetDeprecatedInterface());
    (void)device_context->GetDeprecatedInterface()->OpenTsd(ms_context);

    if (!runtime_instance->Init()) {
      MS_LOG(EXCEPTION) << "Runtime init failed.";
    }
  }
}
}  // namespace
AscendGraphImpl::AscendGraphImpl() : device_type_("Ascend"), context_(nullptr) {}

AscendGraphImpl::~AscendGraphImpl() {}

Status AscendGraphImpl::InitEnv() {
  MS_LOG(INFO) << "Start to init env.";
  env_guard_ = MsEnvGuard::GetEnv(device_id_);
  if (env_guard_ == nullptr) {
    MS_LOG(ERROR) << "Env init failed.";
    return kMCDeviceError;
  }

  backend_ = std::make_shared<backend::ms_backend::MSBackend>();
  if (backend_ == nullptr) {
    MS_LOG(ERROR) << "DeviceContext create failed!, please make sure target device:" << kAscendDevice
                  << " is available.";
    return kMCFailed;
  }

  MS_LOG(INFO) << "InitEnv success.";
  return kSuccess;
}

Status AscendGraphImpl::CompileGraph(const std::shared_ptr<FuncGraph> &func_graph) {
  MS_EXCEPTION_IF_NULL(backend_);
  try {
    MS_EXCEPTION_IF_NULL(func_graph);
    // perpare func graph
    auto manager = MakeManager();
    MS_EXCEPTION_IF_NULL(manager);
    manager->AddFuncGraph(func_graph);
    func_graph->set_manager(manager);
    BackendJitConfig &backend_jit_config = backend::BackendJitConfig::ParseBackendJitConfig();
    graph_id_ = backend_->Build(func_graph, backend_jit_config);
    kernel_graph_ = backend_->GetGraphById(graph_id_);
    return kSuccess;
  } catch (std::exception &e) {
    MS_LOG(ERROR) << "CompileGraph failed: " << e.what();
    return kMCFailed;
  }
}

std::vector<tensor::TensorPtr> AscendGraphImpl::RunGraph(const std::vector<tensor::TensorPtr> &inputs) {
  MS_EXCEPTION_IF_NULL(backend_);
  try {
    VectorRef outputs;
    backend_->Run(graph_id_, GraphImpl::GenerateInputsRef(inputs, func_graph_.lock()), &outputs);
    return TransformVectorRefToMultiTensor(outputs);
  } catch (std::exception &e) {
    MS_LOG(ERROR) << "RunGraph failed: " << e.what();
    return std::vector<tensor::TensorPtr>();
  }
}

Status AscendGraphImpl::ExecuteModel(const std::vector<MSTensor> &request, std::vector<MSTensor> *reply) {
  MS_EXCEPTION_IF_NULL(reply);
  if (context_ == nullptr) {
    MS_LOG(ERROR) << "rtCtx is nullptr";
    return kMCDeviceError;
  }
  auto rt_ret = CALL_ASCEND_API(aclrtSetCurrentContext, context_);
  if (rt_ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Set Ascend rtCtx failed";
    return kMCDeviceError;
  }

  vector<tensor::TensorPtr> inputs;
  for (size_t i = 0; i < request.size(); i++) {
    auto item = request[i];
    auto input = inputs_info_[i];
    MS_EXCEPTION_IF_NULL(item);
    MS_EXCEPTION_IF_NULL(input);
    if (input->Size() != item.DataSize()) {
      MS_LOG(ERROR) << "Input " << i << " data size " << item.DataSize() << " not match model input data size "
                    << input->Size();
      return kMCInvalidInput;
    }
    auto ret = memcpy_s(input->data_c(), input->Size(), item.MutableData(), item.DataSize());
    if (ret != EOK) {
      MS_LOG(ERROR) << "MSTensor copy failed";
      return kMCFailed;
    }
    inputs.push_back(input);
  }
  last_inputs_ = inputs;
  std::vector<tensor::TensorPtr> outputs = RunGraph(inputs);
  if (outputs.empty()) {
    MS_LOG(ERROR) << "Execute Model Failed";
    return kMCFailed;
  }
  for (const auto &out : outputs) {
    MS_EXCEPTION_IF_NULL(out);
    out->data_sync();
  }
  last_outputs_ = outputs;
  reply->clear();
  *reply = GetOutputs();
  return kSuccess;
}

std::vector<MSTensor> AscendGraphImpl::GetInputs() {
  if (!load_flag_) {
    Status ret = Load(device_id_);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "PrepareModel failed.";
      return {};
    }
  }

  std::vector<MSTensor> result(inputs_info_.size());
  for (size_t i = 0; i < inputs_info_.size(); ++i) {
    auto &tensor = inputs_info_[i];
    MS_EXCEPTION_IF_NULL(tensor);
    void *data = nullptr;
    size_t data_size = tensor->Size();
    if (i < last_inputs_.size()) {
      MS_EXCEPTION_IF_NULL(last_inputs_[i]);
      data = last_inputs_[i]->data_c();
      data_size = last_inputs_[i]->Size();
    }
    result[i] =
      MSTensor(input_names_[i], static_cast<enum DataType>(tensor->data_type()), tensor->shape(), data, data_size);
  }
  return result;
}

std::vector<MSTensor> AscendGraphImpl::GetOutputs() {
  if (!load_flag_) {
    Status ret = Load(device_id_);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "PrepareModel failed.";
      return {};
    }
  }

  std::vector<MSTensor> result(outputs_info_.size());
  for (size_t i = 0; i < outputs_info_.size(); ++i) {
    auto &tensor = outputs_info_[i];
    MS_EXCEPTION_IF_NULL(tensor);
    void *data = nullptr;
    size_t data_size = tensor->Size();
    if (i < last_outputs_.size()) {
      MS_EXCEPTION_IF_NULL(last_outputs_[i]);
      data = last_outputs_[i]->data_c();
      data_size = last_outputs_[i]->Size();
    }
    result[i] =
      MSTensor(output_names_[i], static_cast<enum DataType>(tensor->data_type()), tensor->shape(), data, data_size);
  }
  return result;
}

Status AscendGraphImpl::Load(uint32_t device_id) {
  // check graph type
  if (graph_->ModelType() != ModelType::kMindIR) {
    MS_LOG(ERROR) << "Unsupported model type " << graph_->ModelType();
    return kMCInvalidInput;
  }

  const auto &graph_data = GraphImpl::MutableGraphData();
  MS_EXCEPTION_IF_NULL(graph_data);
  auto func_graph = graph_data->GetFuncGraph();
  func_graph_ = func_graph;

  // init
  device_id_ = device_id;
  Status ret = InitEnv();
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "InitEnv failed.";
    return ret;
  }

  // load model
  if (!load_flag_) {
    ret = CompileGraph(func_graph);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "Compile graph model failed";
      return ret;
    }
    auto kg = kernel_graph_.lock();
    MS_EXCEPTION_IF_NULL(backend_);
    MS_EXCEPTION_IF_NULL(kg);
    GraphImpl::GetModelInputsInfo(kg, &inputs_info_, &input_names_);
    GraphImpl::GetModelOutputsInfo(kg, &outputs_info_, &output_names_);
    if (inputs_info_.size() != input_names_.size()) {
      MS_LOG(ERROR) << "Get model inputs info failed";
      return kMCInvalidInput;
    }
    if (outputs_info_.size() != output_names_.size()) {
      MS_LOG(ERROR) << "Get model outputs info failed";
      return kMCInvalidInput;
    }

    // save d context
    auto rt_ret = CALL_ASCEND_API(aclrtGetCurrentContext, &context_);
    if (rt_ret != ACL_SUCCESS || context_ == nullptr) {
      MS_LOG(ERROR) << "the ascend device context is null";
      return kMCDeviceError;
    }

    MS_LOG(INFO) << "Load model success";
    load_flag_ = true;
  }

  auto rt_ret = CALL_ASCEND_API(aclrtSetCurrentContext, context_);
  if (rt_ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Set the ascend device context failed";
    return kMCDeviceError;
  }

  return kSuccess;
}

Status AscendGraphImpl::Run(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) {
  MS_EXCEPTION_IF_NULL(outputs);
  if (!load_flag_) {
    Status ret = Load(device_id_);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "PrepareModel failed.";
      return ret;
    }
  }

  if (inputs.size() != inputs_info_.size()) {
    MS_LOG(ERROR) << "inputs count not match, required count " << inputs_info_.size() << ", given count "
                  << inputs.size();
    return kMCInvalidInput;
  }

  for (size_t i = 0; i < inputs_info_.size(); ++i) {
    if (inputs[i].DataSize() != inputs_info_[i]->Size()) {
      MS_LOG(ERROR) << "input " << i << " data size not match, required size " << inputs_info_[i]->Size()
                    << ", given count " << inputs[i].DataSize();
      return kMCInvalidInput;
    }
  }

  Status ret = ExecuteModel(inputs, outputs);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Execute Model Failed";
    return ret;
  }
  if (outputs_info_.size() != outputs->size()) {
    MS_LOG(ERROR) << "Predict output size " << outputs->size() << " not match output size got from model info "
                  << outputs_info_.size();
    return kMCFailed;
  }

  return kSuccess;
}

AscendGraphImpl::MsEnvGuard::MsEnvGuard(uint32_t device_id) : device_id_(device_id) {
  MS_LOG(INFO) << "Start to init device " << device_id;
  RegAllOp();
  auto ms_context = MsContext::GetInstance();
  if (ms_context == nullptr) {
    MS_LOG(ERROR) << "Get Context failed!";
    errno_ = kMCFailed;
    return;
  }

  auto env_hccl_mode = common::GetEnv(kHcclEnable);
  if (!env_hccl_mode.empty() && env_hccl_mode != std::to_string(0)) {
    MS_LOG(INFO) << "Enable hccl parallel mode.";
    ms_context->set_param<bool>(MS_CTX_ENABLE_HCCL, true);
  }

  ms_context->set_param<int>(MS_CTX_EXECUTION_MODE, kGraphMode);
  ms_context->set_param_inner<uint32_t>(MS_CTX_DEVICE_ID, device_id_);
  ms_context->set_param<std::string>(MS_CTX_DEVICE_TARGET, kAscendDevice);
  ms_context->set_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK, true);

  if (ms_context->get_param<bool>(MS_CTX_ENABLE_HCCL)) {
    InitHccl();
    auto para_group_file = common::GetEnv(kHcclGroupFile);
    if (para_group_file.empty()) {
      MS_LOG(INFO) << "Cannot get Env " << kHcclGroupFile << ", skip.";
    } else {
      MS_LOG(INFO) << "Get env " << kHcclGroupFile << " success: " << para_group_file;
      if (!CreateGroupsByCkptFile(para_group_file)) {
        MS_LOG(ERROR) << "CreateGroupsByCkptFile failed.";
        errno_ = kMCFailed;
        return;
      }
    }
  } else {
    auto ret = CALL_ASCEND_API(aclrtSetDevice, static_cast<int32_t>(device_id_));
    if (ret != ACL_SUCCESS) {
      MS_LOG(EXCEPTION) << "Device " << device_id_ << " call aclrtSetDevice failed, ret[" << static_cast<int>(ret)
                        << "]";
    }
  }

  MS_LOG(INFO) << "Device " << device_id << " init env success.";
  errno_ = kSuccess;
}

AscendGraphImpl::MsEnvGuard::~MsEnvGuard() {
  MS_LOG(INFO) << "Start finalize device " << device_id_;
  try {
    session::ExecutorManager::Instance().Clear();
    device::KernelRuntimeManager::Instance().ClearRuntimeResource();

    auto ms_context = MsContext::GetInstance();
    if (ms_context == nullptr) {
      MS_LOG(ERROR) << "Get Context failed!";
      return;
    }

    if (ms_context->get_param<bool>(MS_CTX_ENABLE_HCCL)) {
      PythonEnvGuard guard;
      const auto &device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
        {kAscendDevice, ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
      MS_EXCEPTION_IF_NULL(device_context);
      MS_EXCEPTION_IF_NULL(device_context->GetDeprecatedInterface());
      if (!device_context->GetDeprecatedInterface()->CloseTsd(ms_context, false)) {
        MS_LOG(ERROR) << "CloseTsd failed!";
        return;
      }
    } else {
      auto ret = CALL_ASCEND_API(aclrtResetDevice, static_cast<int32_t>(device_id_));
      if (ret != ACL_SUCCESS) {
        MS_LOG(ERROR) << "Device " << device_id_ << " call aclrtResetDevice failed, ret[" << static_cast<int>(ret)
                      << "]";
        return;
      }
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "AscendGraphImpl MsEnvGuard destructor run failed, error message : " << e.what();
  } catch (...) {
    MS_LOG(ERROR) << "AscendGraphImpl MsEnvGuard destructor run failed, unknown error occurred.";
  }
  MS_LOG(INFO) << "End finalize device " << device_id_;
}

std::shared_ptr<AscendGraphImpl::MsEnvGuard> AscendGraphImpl::MsEnvGuard::GetEnv(uint32_t device_id) {
  std::shared_ptr<MsEnvGuard> acl_env;
  std::lock_guard<std::mutex> lock(global_ms_env_mutex_);
  auto iter = global_ms_env_.find(device_id);
  if (iter != global_ms_env_.end()) {
    acl_env = iter->second.lock();
  }

  if (acl_env != nullptr) {
    MS_LOG(INFO) << "Env has been initialized, skip.";
    return acl_env;
  }

  acl_env = std::make_shared<MsEnvGuard>(device_id);
  if (acl_env->GetErrno() != kSuccess) {
    MS_LOG(ERROR) << "Init ascend env Failed";
    return nullptr;
  }

  global_ms_env_.emplace(device_id, acl_env);
  MS_LOG(INFO) << "Env init success";
  return acl_env;
}

bool AscendGraphImpl::CheckDeviceSupport(mindspore::DeviceType device_type) {
  // for Ascend, only support kAscend and kAscend910
  if (device_type != kAscend && device_type != kAscend910) {
    return false;
  }
  return IsAscend910Soc();
}

std::map<uint32_t, std::weak_ptr<AscendGraphImpl::MsEnvGuard>> AscendGraphImpl::MsEnvGuard::global_ms_env_;
std::mutex AscendGraphImpl::MsEnvGuard::global_ms_env_mutex_;

PythonEnvGuard::PythonEnvGuard() : origin_init_status_(PythonIsInited()) { InitPython(); }

PythonEnvGuard::~PythonEnvGuard() {
  // finalize when init by this
  try {
    if (!origin_init_status_) {
      FinalizePython();
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "PythonEnvGuard destructor run failed, error message : " << e.what();
  } catch (...) {
    MS_LOG(ERROR) << "PythonEnvGuard destructor run failed, unknown error occurred.";
  }
}

bool PythonEnvGuard::PythonIsInited() const { return Py_IsInitialized() != 0; }

void PythonEnvGuard::InitPython() const {
  if (!PythonIsInited()) {
    Py_Initialize();
  }
}

void PythonEnvGuard::FinalizePython() const {
  if (PythonIsInited()) {
    Py_Finalize();
  }
}
}  // namespace mindspore
