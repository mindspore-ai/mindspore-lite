/**
 * Copyright 2021~2024 Huawei Technologies Co., Ltd
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

#include "include/api/model.h"
#include <mutex>
#ifdef GPU_TENSORRT
#include <cuda_runtime.h>
#endif
#include "flatbuffers/flatbuffers.h"
#include "include/api/context.h"
#include "include/api/dual_abi_helper.h"
#include "include/api/types.h"
#include "include/api/serialization.h"
#include "include/api/graph.h"
#include "src/common/log_adapter.h"
#if defined(ENABLE_PRE_INFERENCE) && defined(__linux__) && !defined(Debug)
#include "src/common/thread_utils.h"
#endif
#include "src/litert/cxx_api/model/model_impl.h"

namespace mindspore {
#ifdef USE_GLOG
extern "C" {
extern void mindspore_log_init();
}
#endif

bool VerifyDeviceInfo(const std::shared_ptr<Context> &model_context) {
  if (model_context == nullptr) {
    MS_LOG(ERROR) << "model_context is nullptr!";
    return false;
  }
  const auto &device_infos = model_context->MutableDeviceInfo();
  if (device_infos.empty()) {
    MS_LOG(ERROR) << "Device info list is empty!";
    return false;
  }
  const std::vector<DeviceType> white_list = {kCPU, kGPU, kDSP, kKirinNPU};
  for (const auto &device_info : device_infos) {
    if (device_info == nullptr) {
      continue;
    }
    auto type = device_info->GetDeviceType();
    if (std::find(white_list.begin(), white_list.end(), type) == white_list.end()) {
      MS_LOG(ERROR) << "Edge inference only supports CPU, GPU, DSP, or KirinNPU backends. Current type: " << type;
      return false;
    }
  }
  return true;
}

Status Model::Build(const void *model_data, size_t data_size, ModelType model_type,
                    const std::shared_ptr<Context> &model_context, const Key &dec_key,
                    const std::vector<char> &dec_mode, const std::vector<char> &cropto_lib_path) {
  MS_LOG(ERROR) << "This interface has been deprecated.";
  return kLiteNotSupport;
}

Status Model::Build(const void *model_data, size_t data_size, ModelType model_type,
                    const std::shared_ptr<Context> &model_context) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return kLiteNullptr;
  }
  bool valid = VerifyDeviceInfo(model_context);
  if (!valid) {
    return Status(kLiteParamInvalid, "Device info is invalid");
  }
  Status ret;
#if defined(ENABLE_PRE_INFERENCE) && defined(__linux__) && !defined(Debug)
  if (lite::GetNumThreads() == lite::kSingleThread && impl_->IsEnablePreInference()) {
    pid_t pid = fork();
    if (pid < 0) {
      return kLiteError;
    } else if (pid == 0) {  // child process
      auto impl = std::make_shared<ModelImpl>();
      if (impl != nullptr) {
        (void)impl->BuildAndRun(model_data, data_size, model_type, model_context);
        impl.reset();
        const_cast<std::shared_ptr<Context> &>(model_context).reset();
      } else {
        MS_LOG(WARNING) << "new ModelImpl failed, pre inference failed.";
      }
      exit(lite::kProcessSuccess);
    }
    ret = lite::CheckPidStatus(pid);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "PreBuild or PreInference failed.";
      return ret;
    }
  }
#endif
  ret = impl_->Build(model_data, data_size, model_type, model_context);
  if (ret != kSuccess) {
    return ret;
  }
  return kSuccess;
}

Status Model::Build(const void *model_data, size_t data_size, const void *weight_data, size_t weight_size,
                    ModelType model_type, const std::shared_ptr<Context> &model_context) {
  MS_LOG(ERROR) << "Build with weight buffer is only support for mindspore_lite's ascend backend.";
  return kLiteError;
}

Status Model::Build(const std::vector<char> &model_path, ModelType model_type,
                    const std::shared_ptr<Context> &model_context, const Key &dec_key,
                    const std::vector<char> &dec_mode, const std::vector<char> &cropto_lib_path) {
  MS_LOG(ERROR) << "This interface has been deprecated.";
  return kLiteNotSupport;
}

Status Model::Build(const std::vector<char> &model_path, ModelType model_type,
                    const std::shared_ptr<Context> &model_context) {
  std::stringstream err_msg;
  if (model_context == nullptr) {
    err_msg << "Invalid null context.";
    MS_LOG(ERROR) << err_msg.str();
    return Status(kLiteNullptr, err_msg.str());
  }
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return kLiteNullptr;
  }
  bool valid = VerifyDeviceInfo(model_context);
  if (!valid) {
    return Status(kLiteParamInvalid, "Device info is invalid");
  }
  Status ret;
#if defined(ENABLE_PRE_INFERENCE) && defined(__linux__) && !defined(Debug)
  if (lite::GetNumThreads() == lite::kSingleThread && impl_->IsEnablePreInference()) {
    pid_t pid = fork();
    if (pid < 0) {
      return kLiteError;
    } else if (pid == 0) {  // child process
      auto impl = std::make_shared<ModelImpl>();
      if (impl != nullptr) {
        (void)impl->BuildAndRun(CharToString(model_path), model_type, model_context);
        impl.reset();
        const_cast<std::shared_ptr<Context> &>(model_context).reset();
      } else {
        MS_LOG(WARNING) << "new ModelImpl failed, pre inference failed.";
      }
      exit(lite::kProcessSuccess);
    }
    ret = lite::CheckPidStatus(pid);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "PreBuild or PreInference failed.";
      return ret;
    }
  }
#endif
  ret = impl_->Build(CharToString(model_path), model_type, model_context);
  if (ret != kSuccess) {
    return ret;
  }
  return kSuccess;
}

Status Model::Build(GraphCell graph, const std::shared_ptr<Context> &model_context,
                    const std::shared_ptr<TrainCfg> &train_cfg) {
  std::stringstream err_msg;
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return kLiteNullptr;
  }

  if (graph.GetGraph() == nullptr) {
    err_msg << "Invalid null graph.";
    MS_LOG(ERROR) << err_msg.str();
    return Status(kLiteNullptr, err_msg.str());
  }
  if (model_context == nullptr) {
    err_msg << "Invalid null context.";
    MS_LOG(ERROR) << err_msg.str();
    return Status(kLiteNullptr, err_msg.str());
  }
  bool valid = VerifyDeviceInfo(model_context);
  if (!valid) {
    return Status(kLiteParamInvalid, "Device info is invalid");
  }
#if defined(ENABLE_PRE_INFERENCE) && defined(__linux__) && !defined(Debug)
  if (lite::GetNumThreads() == lite::kSingleThread && impl_->IsEnablePreInference()) {
    pid_t pid = fork();
    if (pid < 0) {
      return kLiteError;
    } else if (pid == 0) {  // child process
      auto impl = std::make_shared<ModelImpl>();
      if (impl != nullptr) {
        impl->SetContext(model_context);
        impl->SetGraph(graph.GetGraph());
        impl->SetConfig(train_cfg);
        (void)impl->BuildAndRun();
        impl.reset();
        const_cast<std::shared_ptr<Context> &>(model_context).reset();
      } else {
        MS_LOG(WARNING) << "new ModelImpl failed, pre inference failed.";
      }
      exit(lite::kProcessSuccess);
    }
    auto ret = lite::CheckPidStatus(pid);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "PreResize or PreInference failed.";
      return ret;
    }
  }
#endif
  impl_->SetContext(model_context);
  impl_->SetGraph(graph.GetGraph());
  impl_->SetConfig(train_cfg);
  return impl_->Build();
}

Status Model::Resize(const std::vector<MSTensor> &inputs, const std::vector<std::vector<int64_t>> &dims) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return kLiteNullptr;
  }
  return impl_->Resize(inputs, dims);
}

Status Model::UpdateWeights(const std::vector<MSTensor> &new_weights) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return kLiteNullptr;
  }
  return impl_->UpdateWeights(new_weights);
}

Status Model::RunStep(const MSKernelCallBack &before, const MSKernelCallBack &after) {
  if (MS_UNLIKELY(impl_ == nullptr)) {
    MS_LOG(ERROR) << "Model implement is null!";
    return kLiteNullptr;
  }
  auto inputs = impl_->GetInputs();
  auto outputs = impl_->GetOutputs();
  return impl_->Predict(inputs, &outputs, before, after);
}

Status Model::Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                      const MSKernelCallBack &before, const MSKernelCallBack &after) {
  if (MS_UNLIKELY(impl_ == nullptr)) {
    MS_LOG(ERROR) << "Model implement is null!";
    return kLiteNullptr;
  }
  return impl_->Predict(inputs, outputs, before, after);
}

Status Model::Predict(const MSKernelCallBack &before, const MSKernelCallBack &after) {
  if (MS_UNLIKELY(impl_ == nullptr)) {
    MS_LOG(ERROR) << "Model implement is null!";
    return kLiteNullptr;
  }
  return impl_->Predict(before, after);
}

Status Model::PredictWithPreprocess(const std::vector<std::vector<MSTensor>> &inputs, std::vector<MSTensor> *outputs,
                                    const MSKernelCallBack &before, const MSKernelCallBack &after) {
  MS_LOG(ERROR) << "Unsupported Feature!";
  return kLiteNotSupport;
}

Status Model::Preprocess(const std::vector<std::vector<MSTensor>> &inputs, std::vector<MSTensor> *outputs) {
  MS_LOG(ERROR) << "Unsupported Feature!";
  return kLiteNotSupport;
}

bool Model::HasPreprocess() {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return false;
}

Model::Model() {
#ifdef USE_GLOG
  mindspore::mindspore_log_init();
#endif
  impl_ = std::make_shared<ModelImpl>();
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Failed to create ModelImpl!";
  }
}

Model::~Model() {}

bool Model::CheckModelSupport(DeviceType device_type, ModelType model_type) {
  if (device_type == kCPU) {
    return true;
  }
#ifdef GPU_TENSORRT
  if (device_type == kGPU) {
    int driver_version = 0;
    int ret = cudaDriverGetVersion(&driver_version);
    if (ret != cudaSuccess || driver_version == 0) {
      MS_LOG(ERROR) << "No nvidia GPU driver.";
      return false;
    }
    return true;
  }
#endif
  return false;
}

std::vector<MSTensor> Model::GetInputs() {
  std::vector<MSTensor> empty;
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return empty;
  }
  return impl_->GetInputs();
}

std::vector<MSTensor> Model::GetOutputs() {
  std::vector<MSTensor> empty;
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return empty;
  }
  return impl_->GetOutputs();
}

Status Model::BindGLTexture2DMemory(const std::map<std::string, unsigned int> &inputGLTexture,
                                    std::map<std::string, unsigned int> *outputGLTexture) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return kLiteError;
  }
  return impl_->BindGLTexture2DMemory(inputGLTexture, outputGLTexture);
}

MSTensor Model::GetInputByTensorName(const std::vector<char> &name) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return MSTensor(nullptr);
  }
  return impl_->GetInputByTensorName(CharToString(name));
}

std::vector<std::vector<char>> Model::GetOutputTensorNamesChar() {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    std::vector<std::vector<char>> empty;
    return empty;
  }
  return VectorStringToChar(impl_->GetOutputTensorNames());
}

MSTensor Model::GetOutputByTensorName(const std::vector<char> &name) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return MSTensor(nullptr);
  }
  return impl_->GetOutputByTensorName(CharToString(name));
}

std::vector<MSTensor> Model::GetOutputsByNodeName(const std::vector<char> &node_name) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    std::vector<MSTensor> empty;
    return empty;
  }
  return impl_->GetOutputsByNodeName(CharToString(node_name));
}

Status Model::LoadConfig(const std::vector<char> &config_path) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return kLiteNullptr;
  }

  auto ret = impl_->LoadConfig(CharToString(config_path));
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "impl_ LoadConfig failed,";
    return Status(kLiteFileError, "Invalid config file.");
  }
  return kSuccess;
}

Status Model::UpdateConfig(const std::vector<char> &section,
                           const std::pair<std::vector<char>, std::vector<char>> &config) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return kLiteNullptr;
  }
  return impl_->UpdateConfig(CharToString(section), {CharToString(config.first), CharToString(config.second)});
}

Status Model::SetTrainMode(bool train) {
  if ((impl_ == nullptr) || (impl_->session_ == nullptr)) {
    MS_LOG(ERROR) << "Model is null.";
    return kLiteUninitializedObj;
  }
  auto ret = (train) ? impl_->session_->Train() : impl_->session_->Eval();
  return (ret == mindspore::lite::RET_OK) ? kSuccess : kLiteError;
}

bool Model::GetTrainMode() const { return ((impl_ != nullptr) && (impl_->session_) && (impl_->session_->IsTrain())); }

std::vector<MSTensor> Model::GetGradients() const {
  std::vector<MSTensor> empty;
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return empty;
  }
  return impl_->GetGradients();
}

Status Model::ApplyGradients(const std::vector<MSTensor> &gradients) {
  if ((impl_ == nullptr) || (impl_->session_ == nullptr)) {
    MS_LOG(ERROR) << "Model is null.";
    return kLiteUninitializedObj;
  }
  return impl_->ApplyGradients(gradients);
}

std::vector<MSTensor> Model::GetFeatureMaps() const {
  std::vector<MSTensor> empty;
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return empty;
  }
  return impl_->GetFeatureMaps();
}

std::vector<MSTensor> Model::GetTrainableParams() const {
  std::vector<MSTensor> empty;
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return empty;
  }
  return impl_->GetTrainableParams();
}

Status Model::UpdateFeatureMaps(const std::vector<MSTensor> &new_weights) {
  if ((impl_ == nullptr) || (impl_->session_ == nullptr)) {
    MS_LOG(ERROR) << "Model is null.";
    return kLiteUninitializedObj;
  }
  return impl_->UpdateFeatureMaps(new_weights);
}

std::vector<MSTensor> Model::GetOptimizerParams() const {
  std::vector<MSTensor> empty;
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return empty;
  }
  auto res = impl_->GetOptimizerParams();
  return res;
}

Status Model::SetOptimizerParams(const std::vector<MSTensor> &params) {
  if ((impl_ == nullptr) || (impl_->session_ == nullptr)) {
    MS_LOG(ERROR) << "Model is null.";
    return kLiteUninitializedObj;
  }
  return impl_->SetOptimizerParams(params);
}

Status Model::InitMetrics(const std::vector<Metrics *> metrics) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return kLiteUninitializedObj;
  }
  return impl_->InitMetrics(metrics);
}

std::vector<Metrics *> Model::GetMetrics() {
  std::vector<Metrics *> empty;
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return empty;
  }
  return impl_->GetMetrics();
}

Status Model::SetupVirtualBatch(int virtual_batch_multiplier, float lr, float momentum) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return kLiteUninitializedObj;
  }
  return impl_->SetupVirtualBatch(virtual_batch_multiplier, lr, momentum);
}

Status Model::SetLearningRate(float learning_rate) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return kLiteUninitializedObj;
  }
  return impl_->SetLearningRate(learning_rate);
}

float Model::GetLearningRate() {
  if (impl_ == nullptr) {
    MS_LOG(WARNING) << "Model implement is null.";
    return 0.0;
  }
  return impl_->GetLearningRate();
}

std::vector<char> Model::GetModelInfo(const std::vector<char> &key) {
  MS_LOG(WARNING) << "Device-side inference does not support get model info";
  std::vector<char> empty;
  return empty;
}

Status Model::Finalize() {
  MS_LOG(INFO) << "Finalize is only support for mindspore_lite's ascend backend.";
  return kSuccess;
}
}  // namespace mindspore
