/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include <functional>
#include <map>
#include <utility>
#include <memory>
#include <string>
#include <vector>
#include "include/api/context.h"
#include "extendrt/cxx_api/model/model_impl.h"
#include "src/common/config_file.h"
#include "src/common/common.h"
#include "src/common/utils.h"

namespace mindspore {
namespace {
#ifdef USE_GLOG
extern "C" {
extern void mindspore_log_init();
}
#endif
std::mutex g_log_lock;

// Shared null-check + try-catch wrapper for Status-returning methods.
// Used by Resize, Predict, etc. to eliminate duplicate error handling.
Status TryExecute(const std::shared_ptr<ModelImpl> &impl, const std::function<Status()> &exec_fn) {
  if (impl == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return Status(kLiteNullptr, "Model implement is nullptr.");
  }
  try {
    return exec_fn();
  } catch (const std::exception &exe) {
    MS_LOG(ERROR) << "Catch exception: " << exe.what();
    return kCoreFailed;
  }
}

// Try-catch wrapper for Build methods with timing.
Status TryBuildWithTiming(const std::function<Status()> &build_fn) {
  try {
    auto start_time = lite::GetTimeUs();
    auto ret = build_fn();
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "impl_->Build failed! ret = " << ret;
      return ret;
    }
    auto end_time = lite::GetTimeUs();
    auto cost = end_time - start_time;
    MS_LOG(INFO) << "[init time] model build cost " << cost << " us";
    return kSuccess;
  } catch (const std::exception &exe) {
    MS_LOG(ERROR) << "Catch exception: " << exe.what();
    return kCoreFailed;
  }
}

// Null-check + try-catch wrapper for tensor getter methods.
std::vector<MSTensor> TryGetTensors(const std::shared_ptr<ModelImpl> &impl,
                                    const std::function<std::vector<MSTensor>()> &get_fn) {
  if (impl == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return {};
  }
  try {
    return get_fn();
  } catch (const std::exception &exe) {
    MS_LOG(ERROR) << "Catch exception: " << exe.what();
    return {};
  }
}
}  // namespace

Model::Model() {
  {
    std::lock_guard lock(g_log_lock);
#ifdef USE_GLOG
#if defined(_WIN32) || defined(_WIN64) || defined(__APPLE__)
#ifdef _MSC_VER
    mindspore::mindspore_log_init();
#endif
#else
    mindspore::mindspore_log_init();
#endif
#endif
  }
  impl_ = std::make_shared<ModelImpl>();
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Failed to create ModelImpl";
  }
}

Model::~Model() {}

Status Model::Build(const void *model_data, size_t data_size, ModelType model_type,
                    const std::shared_ptr<Context> &model_context) {
  MS_CHECK_TRUE_MSG(impl_ != nullptr, Status(kLiteNullptr, "Model implement is nullptr!"),
                    "Model implement is nullptr!");
  MS_CHECK_TRUE_MSG(model_data != nullptr, Status(kLiteNullptr, "model_data is nullptr!"), "model_data is nullptr!");
  MS_CHECK_TRUE_MSG(model_context != nullptr, Status(kLiteNullptr, "model_context is nullptr!"),
                    "model_context is nullptr!");
  auto ret = impl_->PreInfer(model_data, data_size, model_type, model_context);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "PreInfer failed!";
    return ret;
  }
  return TryBuildWithTiming([=]() { return impl_->Build(model_data, data_size, model_type, model_context); });
}

Status Model::Build(const void *model_data, size_t data_size, const void *weight_data, size_t weight_size,
                    ModelType model_type, const std::shared_ptr<Context> &model_context) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return kLiteNullptr;
  }
  return TryBuildWithTiming(
    [=]() { return impl_->Build(model_data, data_size, weight_data, weight_size, model_type, model_context); });
}

Status Model::Build(const std::vector<char> &model_path, ModelType model_type,
                    const std::shared_ptr<Context> &model_context) {
  MS_CHECK_TRUE_MSG(impl_ != nullptr, Status(kLiteNullptr, "Model implement is nullptr!"),
                    "Model implement is nullptr!");
  MS_CHECK_TRUE_MSG(model_context != nullptr, Status(kLiteNullptr, "context is nullptr!"), "context is nullptr!");
  auto ret = impl_->PreInfer(CharToString(model_path), model_type, model_context);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "PreInfer failed!";
    return ret;
  }
  auto model_path_str = CharToString(model_path);
  return TryBuildWithTiming([=]() { return impl_->Build(model_path_str, model_type, model_context); });
}

Status Model::Build(const std::vector<char> &model_path, ModelType model_type,
                    const std::shared_ptr<Context> &model_context, const Key &dec_key,
                    const std::vector<char> &dec_mode, const std::vector<char> &cropto_lib_path) {
  MS_LOG(ERROR) << "This interface has been deprecated.";
  return kLiteNotSupport;
}

Status Model::Build(const void *model_data, size_t data_size, ModelType model_type,
                    const std::shared_ptr<Context> &model_context, const Key &dec_key,
                    const std::vector<char> &dec_mode, const std::vector<char> &cropto_lib_path) {
  MS_LOG(ERROR) << "This interface has been deprecated.";
  return kLiteNotSupport;
}

Status Model::Build(GraphCell graph, const std::shared_ptr<Context> &model_context,
                    const std::shared_ptr<TrainCfg> &train_cfg) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kLiteNotSupport;
}

Status BuildTransferLearning(GraphCell backbone, GraphCell head, const std::shared_ptr<Context> &context,
                             const std::shared_ptr<TrainCfg> &train_cfg = nullptr) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kLiteNotSupport;
}

Status Model::Resize(const std::vector<MSTensor> &inputs, const std::vector<std::vector<int64_t>> &dims) {
  return TryExecute(impl_, [this, &inputs, &dims]() { return impl_->Resize(inputs, dims); });
}

Status Model::RunStep(const MSKernelCallBack &before, const MSKernelCallBack &after) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return kLiteNullptr;
  }
  auto inputs = impl_->GetInputs();
  auto outputs = impl_->GetOutputs();
  return impl_->Predict(inputs, &outputs);
}

Status Model::Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                      const MSKernelCallBack &before, const MSKernelCallBack &after) {
  return TryExecute(
    impl_, [this, &inputs, outputs, &before, &after]() { return impl_->Predict(inputs, outputs, before, after); });
}

Status Model::Predict(const MSKernelCallBack &before, const MSKernelCallBack &after) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kLiteNotSupport;
}

Status Model::PredictWithPreprocess(const std::vector<std::vector<MSTensor>> &inputs, std::vector<MSTensor> *outputs,
                                    const MSKernelCallBack &before, const MSKernelCallBack &after) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kLiteNotSupport;
}

Status Model::Preprocess(const std::vector<std::vector<MSTensor>> &inputs, std::vector<MSTensor> *outputs) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kLiteNotSupport;
}

bool Model::HasPreprocess() {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return false;
}

std::vector<MSTensor> Model::GetInputs() {
  return TryGetTensors(impl_, [this]() { return impl_->GetInputs(); });
}

std::vector<MSTensor> Model::GetOutputs() {
  return TryGetTensors(impl_, [this]() { return impl_->GetOutputs(); });
}

MSTensor Model::GetInputByTensorName(const std::vector<char> &name) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return MSTensor(nullptr);
  }
  try {
    return impl_->GetInputByTensorName(CharToString(name));
  } catch (const std::exception &exe) {
    MS_LOG(ERROR) << "Catch exception: " << exe.what();
    return MSTensor(nullptr);
  }
}

std::vector<std::vector<char>> Model::GetOutputTensorNamesChar() {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return {};
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
  MS_LOG(ERROR) << "Unsupported Feature.";
  return {};
}

Status Model::BindGLTexture2DMemory(const std::map<std::string, unsigned int> &inputGLTexture,
                                    std::map<std::string, unsigned int> *outputGLTexture) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kLiteNotSupport;
}

Status Model::LoadConfig(const std::vector<char> &config_path) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return Status(kLiteNullptr, "Model implement is nullptr, failed to load config file.");
  }

  auto ret = impl_->LoadConfig(CharToString(config_path));
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "The config file is invalid, failed to load config file.";
    return ret;
  }
  return kSuccess;
}

Status Model::UpdateConfig(const std::vector<char> &section,
                           const std::pair<std::vector<char>, std::vector<char>> &config) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return Status(kLiteNullptr, "Model implement is nullptr, failed to update config file.");
  }
  auto ret = impl_->UpdateConfig(CharToString(section), {CharToString(config.first), CharToString(config.second)});
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Failed to update config file.";
    return ret;
  }
  return kSuccess;
}

bool Model::CheckModelSupport(DeviceType device_type, ModelType model_type) {
  return ModelImpl::CheckModelSupport(device_type, model_type);
}

Status Model::UpdateWeights(const std::vector<std::vector<MSTensor>> &new_weights) {
  return impl_->UpdateWeights(new_weights);
}

Status Model::UpdateWeights(const std::vector<MSTensor> &new_weights) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kLiteNotSupport;
}

std::vector<MSTensor> Model::GetTrainableParams() const {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return {};
}

std::vector<MSTensor> Model::GetGradients() const {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return {};
}

Status Model::ApplyGradients(const std::vector<MSTensor> &gradients) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kLiteNotSupport;
}

std::vector<MSTensor> Model::GetFeatureMaps() const {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return {};
}

Status Model::UpdateFeatureMaps(const std::vector<MSTensor> &new_weights) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kLiteNotSupport;
}

std::vector<MSTensor> Model::GetOptimizerParams() const {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return {};
}

Status Model::SetOptimizerParams(const std::vector<MSTensor> &params) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kLiteNotSupport;
}

Status Model::SetupVirtualBatch(int virtual_batch_multiplier, float lr, float momentum) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kLiteNotSupport;
}

Status Model::SetLearningRate(float learning_rate) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kLiteNotSupport;
}

float Model::GetLearningRate() {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return 0.0;
}

Status Model::InitMetrics(const std::vector<Metrics *> metrics) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kLiteNotSupport;
}

std::vector<Metrics *> Model::GetMetrics() {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return {};
}

Status Model::SetTrainMode(bool train) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kLiteNotSupport;
}

bool Model::GetTrainMode() const {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return false;
}

std::vector<char> Model::GetModelInfo(const std::vector<char> &key) {
  std::vector<char> ret;
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return ret;
  }
  auto string_key = CharToString(key);
  std::vector<std::string> supported_key = {lite::KModelUserInfo, lite::KModelInputShape, lite::kDynamicDimsKey,
                                            lite::KCurrentPid, lite::kSharableWeightMemHandle};
  if (std::find(supported_key.begin(), supported_key.end(), string_key) == supported_key.end()) {
    MS_LOG(WARNING) << "Unsupported key, current supported key:" << supported_key << ", input key:" << string_key;
    return ret;
  }
  auto model_info = impl_->GetModelInfo();
  auto it = model_info.find(CharToString(key));
  if (it == model_info.end()) {
    return ret;
  }
  return StringToChar(it->second);
}

Status Model::Finalize() {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is nullptr.";
    return Status(kLiteNullptr, "Model implement is nullptr.");
  }
  return impl_->Finalize();
}
}  // namespace mindspore
