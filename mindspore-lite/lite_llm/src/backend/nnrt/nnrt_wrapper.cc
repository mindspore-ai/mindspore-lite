/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "backend/nnrt/nnrt_wrapper.h"

#include <dlfcn.h>

#include "backend/nnrt/nnrt_log.h"

namespace mslite {
namespace backend {
namespace nnrt {

NNRTWrapper *NNRTWrapper::instance_{nullptr};

// Not thread-safe: the LLM inference path serializes the first call. Two concurrent
// first-calls could both dlopen and the loser leaks its handles. Caller must serialize init.
NNRTWrapper *NNRTWrapper::GetInstance() {
  if (instance_ == nullptr) {
    auto *wrapper = new NNRTWrapper();
    if (!wrapper->LoadLibraries()) {
      delete wrapper;
      return nullptr;
    }
    instance_ = wrapper;
  }
  return instance_;
}

void NNRTWrapper::SetApiForTesting(const NNRTFunctions &funcs) {
  if (instance_ == nullptr) {
    instance_ = new NNRTWrapper();
  }
  instance_->api_ = funcs;
}

const NNRTFunctions &NNRTWrapper::GetApi() {
  static NNRTFunctions empty{};
  if (instance_ == nullptr) {
    return empty;
  }
  return instance_->api_;
}

NNRTWrapper::~NNRTWrapper() {
  if (nncore_handle_ != nullptr) {
    dlclose(nncore_handle_);
    nncore_handle_ = nullptr;
  }
  if (hiai_handle_ != nullptr) {
    dlclose(hiai_handle_);
    hiai_handle_ = nullptr;
  }
}

bool NNRTWrapper::LoadLibraries() {
  if (!LoadNeuralNetworkCore()) {
    MS_LOG(ERROR) << "Failed to load libneural_network_core.so";
    return false;
  }
  if (!LoadHiAIFoundation()) {
    MS_LOG(ERROR) << "Failed to load libhiai_foundation.so";
    return false;
  }
  return true;
}

// Loads and verifies a required symbol from handle. Missing symbols are logged
// and clear ok, but loading continues so every gap is reported at once.
template <typename T>
void LoadRequiredSymbol(void *handle, T &field, const char *name, bool &ok) {
  field = reinterpret_cast<T>(dlsym(handle, name));
  if (field == nullptr) {
    MS_LOG(ERROR) << "dlsym failed for " << name << ": " << dlerror();
    ok = false;
  }
}

// Loads an optional symbol; missing symbols are logged as a warning but do not
// fail loading, so older NNRT versions remain compatible.
template <typename T>
void LoadOptionalSymbol(void *handle, T &field, const char *name) {
  field = reinterpret_cast<T>(dlsym(handle, name));
  if (field == nullptr) {
    MS_LOG(WARNING) << "Optional NNRT symbol missing " << name << ": " << dlerror();
  }
}

bool NNRTWrapper::LoadNeuralNetworkCore() {
  bool ok = true;
  dlerror();  // clear stale error
  nncore_handle_ = dlopen("libneural_network_core.so", RTLD_LAZY | RTLD_LOCAL);
  if (nncore_handle_ == nullptr) {
    MS_LOG(ERROR) << "dlopen libneural_network_core.so failed: " << dlerror();
    return false;
  }

  LoadRequiredSymbol(nncore_handle_, api_.Compilation_ConstructWithOfflineModelFile,
                     "OH_NNCompilation_ConstructWithOfflineModelFile", ok);
  LoadRequiredSymbol(nncore_handle_, api_.Compilation_Build, "OH_NNCompilation_Build", ok);
  LoadRequiredSymbol(nncore_handle_, api_.Compilation_Destroy, "OH_NNCompilation_Destroy", ok);
  LoadRequiredSymbol(nncore_handle_, api_.Compilation_SetDevice, "OH_NNCompilation_SetDevice", ok);
  LoadRequiredSymbol(nncore_handle_, api_.Compilation_SetPerformanceMode, "OH_NNCompilation_SetPerformanceMode", ok);
  LoadRequiredSymbol(nncore_handle_, api_.Executor_Construct, "OH_NNExecutor_Construct", ok);
  LoadRequiredSymbol(nncore_handle_, api_.Executor_Destroy, "OH_NNExecutor_Destroy", ok);
  LoadRequiredSymbol(nncore_handle_, api_.Executor_RunSync, "OH_NNExecutor_RunSync", ok);
  LoadRequiredSymbol(nncore_handle_, api_.Executor_CreateInputTensorDesc, "OH_NNExecutor_CreateInputTensorDesc", ok);
  LoadRequiredSymbol(nncore_handle_, api_.Executor_CreateOutputTensorDesc, "OH_NNExecutor_CreateOutputTensorDesc", ok);
  LoadOptionalSymbol(nncore_handle_, api_.Executor_GetInputCount, "OH_NNExecutor_GetInputCount");
  LoadOptionalSymbol(nncore_handle_, api_.Executor_GetOutputCount, "OH_NNExecutor_GetOutputCount");
  LoadOptionalSymbol(nncore_handle_, api_.Compilation_ConstructWithOfflineModelBuffer,
                     "OH_NNCompilation_ConstructWithOfflineModelBuffer");
  LoadRequiredSymbol(nncore_handle_, api_.TensorDesc_SetShape, "OH_NNTensorDesc_SetShape", ok);
  LoadRequiredSymbol(nncore_handle_, api_.TensorDesc_SetDataType, "OH_NNTensorDesc_SetDataType", ok);
  LoadOptionalSymbol(nncore_handle_, api_.TensorDesc_GetName, "OH_NNTensorDesc_GetName");
  LoadOptionalSymbol(nncore_handle_, api_.TensorDesc_GetDataType, "OH_NNTensorDesc_GetDataType");
  LoadOptionalSymbol(nncore_handle_, api_.TensorDesc_GetShape, "OH_NNTensorDesc_GetShape");
  LoadOptionalSymbol(nncore_handle_, api_.TensorDesc_GetByteSize, "OH_NNTensorDesc_GetByteSize");
  LoadRequiredSymbol(nncore_handle_, api_.TensorDesc_Destroy, "OH_NNTensorDesc_Destroy", ok);
  LoadRequiredSymbol(nncore_handle_, api_.Tensor_Create, "OH_NNTensor_Create", ok);
  LoadRequiredSymbol(nncore_handle_, api_.Tensor_Destroy, "OH_NNTensor_Destroy", ok);
  LoadRequiredSymbol(nncore_handle_, api_.Tensor_GetDataBuffer, "OH_NNTensor_GetDataBuffer", ok);

  return ok;
}

bool NNRTWrapper::LoadHiAIFoundation() {
  dlerror();
  hiai_handle_ = dlopen("libhiai_foundation.so", RTLD_LAZY | RTLD_LOCAL);
  if (hiai_handle_ == nullptr) {
    MS_LOG(ERROR) << "dlopen libhiai_foundation.so failed: " << dlerror();
    return false;
  }

  // The device libhiai_foundation.so exports the mixed-case spelling (verified on
  // kirin9020); the all-caps "HIAI" variant does not exist and dlsym would fail.
  api_.HIAIOptions_SetAsyncModeEnable = reinterpret_cast<decltype(api_.HIAIOptions_SetAsyncModeEnable)>(
    dlsym(hiai_handle_, "HMS_HiAIOptions_SetAsyncModeEnable"));
  if (api_.HIAIOptions_SetAsyncModeEnable == nullptr) {
    MS_LOG(ERROR) << "Missing HMS_HiAIOptions_SetAsyncModeEnable in libhiai_foundation.so: " << dlerror();
    return false;
  }
  api_.HIAIOptions_SetOmOptions =
    reinterpret_cast<decltype(api_.HIAIOptions_SetOmOptions)>(dlsym(hiai_handle_, "HMS_HiAIOptions_SetOmOptions"));
  api_.HIAIExecutor_InitWeights =
    reinterpret_cast<decltype(api_.HIAIExecutor_InitWeights)>(dlsym(hiai_handle_, "HMS_HiAIExecutor_InitWeights"));
  if (api_.HIAIExecutor_InitWeights == nullptr) {
    MS_LOG(WARNING) << "Optional HMS_HiAIExecutor_InitWeights missing in libhiai_foundation.so: " << dlerror();
  }
  return true;
}

}  // namespace nnrt
}  // namespace backend
}  // namespace mslite
