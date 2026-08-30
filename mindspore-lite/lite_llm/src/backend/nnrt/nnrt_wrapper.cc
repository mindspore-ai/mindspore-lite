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

// LOAD_SYM loads + verifies a required symbol from nncore_handle_, failing on missing.
#define LOAD_SYM(field, name)                                                         \
  do {                                                                                \
    api_.field = reinterpret_cast<decltype(api_.field)>(dlsym(nncore_handle_, name)); \
    if (api_.field == nullptr) {                                                      \
      MS_LOG(ERROR) << "dlsym failed for " name ": " << dlerror();                    \
      return false;                                                                   \
    }                                                                                 \
  } while (0)

// LOAD_SYM_OPTIONAL loads an optional symbol; missing symbols are logged as a warning
// but do not fail loading, so older NNRT versions remain compatible.
#define LOAD_SYM_OPTIONAL(field, name)                                                \
  do {                                                                                \
    api_.field = reinterpret_cast<decltype(api_.field)>(dlsym(nncore_handle_, name)); \
    if (api_.field == nullptr) {                                                      \
      MS_LOG(WARNING) << "Optional NNRT symbol missing " name ": " << dlerror();      \
    }                                                                                 \
  } while (0)

bool NNRTWrapper::LoadNeuralNetworkCore() {
  dlerror();  // clear stale error
  nncore_handle_ = dlopen("libneural_network_core.so", RTLD_LAZY | RTLD_LOCAL);
  if (nncore_handle_ == nullptr) {
    MS_LOG(ERROR) << "dlopen libneural_network_core.so failed: " << dlerror();
    return false;
  }

  LOAD_SYM(Compilation_ConstructWithOfflineModelFile, "OH_NNCompilation_ConstructWithOfflineModelFile");
  LOAD_SYM(Compilation_Build, "OH_NNCompilation_Build");
  LOAD_SYM(Compilation_Destroy, "OH_NNCompilation_Destroy");
  LOAD_SYM(Compilation_SetDevice, "OH_NNCompilation_SetDevice");
  LOAD_SYM(Executor_Construct, "OH_NNExecutor_Construct");
  LOAD_SYM(Executor_Destroy, "OH_NNExecutor_Destroy");
  LOAD_SYM(Executor_RunSync, "OH_NNExecutor_RunSync");
  LOAD_SYM(Executor_CreateInputTensorDesc, "OH_NNExecutor_CreateInputTensorDesc");
  LOAD_SYM(Executor_CreateOutputTensorDesc, "OH_NNExecutor_CreateOutputTensorDesc");
  LOAD_SYM_OPTIONAL(Executor_GetInputCount, "OH_NNExecutor_GetInputCount");
  LOAD_SYM_OPTIONAL(Executor_GetOutputCount, "OH_NNExecutor_GetOutputCount");
  LOAD_SYM_OPTIONAL(Compilation_ConstructWithOfflineModelBuffer, "OH_NNCompilation_ConstructWithOfflineModelBuffer");
  LOAD_SYM(TensorDesc_SetShape, "OH_NNTensorDesc_SetShape");
  LOAD_SYM(TensorDesc_SetDataType, "OH_NNTensorDesc_SetDataType");
  LOAD_SYM_OPTIONAL(TensorDesc_GetName, "OH_NNTensorDesc_GetName");
  LOAD_SYM_OPTIONAL(TensorDesc_GetDataType, "OH_NNTensorDesc_GetDataType");
  LOAD_SYM_OPTIONAL(TensorDesc_GetShape, "OH_NNTensorDesc_GetShape");
  LOAD_SYM_OPTIONAL(TensorDesc_GetByteSize, "OH_NNTensorDesc_GetByteSize");
  LOAD_SYM(TensorDesc_Destroy, "OH_NNTensorDesc_Destroy");
  LOAD_SYM(Tensor_Create, "OH_NNTensor_Create");
  LOAD_SYM(Tensor_Destroy, "OH_NNTensor_Destroy");
  LOAD_SYM(Tensor_GetDataBuffer, "OH_NNTensor_GetDataBuffer");

  return true;
}
#undef LOAD_SYM
#undef LOAD_SYM_OPTIONAL

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
  return true;
}

}  // namespace nnrt
}  // namespace backend
}  // namespace mslite
