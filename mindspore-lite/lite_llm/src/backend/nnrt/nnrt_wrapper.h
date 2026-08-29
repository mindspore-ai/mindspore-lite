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

#ifndef MSLLM_NNRT_WRAPPER_H
#define MSLLM_NNRT_WRAPPER_H

#include <cstddef>
#include <cstdint>

// Opaque NNRT types (verified in neural_network_core.h / neural_network_runtime_type.h)
struct OH_NNCompilation;
struct OH_NNExecutor;
struct NN_TensorDesc;
struct NN_Tensor;

// OH_NN_DataType enum values (neural_network_runtime_type.h) — dlopen path uses int constants
constexpr int32_t kOhNnInt8 = 2;
constexpr int32_t kOhNnInt32 = 4;
constexpr int32_t kOhNnUint8 = 6;
constexpr int32_t kOhNnFloat16 = 10;
constexpr int32_t kOhNnFloat32 = 11;

namespace mslite {
namespace backend {
namespace nnrt {

// OH_NN_ReturnCode is int; OH_NN_SUCCESS == 0
using NnrtReturnCode = int;

/// @brief Function pointer table for NNRT core + HiAI APIs (all from libneural_network_core.so
///        except HIAIOptions which is from libhiai_foundation.so). Verified against NDK headers.
struct NNRTFunctions {
  // === OH_NNCompilation (core.so) ===
  OH_NNCompilation *(*Compilation_ConstructWithOfflineModelFile)(const char *modelPath);
  NnrtReturnCode (*Compilation_Build)(OH_NNCompilation *compilation);
  void (*Compilation_Destroy)(OH_NNCompilation **compilation);
  NnrtReturnCode (*Compilation_SetDevice)(OH_NNCompilation *compilation, size_t deviceID);

  // === OH_NNExecutor (core.so) ===
  OH_NNExecutor *(*Executor_Construct)(OH_NNCompilation *compilation);
  void (*Executor_Destroy)(OH_NNExecutor **executor);
  NnrtReturnCode (*Executor_RunSync)(OH_NNExecutor *executor, NN_Tensor *inputTensor[], size_t inputCount,
                                     NN_Tensor *outputTensor[], size_t outputCount);

  // === OH_NNExecutor tensor desc (core.so) ===
  NN_TensorDesc *(*Executor_CreateInputTensorDesc)(const OH_NNExecutor *executor, size_t index);
  NN_TensorDesc *(*Executor_CreateOutputTensorDesc)(const OH_NNExecutor *executor, size_t index);

  // === OH_NNExecutor I/O count (core.so) ===
  NnrtReturnCode (*Executor_GetInputCount)(const OH_NNExecutor *executor, size_t *inputCount);
  NnrtReturnCode (*Executor_GetOutputCount)(const OH_NNExecutor *executor, size_t *outputCount);

  // === OH_NNCompilation from memory buffer (core.so) ===
  OH_NNCompilation *(*Compilation_ConstructWithOfflineModelBuffer)(const void *modelBuffer, size_t modelSize);

  // === OH_NNTensorDesc (core.so) ===
  NnrtReturnCode (*TensorDesc_SetShape)(NN_TensorDesc *tensorDesc, const int32_t *shape, size_t shapeLen);
  NnrtReturnCode (*TensorDesc_SetDataType)(NN_TensorDesc *tensorDesc, int32_t dataType);
  NnrtReturnCode (*TensorDesc_GetName)(const NN_TensorDesc *tensorDesc, const char **name);
  NnrtReturnCode (*TensorDesc_GetDataType)(const NN_TensorDesc *tensorDesc, int32_t *dataType);
  NnrtReturnCode (*TensorDesc_GetShape)(const NN_TensorDesc *tensorDesc, int32_t **shape, size_t *shapeLength);
  NnrtReturnCode (*TensorDesc_GetByteSize)(const NN_TensorDesc *tensorDesc, size_t *byteSize);
  NnrtReturnCode (*TensorDesc_Destroy)(NN_TensorDesc **tensorDesc);

  // === OH_NNTensor (core.so) — ION-backed ===
  NN_Tensor *(*Tensor_Create)(size_t deviceID, NN_TensorDesc *tensorDesc);
  NnrtReturnCode (*Tensor_Destroy)(NN_Tensor **tensor);
  void *(*Tensor_GetDataBuffer)(const NN_Tensor *tensor);

  // === HiAI Foundation (libhiai_foundation.so) — 分档模式必须关异步 ===
  int (*HIAIOptions_SetAsyncModeEnable)(OH_NNCompilation *compilation, bool enable);
};

class NNRTWrapper {
 public:
  static NNRTWrapper *GetInstance();
  static void SetApiForTesting(const NNRTFunctions &funcs);
  static const NNRTFunctions &GetApi();

 private:
  NNRTWrapper() = default;
  ~NNRTWrapper();

  bool LoadLibraries();
  bool LoadNeuralNetworkCore();
  bool LoadHiAIFoundation();

  NNRTFunctions api_{};
  void *nncore_handle_{nullptr};
  void *hiai_handle_{nullptr};

  static NNRTWrapper *instance_;
};

}  // namespace nnrt
}  // namespace backend
}  // namespace mslite

#endif  // MSLLM_NNRT_WRAPPER_H
