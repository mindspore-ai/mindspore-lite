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
#ifndef MSLLM_NNRT_BACKEND_H
#define MSLLM_NNRT_BACKEND_H

#include <memory>
#include <string>
#include <vector>

#include "backend/common/backend.h"

// Real definitions live in the global mslite_llm namespace (engine contract +
// manifest); forward-declare them here so this header stays self-contained.
namespace mslite_llm {
struct ModelManifest;
struct ModelResources;
}  // namespace mslite_llm

namespace mslite {
namespace backend {
namespace nnrt {
using ::MSLlmStatus;  // global C typedef
using ::mslite_llm::Backend;
using ::mslite_llm::BackendConfig;
using ::mslite_llm::BackendInput;
using ::mslite_llm::BackendOutput;
using ::mslite_llm::ModelManifest;
using ::mslite_llm::ModelResources;

class NnrtExecutor;
struct NnrtConfig;
}  // namespace nnrt
}  // namespace backend
}  // namespace mslite

/// @brief NNRT backend that delegates all inference to the NNRT executor.
///
/// This backend implements the unified Backend contract (Init / Prefill / Decode /
/// Reset) and adapts the generic BackendConfig + ModelManifest into
/// the NNRT-specific NnrtConfig consumed by NnrtExecutor.
///
/// NNRT (Neural Network Runtime) loads a single .omc model and uses gear-based
/// dynamic batching (chunk_size vs seq=1 shapes) to distinguish prefill from
/// decode — validated on Kirin NPU.
namespace mslite {
namespace backend {
namespace nnrt {
class NNRTBackend : public Backend {
 public:
  NNRTBackend() = default;
  ~NNRTBackend() override;

  // ── Unified Backend interface ─────────────────────────────────────────

  /// Build executor from manifest and resource paths in config.
  MSLlmStatus Init(const BackendConfig &config) override;

  /// Prefill: feed the full prompt, return the next token via output->next_token_id.
  MSLlmStatus Prefill(const BackendInput &input, BackendOutput *output) override;

  /// Decode: feed a single token, return the next token via output->next_token_id.
  MSLlmStatus Decode(const BackendInput &input, BackendOutput *output) override;

  /// Reset per-request KV cache state.
  MSLlmStatus Reset() override;

 private:
  /// Populate NnrtConfig from BackendConfig.resources and BackendConfig.manifest.
  bool BuildNnrtConfig(const BackendConfig &config, backend::nnrt::NnrtConfig *nnrt);

  std::unique_ptr<backend::nnrt::NnrtExecutor> executor_;
  bool built_ = false;
};

/// Factory: keeps NnrtExecutor's complete type out of translation units that
/// only hold a unique_ptr<Backend> (GCC requires the member's complete type at
/// make_unique instantiation points).
std::unique_ptr<Backend> CreateNNRTBackend();

}  // namespace nnrt
}  // namespace backend
}  // namespace mslite

#endif  // MSLLM_NNRT_BACKEND_H
