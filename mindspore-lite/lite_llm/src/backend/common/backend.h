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
#ifndef MSLLM_BACKEND_H
#define MSLLM_BACKEND_H

#include <cstdint>
#include <string>
#include <vector>

#include "../../llm_types_internal.h"

namespace mslite_llm {

struct ModelManifest;
struct ModelResources;

enum class BackendExecutionPhase {
  kPrefill,
  kDecode,
};

// ─── Unified Backend Contract ──────────────────────────────────────────────

struct BackendConfig {
  const ModelResources *resources = nullptr;
  const ModelManifest *manifest = nullptr;
  int32_t num_threads = 2;
  int32_t npu_device_id = 0;
};

struct BackendInput {
  std::vector<int32_t> input_ids;
  int32_t valid_seq_len = 0;  // length of valid input (excludes padding)
  bool is_prefill = false;
};

struct BackendOutput {
  std::vector<float> logits;
  int32_t next_token_id = -1;  // argmax of logits, set by caller or backend
};

class Backend {
 public:
  virtual ~Backend() = default;

  /// One-time initialisation with backend configuration and resource references.
  virtual MSLlmStatus Init(const BackendConfig & /*config*/) = 0;

  /// Run a prefill step: feed the full prompt, produce logits for the next token.
  virtual MSLlmStatus Prefill(const BackendInput &input, BackendOutput *output) = 0;

  /// Run a decode step: feed a single token, produce logits for the next token.
  virtual MSLlmStatus Decode(const BackendInput &input, BackendOutput *output) = 0;

  /// Reset per-request state (KV cache, sequence length, etc.).
  virtual MSLlmStatus Reset() = 0;
};

}  // namespace mslite_llm

#endif
