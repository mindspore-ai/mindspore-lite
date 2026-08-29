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
#ifndef MSLLM_MODEL_INSTANCE_H
#define MSLLM_MODEL_INSTANCE_H

#include <string>
#include <vector>
#include <memory>
#include "../llm_types_internal.h"
#include "backend/common/backend.h"
#include "manifest/model_manifest.h"

namespace mslite_llm {

struct ModelWeights {
  std::string model_path;
  MSLlmDType dtype = MSLLM_DTYPE_FLOAT32;

  int32_t num_layers = 0;
  int32_t hidden_size = 0;
  int32_t intermediate_size = 0;
  int32_t num_heads = 0;
  int32_t num_kv_heads = 0;
  int32_t head_dim = 0;
  int32_t vocab_size = 0;
  int32_t max_position_embeddings = 0;
  float rope_theta = 10000.0f;
  float norm_eps = 1e-6f;
  int32_t tie_word_embeddings = 0;

  int32_t max_seq_len = 0;
};

class ModelInstance {
 public:
  ModelInstance();
  ~ModelInstance();

  /// Load model from .msl package. Architecture params read from manifest.json.
  /// Backend execution delegated to mindspore::lite session.
  MSLlmStatus Load(const std::string &model_path, const MSLlmModelConfig &config,
                   const MSLlmEngineConfig &engine_config);
  /// Load with a manifest already parsed by the caller (single-file .msl path,
  /// where manifest.json lives inside the container rather than on disk).
  MSLlmStatus Load(const std::string &model_path, const ModelManifest &manifest, const MSLlmModelConfig &config,
                   const MSLlmEngineConfig &engine_config);
  MSLlmStatus Unload();

  /// Run one forward step. phase selects prefill (full prompt) vs decode
  /// (single token); logits receives the next-token distribution. position_ids
  /// is reserved for future positional backends (NNRT ignores it).
  MSLlmStatus Execute(const std::vector<int32_t> &input_ids, const std::vector<int32_t> &position_ids,
                      BackendExecutionPhase phase, std::vector<float> &logits);

  MSLlmStatus ExecuteTensor(const std::vector<float> &input_tensor, const std::vector<int64_t> &input_shape,
                            std::vector<float> &output_tensor, std::vector<int64_t> &output_shape);

  const ModelWeights &GetWeights() const;
  Backend *GetBackend();
  const ModelManifest &GetManifest() const;

  /// Forward BackendConfig to the backend (NPU path needs resources + manifest).
  MSLlmStatus InitBackend(const BackendConfig &config);

  int32_t GetModelId() const;
  void SetModelId(int32_t id);

  bool IsLoaded() const;

  MSLlmStatus ResetGenerationState();
  int32_t GetContextLimit() const;

 private:
  /// Shared tail of Load: populate weights from manifest_, create the backend.
  /// Assumes manifest_ is already set.
  MSLlmStatus LoadWithManifest(const std::string &model_path, const MSLlmModelConfig &config,
                               const MSLlmEngineConfig &engine_config);

  ModelWeights weights_;
  ModelManifest manifest_;
  std::unique_ptr<Backend> backend_;
  int32_t model_id_;
  bool loaded_;
  MSLlmModelConfig model_config_;
};

}  // namespace mslite_llm

#endif
