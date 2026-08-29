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
#include "pipeline/model_instance.h"

#include <algorithm>
#include <cstring>
#include <cmath>
#include <fstream>
#include <functional>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "backend/common/backend.h"
#include "backend/common/backend_factory.h"

namespace mslite_llm {

namespace {

bool FileExists(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  return file.good();
}

}  // namespace

ModelInstance::ModelInstance() : model_id_(0), loaded_(false) {}

ModelInstance::~ModelInstance() {
  if (loaded_) {
    Unload();
  }
}

MSLlmStatus ModelInstance::Load(const std::string &model_path, const MSLlmModelConfig &config,
                                const MSLlmEngineConfig &engine_config) {
  if (loaded_) {
    return MSLLM_ERROR_INVALID_ARGS;
  }

  const std::string manifest_path = model_path + "/manifest.json";
  const bool has_manifest = FileExists(manifest_path);
  if (!has_manifest && !FileExists(model_path) && !FileExists(model_path + "/npu/graph.bin")) {
    return MSLLM_ERROR_IO;
  }

  if (has_manifest) {
    std::string manifest_error;
    auto status = LoadModelManifest(manifest_path, &manifest_, &manifest_error);
    if (status != MSLLM_SUCCESS) {
      return status;
    }
  } else {
    manifest_ = ModelManifest{};
  }

  return LoadWithManifest(model_path, config, engine_config);
}

MSLlmStatus ModelInstance::Load(const std::string &model_path, const ModelManifest &manifest,
                                const MSLlmModelConfig &config, const MSLlmEngineConfig &engine_config) {
  if (loaded_) {
    return MSLLM_ERROR_INVALID_ARGS;
  }

  manifest_ = manifest;

  return LoadWithManifest(model_path, config, engine_config);
}

MSLlmStatus ModelInstance::LoadWithManifest(const std::string &model_path, const MSLlmModelConfig &config,
                                            const MSLlmEngineConfig &engine_config) {
  weights_.model_path = model_path;

  const auto &arch = manifest_.architecture;
  weights_.dtype = manifest_.dtype;
  weights_.num_layers = arch.num_layers > 0 ? arch.num_layers : config.num_layers;
  weights_.hidden_size = arch.hidden_size > 0 ? arch.hidden_size : config.hidden_size;
  weights_.intermediate_size = arch.intermediate_size > 0 ? arch.intermediate_size : config.intermediate_size;
  weights_.num_heads = arch.num_heads > 0 ? arch.num_heads : config.num_heads;
  weights_.num_kv_heads = arch.num_kv_heads > 0 ? arch.num_kv_heads : config.num_kv_heads;
  weights_.head_dim = arch.head_dim > 0 ? arch.head_dim : config.head_dim;
  weights_.vocab_size = arch.vocab_size > 0 ? arch.vocab_size : config.vocab_size;
  weights_.max_position_embeddings =
    arch.max_position_embeddings > 0
      ? arch.max_position_embeddings
      : (config.max_position_embeddings > 0 ? config.max_position_embeddings : config.max_context_len);
  weights_.rope_theta = arch.rope_theta > 0.0f ? arch.rope_theta : config.rope_theta;
  weights_.norm_eps = arch.norm_eps > 0.0f ? arch.norm_eps : config.norm_eps;
  weights_.tie_word_embeddings = arch.present ? arch.tie_word_embeddings : config.tie_word_embeddings;
  weights_.max_seq_len = arch.max_position_embeddings;
  if (weights_.max_seq_len <= 0) {
    weights_.max_seq_len = weights_.max_position_embeddings;
  }

  model_config_ = config;

  backend_ = CreateBackend(engine_config.backend_type);
  if (!backend_) {
    return MSLLM_ERROR_INVALID_ARGS;
  }
  // Backend initialisation (Init) is deferred to InitBackend(): the caller
  // supplies the BackendConfig with resource/manifest references after Load.
  loaded_ = true;
  return MSLLM_SUCCESS;
}

MSLlmStatus ModelInstance::Unload() {
  if (!loaded_) {
    return MSLLM_ERROR_INVALID_ARGS;
  }

  backend_.reset();  // backend destructor tears down the executor

  weights_ = ModelWeights();
  loaded_ = false;
  return MSLLM_SUCCESS;
}

MSLlmStatus ModelInstance::Execute(const std::vector<int32_t> &input_ids, const std::vector<int32_t> &position_ids,
                                   BackendExecutionPhase phase, std::vector<float> &logits) {
  if (!loaded_) {
    return MSLLM_ERROR_INVALID_ARGS;
  }

  if (input_ids.empty()) {
    return MSLLM_ERROR_INVALID_ARGS;
  }

  if (backend_) {
    BackendInput input;
    input.input_ids = input_ids;
    input.valid_seq_len = static_cast<int32_t>(input_ids.size());
    input.is_prefill = (phase == BackendExecutionPhase::kPrefill);
    BackendOutput output;
    MSLlmStatus status;
    if (input.is_prefill) {
      status = backend_->Prefill(input, &output);
    } else {
      status = backend_->Decode(input, &output);
    }
    if (status != MSLLM_SUCCESS) {
      return status;
    }
    logits = std::move(output.logits);
    return MSLLM_SUCCESS;
  }

  int32_t vocab_size = weights_.vocab_size > 0 ? weights_.vocab_size : 32000;
  logits.resize(vocab_size, 0.0f);

  return MSLLM_SUCCESS;
}

MSLlmStatus ModelInstance::ExecuteTensor(const std::vector<float> &input_tensor,
                                         const std::vector<int64_t> &input_shape, std::vector<float> &output_tensor,
                                         std::vector<int64_t> &output_shape) {
  if (!loaded_) {
    return MSLLM_ERROR_INVALID_ARGS;
  }

  if (input_tensor.empty() || input_shape.empty()) {
    return MSLLM_ERROR_INVALID_ARGS;
  }

  int64_t total_input =
    std::accumulate(input_shape.begin(), input_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());

  if (static_cast<int64_t>(input_tensor.size()) != total_input) {
    return MSLLM_ERROR_INVALID_ARGS;
  }

  int64_t batch = input_shape.size() > 0 ? input_shape[0] : 1;
  int64_t seq_len = input_shape.size() > 1 ? input_shape[1] : 1;

  // Language-model tensor execution: return an empty logits-shaped output.
  // (Vision encoder / projector execution was removed with multimodal support.)
  int32_t vocab_size = weights_.vocab_size > 0 ? weights_.vocab_size : 32000;
  output_shape = {batch, seq_len, vocab_size};
  output_tensor.resize(static_cast<size_t>(batch * seq_len * vocab_size), 0.0f);

  return MSLLM_SUCCESS;
}

const ModelWeights &ModelInstance::GetWeights() const { return weights_; }

Backend *ModelInstance::GetBackend() { return backend_.get(); }

const ModelManifest &ModelInstance::GetManifest() const { return manifest_; }

int32_t ModelInstance::GetModelId() const { return model_id_; }

void ModelInstance::SetModelId(int32_t id) { model_id_ = id; }

bool ModelInstance::IsLoaded() const { return loaded_; }

MSLlmStatus ModelInstance::ResetGenerationState() {
  if (backend_) {
    auto status = backend_->Reset();
    if (status != MSLLM_SUCCESS) {
      return status;
    }
  }
  return MSLLM_SUCCESS;
}

int32_t ModelInstance::GetContextLimit() const {
  int32_t limit = weights_.max_position_embeddings;
  if (model_config_.max_context_len > 0 && (limit <= 0 || model_config_.max_context_len < limit)) {
    limit = model_config_.max_context_len;
  }
  return limit;
}

MSLlmStatus ModelInstance::InitBackend(const BackendConfig &config) {
  if (!backend_) {
    return MSLLM_ERROR_INVALID_ARGS;
  }
  return backend_->Init(config);
}

}  // namespace mslite_llm
