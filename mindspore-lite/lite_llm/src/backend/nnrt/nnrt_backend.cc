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
#include "backend/nnrt/nnrt_backend.h"

#include "backend/common/backend_factory.h"
#include "backend/nnrt/nnrt_config.h"
#include "backend/nnrt/nnrt_executor.h"
#include "manifest/model_manifest.h"

namespace mslite {
namespace backend {
namespace nnrt {

// Helpers

bool NNRTBackend::BuildNnrtConfig(const BackendConfig &config, backend::nnrt::NnrtConfig *nnrt) {
  if (config.resources == nullptr || config.manifest == nullptr) {
    return false;
  }

  const auto &res = *config.resources;
  const auto &man = *config.manifest;

  // ── Architecture (from manifest) ──────────────────────────────────────
  nnrt->num_layers = man.architecture.num_layers;
  nnrt->hidden_size = man.architecture.hidden_size;
  nnrt->num_key_value_heads = man.architecture.num_kv_heads;
  nnrt->head_dim = man.architecture.head_dim;
  nnrt->vocab_size = man.architecture.vocab_size;
  nnrt->max_position_embeddings = man.architecture.max_position_embeddings;

  // ── Graph paths (from resources) ──────────────────────────────────────
  nnrt->prefill_path = res.prefill_path;
  nnrt->decode_path = res.decode_path;

  // ── Asset paths (from resources) ──────────────────────────────────────
  nnrt->embedding_path = res.embedding_path;
  nnrt->embedding_fp16_path = res.embedding_fp16_path;
  nnrt->rope_sin_path = res.rope_sin_path;
  nnrt->rope_cos_path = res.rope_cos_path;
  nnrt->attention_mask_path = res.attention_mask_path;

  // ── NNRT-specific ─────────────────────────────────────────────────────
  nnrt->device_id = static_cast<size_t>(config.npu_device_id);
  nnrt->eos_id =
    man.generation.present && !man.generation.stop_token_ids.empty() ? man.generation.stop_token_ids.front() : -1;

  // ── NPU runtime params (from manifest.npu, fixed at export) ──────────
  nnrt->max_length = man.npu.max_length;
  nnrt->chunk_size = man.npu.chunk_size;
  nnrt->embedding_quant = man.npu.embedding_quant;
  if (man.npu.scale_gp_size > 0) {
    nnrt->scale_gp_size = man.npu.scale_gp_size;
  }

  // ── Single-file .msl container ───────────────────────────────────────
  nnrt->single_file = res.single_file;
  nnrt->package_reader = res.package_reader;

  return true;
}

// Construction / Destruction

NNRTBackend::~NNRTBackend() {
  executor_.reset();
  built_ = false;
}

// Unified Backend Interface

MSLlmStatus NNRTBackend::Init(const BackendConfig &config) {
  if (built_) {
    return MSLLM_SUCCESS;
  }

  executor_ = std::make_unique<backend::nnrt::NnrtExecutor>();

  backend::nnrt::NnrtConfig nnrt;
  if (!BuildNnrtConfig(config, &nnrt)) {
    executor_.reset();
    return MSLLM_ERROR_INVALID_ARGS;
  }

  if (!executor_->Build(nnrt)) {
    executor_.reset();
    return MSLLM_ERROR_MODEL_LOAD;
  }

  built_ = true;
  return MSLLM_SUCCESS;
}

MSLlmStatus NNRTBackend::Prefill(const BackendInput &input, BackendOutput *output) {
  if (!built_ || !executor_) {
    return MSLLM_ERROR_INVALID_ARGS;
  }
  if (output == nullptr) {
    return MSLLM_ERROR_INVALID_ARGS;
  }

  std::vector<int> ids(input.input_ids.begin(), input.input_ids.end());

  int next_token = -1;
  if (!executor_->Forward(ids, &next_token, /*is_prefill=*/true, &output->logits)) {
    return MSLLM_ERROR_INFERENCE;
  }

  // Sampling (argmax or stochastic) happens in Pipeline::Sampler on the host;
  // the executor never picks a token when logits are requested.
  output->next_token_id = -1;
  return MSLLM_SUCCESS;
}

MSLlmStatus NNRTBackend::Decode(const BackendInput &input, BackendOutput *output) {
  if (!built_ || !executor_) {
    return MSLLM_ERROR_INVALID_ARGS;
  }
  if (output == nullptr) {
    return MSLLM_ERROR_INVALID_ARGS;
  }
  if (input.input_ids.empty()) {
    return MSLLM_ERROR_INVALID_ARGS;
  }

  std::vector<int> ids = {static_cast<int>(input.input_ids.back())};

  int next_token = -1;
  if (!executor_->Forward(ids, &next_token, /*is_prefill=*/false, &output->logits)) {
    return MSLLM_ERROR_INFERENCE;
  }

  // Sampling (argmax or stochastic) happens in Pipeline::Sampler on the host.
  output->next_token_id = -1;
  return MSLLM_SUCCESS;
}

MSLlmStatus NNRTBackend::Reset() {
  if (!executor_) {
    return MSLLM_ERROR_INVALID_ARGS;
  }
  if (!executor_->Reset()) {
    return MSLLM_ERROR_INFERENCE;
  }
  return MSLLM_SUCCESS;
}

std::unique_ptr<Backend> CreateNNRTBackend() { return std::make_unique<NNRTBackend>(); }

// Static registration: the anonymous-namespace static registers the NNRT
// creator at startup, so the engine discovers this backend without a
// hard-coded switch in the pipeline.
MSLLM_REGISTER_BACKEND(MSLLM_BACKEND_NNRT, CreateNNRTBackend);

}  // namespace nnrt
}  // namespace backend
}  // namespace mslite
