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
/**
 * @file inference_engine.cpp
 * @brief Internal engine implementing the LLM-API.md public C API.
 *
 * State machine: CREATED → READY → GENERATING → READY.
 * Streaming (MSLLMStreamGenerate) blocks on the caller's thread and invokes
 * a callback per token.
 *
 * Buffer contract: caller pre-allocates; BUFFER_TOO_SMALL on overflow.
 */

#include "llm/llm.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <fstream>
#include <mutex>
#include <memory>
#include <string>
#include <vector>

#include "../llm_types_internal.h"
#include "manifest/model_manifest.h"
#include "manifest/msl_package_reader.h"
#include "pipeline/model_instance.h"
#include "tokenizer/tokenizer.h"
#include "sampler/sampler.h"
#include "backend/common/backend.h"
// Internal Engine

namespace {

struct EngineState {
  enum Value { kCreated = 0, kReady, kGenerating };
};

struct InternalEngine {
  // ── Lifecycle ───────────────────────────────────────────────────────
  std::atomic<EngineState::Value> state{EngineState::kCreated};

  // ── Model ───────────────────────────────────────────────────────────
  std::unique_ptr<mslite_llm::ModelInstance> model;
  mslite_llm::ModelResources resources;
  mslite_llm::ModelManifest manifest;

  // ── Tokenizer / Sampler ─────────────────────────────────────────────
  std::unique_ptr<mslite_llm::Tokenizer> tokenizer;
  std::unique_ptr<mslite_llm::Sampler> sampler;

  // ── Generation config ───────────────────────────────────────────────
  MSLLMGenerationConfig gen_config;
  std::mutex config_mutex;

  // ── Concurrency ─────────────────────────────────────────────────────
  std::mutex engine_mutex;
  std::atomic<bool> abort_flag{false};

  // ── Error ───────────────────────────────────────────────────────────
  std::string last_error;
  std::mutex error_mutex;
};

void SetError(InternalEngine *e, const std::string &msg) {
  std::lock_guard<std::mutex> lock(e->error_mutex);
  e->last_error = msg;
}

// ─── Config conversion ──────────────────────────────────────────────────────

MSLlmGenerateConfig ToInternalGenConfig(const MSLLMGenerationConfig &c) {
  MSLlmGenerateConfig gc = {};
  gc.max_new_tokens = c.max_new_tokens > 0 ? c.max_new_tokens : 256;
  gc.temperature = c.do_sample ? c.temperature : 0.0f;
  gc.top_k = c.top_k;
  gc.top_p = c.top_p;
  gc.repetition_penalty = c.repetition_penalty;
  gc.override_sampler = 1;
  return gc;
}

MSLlmGenerateConfig DefaultGenConfig() {
  MSLlmGenerateConfig gc = {};
  gc.max_new_tokens = 256;
  gc.temperature = 0.0f;
  gc.top_k = 1;
  gc.top_p = 1.0f;
  gc.repetition_penalty = 1.0f;
  gc.override_sampler = 1;
  return gc;
}

// ─── Helpers ────────────────────────────────────────────────────────────────

int32_t GetMaxSeqLen(const InternalEngine *e) {
  if (e->model && e->model->IsLoaded()) {
    return e->model->GetWeights().max_seq_len;
  }
  return 0;
}

bool IsEosToken(int32_t token_id, const InternalEngine *e) { return e->tokenizer->IsStopTokenId(token_id); }

// ─── Generation loop helper ─────────────────────────────────────────────────

struct GenContext {
  mslite_llm::Backend *backend;
  InternalEngine *engine;
  bool is_first_step;
};

/// Run one forward step: prefill on the first call, decode on subsequent calls.
/// Returns false on error.
bool StepForward(GenContext *ctx, std::vector<int32_t> &token_ids, int32_t position, std::vector<float> &logits) {
  auto *backend = ctx->backend;
  auto *model = ctx->engine->model.get();
  if (!backend || !model) return false;

  // Build position_ids
  std::vector<int32_t> position_ids(token_ids.size());
  for (size_t i = 0; i < token_ids.size(); ++i) {
    position_ids[i] = position + static_cast<int32_t>(i);
  }

  mslite_llm::BackendExecutionPhase phase;
  if (ctx->is_first_step) {
    phase = mslite_llm::BackendExecutionPhase::kPrefill;
    ctx->is_first_step = false;
  } else {
    phase = mslite_llm::BackendExecutionPhase::kDecode;
  }

  auto status = model->Execute(token_ids, position_ids, phase, logits);
  return status == MSLLM_SUCCESS;
}

}  // namespace

// Public C API Implementation

extern "C" {

MSLLMModelHandle MSLLMCreateModel(void) {
  auto *e = new InternalEngine();
  // Set sensible defaults
  e->gen_config.max_new_tokens = 256;
  e->gen_config.do_sample = false;
  e->gen_config.temperature = 1.0f;
  e->gen_config.top_k = 1;
  e->gen_config.top_p = 1.0f;
  e->gen_config.repetition_penalty = 1.0f;
  return reinterpret_cast<MSLLMModelHandle>(e);
}

MSLLMStatus MSLLMDestroyModel(MSLLMModelHandle llm_model) {
  if (llm_model == nullptr) return kMSLLM_ERROR_INVALID_ARGS;
  auto *e = reinterpret_cast<InternalEngine *>(llm_model);

  // Refuse to destroy while a generation is in-flight (use-after-free
  // otherwise). Caller sequence: Abort → wait for StreamGenerate to return →
  // Destroy (#16).
  if (e->state.load() == EngineState::kGenerating) return kMSLLM_ERROR_BUSY;

  delete e;
  return kMSLLM_SUCCESS;
}

MSLLMStatus MSLLMBuildModel(MSLLMModelHandle llm_model, const char *model_path) {
  if (llm_model == nullptr || model_path == nullptr) return kMSLLM_ERROR_INVALID_ARGS;
  auto *e = reinterpret_cast<InternalEngine *>(llm_model);

  std::lock_guard<std::mutex> lock(e->engine_mutex);
  if (e->state.load() == EngineState::kGenerating) return kMSLLM_ERROR_BUSY;
  if (e->state.load() == EngineState::kReady) return kMSLLM_ERROR_NOT_SUPPORTED;
  if (e->state.load() != EngineState::kCreated) return kMSLLM_ERROR_INVALID_ARGS;

  std::string path(model_path);
  if (path.empty()) return kMSLLM_ERROR_INVALID_ARGS;

  // ── Determine backend type ──────────────────────────────────────────
  auto backend_type = MSLLM_BACKEND_NNRT;

  // ── Load resources (directory or single-file .msl) ──────────────────
  std::string resource_error;
  auto status = mslite_llm::LoadModelResources(path, &e->resources, backend_type, &resource_error);
  if (status != MSLLM_SUCCESS) {
    SetError(e, "resource load: " + resource_error);
    return kMSLLM_ERROR_MODEL_LOAD;
  }

  // ── Load manifest (directory file or single-file entry) ─────────────
  std::string manifest_error;
  if (e->resources.single_file) {
    // Single-file .msl stores the manifest as flat KV entries (msl_pack v1);
    // rebuild the typed manifest from the package reader, same as
    // LoadModelResourcesFromSingleFile does for validation.
    status = mslite_llm::BuildModelManifestFromKv(*e->resources.package_reader, &e->manifest, &manifest_error);
  } else {
    status = mslite_llm::LoadModelManifest(path + "/manifest.json", &e->manifest, &manifest_error);
  }
  if (status != MSLLM_SUCCESS) {
    SetError(e, "manifest load: " + manifest_error);
    return kMSLLM_ERROR_MODEL_LOAD;
  }

  MSLlmModelConfig model_cfg = {};
  model_cfg.max_context_len = e->manifest.architecture.max_position_embeddings;
  model_cfg.max_batch_size = 1;

  MSLlmEngineConfig engine_cfg = {};
  engine_cfg.backend_type = backend_type;
  engine_cfg.num_threads = 2;

  // ── Create and load ModelInstance (manifest pre-parsed for single-file) ──
  e->model = std::make_unique<mslite_llm::ModelInstance>();
  if (e->resources.single_file) {
    status = e->model->Load(path, e->manifest, model_cfg, engine_cfg);
  } else {
    status = e->model->Load(path, model_cfg, engine_cfg);
  }
  if (status != MSLLM_SUCCESS) {
    SetError(e, "model load failed");
    e->model.reset();
    return kMSLLM_ERROR_MODEL_LOAD;
  }

  // ── Init backend with resources ─────────────────────────────────────
  mslite_llm::BackendConfig backend_cfg;
  backend_cfg.resources = &e->resources;
  backend_cfg.manifest = &e->manifest;
  backend_cfg.num_threads = 2;
  status = e->model->InitBackend(backend_cfg);
  if (status != MSLLM_SUCCESS) {
    SetError(e, "backend init failed");
    e->model.reset();
    return kMSLLM_ERROR_MODEL_LOAD;
  }

  // ── Create tokenizer (single-file entry or filesystem path) ─────────
  if (e->resources.single_file) {
    std::vector<uint8_t> vocab;
    if (!e->resources.package_reader || !e->resources.package_reader->Read(e->resources.tokenizer_path, &vocab)) {
      SetError(e, "tokenizer entry not found in .msl");
      e->model.reset();
      return kMSLLM_ERROR_MODEL_LOAD;
    }
    e->tokenizer = mslite_llm::CreateTokenizerFromBuffer(vocab.data(), vocab.size());
  } else {
    std::string vocab_path = e->resources.tokenizer_path;
    if (vocab_path.empty()) {
      vocab_path = path + "/vocab.bin";
    }
    // Fallback: look for tokenizer.model (SentencePiece)
    {
      std::ifstream test(vocab_path, std::ios::binary);
      if (!test.good()) {
        vocab_path = path + "/tokenizer.model";
      }
    }
    e->tokenizer = mslite_llm::CreateTokenizer(vocab_path);
  }
  if (!e->tokenizer) {
    SetError(e, "tokenizer creation failed");
    e->model.reset();
    return kMSLLM_ERROR_MODEL_LOAD;
  }

  // ── Create sampler ──────────────────────────────────────────────────
  auto internal_gen_config = DefaultGenConfig();
  e->sampler = std::make_unique<mslite_llm::Sampler>(internal_gen_config);

  e->state.store(EngineState::kReady);
  return kMSLLM_SUCCESS;
}

MSLLMStatus MSLLMSetGenerationConfig(MSLLMModelHandle llm_model, const MSLLMGenerationConfig config) {
  if (llm_model == nullptr) return kMSLLM_ERROR_INVALID_ARGS;
  auto *e = reinterpret_cast<InternalEngine *>(llm_model);

  // Boundary validation (#3/#4/#5/#6): only the parameters actually used by
  // the sampling strategy are validated (whitelist); out-of-range values are
  // rejected and the caller's previous config is preserved on rejection.
  // max_new_tokens: -1 and 0 both mean "no explicit output cap" (#3).
  if (config.max_new_tokens < -1) {
    return kMSLLM_ERROR_INVALID_ARGS;
  }
  if (config.do_sample) {
    // do_sample=true consumes temperature/top_k/top_p; do_sample=false
    // (greedy) ignores them, so they are not validated then (#5).
    // repetition_penalty applies to both strategies (sampler applies it
    // before sampling), has no defined range and 0 falls back to 1.0.
    if (config.temperature < 0.0f || config.temperature > 2.0f || config.top_k < 0 || config.top_p < 0.0f ||
        config.top_p > 1.0f) {
      return kMSLLM_ERROR_INVALID_ARGS;
    }
  }

  std::lock_guard<std::mutex> lock(e->config_mutex);
  if (e->state.load() == EngineState::kGenerating) return kMSLLM_ERROR_BUSY;

  e->gen_config = config;
  return kMSLLM_SUCCESS;
}

MSLLMStatus MSLLMGetGenerationConfig(MSLLMModelHandle llm_model, MSLLMGenerationConfig *config) {
  if (llm_model == nullptr || config == nullptr) return kMSLLM_ERROR_INVALID_ARGS;
  auto *e = reinterpret_cast<InternalEngine *>(llm_model);

  std::lock_guard<std::mutex> lock(e->config_mutex);
  *config = e->gen_config;
  return kMSLLM_SUCCESS;
}

MSLLMStatus MSLLMApplyChatTemplate(MSLLMModelHandle llm_model, const MSLLMChatMessage *messages, int num_messages,
                                   int add_generation_prompt, char *generated_prompt, int prompt_size) {
  if (llm_model == nullptr || messages == nullptr || num_messages <= 0 || generated_prompt == nullptr ||
      prompt_size <= 0) {
    return kMSLLM_ERROR_INVALID_ARGS;
  }
  auto *e = reinterpret_cast<InternalEngine *>(llm_model);

  std::vector<MSLlmChatMessage> msgs;
  msgs.reserve(static_cast<size_t>(num_messages));
  for (int i = 0; i < num_messages; ++i) {
    if (messages[i].content == nullptr) return kMSLLM_ERROR_INVALID_ARGS;  // #10
    MSLlmChatMessage m;
    m.role = static_cast<MSLlmChatRole>(messages[i].role);
    m.content = messages[i].content;
    msgs.push_back(m);
  }

  // D9: during a generation the template interface reports BUSY, regardless
  // of resource availability.
  {
    std::lock_guard<std::mutex> lock(e->engine_mutex);
    if (e->state.load() == EngineState::kGenerating) return kMSLLM_ERROR_BUSY;
  }

  if (!e->tokenizer) return kMSLLM_ERROR_INVALID_ARGS;

  // Template-less packages are rejected: the runtime has no builtin renderer
  // The template is pinned at export time.
  if (!e->tokenizer->HasChatTemplate()) {
    return kMSLLM_ERROR_MODEL_LOAD;
  }

  std::string rendered = e->tokenizer->ApplyChatTemplate(msgs, add_generation_prompt != 0);
  int needed = static_cast<int>(rendered.size()) + 1;

  if (needed > prompt_size) {
    return kMSLLM_ERROR_BUFFER_TOO_SMALL;
  }

  std::memcpy(generated_prompt, rendered.c_str(), static_cast<size_t>(needed));
  return kMSLLM_SUCCESS;
}

MSLLMStatus MSLLMGenerate(MSLLMModelHandle llm_model, const char *prompt, char *generated_text, int text_size) {
  if (llm_model == nullptr || prompt == nullptr || generated_text == nullptr || text_size <= 0) {
    return kMSLLM_ERROR_INVALID_ARGS;
  }
  auto *e = reinterpret_cast<InternalEngine *>(llm_model);

  // ── Acquire engine mutex, validate state and readiness ──────────────
  std::unique_lock<std::mutex> lock(e->engine_mutex);
  if (e->state.load() != EngineState::kReady) {
    if (e->state.load() == EngineState::kGenerating) return kMSLLM_ERROR_BUSY;
    return kMSLLM_ERROR_INVALID_ARGS;
  }
  if (!e->model || !e->tokenizer || !e->sampler || !e->model->IsLoaded()) {
    return kMSLLM_ERROR_INVALID_ARGS;
  }
  e->state.store(EngineState::kGenerating);
  e->abort_flag.store(false);
  lock.unlock();

  // ── Snapshot config ─────────────────────────────────────────────────
  MSLLMGenerationConfig cfg;
  {
    std::lock_guard<std::mutex> cl(e->config_mutex);
    cfg = e->gen_config;
  }
  auto internal_cfg = ToInternalGenConfig(cfg);
  e->sampler->ApplyConfigOverrides(internal_cfg);

  // ── Tokenize ────────────────────────────────────────────────────────
  std::vector<int32_t> input_ids = e->tokenizer->Encode(prompt);
  if (input_ids.empty()) {
    lock.lock();
    e->state.store(EngineState::kReady);
    return kMSLLM_ERROR_INVALID_ARGS;
  }

  int32_t max_seq_len = GetMaxSeqLen(e);
  // max_new_tokens==0 means "no explicit output cap" (#3): generate until
  // EOS or the context window is exhausted.
  int32_t max_new = cfg.max_new_tokens;

  // ── Context overflow check ──────────────────────────────────────────
  if (max_seq_len > 0 && static_cast<int32_t>(input_ids.size()) >= max_seq_len) {
    lock.lock();
    e->state.store(EngineState::kReady);
    return kMSLLM_ERROR_CONTEXT_OVERFLOW;
  }

  // ── Reset backend state ─────────────────────────────────────────────
  e->model->ResetGenerationState();
  e->sampler->Reset();

  // ── Generation loop ─────────────────────────────────────────────────
  std::string output;
  std::vector<int32_t> token_ids = input_ids;
  std::vector<int32_t> generated_ids;
  GenContext ctx;
  ctx.backend = e->model->GetBackend();
  ctx.engine = e;
  ctx.is_first_step = true;

  int32_t position = 0;
  std::vector<float> logits;

  // Prefill: feed all prompt tokens
  if (!StepForward(&ctx, token_ids, position, logits)) {
    lock.lock();
    e->state.store(EngineState::kReady);
    return kMSLLM_ERROR_INFERENCE;
  }

  // Sample first token
  int32_t token_id = e->sampler->Sample(logits);
  position = static_cast<int32_t>(input_ids.size());
  int32_t generated_count = 1;
  bool eos = IsEosToken(token_id, e);

  if (!eos) {
    generated_ids.push_back(token_id);
    output = e->tokenizer->Decode(generated_ids);
  }

  // Decode loop: generate until EOS, explicit output cap, or the context
  // window is exhausted. Non-streaming is not abortable (D8).
  while (!eos) {
    if (max_new > 0 && generated_count >= max_new) break;       // MAX_OUTPUT_LENGTH
    if (max_seq_len > 0 && position + 1 >= max_seq_len) break;  // MAX_CONTEXT_LENGTH

    std::vector<int32_t> single_token = {token_id};
    ctx.is_first_step = false;
    if (!StepForward(&ctx, single_token, position, logits)) {
      lock.lock();
      e->state.store(EngineState::kReady);
      return kMSLLM_ERROR_INFERENCE;
    }

    token_id = e->sampler->Sample(logits);
    ++position;
    ++generated_count;

    eos = IsEosToken(token_id, e);
    if (!eos) {
      generated_ids.push_back(token_id);
      output = e->tokenizer->Decode(generated_ids);
    }
  }

  // ── Buffer contract check ───────────────────────────────────────────
  int needed = static_cast<int>(output.size()) + 1;

  lock.lock();
  e->state.store(EngineState::kReady);
  lock.unlock();

  if (needed > text_size) {
    return kMSLLM_ERROR_BUFFER_TOO_SMALL;
  }

  std::memcpy(generated_text, output.c_str(), static_cast<size_t>(needed));
  return kMSLLM_SUCCESS;
}

MSLLMStatus MSLLMStreamGenerate(MSLLMModelHandle llm_model, const char *prompt, MSLLMStreamCallback callback,
                                void *user_data) {
  if (llm_model == nullptr || prompt == nullptr || callback == nullptr) {
    return kMSLLM_ERROR_INVALID_ARGS;
  }
  auto *e = reinterpret_cast<InternalEngine *>(llm_model);

  // ── Acquire engine mutex, validate state and readiness ──────────────
  std::unique_lock<std::mutex> lock(e->engine_mutex);
  if (e->state.load() != EngineState::kReady) {
    if (e->state.load() == EngineState::kGenerating) return kMSLLM_ERROR_BUSY;
    return kMSLLM_ERROR_INVALID_ARGS;
  }
  if (!e->model || !e->tokenizer || !e->sampler || !e->model->IsLoaded()) {
    return kMSLLM_ERROR_INVALID_ARGS;
  }
  e->state.store(EngineState::kGenerating);
  e->abort_flag.store(false);
  lock.unlock();

  // ── Snapshot config ─────────────────────────────────────────────────
  MSLLMGenerationConfig cfg;
  {
    std::lock_guard<std::mutex> cl(e->config_mutex);
    cfg = e->gen_config;
  }
  auto internal_cfg = ToInternalGenConfig(cfg);
  e->sampler->ApplyConfigOverrides(internal_cfg);

  // ── Tokenize ────────────────────────────────────────────────────────
  std::vector<int32_t> input_ids = e->tokenizer->Encode(prompt);
  if (input_ids.empty()) {
    lock.lock();
    e->state.store(EngineState::kReady);
    return kMSLLM_ERROR_INVALID_ARGS;
  }

  int32_t max_seq_len = GetMaxSeqLen(e);
  // max_new_tokens==0 means "no explicit output cap" (#3).
  int32_t max_new = cfg.max_new_tokens;

  // ── Context overflow check ──────────────────────────────────────────
  if (max_seq_len > 0 && static_cast<int32_t>(input_ids.size()) >= max_seq_len) {
    lock.lock();
    e->state.store(EngineState::kReady);
    return kMSLLM_ERROR_CONTEXT_OVERFLOW;
  }

  // ── Reset backend ───────────────────────────────────────────────────
  e->model->ResetGenerationState();
  e->sampler->Reset();

  // ── Generation loop with callback ───────────────────────────────────
  std::vector<int32_t> token_ids = input_ids;
  std::vector<int32_t> generated_ids;
  GenContext ctx;
  ctx.backend = e->model->GetBackend();
  ctx.engine = e;
  ctx.is_first_step = true;

  int32_t position = 0;
  std::vector<float> logits;

  // Prefill
  if (!StepForward(&ctx, token_ids, position, logits)) {
    callback(nullptr, kMSLLM_FINISHED_BY_ERROR, user_data);
    lock.lock();
    e->state.store(EngineState::kReady);
    return kMSLLM_ERROR_INFERENCE;
  }

  int32_t token_id = e->sampler->Sample(logits);
  position = static_cast<int32_t>(input_ids.size());
  int32_t generated_count = 1;
  bool eos = IsEosToken(token_id, e);

  MSLLMFinishReason finish_reason = kMSLLM_RUNNING;

  if (!eos) {
    generated_ids.push_back(token_id);
    std::string delta = e->tokenizer->DecodeIncremental(token_id);
    callback(delta.c_str(), kMSLLM_RUNNING, user_data);
  }

  // Decode loop: terminate on abort, explicit output cap (wins over context
  // per #14), context window, EOS, or inference error.
  while (!eos) {
    if (e->abort_flag.load()) {
      finish_reason = kMSLLM_STOPPED_BY_USER;
      break;
    }
    if (max_new > 0 && generated_count >= max_new) {
      finish_reason = kMSLLM_FINISHED_BY_MAX_OUTPUT_LENGTH;
      break;
    }
    if (max_seq_len > 0 && position + 1 >= max_seq_len) {
      finish_reason = kMSLLM_FINISHED_BY_MAX_CONTEXT_LENGTH;
      break;
    }

    std::vector<int32_t> single_token = {token_id};
    ctx.is_first_step = false;
    if (!StepForward(&ctx, single_token, position, logits)) {
      finish_reason = kMSLLM_FINISHED_BY_ERROR;
      break;
    }

    token_id = e->sampler->Sample(logits);
    ++position;
    ++generated_count;

    eos = IsEosToken(token_id, e);
    if (!eos) {
      generated_ids.push_back(token_id);
      std::string delta = e->tokenizer->DecodeIncremental(token_id);
      callback(delta.c_str(), kMSLLM_RUNNING, user_data);
    }
  }

  if (eos) {
    finish_reason = kMSLLM_FINISHED_BY_EOS;
  }

  // Flush any buffered incomplete UTF-8 tail before the terminal callback
  // (#17).
  std::string tail = e->tokenizer->FlushDecode();
  if (!tail.empty()) {
    callback(tail.c_str(), kMSLLM_RUNNING, user_data);
  }

  // Terminal callback
  callback(nullptr, finish_reason, user_data);

  lock.lock();
  e->state.store(EngineState::kReady);
  return kMSLLM_SUCCESS;
}

MSLLMStatus MSLLMAbort(MSLLMModelHandle llm_model) {
  if (llm_model == nullptr) return kMSLLM_ERROR_INVALID_ARGS;
  auto *e = reinterpret_cast<InternalEngine *>(llm_model);

  // Only affect an in-flight streaming generation; otherwise no-op (#13).
  if (e->state.load() == EngineState::kGenerating) {
    e->abort_flag.store(true);
  }
  return kMSLLM_SUCCESS;
}

}  // extern "C"
