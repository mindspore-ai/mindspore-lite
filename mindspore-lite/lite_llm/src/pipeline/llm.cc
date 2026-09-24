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

#include <sys/stat.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
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

// Defaults applied when the caller leaves generation knobs at their zero
// value (see ToInternalGenConfig / DefaultGenConfig).
constexpr int kDefaultMaxNewTokens = 256;
constexpr int kDefaultNumThreads = 2;

struct EngineState {
  enum class Value { kCreated = 0, kReady, kGenerating };
};

struct InternalEngine {
  // ── Lifecycle ───────────────────────────────────────────────────────
  std::atomic<EngineState::Value> state{EngineState::Value::kCreated};

  // ── Model ───────────────────────────────────────────────────────────
  std::unique_ptr<mslite_llm::ModelInstance> model;
  mslite_llm::ModelResources resources;
  mslite_llm::ModelManifest manifest;

  // ── Tokenizer / Sampler ─────────────────────────────────────────────
  std::unique_ptr<mslite_llm::Tokenizer> tokenizer;
  std::unique_ptr<mslite_llm::Sampler> sampler;

  // ── Generation config ───────────────────────────────────────────────
  MSLLMGenerationConfig gen_config = {};
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
  gc.max_new_tokens = c.max_new_tokens > 0 ? c.max_new_tokens : kDefaultMaxNewTokens;
  gc.temperature = c.do_sample ? c.temperature : 0.0f;
  gc.top_k = c.top_k;
  gc.top_p = c.top_p;
  gc.repetition_penalty = c.repetition_penalty;
  gc.override_sampler = 1;
  return gc;
}

MSLlmGenerateConfig DefaultGenConfig() {
  MSLlmGenerateConfig gc = {};
  gc.max_new_tokens = kDefaultMaxNewTokens;
  gc.temperature = 0.0f;
  gc.top_k = 1;
  gc.top_p = 1.0f;
  gc.repetition_penalty = 1.0f;
  gc.override_sampler = 1;
  return gc;
}

// ─── Helpers ────────────────────────────────────────────────────────────────

bool PathDoesNotExist(const std::string &path) {
  struct stat path_stat {};
  if (::stat(path.c_str(), &path_stat) == 0) {
    return false;
  }
  return errno == ENOENT || errno == ENOTDIR;
}

int32_t GetMaxSeqLen(const InternalEngine *e) {
  if (e->model && e->model->IsLoaded()) {
    return e->model->GetContextLimit();
  }
  return 0;
}

bool IsEosToken(int32_t token_id, const InternalEngine *e) { return e->tokenizer->IsStopTokenId(token_id); }

int32_t SampleStepLogits(mslite_llm::Sampler *sampler, const mslite_llm::BackendOutput &output) {
  if (output.logits_view != nullptr) {
    return sampler->Sample(output.logits_view, output.logits_view_size);
  }
  return sampler->Sample(output.logits);
}

// ─── Generation loop helper ─────────────────────────────────────────────────

struct GenContext {
  InternalEngine *engine;
  bool is_first_step;
};

/// Run one forward step: prefill on the first call, decode on subsequent calls.
/// Returns false on error.
bool StepForward(GenContext *ctx, std::vector<int32_t> &token_ids, int32_t position,
                 mslite_llm::BackendOutput *output) {
  auto *model = ctx->engine->model.get();
  if (model == nullptr || output == nullptr) {
    return false;
  }

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

  auto status = model->Execute(token_ids, position_ids, phase, output);
  return status == MSLLM_SUCCESS;
}

MSLLMStatus LoadEngineResources(InternalEngine *e, const std::string &path) {
  std::string error;
  auto status = mslite_llm::LoadModelResources(path, &e->resources, MSLLM_BACKEND_NNRT, &error);
  if (status != MSLLM_SUCCESS) {
    SetError(e, "resource load: " + error);
    return kMSLLM_ERROR_MODEL_LOAD;
  }
  if (e->resources.single_file) {
    status = mslite_llm::BuildModelManifestFromKv(*e->resources.package_reader, &e->manifest, &error);
  } else {
    status = mslite_llm::LoadModelManifest(path + "/manifest.json", &e->manifest, &error);
  }
  if (status != MSLLM_SUCCESS) {
    SetError(e, "manifest load: " + error);
    return kMSLLM_ERROR_MODEL_LOAD;
  }
  return kMSLLM_SUCCESS;
}

MSLLMStatus LoadEngineModel(InternalEngine *e, const std::string &path) {
  MSLlmModelConfig model_cfg = {};
  model_cfg.max_context_len =
    e->manifest.npu.present ? e->manifest.npu.max_length : e->manifest.architecture.max_position_embeddings;
  model_cfg.max_batch_size = 1;
  MSLlmEngineConfig engine_cfg = {};
  engine_cfg.backend_type = MSLLM_BACKEND_NNRT;
  engine_cfg.num_threads = kDefaultNumThreads;
  e->model = std::make_unique<mslite_llm::ModelInstance>();
  auto status = e->resources.single_file ? e->model->Load(path, e->manifest, model_cfg, engine_cfg)
                                         : e->model->Load(path, model_cfg, engine_cfg);
  if (status != MSLLM_SUCCESS) {
    SetError(e, "model load failed");
    e->model.reset();
    return kMSLLM_ERROR_MODEL_LOAD;
  }
  mslite_llm::BackendConfig backend_cfg;
  backend_cfg.resources = &e->resources;
  backend_cfg.manifest = &e->manifest;
  backend_cfg.num_threads = kDefaultNumThreads;
  if (e->model->InitBackend(backend_cfg) != MSLLM_SUCCESS) {
    SetError(e, "backend init failed");
    e->model.reset();
    return kMSLLM_ERROR_MODEL_LOAD;
  }
  return kMSLLM_SUCCESS;
}

MSLLMStatus LoadEngineTokenizer(InternalEngine *e, const std::string &path) {
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
    // Fallback: look for tokenizer.model (SentencePiece).
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
  return kMSLLM_SUCCESS;
}

void FinishGeneration(InternalEngine *e) {
  std::lock_guard<std::mutex> lock(e->engine_mutex);
  e->state.store(EngineState::Value::kReady);
}

MSLLMStatus PrepareGeneration(InternalEngine *e, const char *prompt, std::vector<int32_t> *input_ids,
                              MSLLMGenerationConfig *cfg) {
  {
    std::lock_guard<std::mutex> lock(e->engine_mutex);
    if (e->state.load() != EngineState::Value::kReady) {
      if (e->state.load() == EngineState::Value::kGenerating) return kMSLLM_ERROR_BUSY;
      return kMSLLM_ERROR_INVALID_ARGS;
    }
    if (!e->model || !e->tokenizer || !e->sampler || !e->model->IsLoaded()) {
      return kMSLLM_ERROR_INVALID_ARGS;
    }
    e->state.store(EngineState::Value::kGenerating);
    e->abort_flag.store(false);
  }
  {
    std::lock_guard<std::mutex> lock(e->config_mutex);
    *cfg = e->gen_config;
  }
  e->sampler->ApplyConfigOverrides(ToInternalGenConfig(*cfg));
  *input_ids = e->tokenizer->Encode(prompt);
  if (input_ids->empty()) {
    FinishGeneration(e);
    return kMSLLM_ERROR_INVALID_ARGS;
  }
  const int32_t max_seq_len = GetMaxSeqLen(e);
  if (max_seq_len > 0 && static_cast<int32_t>(input_ids->size()) >= max_seq_len) {
    FinishGeneration(e);
    return kMSLLM_ERROR_CONTEXT_OVERFLOW;
  }
  e->model->ResetGenerationState();
  e->sampler->Reset();
  return kMSLLM_SUCCESS;
}

MSLLMFinishReason GenerationLimit(int32_t max_new, int32_t generated_count, int32_t max_seq_len, int32_t position) {
  // Zero means no explicit output cap; an explicit cap wins over context.
  if (max_new > 0 && generated_count >= max_new) return kMSLLM_FINISHED_BY_MAX_OUTPUT_LENGTH;
  if (max_seq_len > 0 && position + 1 >= max_seq_len) return kMSLLM_FINISHED_BY_MAX_CONTEXT_LENGTH;
  return kMSLLM_RUNNING;
}

MSLLMStatus GenerateText(InternalEngine *e, std::vector<int32_t> &token_ids, int32_t max_new, std::string *output) {
  GenContext ctx{e, true};
  mslite_llm::BackendOutput backend_output;
  if (!StepForward(&ctx, token_ids, 0, &backend_output)) return kMSLLM_ERROR_INFERENCE;
  int32_t token_id = SampleStepLogits(e->sampler.get(), backend_output);
  int32_t position = static_cast<int32_t>(token_ids.size());
  int32_t generated_count = 1;
  const int32_t max_seq_len = GetMaxSeqLen(e);
  std::vector<int32_t> generated_ids;
  // Non-streaming generation is not abortable (D8).
  while (!IsEosToken(token_id, e)) {
    generated_ids.push_back(token_id);
    *output = e->tokenizer->Decode(generated_ids);
    if (GenerationLimit(max_new, generated_count, max_seq_len, position) != kMSLLM_RUNNING) {
      break;
    }
    std::vector<int32_t> single_token = {token_id};
    if (!StepForward(&ctx, single_token, position, &backend_output)) return kMSLLM_ERROR_INFERENCE;
    token_id = SampleStepLogits(e->sampler.get(), backend_output);
    ++position;
    ++generated_count;
  }
  return kMSLLM_SUCCESS;
}

MSLLMStatus GenerateStream(InternalEngine *e, std::vector<int32_t> &token_ids, int32_t max_new,
                           MSLLMStreamCallback callback, void *user_data) {
  GenContext ctx{e, true};
  mslite_llm::BackendOutput backend_output;
  if (!StepForward(&ctx, token_ids, 0, &backend_output)) {
    callback(nullptr, kMSLLM_FINISHED_BY_ERROR, user_data);
    return kMSLLM_ERROR_INFERENCE;
  }
  int32_t token_id = SampleStepLogits(e->sampler.get(), backend_output);
  int32_t position = static_cast<int32_t>(token_ids.size());
  int32_t generated_count = 1;
  const int32_t max_seq_len = GetMaxSeqLen(e);
  MSLLMFinishReason finish_reason = kMSLLM_FINISHED_BY_EOS;
  while (!IsEosToken(token_id, e)) {
    std::string delta = e->tokenizer->DecodeIncremental(token_id);
    callback(delta.c_str(), kMSLLM_RUNNING, user_data);
    if (e->abort_flag.load()) {
      finish_reason = kMSLLM_STOPPED_BY_USER;
      break;
    }
    finish_reason = GenerationLimit(max_new, generated_count, max_seq_len, position);
    if (finish_reason != kMSLLM_RUNNING) {
      break;
    }
    std::vector<int32_t> single_token = {token_id};
    if (!StepForward(&ctx, single_token, position, &backend_output)) {
      finish_reason = kMSLLM_FINISHED_BY_ERROR;
      break;
    }
    token_id = SampleStepLogits(e->sampler.get(), backend_output);
    ++position;
    ++generated_count;
  }
  if (IsEosToken(token_id, e)) finish_reason = kMSLLM_FINISHED_BY_EOS;
  // Flush buffered incomplete UTF-8 before the terminal callback (#17).
  std::string tail = e->tokenizer->FlushDecode();
  if (!tail.empty()) callback(tail.c_str(), kMSLLM_RUNNING, user_data);
  callback(nullptr, finish_reason, user_data);
  return kMSLLM_SUCCESS;
}

}  // namespace

// Public C API Implementation

extern "C" {
MSLLMModelHandle MSLLMCreateModel(void) {
  auto *e = new InternalEngine();
  // Set sensible defaults
  e->gen_config.max_new_tokens = kDefaultMaxNewTokens;
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
  if (e->state.load() == EngineState::Value::kGenerating) return kMSLLM_ERROR_BUSY;

  delete e;
  return kMSLLM_SUCCESS;
}

MSLLMStatus MSLLMBuildModel(MSLLMModelHandle llm_model, const char *model_path) {
  if (llm_model == nullptr || model_path == nullptr) return kMSLLM_ERROR_INVALID_ARGS;
  auto *e = reinterpret_cast<InternalEngine *>(llm_model);

  std::lock_guard<std::mutex> lock(e->engine_mutex);
  if (e->state.load() == EngineState::Value::kGenerating) return kMSLLM_ERROR_BUSY;
  if (e->state.load() == EngineState::Value::kReady) return kMSLLM_ERROR_NOT_SUPPORTED;
  if (e->state.load() != EngineState::Value::kCreated) return kMSLLM_ERROR_INVALID_ARGS;

  const std::string path(model_path);
  if (path.empty()) return kMSLLM_ERROR_INVALID_ARGS;
  if (PathDoesNotExist(path)) return kMSLLM_ERROR_INVALID_ARGS;

  auto status = LoadEngineResources(e, path);
  if (status != kMSLLM_SUCCESS) return status;
  status = LoadEngineModel(e, path);
  if (status != kMSLLM_SUCCESS) return status;
  status = LoadEngineTokenizer(e, path);
  if (status != kMSLLM_SUCCESS) return status;
  e->sampler = std::make_unique<mslite_llm::Sampler>(DefaultGenConfig());

  e->state.store(EngineState::Value::kReady);
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

  auto normalized_config = config;
  if (normalized_config.repetition_penalty == 0.0f) {
    normalized_config.repetition_penalty = 1.0f;
  }

  std::lock_guard<std::mutex> lock(e->config_mutex);
  if (e->state.load() == EngineState::Value::kGenerating) return kMSLLM_ERROR_BUSY;

  e->gen_config = normalized_config;
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
                                   char *generated_prompt, int prompt_size) {
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
    if (e->state.load() == EngineState::Value::kGenerating) return kMSLLM_ERROR_BUSY;
  }

  if (!e->tokenizer) return kMSLLM_ERROR_INVALID_ARGS;

  // Template-less packages are rejected: the runtime has no builtin renderer
  // The template is pinned at export time.
  if (!e->tokenizer->HasChatTemplate()) {
    return kMSLLM_ERROR_MODEL_LOAD;
  }

  // Render without appending the generation prompt. If the trailing
  // assistant-start marker is later needed, that will be introduced as a
  // separate configuration; the current behavior is add_generation_prompt=false.
  std::string rendered = e->tokenizer->ApplyChatTemplate(msgs, false);
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
  std::vector<int32_t> input_ids;
  MSLLMGenerationConfig cfg = {};
  auto status = PrepareGeneration(e, prompt, &input_ids, &cfg);
  if (status != kMSLLM_SUCCESS) return status;
  std::string output;
  status = GenerateText(e, input_ids, cfg.max_new_tokens, &output);
  FinishGeneration(e);
  if (status != kMSLLM_SUCCESS) {
    return status;
  }
  int needed = static_cast<int>(output.size()) + 1;
  if (needed > text_size) return kMSLLM_ERROR_BUFFER_TOO_SMALL;
  std::memcpy(generated_text, output.c_str(), static_cast<size_t>(needed));
  return kMSLLM_SUCCESS;
}

MSLLMStatus MSLLMStreamGenerate(MSLLMModelHandle llm_model, const char *prompt, MSLLMStreamCallback callback,
                                void *user_data) {
  if (llm_model == nullptr || prompt == nullptr || callback == nullptr) {
    return kMSLLM_ERROR_INVALID_ARGS;
  }
  auto *e = reinterpret_cast<InternalEngine *>(llm_model);
  std::vector<int32_t> input_ids;
  MSLLMGenerationConfig cfg = {};
  auto status = PrepareGeneration(e, prompt, &input_ids, &cfg);
  if (status != kMSLLM_SUCCESS) return status;
  status = GenerateStream(e, input_ids, cfg.max_new_tokens, callback, user_data);
  FinishGeneration(e);
  return status;
}

MSLLMStatus MSLLMAbort(MSLLMModelHandle llm_model) {
  if (llm_model == nullptr) return kMSLLM_ERROR_INVALID_ARGS;
  auto *e = reinterpret_cast<InternalEngine *>(llm_model);

  // Only affect an in-flight streaming generation; otherwise no-op (#13).
  if (e->state.load() == EngineState::Value::kGenerating) {
    e->abort_flag.store(true);
  }
  return kMSLLM_SUCCESS;
}

}  // extern "C"
