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
#include <sys/stat.h>

#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "llm/llm.h"
#include "manifest/model_manifest.h"
#include "manifest/msl_package_reader.h"
namespace {

void DebugLog(const char *tag) { std::cout << "[m] " << tag << '\n' << std::flush; }

bool ParsePositiveInt(const char *value, int32_t *out) {
  if (value == nullptr || out == nullptr) {
    return false;
  }
  errno = 0;
  char *end = nullptr;
  const int64_t parsed = std::strtoll(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed <= 0 || parsed > std::numeric_limits<int32_t>::max()) {
    return false;
  }
  *out = static_cast<int32_t>(parsed);
  return true;
}

void PrintUsage(const char *program) { std::cerr << "Usage: " << program << " MODEL_PACKAGE PROMPT [MAX_TOKENS=64]\n"; }

const char *StatusName(MSLLMStatus status) {
  switch (status) {
    case kMSLLM_SUCCESS:
      return "SUCCESS";
    case kMSLLM_ERROR_INVALID_ARGS:
      return "INVALID_ARGS";
    case kMSLLM_ERROR_MODEL_LOAD:
      return "MODEL_LOAD";
    case kMSLLM_ERROR_INFERENCE:
      return "INFERENCE";
    case kMSLLM_ERROR_OOM:
      return "OOM";
    case kMSLLM_ERROR_IO:
      return "IO";
    case kMSLLM_ERROR_BUSY:
      return "BUSY";
    case kMSLLM_ERROR_CONTEXT_OVERFLOW:
      return "CONTEXT_OVERFLOW";
    case kMSLLM_ERROR_BUFFER_TOO_SMALL:
      return "BUFFER_TOO_SMALL";
    default:
      return "UNKNOWN";
  }
}

const char *ReasonName(MSLLMFinishReason reason) {
  switch (reason) {
    case kMSLLM_RUNNING:
      return "RUNNING";
    case kMSLLM_FINISHED_BY_EOS:
      return "EOS";
    case kMSLLM_FINISHED_BY_MAX_CONTEXT_LENGTH:
      return "MAX_CONTEXT_LENGTH";
    case kMSLLM_FINISHED_BY_MAX_OUTPUT_LENGTH:
      return "MAX_OUTPUT_LENGTH";
    case kMSLLM_STOPPED_BY_USER:
      return "STOPPED_BY_USER";
    case kMSLLM_FINISHED_BY_ERROR:
      return "ERROR";
    default:
      return "UNKNOWN";
  }
}

std::string FormatSize(uint64_t bytes) {
  char buf[64];
  if (bytes >= 1024ULL * 1024 * 1024) {
    std::snprintf(buf, sizeof(buf), "%.1f GiB", static_cast<double>(bytes) / (1024.0 * 1024 * 1024));
  } else if (bytes >= 1024ULL * 1024) {
    std::snprintf(buf, sizeof(buf), "%.1f MiB", static_cast<double>(bytes) / (1024.0 * 1024));
  } else if (bytes >= 1024ULL) {
    std::snprintf(buf, sizeof(buf), "%.1f KiB", static_cast<double>(bytes) / 1024.0);
  } else {
    std::snprintf(buf, sizeof(buf), "%" PRIu64 " B", static_cast<uint64_t>(bytes));
  }
  return buf;
}

// ── Model package info (read from the .msl manifest; does not load the graph) ──
bool PrintModelInfo(const char *path) {
  struct stat st;
  if (stat(path, &st) != 0) {
    std::cerr << "[info] stat failed: " << path << '\n';
    return false;
  }
  std::cout << "[info] model file: " << path << " (" << FormatSize(static_cast<uint64_t>(st.st_size)) << ")\n"
            << std::flush;

  mslite_llm::MslPackageReader reader;
  std::string err;
  if (!reader.Open(path, &err)) {
    std::cerr << "[info] open .msl failed: " << err << '\n';
    return false;
  }
  std::cout << "[info] msl entries (" << reader.entry_count() << "):\n";
  for (const auto &e : reader.entries()) {
    std::cout << "[info]   " << e.name << "  " << FormatSize(e.size) << (e.access == 0 ? "  mmap" : "  read") << '\n';
  }

  mslite_llm::ModelManifest m;
  if (mslite_llm::BuildModelManifestFromKv(reader, &m, &err) != MSLLM_SUCCESS) {
    std::cerr << "[info] parse manifest failed: " << err << '\n';
    return false;
  }
  const auto &a = m.architecture;
  std::cout << "[info] manifest: model=" << m.model_name << " layers=" << a.num_layers << " hidden=" << a.hidden_size
            << " heads=" << a.num_heads << " kv=" << a.num_kv_heads << " head_dim=" << a.head_dim
            << " vocab=" << a.vocab_size << " max_pos=" << a.max_position_embeddings << " rope=" << a.rope_theta
            << " eps=" << a.norm_eps << " tie=" << a.tie_word_embeddings << '\n';
  if (m.npu.present) {
    std::cout << "[info] npu: max_length=" << m.npu.max_length << " chunk_size=" << m.npu.chunk_size
              << " embedding_quant=" << (m.npu.embedding_quant ? "on" : "off")
              << " scale_gp_size=" << m.npu.scale_gp_size << '\n';
  }
  if (m.generation.present) {
    std::cout << "[info] generation policy: stop=[";
    for (size_t i = 0; i < m.generation.stop_token_ids.size(); ++i) {
      if (i != 0) std::cout << ",";
      std::cout << m.generation.stop_token_ids[i];
    }
    std::cout << "] suppress=[";
    for (size_t i = 0; i < m.generation.suppress_token_ids.size(); ++i) {
      if (i != 0) std::cout << ",";
      std::cout << m.generation.suppress_token_ids[i];
    }
    std::cout << "]\n" << std::flush;
  }
  return true;
}

void PrintSamplingConfig(const MSLLMGenerationConfig &cfg) {
  std::cout << "[info] sampling: max_new_tokens=" << cfg.max_new_tokens
            << " do_sample=" << (cfg.do_sample ? "true" : "false") << " temperature=" << cfg.temperature
            << " top_k=" << cfg.top_k << " top_p=" << cfg.top_p << " repetition_penalty=" << cfg.repetition_penalty
            << "\n"
            << std::flush;
}

// ── Streaming pipeline ─────────────────────────────────────────────────────
// MSLLMStreamGenerate runs on a worker thread; its per-token callback only
// queues the token and signals a condition variable.  The main thread waits on
// the condvar and prints tokens as they arrive (plus per-token timing stats),
// so console I/O never blocks the inference loop.
struct StreamSink {
  std::mutex mtx;
  std::condition_variable cv;
  std::deque<std::string> tokens;  // pending tokens for the main thread
  bool done = false;               // generator finished (null token seen)
  MSLLMFinishReason final_reason = kMSLLM_FINISHED_BY_EOS;
  MSLLMStatus generate_status = kMSLLM_SUCCESS;

  // timing stats, computed on the main thread as tokens are printed
  std::chrono::steady_clock::time_point t_start;
  std::chrono::steady_clock::time_point t_last;
  int prefill_tokens = 0;  // 1st generated token comes from the prefill step
  int decode_tokens = 0;   // subsequent tokens come from decode steps
  double prefill_ms = 0.0;
  double decode_ms = 0.0;
  std::string output;
  bool token_written = false;
};

// Worker-thread callback: enqueue the token (or the end marker) and wake the
// main thread.  Never performs I/O itself.
void OnStreamToken(const char *token, MSLLMFinishReason reason, void *user_data) {
  auto *s = static_cast<StreamSink *>(user_data);
  {
    std::lock_guard<std::mutex> lock(s->mtx);
    if (token != nullptr) {
      s->tokens.emplace_back(token);
    } else {
      s->final_reason = reason;
      s->done = true;
    }
  }
  s->cv.notify_one();
}

}  // namespace

int main(int argc, char **argv) {
  DebugLog("start");
  if (argc < 3) {
    PrintUsage(argv[0]);
    return 2;
  }

  int32_t max_tokens = 64;
  if (argc > 3 && !ParsePositiveInt(argv[3], &max_tokens)) {
    PrintUsage(argv[0]);
    return 2;
  }

  const char *model_path = argv[1];
  const char *prompt = argv[2];

  if (!PrintModelInfo(model_path)) {
    return 1;
  }
  DebugLog("info-ok");

  MSLLMModelHandle model = MSLLMCreateModel();
  if (model == nullptr) {
    std::cerr << "[error] MSLLMCreateModel failed\n";
    return 1;
  }

  MSLLMStatus status = MSLLMBuildModel(model, model_path);
  if (status != kMSLLM_SUCCESS) {
    std::cerr << "[error] MSLLMBuildModel failed: " << StatusName(status) << '\n';
    MSLLMDestroyModel(model);
    return 1;
  }

  MSLLMGenerationConfig config = {};
  config.max_new_tokens = max_tokens;
  config.do_sample = false;  // greedy (argmax)
  config.temperature = 1.0f;
  config.top_k = 1;
  config.top_p = 1.0f;
  config.repetition_penalty = 1.0f;
  PrintSamplingConfig(config);
  status = MSLLMSetGenerationConfig(model, config);
  if (status != kMSLLM_SUCCESS) {
    std::cerr << "[error] MSLLMSetGenerationConfig failed: " << StatusName(status) << '\n';
    MSLLMDestroyModel(model);
    return 1;
  }
  DebugLog("config-ok");

  StreamSink sink;
  sink.t_start = std::chrono::steady_clock::now();
  sink.t_last = sink.t_start;
  DebugLog("stream-begin");

  // Worker thread runs the blocking stream generator; its callback queues
  // tokens and signals the condvar.  The main thread below prints tokens as
  // they arrive.
  std::thread worker([&] {
    sink.generate_status = MSLLMStreamGenerate(model, prompt, OnStreamToken, &sink);
    {  // generator returned: no more tokens will be queued
      std::lock_guard<std::mutex> lock(sink.mtx);
      sink.done = true;
    }
    sink.cv.notify_all();
  });

  for (;;) {
    std::string token;
    {
      std::unique_lock<std::mutex> lock(sink.mtx);
      sink.cv.wait(lock, [&] { return !sink.tokens.empty() || sink.done; });
      if (!sink.tokens.empty()) {
        token = std::move(sink.tokens.front());
        sink.tokens.pop_front();
      } else if (sink.done) {
        break;
      }
    }
    const auto now = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(now - sink.t_last).count();
    if (sink.prefill_tokens == 0 && sink.decode_tokens == 0) {
      sink.prefill_ms = ms;  // time-to-first-token == prefill + first sample
      ++sink.prefill_tokens;
    } else {
      sink.decode_ms += ms;  // inter-token gap == one decode step
      ++sink.decode_tokens;
    }
    sink.t_last = now;
    std::cout << token << std::flush;
    sink.output += token;
    sink.token_written = true;
  }
  worker.join();
  DebugLog("stream-done");

  status = sink.generate_status;
  if (status != kMSLLM_SUCCESS) {
    std::cerr << "[error] MSLLMStreamGenerate failed: " << StatusName(status) << '\n';
    MSLLMDestroyModel(model);
    return 1;
  }
  std::cout << "\n[finish reason] " << ReasonName(sink.final_reason) << '\n' << std::flush;
  if (!sink.token_written) {
    std::cout << '\n';  // immediate EOS: nothing generated
  }

  std::cout << "[stats] prefill: " << sink.prefill_tokens << " token, " << sink.prefill_ms << " ms\n";
  if (sink.decode_tokens > 0) {
    std::cout << "[stats] decode: " << sink.decode_tokens << " tokens, total " << sink.decode_ms << " ms, avg "
              << (sink.decode_ms / sink.decode_tokens) << " ms/token\n";
  }

  MSLLMStatus destroy_status = MSLLMDestroyModel(model);
  return destroy_status == kMSLLM_SUCCESS ? 0 : 1;
}
