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
 * @file test_llm_engine.cpp
 * @brief Seam ② — Public C API driven by an injected FakeBackend + a
 * runtime-generated model fixture (no committed test data files, no NPU).
 *
 * The FakeBackend substitutes the NNRT/NPU hardware boundary so the C API
 * state machine (BuildModel → Generate/StreamGenerate → Destroy) runs on an
 * x86 host. All fixture bytes (manifest.json, vocab.bin, dummy assets) are
 * written into a tempdir at test setup by llm_test_fixture.h.
 */

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "backend/common/backend_factory.h"
#include "llm/llm.h"
#include "llm/llm_types.h"
#include "tokenizer/tokenizer.h"

#include "ut/fake_backend.h"
#include "ut/llm_test_fixture.h"

namespace {

/// Build a model with an injectable FakeBackend; returns the raw backend
/// pointer so tests can script it.
struct TestModel {
  mslite_llm_test::FakeBackend *backend = nullptr;
  MSLLMModelHandle handle = nullptr;

  ~TestModel() {
    if (handle != nullptr) MSLLMDestroyModel(handle);
  }
};

TestModel BuildTestModel() {
  TestModel tm;
  mslite_llm::SetBackendFactory([&tm] {
    auto *b = new mslite_llm_test::FakeBackend();
    tm.backend = b;
    return std::unique_ptr<mslite_llm::Backend>(b);
  });
  auto fixture = mslite_llm_test::WriteMinimalModelDir();
  tm.handle = MSLLMCreateModel();
  MSLLMBuildModel(tm.handle, fixture.dir.c_str());
  mslite_llm::SetBackendFactory(nullptr);
  // Note: fixture's tempdir is removed when it goes out of scope; the model
  // has already loaded its resources into memory by now.
  return tm;
}

// ─── BuildModel with injected backend ────────────────────────────────────────

TEST(BuildModel, SucceedsWithInjectedFakeBackend) {
  mslite_llm::SetBackendFactory([] { return std::make_unique<mslite_llm_test::FakeBackend>(); });

  auto fixture = mslite_llm_test::WriteMinimalModelDir();
  ASSERT_FALSE(fixture.dir.empty());

  auto *h = MSLLMCreateModel();
  ASSERT_NE(h, nullptr);
  EXPECT_EQ(MSLLMBuildModel(h, fixture.dir.c_str()), kMSLLM_SUCCESS);
  MSLLMDestroyModel(h);

  mslite_llm::SetBackendFactory(nullptr);
}

TEST(BuildModel, SecondBuildReturnsNotSupported) {
  mslite_llm_test::FakeBackend *raw = nullptr;
  mslite_llm::SetBackendFactory([&raw] {
    auto *b = new mslite_llm_test::FakeBackend();
    raw = b;
    return std::unique_ptr<mslite_llm::Backend>(b);
  });

  auto fixture = mslite_llm_test::WriteMinimalModelDir();
  ASSERT_FALSE(fixture.dir.empty());

  auto *h = MSLLMCreateModel();
  ASSERT_NE(h, nullptr);
  ASSERT_EQ(MSLLMBuildModel(h, fixture.dir.c_str()), kMSLLM_SUCCESS);
  // Re-compile in READY state is rejected (#20).
  EXPECT_EQ(MSLLMBuildModel(h, fixture.dir.c_str()), kMSLLM_ERROR_NOT_SUPPORTED);
  MSLLMDestroyModel(h);

  mslite_llm::SetBackendFactory(nullptr);
}

TEST(BuildModel, NullHandleReturnsInvalidArgs) {
  EXPECT_EQ(MSLLMBuildModel(nullptr, "/tmp/x"), kMSLLM_ERROR_INVALID_ARGS);
}

TEST(BuildModel, NullPathReturnsInvalidArgs) {
  auto *h = MSLLMCreateModel();
  ASSERT_NE(h, nullptr);
  EXPECT_EQ(MSLLMBuildModel(h, nullptr), kMSLLM_ERROR_INVALID_ARGS);
  MSLLMDestroyModel(h);
}

TEST(BuildModel, EmptyPathReturnsInvalidArgs) {
  auto *h = MSLLMCreateModel();
  ASSERT_NE(h, nullptr);
  EXPECT_EQ(MSLLMBuildModel(h, ""), kMSLLM_ERROR_INVALID_ARGS);
  MSLLMDestroyModel(h);
}

TEST(BuildModel, NonexistentPathReturnsModelLoad) {
  auto *h = MSLLMCreateModel();
  ASSERT_NE(h, nullptr);
  EXPECT_EQ(MSLLMBuildModel(h, "/tmp/definitely_does_not_exist_xyz"), kMSLLM_ERROR_MODEL_LOAD);
  MSLLMDestroyModel(h);
}

// ─── Destroy during generation (#16) ────────────────────────────────────────

TEST(Lifecycle, DestroyDuringGeneratingReturnsBusy) {
  auto tm = BuildTestModel();
  ASSERT_NE(tm.handle, nullptr);
  ASSERT_NE(tm.backend, nullptr);

  std::vector<float> logits(8, 0.0f);
  logits[3] = 1.0f;  // argmax → token id 3
  tm.backend->QueueLogits(logits);
  tm.backend->BlockExecute();

  std::atomic<bool> started{false};
  std::thread t([&] {
    started.store(true);
    MSLLMStreamGenerate(tm.handle, "a", [](const char *, MSLLMFinishReason, void *) {}, nullptr);
  });

  while (!started.load()) std::this_thread::yield();
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Generation is in-flight (blocked in Execute): Destroy must refuse.
  EXPECT_EQ(MSLLMDestroyModel(tm.handle), kMSLLM_ERROR_BUSY);

  tm.backend->UnblockExecute();
  t.join();

  // After the generation returns, Destroy succeeds.
  EXPECT_EQ(MSLLMDestroyModel(tm.handle), kMSLLM_SUCCESS);
  tm.handle = nullptr;
}

// ─── Abort semantics (#13/#15) ─────────────────────────────────────────────

TEST(Abort, IdleAbortReturnsSuccess) {
  auto tm = BuildTestModel();
  ASSERT_NE(tm.handle, nullptr);
  // No generation in flight: Abort is a no-op but still succeeds.
  EXPECT_EQ(MSLLMAbort(tm.handle), kMSLLM_SUCCESS);
}

TEST(Abort, NullHandleReturnsInvalidArgs) { EXPECT_EQ(MSLLMAbort(nullptr), kMSLLM_ERROR_INVALID_ARGS); }

// ─── ApplyChatTemplate message validation (#10) ────────────────────────────

TEST(ApplyChatTemplate, NullContentReturnsInvalidArgs) {
  auto tm = BuildTestModel();
  ASSERT_NE(tm.handle, nullptr);

  MSLLMChatMessage msgs[] = {{MSLLM_ROLE_USER, nullptr}};
  char buf[256];
  EXPECT_EQ(MSLLMApplyChatTemplate(tm.handle, msgs, 1, 0, buf, sizeof(buf)), kMSLLM_ERROR_INVALID_ARGS);
}

TEST(ApplyChatTemplate, EmptyContentAccepted) {
  auto tm = BuildTestModel();
  ASSERT_NE(tm.handle, nullptr);

  MSLLMChatMessage msgs[] = {{MSLLM_ROLE_USER, ""}};
  char buf[256];
  // Empty content is legal (#10); the fixture package carries a minimal chat
  // template, so the render succeeds and emits "role:content\n".
  EXPECT_EQ(MSLLMApplyChatTemplate(tm.handle, msgs, 1, 0, buf, sizeof(buf)), kMSLLM_SUCCESS);
  EXPECT_STREQ(buf, "user:\n");
}

TEST(ApplyChatTemplate, SmallBufferReturnsBufferTooSmall) {
  auto tm = BuildTestModel();
  ASSERT_NE(tm.handle, nullptr);

  MSLLMChatMessage msgs[] = {{MSLLM_ROLE_USER, "hello world"}};
  char tiny[4] = {};  // rendered "user:hello world\n" (17 bytes) does not fit
  EXPECT_EQ(MSLLMApplyChatTemplate(tm.handle, msgs, 1, 0, tiny, sizeof(tiny)), kMSLLM_ERROR_BUFFER_TOO_SMALL);
}

TEST(ApplyChatTemplate, LastMessageAssistantAccepted) {
  auto tm = BuildTestModel();
  ASSERT_NE(tm.handle, nullptr);

  // #9: a conversation whose last message is an assistant turn is legal; the
  // template renders it verbatim (model decides how to continue).
  MSLLMChatMessage msgs[] = {{MSLLM_ROLE_USER, "hi"}, {MSLLM_ROLE_ASSISTANT, "hello!"}};
  char buf[256];
  EXPECT_EQ(MSLLMApplyChatTemplate(tm.handle, msgs, 2, 0, buf, sizeof(buf)), kMSLLM_SUCCESS);
  EXPECT_STREQ(buf, "user:hiassistant:hello!\n");
}

// ─── Generation semantics (#3/#14) ────────────────────────────────────────

struct StreamResult {
  std::vector<std::string> tokens;
  MSLLMFinishReason reason = kMSLLM_RUNNING;
};

void CollectTokens(const char *token, MSLLMFinishReason reason, void *data) {
  auto *r = static_cast<StreamResult *>(data);
  if (token != nullptr) r->tokens.emplace_back(token);
  if (reason != kMSLLM_RUNNING) r->reason = reason;
}

// Scripted logits of vocab_size 8 with argmax at `token_id`.
std::vector<float> LogitsFor(int token_id) {
  std::vector<float> logits(8, 0.0f);
  logits[token_id] = 1.0f;
  return logits;
}

void SetConfig(MSLLMModelHandle h, int32_t max_new_tokens) {
  MSLLMGenerationConfig cfg = {};
  cfg.max_new_tokens = max_new_tokens;
  cfg.do_sample = false;
  cfg.temperature = 1.0f;
  cfg.top_k = 1;
  cfg.top_p = 1.0f;
  cfg.repetition_penalty = 1.0f;
  ASSERT_EQ(MSLLMSetGenerationConfig(h, cfg), kMSLLM_SUCCESS);
}

TEST(Generate, StopsAtEos) {
  auto tm = BuildTestModel();
  ASSERT_NE(tm.handle, nullptr);
  SetConfig(tm.handle, 0);

  tm.backend->QueueLogits(LogitsFor(3));  // prefill → 'a'
  tm.backend->QueueLogits(LogitsFor(2));  // decode → EOS

  char buf[256] = {};
  EXPECT_EQ(MSLLMGenerate(tm.handle, "a", buf, sizeof(buf)), kMSLLM_SUCCESS);
  EXPECT_STREQ(buf, "a");
}

TEST(Generate, PromptOverflowReturnsContextOverflow) {
  auto tm = BuildTestModel();
  ASSERT_NE(tm.handle, nullptr);
  SetConfig(tm.handle, 0);

  // Fixture max_position_embeddings = 64; 64 'a' tokens == the window.
  std::string prompt(64, 'a');
  char buf[256] = {};
  EXPECT_EQ(MSLLMGenerate(tm.handle, prompt.c_str(), buf, sizeof(buf)), kMSLLM_ERROR_CONTEXT_OVERFLOW);
}

TEST(Generate, BufferTooSmallReturnsError) {
  auto tm = BuildTestModel();
  ASSERT_NE(tm.handle, nullptr);
  SetConfig(tm.handle, 5);  // 5 'a' tokens → "aaaaa\0" = 6 bytes

  tm.backend->QueueLogits(LogitsFor(3));  // never EOS

  char buf[3] = {};
  EXPECT_EQ(MSLLMGenerate(tm.handle, "a", buf, sizeof(buf)), kMSLLM_ERROR_BUFFER_TOO_SMALL);
  // Nothing written on overflow (#1): buffer unchanged.
  EXPECT_EQ(buf[0], '\0');
}

TEST(Generate, DoSampleFalseIgnoresSamplingParams) {
  auto tm = BuildTestModel();
  ASSERT_NE(tm.handle, nullptr);

  // do_sample=false with non-default sampling params: greedy (argmax) still
  // wins (#5), sampling params are silently ignored.
  MSLLMGenerationConfig cfg = {};
  cfg.max_new_tokens = 3;
  cfg.do_sample = false;
  cfg.temperature = 1.5f;
  cfg.top_k = 4;
  cfg.top_p = 0.5f;
  cfg.repetition_penalty = 1.0f;
  ASSERT_EQ(MSLLMSetGenerationConfig(tm.handle, cfg), kMSLLM_SUCCESS);

  tm.backend->QueueLogits(LogitsFor(3));  // argmax = 3

  char buf[256] = {};
  EXPECT_EQ(MSLLMGenerate(tm.handle, "a", buf, sizeof(buf)), kMSLLM_SUCCESS);
  EXPECT_STREQ(buf, "aaa");  // deterministic argmax, not sampled
}

TEST(StreamGenerate, EosFinishReason) {
  auto tm = BuildTestModel();
  ASSERT_NE(tm.handle, nullptr);
  SetConfig(tm.handle, 0);

  tm.backend->QueueLogits(LogitsFor(3));  // prefill → 'a'
  tm.backend->QueueLogits(LogitsFor(2));  // decode → EOS

  StreamResult r;
  EXPECT_EQ(MSLLMStreamGenerate(tm.handle, "a", CollectTokens, &r), kMSLLM_SUCCESS);
  ASSERT_EQ(r.tokens.size(), 1u);
  EXPECT_EQ(r.tokens[0], "a");
  EXPECT_EQ(r.reason, kMSLLM_FINISHED_BY_EOS);
}

TEST(StreamGenerate, MaxOutputLengthFinishReason) {
  auto tm = BuildTestModel();
  ASSERT_NE(tm.handle, nullptr);
  SetConfig(tm.handle, 2);

  tm.backend->QueueLogits(LogitsFor(3));  // never EOS

  StreamResult r;
  EXPECT_EQ(MSLLMStreamGenerate(tm.handle, "a", CollectTokens, &r), kMSLLM_SUCCESS);
  EXPECT_EQ(r.tokens.size(), 2u);
  EXPECT_EQ(r.reason, kMSLLM_FINISHED_BY_MAX_OUTPUT_LENGTH);
}

TEST(StreamGenerate, MaxContextLengthFinishReason) {
  auto tm = BuildTestModel();
  ASSERT_NE(tm.handle, nullptr);
  SetConfig(tm.handle, 0);  // no output cap

  tm.backend->QueueLogits(LogitsFor(3));  // never EOS

  StreamResult r;
  EXPECT_EQ(MSLLMStreamGenerate(tm.handle, "a", CollectTokens, &r), kMSLLM_SUCCESS);
  // prompt = BOS + 'a' = 2 tokens, window = 64 → 62 generated tokens.
  EXPECT_EQ(r.tokens.size(), 62u);
  EXPECT_EQ(r.reason, kMSLLM_FINISHED_BY_MAX_CONTEXT_LENGTH);
}

TEST(StreamGenerate, AbortDuringGenerationYieldsStoppedByUser) {
  auto tm = BuildTestModel();
  ASSERT_NE(tm.handle, nullptr);
  SetConfig(tm.handle, 1000);             // long generation, never EOS
  tm.backend->QueueLogits(LogitsFor(3));  // argmax 'a' repeatedly

  // Hold the first backend call so the generation is deterministically
  // in-flight when the aborter thread fires (#15).
  tm.backend->BlockExecute();

  StreamResult r;
  std::thread aborter([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_EQ(MSLLMAbort(tm.handle), kMSLLM_SUCCESS);  // external thread, allowed
    tm.backend->UnblockExecute();                      // let the loop observe the flag
  });
  EXPECT_EQ(MSLLMStreamGenerate(tm.handle, "a", CollectTokens, &r), kMSLLM_SUCCESS);
  aborter.join();
  EXPECT_EQ(r.reason, kMSLLM_STOPPED_BY_USER);  // #15: terminal reason reported
}

// ─── Callback reentry matrix (#19) ────────────────────────────────────────

enum class ReentryOp {
  kGenerate,
  kStreamGenerate,
  kSetConfig,
  kApplyChatTemplate,
  kAbort,
  kDestroy,
};

struct ReentryCtx {
  MSLLMModelHandle handle;
  ReentryOp op;
  MSLLMStatus result = kMSLLM_SUCCESS;
};

void ReentrantCallback(const char *token, MSLLMFinishReason reason, void *data) {
  auto *ctx = static_cast<ReentryCtx *>(data);
  if (token == nullptr || reason != kMSLLM_RUNNING) return;

  char buf[64];
  switch (ctx->op) {
    case ReentryOp::kGenerate:
      ctx->result = MSLLMGenerate(ctx->handle, "a", buf, sizeof(buf));
      break;
    case ReentryOp::kStreamGenerate: {
      StreamResult inner;
      ctx->result = MSLLMStreamGenerate(ctx->handle, "a", CollectTokens, &inner);
      break;
    }
    case ReentryOp::kSetConfig: {
      MSLLMGenerationConfig cfg = {};
      cfg.max_new_tokens = 1;
      ctx->result = MSLLMSetGenerationConfig(ctx->handle, cfg);
      break;
    }
    case ReentryOp::kApplyChatTemplate: {
      MSLLMChatMessage m[] = {{MSLLM_ROLE_USER, "x"}};
      ctx->result = MSLLMApplyChatTemplate(ctx->handle, m, 1, 0, buf, sizeof(buf));
      break;
    }
    case ReentryOp::kAbort:
      ctx->result = MSLLMAbort(ctx->handle);
      break;
    case ReentryOp::kDestroy:
      ctx->result = MSLLMDestroyModel(ctx->handle);
      break;
  }
}

MSLLMStatus RunReentrant(MSLLMModelHandle handle, mslite_llm_test::FakeBackend *backend, ReentryOp op) {
  SetConfig(handle, 4);                // a few tokens so the callback fires
  backend->QueueLogits(LogitsFor(3));  // never EOS

  ReentryCtx ctx{handle, op};
  MSLLMStreamGenerate(handle, "a", ReentrantCallback, &ctx);
  return ctx.result;
}

TEST(Reentry, GenerateReturnsBusy) {
  auto tm = BuildTestModel();
  EXPECT_EQ(RunReentrant(tm.handle, tm.backend, ReentryOp::kGenerate), kMSLLM_ERROR_BUSY);
}

TEST(Reentry, StreamGenerateReturnsBusy) {
  auto tm = BuildTestModel();
  EXPECT_EQ(RunReentrant(tm.handle, tm.backend, ReentryOp::kStreamGenerate), kMSLLM_ERROR_BUSY);
}

TEST(Reentry, SetGenerationConfigReturnsBusy) {
  auto tm = BuildTestModel();
  EXPECT_EQ(RunReentrant(tm.handle, tm.backend, ReentryOp::kSetConfig), kMSLLM_ERROR_BUSY);
}

TEST(Reentry, ApplyChatTemplateReturnsBusy) {
  auto tm = BuildTestModel();
  EXPECT_EQ(RunReentrant(tm.handle, tm.backend, ReentryOp::kApplyChatTemplate), kMSLLM_ERROR_BUSY);
}

TEST(Reentry, AbortReturnsSuccess) {
  auto tm = BuildTestModel();
  EXPECT_EQ(RunReentrant(tm.handle, tm.backend, ReentryOp::kAbort), kMSLLM_SUCCESS);
}

TEST(Reentry, DestroyReturnsBusy) {
  auto tm = BuildTestModel();
  EXPECT_EQ(RunReentrant(tm.handle, tm.backend, ReentryOp::kDestroy), kMSLLM_ERROR_BUSY);
}

// ─── Incremental decode (#17) ────────────────────────────────────────────

TEST(TokenizerIncremental, SplitsMultibyteUtf8) {
  auto vocab = mslite_llm_test::BuildMinimalVocabBin();
  auto tok = mslite_llm::CreateTokenizerFromBuffer(vocab.data(), vocab.size());
  ASSERT_NE(tok, nullptr);

  // Ground truth: ids 5/6/7 decode to the three bytes of '你'.
  EXPECT_EQ(tok->Decode({5, 6, 7}), "你");

  // Incremental: incomplete UTF-8 bytes are buffered until the char completes.
  EXPECT_EQ(tok->DecodeIncremental(5), "");
  EXPECT_EQ(tok->DecodeIncremental(6), "");
  EXPECT_EQ(tok->DecodeIncremental(7), "你");
  EXPECT_EQ(tok->FlushDecode(), "");
}

TEST(StreamGenerate, EmitsCompleteUtf8Char) {
  auto tm = BuildTestModel();
  ASSERT_NE(tm.handle, nullptr);
  SetConfig(tm.handle, 0);

  // Script: prefill → token 5, then 6, 7, then EOS.
  tm.backend->QueueLogits(LogitsFor(5));
  tm.backend->QueueLogits(LogitsFor(6));
  tm.backend->QueueLogits(LogitsFor(7));
  tm.backend->QueueLogits(LogitsFor(2));

  StreamResult r;
  EXPECT_EQ(MSLLMStreamGenerate(tm.handle, "a", CollectTokens, &r), kMSLLM_SUCCESS);
  // The three byte-level tokens must be delivered as one complete char, not
  // as mojibake fragments ("ä½ł" with the old per-token Decode).
  ASSERT_EQ(r.tokens.size(), 3u);
  EXPECT_EQ(r.tokens[0], "");
  EXPECT_EQ(r.tokens[1], "");
  EXPECT_EQ(r.tokens[2], "你");
  EXPECT_EQ(r.reason, kMSLLM_FINISHED_BY_EOS);
}

}  // namespace
