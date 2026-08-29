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
 * @file test_llm_lifecycle.cpp
 * @brief Seam ① — Test MSLLMCreateModel / MSLLMDestroyModel / Config lifecycle.
 *
 * Pure memory operations: no .msl model, no backend, no tokenizer required.
 * These tests validate the state machine boundary for the new LLM-API.
 */

#include <gtest/gtest.h>
#include "llm/llm.h"
#include "llm/llm_types.h"

// ─── Create ──────────────────────────────────────────────────────────────────

TEST(Lifecycle, CreateReturnsNonNullHandle) {
  auto *h = MSLLMCreateModel();
  ASSERT_NE(h, nullptr);
  MSLLMDestroyModel(h);
}

TEST(Lifecycle, CreateReturnsDistinctHandles) {
  auto *h1 = MSLLMCreateModel();
  auto *h2 = MSLLMCreateModel();
  ASSERT_NE(h1, nullptr);
  ASSERT_NE(h2, nullptr);
  ASSERT_NE(h1, h2);
  MSLLMDestroyModel(h2);
  MSLLMDestroyModel(h1);
}

// ─── Destroy ─────────────────────────────────────────────────────────────────

TEST(Lifecycle, DestroyNullReturnsError) {
  auto status = MSLLMDestroyModel(nullptr);
  EXPECT_EQ(status, kMSLLM_ERROR_INVALID_ARGS);
}

// ─── Config — set / get round-trip ───────────────────────────────────────────

TEST(Config, SetGetRoundTrip) {
  auto *h = MSLLMCreateModel();
  ASSERT_NE(h, nullptr);

  MSLLMGenerationConfig in = {};
  in.max_new_tokens = 100;
  in.do_sample = true;
  in.temperature = 0.8f;
  in.top_k = 50;
  in.top_p = 0.95f;
  in.repetition_penalty = 1.1f;

  EXPECT_EQ(MSLLMSetGenerationConfig(h, in), kMSLLM_SUCCESS);

  MSLLMGenerationConfig out = {};
  EXPECT_EQ(MSLLMGetGenerationConfig(h, &out), kMSLLM_SUCCESS);
  EXPECT_EQ(out.max_new_tokens, 100);
  EXPECT_TRUE(out.do_sample);
  EXPECT_FLOAT_EQ(out.temperature, 0.8f);
  EXPECT_EQ(out.top_k, 50);
  EXPECT_FLOAT_EQ(out.top_p, 0.95f);
  EXPECT_FLOAT_EQ(out.repetition_penalty, 1.1f);

  MSLLMDestroyModel(h);
}

TEST(Config, SetConfigNullHandle) {
  MSLLMGenerationConfig cfg = {};
  EXPECT_EQ(MSLLMSetGenerationConfig(nullptr, cfg), kMSLLM_ERROR_INVALID_ARGS);
}

TEST(Config, GetConfigNullHandleReturnsInvalidArgs) {
  MSLLMGenerationConfig cfg = {};
  EXPECT_EQ(MSLLMGetGenerationConfig(nullptr, &cfg), kMSLLM_ERROR_INVALID_ARGS);
}

TEST(Config, GetConfigNullOutParamReturnsInvalidArgs) {
  auto *h = MSLLMCreateModel();
  ASSERT_NE(h, nullptr);
  EXPECT_EQ(MSLLMGetGenerationConfig(h, nullptr), kMSLLM_ERROR_INVALID_ARGS);
  MSLLMDestroyModel(h);
}

// ─── Config — defaults ──────────────────────────────────────────────────────

TEST(Config, DefaultConfigAfterCreate) {
  auto *h = MSLLMCreateModel();
  ASSERT_NE(h, nullptr);

  MSLLMGenerationConfig cfg = {};
  EXPECT_EQ(MSLLMGetGenerationConfig(h, &cfg), kMSLLM_SUCCESS);
  // After Create, expect sensible defaults
  EXPECT_EQ(cfg.max_new_tokens, 256);
  EXPECT_FALSE(cfg.do_sample);
  EXPECT_EQ(cfg.top_k, 1);
  EXPECT_FLOAT_EQ(cfg.top_p, 1.0f);
  EXPECT_FLOAT_EQ(cfg.repetition_penalty, 1.0f);

  MSLLMDestroyModel(h);
}

TEST(Config, OverwriteThenReadbackPreservesLastValue) {
  auto *h = MSLLMCreateModel();
  ASSERT_NE(h, nullptr);

  MSLLMGenerationConfig c1 = {};
  c1.max_new_tokens = 50;
  c1.do_sample = true;
  MSLLMSetGenerationConfig(h, c1);

  MSLLMGenerationConfig c2 = {};
  c2.max_new_tokens = 200;
  c2.do_sample = false;
  MSLLMSetGenerationConfig(h, c2);

  MSLLMGenerationConfig out = {};
  EXPECT_EQ(MSLLMGetGenerationConfig(h, &out), kMSLLM_SUCCESS);
  EXPECT_EQ(out.max_new_tokens, 200);
  EXPECT_FALSE(out.do_sample);

  MSLLMDestroyModel(h);
}

// ─── Config — validation (#3/#4/#5/#6) ─────────────────────────────────────
// max_new_tokens: -1 and 0 both mean "no explicit output cap"; other negative
// values are rejected. Sampling fields are validated only when the strategy
// consumes them (do_sample=true → temperature/top_k/top_p whitelist).

TEST(Config, AcceptsMaxNewTokensUnlimited) {
  auto *h = MSLLMCreateModel();
  ASSERT_NE(h, nullptr);

  MSLLMGenerationConfig cfg = {};
  cfg.max_new_tokens = -1;  // #3: -1 = no explicit cap
  EXPECT_EQ(MSLLMSetGenerationConfig(h, cfg), kMSLLM_SUCCESS);
  cfg.max_new_tokens = 0;  // #3: 0 = no explicit cap
  EXPECT_EQ(MSLLMSetGenerationConfig(h, cfg), kMSLLM_SUCCESS);

  MSLLMDestroyModel(h);
}

TEST(Config, RejectsMaxNewTokensBelowMinusOne) {
  auto *h = MSLLMCreateModel();
  ASSERT_NE(h, nullptr);

  MSLLMGenerationConfig cfg = {};
  cfg.max_new_tokens = -2;
  EXPECT_EQ(MSLLMSetGenerationConfig(h, cfg), kMSLLM_ERROR_INVALID_ARGS);

  MSLLMDestroyModel(h);
}

TEST(Config, RejectsTemperatureBelowZeroWhenSampling) {
  auto *h = MSLLMCreateModel();
  ASSERT_NE(h, nullptr);
  MSLLMGenerationConfig cfg = {};
  cfg.do_sample = true;  // #5: sampling fields validated only when used
  cfg.temperature = -0.1f;
  EXPECT_EQ(MSLLMSetGenerationConfig(h, cfg), kMSLLM_ERROR_INVALID_ARGS);
  MSLLMDestroyModel(h);
}

TEST(Config, IgnoresSamplingFieldsWhenGreedy) {
  auto *h = MSLLMCreateModel();
  ASSERT_NE(h, nullptr);
  // #5 whitelist: do_sample=false (greedy) ignores temperature/top_k/top_p,
  // so out-of-range values are accepted (kept for a later do_sample=true).
  MSLLMGenerationConfig cfg = {};
  cfg.temperature = 2.1f;
  cfg.top_k = -5;
  cfg.top_p = 3.0f;
  EXPECT_EQ(MSLLMSetGenerationConfig(h, cfg), kMSLLM_SUCCESS);
  MSLLMDestroyModel(h);
}

TEST(Config, RejectsTemperatureAboveTwoWhenSampling) {
  auto *h = MSLLMCreateModel();
  ASSERT_NE(h, nullptr);
  MSLLMGenerationConfig cfg = {};
  cfg.do_sample = true;  // #5: sampling fields validated only when used
  cfg.temperature = 2.1f;
  EXPECT_EQ(MSLLMSetGenerationConfig(h, cfg), kMSLLM_ERROR_INVALID_ARGS);
  MSLLMDestroyModel(h);
}

TEST(Config, AcceptsTemperatureBoundaries) {
  auto *h = MSLLMCreateModel();
  ASSERT_NE(h, nullptr);

  MSLLMGenerationConfig c0 = {};
  c0.temperature = 0.0f;
  EXPECT_EQ(MSLLMSetGenerationConfig(h, c0), kMSLLM_SUCCESS);

  MSLLMGenerationConfig c2 = {};
  c2.temperature = 2.0f;
  EXPECT_EQ(MSLLMSetGenerationConfig(h, c2), kMSLLM_SUCCESS);

  MSLLMDestroyModel(h);
}

TEST(Config, RejectsNegativeTopKWhenSampling) {
  auto *h = MSLLMCreateModel();
  ASSERT_NE(h, nullptr);
  MSLLMGenerationConfig cfg = {};
  cfg.do_sample = true;  // #5: sampling fields validated only when used
  cfg.top_k = -1;
  EXPECT_EQ(MSLLMSetGenerationConfig(h, cfg), kMSLLM_ERROR_INVALID_ARGS);
  MSLLMDestroyModel(h);
}

TEST(Config, RejectsNegativeTopPWhenSampling) {
  auto *h = MSLLMCreateModel();
  ASSERT_NE(h, nullptr);
  MSLLMGenerationConfig cfg = {};
  cfg.do_sample = true;  // #5: sampling fields validated only when used
  cfg.top_p = -0.1f;
  EXPECT_EQ(MSLLMSetGenerationConfig(h, cfg), kMSLLM_ERROR_INVALID_ARGS);
  MSLLMDestroyModel(h);
}

TEST(Config, RejectsTopPAboveOneWhenSampling) {
  auto *h = MSLLMCreateModel();
  ASSERT_NE(h, nullptr);
  MSLLMGenerationConfig cfg = {};
  cfg.do_sample = true;  // #5: sampling fields validated only when used
  cfg.top_p = 1.1f;
  EXPECT_EQ(MSLLMSetGenerationConfig(h, cfg), kMSLLM_ERROR_INVALID_ARGS);
  MSLLMDestroyModel(h);
}

TEST(Config, AcceptsTopKTopPBoundaries) {
  auto *h = MSLLMCreateModel();
  ASSERT_NE(h, nullptr);

  MSLLMGenerationConfig ck = {};
  ck.top_k = 0;  // disabled
  EXPECT_EQ(MSLLMSetGenerationConfig(h, ck), kMSLLM_SUCCESS);

  MSLLMGenerationConfig cp0 = {};
  cp0.top_p = 0.0f;  // disabled
  EXPECT_EQ(MSLLMSetGenerationConfig(h, cp0), kMSLLM_SUCCESS);

  MSLLMGenerationConfig cp1 = {};
  cp1.top_p = 1.0f;  // disabled
  EXPECT_EQ(MSLLMSetGenerationConfig(h, cp1), kMSLLM_SUCCESS);

  MSLLMDestroyModel(h);
}

TEST(Config, RejectionPreservesPreviousConfig) {
  auto *h = MSLLMCreateModel();
  ASSERT_NE(h, nullptr);

  MSLLMGenerationConfig good = {};
  good.do_sample = true;
  good.max_new_tokens = 100;
  good.temperature = 0.8f;
  EXPECT_EQ(MSLLMSetGenerationConfig(h, good), kMSLLM_SUCCESS);

  MSLLMGenerationConfig bad = good;
  bad.temperature = 3.0f;  // out of range
  EXPECT_EQ(MSLLMSetGenerationConfig(h, bad), kMSLLM_ERROR_INVALID_ARGS);

  MSLLMGenerationConfig out = {};
  ASSERT_EQ(MSLLMGetGenerationConfig(h, &out), kMSLLM_SUCCESS);
  EXPECT_EQ(out.max_new_tokens, 100);
  EXPECT_FLOAT_EQ(out.temperature, 0.8f);

  MSLLMDestroyModel(h);
}

// ─── Generate rejected before BuildModel ────────────────────────────────────

TEST(Lifecycle, GenerateRejectedBeforeBuild) {
  auto *h = MSLLMCreateModel();
  ASSERT_NE(h, nullptr);

  char buf[256];
  EXPECT_EQ(MSLLMGenerate(h, "hello", buf, sizeof(buf)), kMSLLM_ERROR_INVALID_ARGS);

  MSLLMDestroyModel(h);
}

TEST(Lifecycle, StreamGenerateRejectedBeforeBuild) {
  auto *h = MSLLMCreateModel();
  ASSERT_NE(h, nullptr);

  EXPECT_EQ(MSLLMStreamGenerate(h, "hello", nullptr, nullptr), kMSLLM_ERROR_INVALID_ARGS);

  MSLLMDestroyModel(h);
}
