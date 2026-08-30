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
// Sampling-chain unit tests (host-side, no backend/hardware required).
//
// Covers the chain that routes NPU logits to host sampling: the executor never
// picks a token; Pipeline::Sampler owns argmax / top-k / top-p / temperature /
// repetition-penalty / logit-bias semantics. These tests pin that semantics so
// the NPU host path and the CPU path behave identically.

#include <gtest/gtest.h>

#include <vector>

#include "sampler/sampler.h"

namespace {

mslite_llm::Sampler MakeSampler(float temperature = 0.0f, int32_t top_k = 0, float top_p = 0.0f,
                                float repetition_penalty = 1.0f, int32_t seed = 0) {
  MSLlmGenerateConfig cfg = {};
  cfg.max_new_tokens = 32;
  cfg.temperature = temperature;
  cfg.top_k = top_k;
  cfg.top_p = top_p;
  cfg.repetition_penalty = repetition_penalty;
  cfg.seed = seed;
  cfg.override_sampler = 1;
  return mslite_llm::Sampler(cfg);
}

TEST(Sampler, GreedyArgmax) {
  auto sampler = MakeSampler(/*temperature=*/0.0f);
  std::vector<float> logits = {0.1f, 0.9f, 0.2f, 0.3f, 0.05f};
  EXPECT_EQ(sampler.Sample(logits), 1) << "greedy picks the argmax token";
}

TEST(Sampler, GreedyTieBreak) {
  // All-equal logits must deterministically pick the first index (strict >).
  auto sampler = MakeSampler(0.0f);
  std::vector<float> logits(8, 0.5f);
  EXPECT_EQ(sampler.Sample(logits), 0) << "greedy tie-break picks index 0";
}

TEST(Sampler, TemperatureDeterminism) {
  // Two samplers with the same seed must reproduce the same token sequence.
  auto s1 = MakeSampler(0.8f, 0, 0.0f, 1.0f, 42);
  auto s2 = MakeSampler(0.8f, 0, 0.0f, 1.0f, 42);
  std::vector<float> logits = {0.1f, 0.7f, 0.2f};
  for (int i = 0; i < 50; ++i) {
    EXPECT_EQ(s1.Sample(logits), s2.Sample(logits)) << "same seed reproduces the same sequence at step " << i;
  }
}

TEST(Sampler, TopKOnlyTopCandidates) {
  // top_k=2 over {0,0,10,9,-1000} may only ever yield tokens 2 or 3.
  auto sampler = MakeSampler(1.0f, 2, 0.0f, 1.0f, 7);
  std::vector<float> logits = {0.0f, 0.0f, 10.0f, 9.0f, -1000.0f};
  for (int i = 0; i < 200; ++i) {
    const int t = sampler.Sample(logits);
    EXPECT_TRUE(t == 2 || t == 3) << "top-k sampled outside the k candidates: " << t;
  }
}

TEST(Sampler, TopPExcludesTail) {
  // top_p=0.9 over {0,0,10,5,0.001,0.001,0.001}: softmax keeps only token 2
  // (its mass alone clears 0.9); the near-zero tail must never be sampled.
  auto sampler = MakeSampler(1.0f, 0, 0.9f, 1.0f, 3);
  std::vector<float> logits = {0.0f, 0.0f, 10.0f, 5.0f, 0.001f, 0.001f, 0.001f};
  for (int i = 0; i < 500; ++i) {
    EXPECT_LT(sampler.Sample(logits), 4) << "top-p sampled the probability tail";
  }
}

TEST(Sampler, RepetitionPenalty) {
  // First sample is the max (1.0 → token 1); the penalty then halves token 1's
  // score (1.0/2 = 0.5 < 0.9), so the second sample must pick token 2.
  auto sampler = MakeSampler(0.0f, 0, 0.0f, 2.0f);
  std::vector<float> logits = {0.1f, 1.0f, 0.9f};
  EXPECT_EQ(sampler.Sample(logits), 1) << "first sample picks the max";
  EXPECT_EQ(sampler.Sample(logits), 2) << "repetition penalty demotes the repeated token";
}

TEST(Sampler, LogitBias) {
  MSLlmGenerateConfig cfg = {};
  cfg.max_new_tokens = 32;
  cfg.temperature = 0.0f;
  cfg.override_sampler = 1;
  int32_t bias_token = 3;
  float bias_val = 5.0f;
  cfg.logit_bias_tokens = &bias_token;
  cfg.logit_bias_values = &bias_val;
  cfg.num_logit_biases = 1;
  mslite_llm::Sampler sampler(cfg);
  // argmax without bias = 1; +5 on token 3 flips the winner.
  std::vector<float> logits = {0.1f, 0.9f, 0.2f, 0.3f};
  EXPECT_EQ(sampler.Sample(logits), 3) << "logit bias promotes the biased token";
}

}  // namespace
