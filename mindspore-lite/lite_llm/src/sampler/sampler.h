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
#ifndef MSLLM_SAMPLER_H
#define MSLLM_SAMPLER_H

#include <vector>
#include <random>
#include <unordered_map>
#include "../llm_types_internal.h"

namespace mslite_llm {

class Sampler {
 public:
  explicit Sampler(const MSLlmGenerateConfig &config);
  ~Sampler();

  int32_t Sample(const std::vector<float> &logits);
  void Reset();

  void SetStrategy(MSLlmSamplerStrategy strategy);
  void SetTemperature(float temperature);
  void SetTopK(int32_t top_k);
  void SetTopP(float top_p);
  void SetRepetitionPenalty(float penalty);
  void SetPresencePenalty(float penalty);
  void SetFrequencyPenalty(float penalty);
  void SetSeed(int32_t seed);

  void ApplyConfigOverrides(const MSLlmGenerateConfig &config);
  void ApplyLogitBias(std::vector<float> &logits) const;

 private:
  MSLlmSamplerStrategy strategy_;
  float temperature_;
  int32_t top_k_;
  float top_p_;
  float repetition_penalty_;
  float presence_penalty_;
  float frequency_penalty_;
  int32_t seed_;
  std::mt19937 rng_;
  std::vector<int32_t> generated_tokens_;
  std::unordered_map<int32_t, float> logit_bias_;
};

}  // namespace mslite_llm

#endif
