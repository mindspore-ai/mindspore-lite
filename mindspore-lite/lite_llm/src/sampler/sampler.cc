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
#include "sampler/sampler.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace mslite_llm {

namespace {

void Softmax(std::vector<float> &logits) {
  float max_val = *std::max_element(logits.begin(), logits.end());
  float sum = 0.0f;
  for (auto &v : logits) {
    v = std::exp(v - max_val);
    sum += v;
  }
  if (sum > 0.0f) {
    std::transform(logits.begin(), logits.end(), logits.begin(), [sum](float v) { return v / sum; });
  }
}

}  // namespace

Sampler::Sampler(const MSLlmGenerateConfig &config)
    : strategy_(MSLLM_SAMPLER_GREEDY),
      temperature_(1.0f),
      top_k_(0),
      top_p_(0.0f),
      repetition_penalty_(1.0f),
      presence_penalty_(0.0f),
      frequency_penalty_(0.0f),
      seed_(0),
      rng_(std::random_device()()) {
  // Constructor always applies config — override_sampler guard is only for
  // mid-generation ApplyConfigOverrides calls.
  if (config.temperature >= 0.0f) {
    temperature_ = config.temperature;
    if (config.temperature <= 0.0f) {
      strategy_ = MSLLM_SAMPLER_GREEDY;
    } else {
      strategy_ = MSLLM_SAMPLER_TEMPERATURE;
    }
  }
  if (config.top_k > 0) {
    top_k_ = config.top_k;
    strategy_ = MSLLM_SAMPLER_TOP_K;
  }
  if (config.top_p > 0.0f) {
    top_p_ = config.top_p;
    if (strategy_ != MSLLM_SAMPLER_TOP_K) {
      strategy_ = MSLLM_SAMPLER_TOP_P;
    }
  }
  if (config.repetition_penalty > 0.0f) {
    repetition_penalty_ = config.repetition_penalty;
  }
  if (std::fabs(config.presence_penalty) > 1e-6f) {
    presence_penalty_ = config.presence_penalty;
  }
  if (std::fabs(config.frequency_penalty) > 1e-6f) {
    frequency_penalty_ = config.frequency_penalty;
  }
  if (config.seed > 0) {
    SetSeed(config.seed);
  }
  if (config.logit_bias_tokens && config.logit_bias_values && config.num_logit_biases > 0) {
    logit_bias_.clear();
    for (size_t i = 0; i < config.num_logit_biases; ++i) {
      logit_bias_[config.logit_bias_tokens[i]] = config.logit_bias_values[i];
    }
  }
}

Sampler::~Sampler() = default;

void Sampler::Reset() { generated_tokens_.clear(); }

int32_t Sampler::Sample(const std::vector<float> &logits) {
  if (logits.empty()) {
    return 0;
  }

  std::vector<float> work(logits.begin(), logits.end());

  // Apply logit bias
  ApplyLogitBias(work);

  // Repetition penalty
  if (repetition_penalty_ > 0.0f && std::fabs(repetition_penalty_ - 1.0f) > 1e-6f && !generated_tokens_.empty()) {
    for (auto tid : generated_tokens_) {
      if (tid >= 0 && static_cast<size_t>(tid) < work.size()) {
        if (work[tid] > 0.0f) {
          work[tid] /= repetition_penalty_;
        } else {
          work[tid] *= repetition_penalty_;
        }
      }
    }
  }

  // Presence / frequency penalty
  if ((std::fabs(presence_penalty_) > 1e-6f || std::fabs(frequency_penalty_) > 1e-6f) && !generated_tokens_.empty()) {
    std::unordered_map<int32_t, int32_t> token_counts;
    for (auto tid : generated_tokens_) {
      token_counts[tid]++;
    }
    for (const auto &pair : token_counts) {
      int32_t tid = pair.first;
      if (tid >= 0 && static_cast<size_t>(tid) < work.size()) {
        work[tid] -= frequency_penalty_ * static_cast<float>(pair.second);
        work[tid] -= presence_penalty_;
      }
    }
  }

  // Temperature
  if (temperature_ > 0.0f && std::fabs(temperature_ - 1.0f) > 1e-6f) {
    std::transform(work.begin(), work.end(), work.begin(), [this](float v) { return v / temperature_; });
  }

  bool greedy_mode = (strategy_ == MSLLM_SAMPLER_GREEDY) || (temperature_ <= 0.0f);

  if (greedy_mode) {
    int32_t argmax = 0;
    float max_val = work[0];
    for (size_t i = 1; i < work.size(); ++i) {
      if (work[i] > max_val) {
        max_val = work[i];
        argmax = static_cast<int32_t>(i);
      }
    }
    generated_tokens_.push_back(argmax);
    return argmax;
  }

  Softmax(work);

  if (strategy_ == MSLLM_SAMPLER_TOP_K && top_k_ > 0 && top_k_ < static_cast<int32_t>(work.size())) {
    std::vector<size_t> indices(work.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + top_k_, indices.end(),
                      [&work](size_t a, size_t b) { return work[a] > work[b]; });

    std::vector<float> filtered(work.size(), 0.0f);
    for (int32_t i = 0; i < top_k_; ++i) {
      filtered[indices[i]] = work[indices[i]];
    }
    work = std::move(filtered);

    float sum = std::accumulate(work.begin(), work.end(), 0.0f);
    if (sum > 0.0f) {
      std::transform(work.begin(), work.end(), work.begin(), [sum](float v) { return v / sum; });
    }
  } else if (strategy_ == MSLLM_SAMPLER_TOP_P && top_p_ > 0.0f && top_p_ < 1.0f) {
    std::vector<size_t> indices(work.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&work](size_t a, size_t b) { return work[a] > work[b]; });

    float cumulative = 0.0f;
    size_t cutoff = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
      cumulative += work[indices[i]];
      cutoff = i + 1;
      if (cumulative >= top_p_) {
        break;
      }
    }

    std::vector<float> filtered(work.size(), 0.0f);
    for (size_t i = 0; i < cutoff; ++i) {
      filtered[indices[i]] = work[indices[i]];
    }
    work = std::move(filtered);

    float sum = std::accumulate(work.begin(), work.end(), 0.0f);
    if (sum > 0.0f) {
      std::transform(work.begin(), work.end(), work.begin(), [sum](float v) { return v / sum; });
    }
  }

  std::discrete_distribution<int32_t> dist(work.begin(), work.end());
  int32_t token_id = dist(rng_);
  generated_tokens_.push_back(token_id);
  return token_id;
}

void Sampler::SetStrategy(MSLlmSamplerStrategy strategy) { strategy_ = strategy; }

void Sampler::SetTemperature(float temperature) { temperature_ = temperature; }

void Sampler::SetTopK(int32_t top_k) { top_k_ = top_k; }

void Sampler::SetTopP(float top_p) { top_p_ = top_p; }

void Sampler::SetRepetitionPenalty(float penalty) { repetition_penalty_ = penalty; }

void Sampler::SetPresencePenalty(float penalty) { presence_penalty_ = penalty; }

void Sampler::SetFrequencyPenalty(float penalty) { frequency_penalty_ = penalty; }

void Sampler::SetSeed(int32_t seed) {
  seed_ = seed;
  if (seed != 0) {
    rng_.seed(static_cast<uint32_t>(seed));
  }
}

void Sampler::ApplyConfigOverrides(const MSLlmGenerateConfig &config) {
  if (!config.override_sampler) {
    return;
  }
  if (config.temperature >= 0.0f) {
    temperature_ = config.temperature;
    if (config.temperature <= 0.0f) {
      strategy_ = MSLLM_SAMPLER_GREEDY;
    }
  }
  if (config.top_k > 0) {
    top_k_ = config.top_k;
    strategy_ = MSLLM_SAMPLER_TOP_K;
  }
  if (config.top_p > 0.0f) {
    top_p_ = config.top_p;
    if (strategy_ != MSLLM_SAMPLER_TOP_K) {
      strategy_ = MSLLM_SAMPLER_TOP_P;
    }
  }
  if (config.repetition_penalty > 0.0f) {
    repetition_penalty_ = config.repetition_penalty;
  }
  if (std::fabs(config.presence_penalty) > 1e-6f) {
    presence_penalty_ = config.presence_penalty;
  }
  if (std::fabs(config.frequency_penalty) > 1e-6f) {
    frequency_penalty_ = config.frequency_penalty;
  }
  if (config.seed > 0) {
    SetSeed(config.seed);
  }

  // Rebuild logit bias map
  logit_bias_.clear();
  if (config.logit_bias_tokens && config.logit_bias_values && config.num_logit_biases > 0) {
    for (size_t i = 0; i < config.num_logit_biases; ++i) {
      logit_bias_[config.logit_bias_tokens[i]] = config.logit_bias_values[i];
    }
  }
}

void Sampler::ApplyLogitBias(std::vector<float> &logits) const {
  for (const auto &pair : logit_bias_) {
    if (pair.first >= 0 && static_cast<size_t>(pair.first) < logits.size()) {
      logits[pair.first] += pair.second;
    }
  }
}

}  // namespace mslite_llm
