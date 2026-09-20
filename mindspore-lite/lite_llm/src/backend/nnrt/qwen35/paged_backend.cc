// Copyright 2026 Huawei Technologies Co., Ltd
#include "backend/nnrt/qwen35/paged_backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

#include "backend/nnrt/qwen35/half.h"
#include "manifest/model_manifest.h"

namespace mslite_llm::qwen35 {
namespace {
template <typename T>
T *InputBuffer(Qwen35MoeModel &model, ModelGear gear, const char *name, size_t count) {
  const auto buffer = model.Input(gear, name);
  return buffer.bytes == count * sizeof(T) ? static_cast<T *>(buffer.data) : nullptr;
}
}  // namespace

MSLlmStatus PagedBackend::Init(const BackendConfig &config) {
  if (initialized_ || config.resources == nullptr || config.manifest == nullptr) {
    return MSLLM_ERROR_INVALID_ARGS;
  }
  const auto &root = config.resources->package_root;
  if (root.empty() || !model_.Load(root, 16, &error_)) {
    std::cerr << "Qwen3.5 model load failed: " << error_ << '\n';
    return MSLLM_ERROR_MODEL_LOAD;
  }
  const auto &metadata = model_.Metadata();
  if (!embedding_.Load(root + "/global/embedding_fp16.bin", metadata.vocab_size, metadata.hidden_size, &error_)) {
    return MSLLM_ERROR_MODEL_LOAD;
  }
  BuildRopeTable();
  initialized_ = true;
  return Reset();
}

void PagedBackend::BuildRopeTable() {
  const auto &metadata = model_.Metadata();
  const int32_t rotary = metadata.rotary_dim;
  cosine_.resize(static_cast<size_t>(model_.CacheLength()) * rotary);
  sine_.resize(cosine_.size());
  for (int32_t position = 0; position < model_.CacheLength(); ++position) {
    for (int32_t index = 0; index < rotary / 2; ++index) {
      const float exponent = static_cast<float>(2 * index) / rotary;
      const float angle = position / std::pow(metadata.rope_theta, exponent);
      const size_t offset = static_cast<size_t>(position) * rotary + index;
      cosine_[offset] = cosine_[offset + rotary / 2] = FloatToHalf(std::cos(angle));
      sine_[offset] = sine_[offset + rotary / 2] = FloatToHalf(std::sin(angle));
    }
  }
}

bool PagedBackend::PrepareInputs(const std::vector<int32_t> &tokens, ModelGear gear) {
  const auto &metadata = model_.Metadata();
  const size_t sequence = tokens.size();
  const size_t hidden = metadata.hidden_size;
  const size_t rotary = metadata.rotary_dim;
  const size_t cache = model_.CacheLength();
  auto *embedding = InputBuffer<uint16_t>(model_, gear, "inputs_embeds", sequence * hidden);
  auto *attention = InputBuffer<uint16_t>(model_, gear, "attention_mask", sequence);
  auto *mask = InputBuffer<uint16_t>(model_, gear, "kvcache_mask", sequence * cache);
  auto *cosine = InputBuffer<uint16_t>(model_, gear, "rope_cos", sequence * rotary);
  auto *sine = InputBuffer<uint16_t>(model_, gear, "cope_sin", sequence * rotary);
  auto *last = InputBuffer<int32_t>(model_, gear, "lmhead_idx", 1);
  if (!embedding || !attention || !mask || !cosine || !sine || !last) return false;
  std::fill(mask, mask + sequence * cache, uint16_t{0xfbff});
  for (size_t row = 0; row < sequence; ++row) {
    const auto *values = embedding_.Lookup(tokens[row]);
    if (values == nullptr) return false;
    std::memcpy(embedding + row * hidden, values, hidden * sizeof(uint16_t));
    const int32_t physical = cache_position_ + static_cast<int32_t>(row);
    const int32_t position = std::max(0, physical - left_padding_);
    attention[row] = physical < left_padding_ ? 0 : 0x3c00;
    std::memcpy(cosine + row * rotary, cosine_.data() + position * rotary, rotary * sizeof(uint16_t));
    std::memcpy(sine + row * rotary, sine_.data() + position * rotary, rotary * sizeof(uint16_t));
    if (physical < left_padding_) {
      mask[row * cache + physical] = 0;
    } else {
      std::fill(mask + row * cache + left_padding_, mask + row * cache + physical + 1, uint16_t{0});
    }
  }
  *last = static_cast<int32_t>(sequence) - 1;
  return true;
}

bool PagedBackend::RunChunk(const std::vector<int32_t> &tokens, ModelGear gear) {
  if (tokens.empty() || tokens.size() > static_cast<size_t>(model_.CacheLength() - cache_position_)) return false;
  double elapsed = 0;
  if (!model_.PrepareGear(gear, &error_) || !PrepareInputs(tokens, gear) || !model_.Run(gear, &elapsed, &error_)) {
    std::cerr << "Qwen3.5 execution failed: " << error_ << '\n';
    return false;
  }
  cache_position_ += static_cast<int32_t>(tokens.size());
  return true;
}

bool PagedBackend::ReadLogits(ModelGear gear, BackendOutput *output) {
  const auto buffer = model_.Logits(gear);
  const size_t count = model_.Metadata().vocab_size;
  if (buffer.data == nullptr || buffer.bytes != count * sizeof(uint16_t)) return false;
  const auto *values = static_cast<const uint16_t *>(buffer.data);
  output->logits.resize(count);
  std::transform(values, values + count, output->logits.begin(), HalfToFloat);
  output->next_token_id = -1;
  return true;
}

MSLlmStatus PagedBackend::Prefill(const BackendInput &input, BackendOutput *output) {
  if (!initialized_ || output == nullptr || input.input_ids.empty() || cache_position_ != 0) {
    return MSLLM_ERROR_INVALID_ARGS;
  }
  const size_t chunk = model_.Metadata().prefill_chunk;
  if (input.input_ids.size() > static_cast<size_t>(model_.CacheLength())) return MSLLM_ERROR_INVALID_ARGS;
  left_padding_ = static_cast<int32_t>((chunk - input.input_ids.size() % chunk) % chunk);
  std::vector<int32_t> padded(left_padding_, model_.Metadata().pad_token_id);
  padded.insert(padded.end(), input.input_ids.begin(), input.input_ids.end());
  for (size_t offset = 0; offset < padded.size(); offset += chunk) {
    std::vector<int32_t> tokens(padded.begin() + offset, padded.begin() + offset + chunk);
    if (!RunChunk(tokens, ModelGear::kPrefill64)) return MSLLM_ERROR_INFERENCE;
  }
  return ReadLogits(ModelGear::kPrefill64, output) ? MSLLM_SUCCESS : MSLLM_ERROR_INFERENCE;
}

MSLlmStatus PagedBackend::Decode(const BackendInput &input, BackendOutput *output) {
  if (!initialized_ || output == nullptr || input.input_ids.size() != 1 || cache_position_ == 0) {
    return MSLLM_ERROR_INVALID_ARGS;
  }
  if (!RunChunk(input.input_ids, ModelGear::kDecode1)) return MSLLM_ERROR_INFERENCE;
  return ReadLogits(ModelGear::kDecode1, output) ? MSLLM_SUCCESS : MSLLM_ERROR_INFERENCE;
}

MSLlmStatus PagedBackend::Reset() {
  if (!initialized_) return MSLLM_ERROR_INVALID_ARGS;
  if (!model_.ResetCache(&error_)) return MSLLM_ERROR_INFERENCE;
  cache_position_ = 0;
  left_padding_ = 0;
  return MSLLM_SUCCESS;
}
}  // namespace mslite_llm::qwen35
