// Copyright 2026 Huawei Technologies Co., Ltd
// Adapted from HMOS-Device-AI llm/engine/runtime; preserves its paged Qwen3.5 execution.
#include "backend/nnrt/qwen35/qwen35_moe_model.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17): lite_llm requires C++17.
#include <iostream>
#include <limits>

#include "manifest/artifact_file.h"

namespace fs = std::filesystem;

namespace mslite_llm::qwen35 {
namespace {

constexpr int32_t kLayers = 40;
constexpr int32_t kHidden = 2048;
constexpr int32_t kExperts = 256;
constexpr int32_t kTopK = 8;
constexpr int32_t kCache = 8192;
constexpr int32_t kRotary = 64;

float HalfToFloat(uint16_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
  const uint32_t exponent = (bits >> 10) & 0x1fu;
  const uint32_t mantissa = bits & 0x03ffu;
  uint32_t value = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      value = sign;
    } else {
      uint32_t normalized = mantissa;
      int32_t shift = 0;
      while ((normalized & 0x0400u) == 0) {
        normalized <<= 1;
        ++shift;
      }
      value = sign | (static_cast<uint32_t>(127 - 15 - shift + 1) << 23) | ((normalized & 0x03ffu) << 13);
    }
  } else if (exponent == 31) {
    value = sign | 0x7f800000u | (mantissa << 13);
  } else {
    value = sign | ((exponent + 112u) << 23) | (mantissa << 13);
  }
  float result = 0.0f;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

uint16_t FloatToHalf(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u;
  const uint32_t mantissa = bits & 0x7fffffu;
  const uint32_t raw_exponent = (bits >> 23) & 0xffu;
  if (raw_exponent == 0xffu) {
    return static_cast<uint16_t>(sign | (mantissa == 0 ? 0x7c00u : 0x7e00u));
  }
  int32_t exponent = static_cast<int32_t>(raw_exponent) - 127 + 15;
  if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
  if (exponent <= 0) {
    if (exponent < -10) return static_cast<uint16_t>(sign);
    const uint32_t normalized = mantissa | 0x800000u;
    const uint32_t shift = static_cast<uint32_t>(14 - exponent);
    uint32_t half_mantissa = normalized >> shift;
    const uint32_t remainder = normalized & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1u);
    if (remainder > halfway || (remainder == halfway && (half_mantissa & 1u) != 0)) {
      ++half_mantissa;
    }
    return static_cast<uint16_t>(sign | half_mantissa);
  }
  uint32_t half = (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13);
  const uint32_t remainder = mantissa & 0x1fffu;
  if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1u) != 0)) {
    ++half;
  }
  return static_cast<uint16_t>(sign | half);
}

bool IsFinite(float value) { return std::isfinite(value); }

int32_t UpdateRouteCache(const std::array<int32_t, kTopK> &experts, size_t capacity, std::vector<int32_t> *cache) {
  int32_t hits = 0;
  for (int32_t expert : experts) {
    const auto found = std::find(cache->begin(), cache->end(), expert);
    if (found != cache->end()) {
      ++hits;
      cache->erase(found);
    }
    cache->push_back(expert);
    if (cache->size() > capacity) cache->erase(cache->begin());
  }
  return hits;
}

std::vector<int32_t> RankRouterExperts(const TensorView &router_logits, size_t count) {
  std::vector<int32_t> indices(kExperts);
  for (int32_t expert = 0; expert < kExperts; ++expert) indices[expert] = expert;
  const auto *values = static_cast<const uint16_t *>(router_logits.data);
  std::partial_sort(indices.begin(), indices.begin() + std::min(count, indices.size()), indices.end(),
                    [&](int32_t left, int32_t right) {
                      const float left_score = HalfToFloat(values[left]);
                      const float right_score = HalfToFloat(values[right]);
                      return left_score > right_score || (left_score == right_score && left < right);
                    });
  indices.resize(std::min(count, indices.size()));
  return indices;
}

int32_t CountPredictedHits(const std::array<int32_t, kTopK> &experts, const std::vector<int32_t> &prediction) {
  int32_t hits = 0;
  for (int32_t expert : experts) {
    if (std::find(prediction.begin(), prediction.end(), expert) != prediction.end()) {
      ++hits;
    }
  }
  return hits;
}

}  // namespace

Qwen35MoeModel::Gear *Qwen35MoeModel::SelectGear(ModelGear gear, Gear *decode, Gear *prefill) {
  if (gear == ModelGear::kDecode1) return decode;
  if (gear == ModelGear::kPrefill64) return prefill;
  return nullptr;
}

const Qwen35MoeModel::Gear *Qwen35MoeModel::SelectGear(ModelGear gear, const Gear *decode, const Gear *prefill) {
  if (gear == ModelGear::kDecode1) return decode;
  if (gear == ModelGear::kPrefill64) return prefill;
  return nullptr;
}

bool Qwen35MoeModel::IsActiveGear(ModelGear gear) {
  return gear == ModelGear::kDecode1 || gear == ModelGear::kPrefill64;
}

bool Qwen35MoeModel::Load(const std::string &artifact_root, size_t expert_cache_shards, std::string *error) {
  if (loaded_) {
    if (error) *error = "Qwen35 MoE model is already loaded";
    return false;
  }
  if (artifact_root.empty() || expert_cache_shards == 0) {
    if (error) *error = "invalid Qwen35 MoE artifact configuration";
    return false;
  }
  if (!IsArtifactPackagePath(artifact_root + "/model.json")) {
    if (error) *error = "Qwen3.5 requires a registered single .msl package";
    return false;
  }
  metadata_.model_family = "qwen3.5-moe";
  metadata_.hidden_size = kHidden;
  metadata_.vocab_size = 248320;
  metadata_.rotary_dim = kRotary;
  metadata_.prefill_chunk = 64;
  metadata_.max_cache_length = kCache;
  metadata_.pad_token_id = 248044;
  metadata_.eos_token_id = 248046;
  metadata_.rope_theta = 10000000.0f;
  const fs::path root(artifact_root);
  if (!ArtifactExists(root / "model.json") || !ArtifactExists(root / "global/embedding_fp16.bin")) {
    if (error) *error = "Qwen35 MoE artifact root is incomplete: " + artifact_root;
    return false;
  }
  artifact_root_ = artifact_root;
  // Initialize the file-backed LM Head before the layer sessions so its four
  // stable weight mappings are established before the decode runtime starts.
  if (!lm_head_.Load(artifact_root, error) || !resident_.Load(artifact_root, error) ||
      !experts_.Load(artifact_root, expert_cache_shards, error) ||
      !fused_decode_.Load(artifact_root, &resident_, &experts_, error)) {
    return false;
  }
  decode_.sequence = 1;
  prefill_.sequence = 64;
  for (Gear *gear : {&decode_, &prefill_}) {
    const size_t sequence = static_cast<size_t>(gear->sequence);
    gear->lmhead_idx.resize(1);
    gear->kvcache_mask.resize(sequence * kCache);
    gear->rope_cos.resize(sequence * kRotary);
    gear->rope_sin.resize(sequence * kRotary);
    gear->inputs_embeds.resize(sequence * kHidden);
    gear->attention_mask.resize(sequence);
    gear->logits.resize(248320);
  }
  hidden_.resize(static_cast<size_t>(64) * kHidden);
  combined_.resize(static_cast<size_t>(64) * kHidden);
  loaded_ = true;
  return ResetCache(error);
}

bool Qwen35MoeModel::ResetCache(std::string *error) {
  if (!loaded_) {
    if (error) *error = "Qwen35 MoE model is not loaded";
    return false;
  }
  return resident_.ResetCaches(error);
}

bool Qwen35MoeModel::SaveCache(CacheSnapshot *snapshot, std::string *error) const {
  if (!loaded_ || snapshot == nullptr) {
    if (error) *error = "Qwen35 MoE cache snapshot is unavailable";
    return false;
  }
  return resident_.SaveCaches(&snapshot->buffers, error);
}

bool Qwen35MoeModel::RestoreCache(const CacheSnapshot &snapshot, std::string *error) {
  if (!loaded_) {
    if (error) *error = "Qwen35 MoE model is not loaded";
    return false;
  }
  return resident_.RestoreCaches(snapshot.buffers, error);
}

bool Qwen35MoeModel::PrepareGear(ModelGear gear, std::string *error) {
  if (!loaded_ || !IsActiveGear(gear)) {
    if (error) *error = "Qwen35 MoE supports only Decode1 and Prefill64 gears";
    return false;
  }
  active_ = SelectGear(gear, &decode_, &prefill_);
  return active_ != nullptr;
}

TensorBuffer Qwen35MoeModel::Input(ModelGear gear, const std::string &name) const {
  const Gear *selected = SelectGear(gear, &decode_, &prefill_);
  if (selected == nullptr || selected != active_) return {};
  if (name == "lmhead_idx") {
    return {const_cast<int32_t *>(selected->lmhead_idx.data()), selected->lmhead_idx.size() * sizeof(int32_t)};
  }
  if (name == "kvcache_mask") {
    return {const_cast<uint16_t *>(selected->kvcache_mask.data()), selected->kvcache_mask.size() * sizeof(uint16_t)};
  }
  if (name == "rope_cos") {
    return {const_cast<uint16_t *>(selected->rope_cos.data()), selected->rope_cos.size() * sizeof(uint16_t)};
  }
  if (name == "cope_sin") {
    return {const_cast<uint16_t *>(selected->rope_sin.data()), selected->rope_sin.size() * sizeof(uint16_t)};
  }
  if (name == "inputs_embeds") {
    return {const_cast<uint16_t *>(selected->inputs_embeds.data()), selected->inputs_embeds.size() * sizeof(uint16_t)};
  }
  if (name == "attention_mask") {
    return {const_cast<uint16_t *>(selected->attention_mask.data()),
            selected->attention_mask.size() * sizeof(uint16_t)};
  }
  return {};
}

TensorBuffer Qwen35MoeModel::Logits(ModelGear gear) const {
  const Gear *selected = SelectGear(gear, &decode_, &prefill_);
  if (selected == nullptr || selected != active_) return {};
  return {const_cast<uint16_t *>(selected->logits.data()), selected->logits.size() * sizeof(uint16_t)};
}

TensorBuffer Qwen35MoeModel::TokenIds(ModelGear /*gear*/) const {
  if (!lm_head_.HasTop1()) return {};
  return {const_cast<int32_t *>(lm_head_.TokenIdData()), sizeof(int32_t)};
}

TensorBuffer Qwen35MoeModel::HiddenStates(ModelGear /*gear*/) const { return {}; }

Qwen35MoeModel::ExpertExecution::ExpertExecution()
    : expert_input(64 * kHidden), gated(64 * 512), expert_output(64 * kHidden) {}

Qwen35MoeModel::FusedExecution::FusedExecution() : final_expert_output(kTopK * kHidden) {}

bool Qwen35MoeModel::BuildRoutes(const TensorView &router_logits, const std::vector<uint16_t> &attention_mask,
                                 int32_t sequence, std::vector<std::vector<Route>> *routes, std::string *error) {
  if (routes == nullptr || router_logits.data == nullptr || sequence <= 0 ||
      router_logits.bytes != static_cast<size_t>(sequence) * kExperts * sizeof(uint16_t) ||
      attention_mask.size() != static_cast<size_t>(sequence)) {
    if (error) *error = "Qwen35 router tensor size mismatch";
    return false;
  }
  routes->assign(kExperts, {});
  const auto *values = static_cast<const uint16_t *>(router_logits.data);
  for (int32_t token = 0; token < sequence; ++token) {
    if (HalfToFloat(attention_mask[token]) == 0.0f) continue;
    if (!BuildTokenRoutes(values + token * kExperts, token, routes, error)) return false;
  }
  return true;
}

bool Qwen35MoeModel::RunGear(Gear *gear, double *elapsed_ms, std::string *error) {
  // Execute one S64 prefill chunk or one S1 decode token through all layers.
  // The host orchestrates routing and merging; runtime sessions execute NPU graphs.
  if (gear == nullptr || gear != active_) {
    if (error) *error = "Qwen35 MoE gear is not active";
    return false;
  }
  // 1. Initialize hidden states from embeddings for the active sequence gear.
  const int32_t sequence = gear->sequence;
  const size_t hidden_elements = static_cast<size_t>(sequence) * kHidden;
  std::memcpy(hidden_.data(), gear->inputs_embeds.data(), hidden_elements * sizeof(uint16_t));
  // 2. Qwen applies the attention mask to padding states before the linear
  // attention block. Zeroing them once is sufficient because all subsequent
  // projections and residual paths are bias-free for these rows.
  for (int32_t token = 0; token < sequence; ++token) {
    if (HalfToFloat(gear->attention_mask[token]) != 0.0f) continue;
    std::memset(hidden_.data() + static_cast<size_t>(token) * kHidden, 0, kHidden * sizeof(uint16_t));
  }

  if (sequence == 1 && fused_decode_.Enabled()) {
    // 3. Fused decode combines a layer's experts/merge with the next resident layer.
    return RunFusedDecode(gear, elapsed_ms, error);
  }

  ExpertExecution execution;
  for (int32_t layer = 0; layer < kLayers; ++layer) {
    if (!RunLayerExperts(gear, layer, execution, error)) return false;
  }
  // 9. Project the selected final hidden row to vocabulary logits on the NPU.
  // Token sampling remains outside this model, in the generation pipeline.
  const int32_t lmhead_row = gear->lmhead_idx[0];
  if (lmhead_row < 0 || lmhead_row >= sequence) {
    if (error) *error = "Qwen35 lmhead_idx is outside the active gear";
    return false;
  }
  double lm_head_elapsed = 0.0;
  if (!lm_head_.Run(hidden_.data() + static_cast<size_t>(lmhead_row) * kHidden, gear->logits.data(), &lm_head_elapsed,
                    error)) {
    return false;
  }
  execution.total += lm_head_elapsed;
  if (sequence == 1 && std::getenv("QWEN35_PERF_STATS") != nullptr) {
    std::cerr << "qwen35_stage_ms=" << execution.resident_total << "," << execution.expert_total << ","
              << lm_head_elapsed << '\n';
  }
  // 10. Sum graph times; CPU routing, copying and merging are not included.
  if (elapsed_ms != nullptr) *elapsed_ms = execution.total;
  return true;
}

bool Qwen35MoeModel::RunFusedDecode(Gear *gear, double *elapsed_ms, std::string *error) {
  if (gear == nullptr || gear->sequence != 1 || !fused_decode_.Enabled() || !experts_.GroupedDecodeEnabled()) {
    if (error) *error = "Qwen35 fused decoder is unavailable";
    return false;
  }

  FusedExecution execution;
  double elapsed = 0.0;

  auto resident_start = std::chrono::steady_clock::now();
  if (!resident_.RunLayer(0, 1, hidden_.data(), gear->rope_cos.data(), gear->rope_sin.data(), gear->kvcache_mask.data(),
                          &execution.resident_outputs, &elapsed, error)) {
    return false;
  }
  execution.resident_wall =
    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - resident_start).count();
  execution.total += elapsed;
  execution.resident_total += elapsed;
  auto route_start = std::chrono::steady_clock::now();
  if (!BuildRoutes(execution.resident_outputs.router_logits, gear->attention_mask, 1, &execution.routes, error)) {
    return false;
  }
  execution.route_build_wall +=
    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - route_start).count();

  for (int32_t layer = 0; layer < kLayers - 1; ++layer) {
    auto select_start = std::chrono::steady_clock::now();
    if (!SelectFusedRoutes(layer, execution.resident_outputs.router_logits, execution, error)) {
      return false;
    }
    execution.route_select_wall +=
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - select_start).count();
    auto boundary_start = std::chrono::steady_clock::now();
    if (!fused_decode_.RunBoundary(
          layer, execution.selected_experts, execution.selected_weights, execution.resident_outputs.expert_input,
          execution.resident_outputs.attention_residual, execution.resident_outputs.shared_expert_output,
          gear->rope_cos.data(), gear->rope_sin.data(), gear->kvcache_mask.data(), &experts_,
          &execution.resident_outputs, &elapsed, error)) {
      return false;
    }
    execution.boundary_wall +=
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - boundary_start).count();
    execution.total += elapsed;
    execution.fused_total += elapsed;
    route_start = std::chrono::steady_clock::now();
    if (!BuildRoutes(execution.resident_outputs.router_logits, gear->attention_mask, 1, &execution.routes, error)) {
      return false;
    }
    execution.route_build_wall +=
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - route_start).count();
  }

  if (!RunFinalExperts(execution, error)) return false;

  double lm_head_elapsed = 0.0;
  auto lm_head_start = std::chrono::steady_clock::now();
  if (!lm_head_.Run(hidden_.data(), gear->logits.data(), &lm_head_elapsed, error)) {
    return false;
  }
  execution.lm_head_wall =
    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - lm_head_start).count();
  execution.total += lm_head_elapsed;
  if (std::getenv("QWEN35_PERF_STATS") != nullptr) {
    std::cerr << "qwen35_fused_ms=" << execution.fused_total << '\n';
    std::cerr << "qwen35_stage_ms=" << execution.resident_total + execution.fused_total << "," << execution.expert_total
              << "," << lm_head_elapsed << '\n';
    std::cerr << "qwen35_fused_host_ms=resident=" << execution.resident_wall - execution.resident_total
              << " boundaries=" << execution.boundary_wall - execution.fused_total
              << " route_build=" << execution.route_build_wall << " route_select=" << execution.route_select_wall
              << " final_expert=" << execution.final_expert_wall - execution.expert_total
              << " lm_head=" << execution.lm_head_wall - lm_head_elapsed << '\n';
  }
  if (elapsed_ms != nullptr) *elapsed_ms = execution.total;
  return true;
}

bool Qwen35MoeModel::Run(ModelGear gear, double *elapsed_ms, std::string *error, bool defer_cache_commit) {
  if (defer_cache_commit) {
    if (error) *error = "Qwen35 MoE runtime does not support deferred cache commits";
    return false;
  }
  if (!loaded_ || SelectGear(gear, &decode_, &prefill_) != active_) {
    if (error) *error = "Qwen35 MoE gear is not prepared";
    return false;
  }
  return RunGear(active_, elapsed_ms, error);
}

bool Qwen35MoeModel::CommitCache(std::string * /*error*/) { return true; }
bool Qwen35MoeModel::CommitCheckpointCache(std::string * /*error*/) { return true; }
bool Qwen35MoeModel::DiscardCache(std::string * /*error*/) { return true; }

bool Qwen35MoeModel::BuildTokenRoutes(const uint16_t *values, int32_t token, std::vector<std::vector<Route>> *routes,
                                      std::string *error) {
  std::array<int32_t, kTopK> indices{};
  std::array<float, kTopK> scores{};
  indices.fill(-1);
  scores.fill(-std::numeric_limits<float>::infinity());
  for (int32_t expert = 0; expert < kExperts; ++expert) {
    const float score = HalfToFloat(values[expert]);
    if (!IsFinite(score)) continue;
    int32_t insert = 0;
    while (insert < kTopK && (score < scores[insert] || (score == scores[insert] && expert > indices[insert]))) {
      ++insert;
    }
    if (insert >= kTopK) continue;
    for (int32_t pos = kTopK - 1; pos > insert; --pos) {
      scores[pos] = scores[pos - 1];
      indices[pos] = indices[pos - 1];
    }
    scores[insert] = score;
    indices[insert] = expert;
  }
  if (indices[kTopK - 1] < 0) {
    if (error) *error = "Qwen35 router produced no finite expert logits";
    return false;
  }
  const float max_score = scores[0];
  std::array<float, kTopK> weights{};
  float total = 0.0f;
  for (int32_t slot = 0; slot < kTopK; ++slot) {
    weights[slot] = std::exp(scores[slot] - max_score);
    total += weights[slot];
  }
  if (!(total > 0.0f) || !IsFinite(total)) {
    if (error) *error = "Qwen35 router softmax failed";
    return false;
  }
  for (int32_t slot = 0; slot < kTopK; ++slot) {
    const float normalized = weights[slot] / total;
    routes->at(indices[slot]).push_back(Route{token, slot, HalfToFloat(FloatToHalf(normalized))});
  }
  return true;
}

bool Qwen35MoeModel::RunLayerExperts(Gear *gear, int32_t layer, ExpertExecution &execution, std::string *error) {
  const int32_t sequence = gear->sequence;
  // 4. NPU: run attention, router and shared expert using this layer's tensor set.
  // RunLayer is synchronous and returns views of the session output buffers;
  // the attention state is updated for subsequent chunks or decode tokens.
  Qwen35ResidentOutputs resident_outputs;
  double elapsed = 0.0;
  if (!resident_.RunLayer(layer, sequence, hidden_.data(), gear->rope_cos.data(), gear->rope_sin.data(),
                          gear->kvcache_mask.data(), &resident_outputs, &elapsed, error)) {
    return false;
  }
  execution.total += elapsed;
  execution.resident_total += elapsed;
  // 5. CPU: choose Top-8 experts for each active token and group routes by expert.
  // Each route retains the original token row and its normalized routing weight.
  if (!BuildRoutes(resident_outputs.router_logits, gear->attention_mask, sequence, &execution.routes, error)) {
    return false;
  }
  const size_t output_elements = static_cast<size_t>(gear->sequence) * kHidden;
  if (resident_outputs.attention_residual.bytes != output_elements * sizeof(uint16_t) ||
      resident_outputs.shared_expert_output.bytes != output_elements * sizeof(uint16_t)) {
    if (error) *error = "Qwen35 resident output shape mismatch";
    return false;
  }
  const auto *residual = static_cast<const uint16_t *>(resident_outputs.attention_residual.data);
  const auto *shared = static_cast<const uint16_t *>(resident_outputs.shared_expert_output.data);
  // 6. CPU: initialize the FP32 accumulator with residual and shared-expert output.
  // Routed-expert contributions are added below before converting back to FP16.
  for (size_t index = 0; index < output_elements; ++index) {
    combined_[index] = HalfToFloat(residual[index]) + HalfToFloat(shared[index]);
  }

  const auto *expert_source = static_cast<const uint16_t *>(resident_outputs.expert_input.data);
  if (sequence == 1 && experts_.GroupedDecodeEnabled()) {
    if (!RunGroupedExperts(layer, expert_source, execution, error)) return false;
  } else {
    if (!RunRoutedExperts(layer, expert_source, execution, error)) return false;
  }
  // 8. The merged hidden becomes the next layer's input in FP16; repeat 4-8.
  for (size_t index = 0; index < output_elements; ++index) {
    hidden_[index] = FloatToHalf(combined_[index]);
  }
  return true;
}

bool Qwen35MoeModel::RunGroupedExperts(int32_t layer, const uint16_t *expert_source, ExpertExecution &execution,
                                       std::string *error) {
  double elapsed = 0.0;
  // 7a. S1 optimization: prepare the eight selected experts for grouped NPU
  // execution, rather than issuing separate GateUp/Down stages per expert.
  std::array<int32_t, kTopK> selected_experts{};
  std::array<float, kTopK> selected_weights{};
  int32_t selected_count = 0;
  for (int32_t expert = 0; expert < kExperts; ++expert) {
    if (execution.routes[expert].empty()) continue;
    if (execution.routes[expert].size() != 1 || selected_count >= kTopK) {
      if (error) *error = "invalid grouped decoder route";
      return false;
    }
    selected_experts[selected_count] = expert;
    selected_weights[selected_count] = execution.routes[expert][0].weight;
    ++selected_count;
  }
  if (selected_count != kTopK || !experts_.RunGroupedDecode(layer, selected_experts, expert_source,
                                                            execution.expert_output.data(), &elapsed, error)) {
    if (selected_count != kTopK && error) {
      *error = "grouped decoder requires exactly eight experts";
    }
    return false;
  }
  execution.total += elapsed;
  execution.expert_total += elapsed;
  const int32_t cache8_hits = UpdateRouteCache(selected_experts, 8, &route_cache_8_[layer]);
  const int32_t cache16_hits = UpdateRouteCache(selected_experts, 16, &route_cache_16_[layer]);
  const int32_t cache32_hits = UpdateRouteCache(selected_experts, 32, &route_cache_32_[layer]);
  if (std::getenv("QWEN35_PERF_STATS") != nullptr) {
    std::cerr << "qwen35_layer=" << layer << " expert_ms=" << elapsed << " cache_hits=" << cache8_hits << ","
              << cache16_hits << "," << cache32_hits << '\n';
  }
  // 7a.1. CPU: weight returned expert slots and accumulate into the single token.
  for (int32_t slot = 0; slot < kTopK; ++slot) {
    const uint16_t *source = execution.expert_output.data() + static_cast<size_t>(slot) * kHidden;
    const float weight = selected_weights[slot];
    for (int32_t channel = 0; channel < kHidden; ++channel) {
      combined_[channel] += HalfToFloat(source[channel]) * weight;
    }
  }
  return true;
}

bool Qwen35MoeModel::RunRoutedExperts(int32_t layer, const uint16_t *expert_source, ExpertExecution &execution,
                                      std::string *error) {
  double elapsed = 0.0;
  // 7b. Prefill and fallback decode: process each expert's assigned token rows.
  for (int32_t expert = 0; expert < kExperts; ++expert) {
    const std::vector<Route> &selected = execution.routes[expert];
    if (selected.empty()) continue;
    if (selected.size() > 64) {
      if (error) *error = "Qwen35 expert route exceeds S64 capacity";
      return false;
    }
    // 7b.1. CPU input gather: pack this expert's activations into dense rows.
    // This gathers activations; expert weight preparation happens in RunStage.
    for (size_t row = 0; row < selected.size(); ++row) {
      std::memcpy(execution.expert_input.data() + row * kHidden,
                  expert_source + static_cast<size_t>(selected[row].token) * kHidden, kHidden * sizeof(uint16_t));
    }
    const int32_t rows = static_cast<int32_t>(selected.size());
    // 7b.2. Each runtime call acquires the expert shard from ExpertCache, copies
    // weights/activations into session inputs, runs synchronously on the NPU,
    // and copies the result back. GateUp returns the gated intermediate;
    // Down projects it back to the hidden width.
    if (!experts_.RunGateUp(layer, expert, rows, execution.expert_input.data(), execution.gated.data(), &elapsed,
                            error)) {
      return false;
    }
    execution.total += elapsed;
    execution.expert_total += elapsed;
    if (!experts_.RunDown(layer, expert, rows, execution.gated.data(), execution.expert_output.data(), &elapsed,
                          error)) {
      return false;
    }
    execution.total += elapsed;
    execution.expert_total += elapsed;
    // 7b.3. CPU output scatter/merge: restore token positions and add each
    // expert's contribution with its routing weight. A token receives eight.
    for (int32_t row = 0; row < rows; ++row) {
      const size_t destination = static_cast<size_t>(selected[row].token) * kHidden;
      const uint16_t *source = execution.expert_output.data() + static_cast<size_t>(row) * kHidden;
      const float weight = selected[row].weight;
      for (int32_t channel = 0; channel < kHidden; ++channel) {
        combined_[destination + channel] += HalfToFloat(source[channel]) * weight;
      }
    }
  }
  return true;
}

bool Qwen35MoeModel::SelectFusedRoutes(int32_t layer, const TensorView &router_logits, FusedExecution &execution,
                                       std::string *error) {
  int32_t selected_count = 0;
  for (int32_t expert = 0; expert < kExperts; ++expert) {
    if (execution.routes[expert].empty()) continue;
    if (execution.routes[expert].size() != 1 || selected_count >= kTopK) {
      if (error) *error = "invalid grouped fused decoder route";
      return false;
    }
    execution.selected_experts[selected_count] = expert;
    execution.selected_weights[selected_count] = FloatToHalf(execution.routes[expert][0].weight);
    ++selected_count;
  }
  if (selected_count != kTopK) {
    if (error) *error = "grouped fused decoder requires exactly eight experts";
    return false;
  }
  const int32_t cache8_hits = UpdateRouteCache(execution.selected_experts, 8, &route_cache_8_[layer]);
  const int32_t cache16_hits = UpdateRouteCache(execution.selected_experts, 16, &route_cache_16_[layer]);
  const int32_t cache32_hits = UpdateRouteCache(execution.selected_experts, 32, &route_cache_32_[layer]);
  if (std::getenv("QWEN35_ROUTE_PREDICT_STATS") != nullptr) {
    const int32_t predicted16 = CountPredictedHits(execution.selected_experts, previous_router_top16_[layer]);
    const int32_t predicted32 = CountPredictedHits(execution.selected_experts, previous_router_top32_[layer]);
    std::cerr << "qwen35_route_predict layer=" << layer << " previous_top16_hits=" << predicted16
              << " previous_top32_hits=" << predicted32 << '\n';
  }
  previous_router_top16_[layer] = RankRouterExperts(router_logits, 16);
  previous_router_top32_[layer] = RankRouterExperts(router_logits, 32);
  if (std::getenv("QWEN35_PERF_STATS") != nullptr) {
    std::cerr << "qwen35_layer=" << layer << " cache_hits=" << cache8_hits << "," << cache16_hits << "," << cache32_hits
              << '\n';
  }
  return true;
}

bool Qwen35MoeModel::RunFinalExperts(FusedExecution &execution, std::string *error) {
  double elapsed = 0.0;
  auto select_start = std::chrono::steady_clock::now();
  if (!SelectFusedRoutes(kLayers - 1, execution.resident_outputs.router_logits, execution, error)) {
    return false;
  }
  execution.route_select_wall +=
    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - select_start).count();
  auto final_expert_start = std::chrono::steady_clock::now();
  if (!experts_.RunGroupedDecode(kLayers - 1, execution.selected_experts,
                                 static_cast<const uint16_t *>(execution.resident_outputs.expert_input.data),
                                 execution.final_expert_output.data(), &elapsed, error)) {
    return false;
  }
  execution.final_expert_wall =
    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - final_expert_start).count();
  execution.total += elapsed;
  execution.expert_total += elapsed;

  if (execution.resident_outputs.attention_residual.bytes != static_cast<size_t>(kHidden) * sizeof(uint16_t) ||
      execution.resident_outputs.shared_expert_output.bytes != static_cast<size_t>(kHidden) * sizeof(uint16_t)) {
    if (error) *error = "Qwen35 fused resident output shape mismatch";
    return false;
  }
  const auto *residual = static_cast<const uint16_t *>(execution.resident_outputs.attention_residual.data);
  const auto *shared = static_cast<const uint16_t *>(execution.resident_outputs.shared_expert_output.data);
  for (int32_t channel = 0; channel < kHidden; ++channel) {
    combined_[channel] = HalfToFloat(residual[channel]) + HalfToFloat(shared[channel]);
  }
  for (int32_t slot = 0; slot < kTopK; ++slot) {
    const uint16_t *source = execution.final_expert_output.data() + static_cast<size_t>(slot) * kHidden;
    const float weight = HalfToFloat(execution.selected_weights[slot]);
    for (int32_t channel = 0; channel < kHidden; ++channel) {
      combined_[channel] += HalfToFloat(source[channel]) * weight;
    }
  }
  for (int32_t channel = 0; channel < kHidden; ++channel) {
    hidden_[channel] = FloatToHalf(combined_[channel]);
  }

  return true;
}

}  // namespace mslite_llm::qwen35
