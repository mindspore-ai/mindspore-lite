// Copyright 2026 Huawei Technologies Co., Ltd
// Adapted from HMOS-Device-AI llm/engine/runtime; preserves its paged Qwen3.5 execution.
#include "backend/nnrt/qwen35/qwen35_fused_decode_runtime.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17): lite_llm requires C++17.
#include <iostream>
#include <unordered_map>

#include "backend/nnrt/qwen35/qwen35_expert_runtime.h"
#include "manifest/artifact_file.h"

namespace fs = std::filesystem;

namespace mslite_llm::qwen35 {
namespace {

constexpr int32_t kHidden = 2048;
constexpr int32_t kCache = 8192;

const TensorShapes kLinearWeights = {
  {"in_norm", {2048}},
  {"post_norm", {2048}},
  {"qkv_z_w", {12288, 2048}},
  {"a_b_w", {64, 2048}},
  {"conv_w", {8192, 1, 4}},
  {"dt_bias", {32}},
  {"a_log_bits", {64}},
  {"gated_norm", {128}},
  {"out_w", {2048, 4096}},
  {"router_w", {256, 2048}},
  {"shared_gate_up_w", {1024, 2048}},
  {"shared_down_w", {2048, 512}},
  {"shared_score_w", {16, 2048}},
};

const TensorShapes kFullWeights = {
  {"in_norm", {2048}},
  {"post_norm", {2048}},
  {"qkv_w", {9216, 2048}},
  {"o_w", {2048, 4096}},
  {"q_norm", {256}},
  {"k_norm", {256}},
  {"router_w", {256, 2048}},
  {"shared_gate_up_w", {1024, 2048}},
  {"shared_down_w", {2048, 512}},
  {"shared_score_w", {16, 2048}},
};

TensorShapes InputShapes(bool full, bool indexed, bool packed, bool zero_copy, bool banked, int32_t packed_slots) {
  TensorShapes shapes = {
    {"current_expert_input", {1, kHidden}},
    {"current_attention_residual", {1, kHidden}},
    {"current_shared_expert_output", {1, kHidden}},
    {"current_route_weights", {8, 64}},
    {"next_output_zero", {1}},
  };
  if (banked) {
    shapes["current_expert_ids"] = {8};
    for (int32_t bank = 0; bank < 6; ++bank) shapes["current_packed_bank_" + std::to_string(bank)] = {64, 1777664};
  } else if (zero_copy) {
    for (int32_t slot = 0; slot < 8; ++slot) {
      shapes["current_expert_" + std::to_string(slot) + "_shard"] = {1777664};
    }
  } else if (packed) {
    shapes["current_expert_ids"] = {8};
    shapes["current_packed_weights"] = {packed_slots, 1777664};
  } else if (indexed) {
    shapes["current_expert_ids"] = {8};
  }
  const int32_t slot_count = indexed ? 16 : 8;
  for (int32_t slot = 0; !packed && !zero_copy && !banked && slot < slot_count; ++slot) {
    const std::string prefix = "current_expert_" + std::to_string(slot);
    shapes[prefix + "_gate_up_w"] = {1024, 1024};
    shapes[prefix + "_gate_up_bias"] = {1024};
    shapes[prefix + "_gate_up_w_scale"] = {8, 16, 128, 8};
    shapes[prefix + "_down_w"] = {2048, 256};
    shapes[prefix + "_down_bias"] = {2048};
    shapes[prefix + "_down_w_scale"] = {16, 4, 128, 8};
  }
  const TensorShapes &weights = full ? kFullWeights : kLinearWeights;
  for (const auto &item : weights) shapes["next_" + item.first] = item.second;
  if (full) {
    shapes["next_rope_cos"] = {1, 1, 64};
    shapes["next_rope_sin"] = {1, 1, 64};
    shapes["next_kvcache_mask"] = {1, 1, 1, kCache};
    shapes["next_past_key"] = {1, 2, kCache, 256};
    shapes["next_past_value"] = {1, 2, kCache, 256};
  } else {
    shapes["next_past_conv_state"] = {1, 8192, 4};
    shapes["next_past_recurrent_state_bits"] = {1, 32, 128, 256};
  }
  return shapes;
}

TensorShapes OutputShapes(bool full) {
  TensorShapes shapes = {
    {"next_attention_residual", {1, 1, kHidden}},
    {"next_expert_input", {1, 1, kHidden}},
    {"next_router_logits", {1, 1, 256}},
    {"next_shared_expert_output", {1, 1, kHidden}},
  };
  if (full) {
    shapes["next_present_key"] = {1, 2, kCache, 256};
    shapes["next_present_value"] = {1, 2, kCache, 256};
  } else {
    shapes["next_present_conv_state"] = {1, 8192, 4};
    shapes["next_present_recurrent_state_bits"] = {1, 32, 128, 256};
  }
  return shapes;
}

bool Copy(TensorView target, const void *source, size_t bytes, const char *name, std::string *error) {
  if (target.data == nullptr || source == nullptr || target.bytes != bytes) {
    if (error) *error = std::string("fused decode tensor mismatch: ") + name;
    return false;
  }
  std::memcpy(target.data, source, bytes);
  return true;
}

bool ModelPairExists(const fs::path &linear, const fs::path &full) {
  return ArtifactExists(linear) && ArtifactExists(full);
}

bool EnvironmentIsOne(const char *name) {
  const char *value = std::getenv(name);
  return value != nullptr && std::string(value) == "1";
}

}  // namespace

bool Qwen35FusedDecodeRuntime::Load(const std::string &artifact_root, Qwen35ResidentRuntime *resident,
                                    Qwen35ExpertRuntime *expert_runtime, std::string *error) {
  // The fused boundary is the accepted decoder path. Keep an explicit escape
  // hatch so the byte-identical grouped path remains available for diagnosis.
  if (std::getenv("QWEN35_DISABLE_FUSED_DECODE") != nullptr) return true;
  const ModelChoice choice = ChooseModels(artifact_root, expert_runtime);
  const auto &linear_model = choice.linear_model;
  const auto &full_model = choice.full_model;
  const auto packed_slots = choice.packed_slots;
  const bool indexed_available = choice.indexed_available;
  const bool packed_available = choice.packed_available;
  const bool zero_copy_available = choice.zero_copy_available;
  const bool banked_available = choice.banked_available;
  if (!ArtifactExists(linear_model) || !ArtifactExists(full_model)) return true;
  if (resident == nullptr || !LoadSessions(choice, expert_runtime, error)) return false;
  indexed_ = indexed_available && !packed_available && !zero_copy_available && !banked_available;
  packed_ = packed_available && !zero_copy_available && !banked_available;
  zero_copy_ = zero_copy_available && !banked_available;
  banked_ = banked_available;
  packed_slots_ = packed_available ? packed_slots : 0;
  for (int32_t layer = 0; layer < 39; ++layer) {
    if (!CreateBoundary(layer, resident, expert_runtime, error)) return false;
  }
  enabled_ = true;
  return true;
}

bool Qwen35FusedDecodeRuntime::CreateBoundary(int32_t current_layer, Qwen35ResidentRuntime *resident,
                                              Qwen35ExpertRuntime *expert_runtime, std::string *error) {
  const int32_t next_layer = current_layer + 1;
  const bool full = Qwen35ResidentRuntime::IsFullLayer(next_layer);
  NnrtSession *session = full ? &full_ : &linear_;
  TensorShapes inputs = InputShapes(full, indexed_, packed_, zero_copy_, banked_, packed_slots_);
  SharedInputKeys shared;
  ExternalInputHandles external;
  FileInputBindings files;
  if (banked_) {
    files = expert_runtime->BankedDmaBindings("current_packed_bank_");
  } else if (zero_copy_) {
    if (!expert_runtime->PrimeZeroCopyInputs(session, SetName(current_layer), "current_expert_", current_layer, &files,
                                             error)) {
      return false;
    }
  } else if (packed_) {
    files = expert_runtime->PackedDmaBindings(current_layer, "current_packed_weights");
  } else {
    files = expert_runtime->FusedDmaBindings(current_layer, "current_expert_", indexed_ ? 16 : 8);
  }
  ShareCurrentInputs(inputs, files, expert_runtime, &shared);
  const TensorShapes &weights = full ? kFullWeights : kLinearWeights;
  for (const auto &item : weights) {
    external["next_" + item.first] = resident->InputHandle(next_layer, 1, item.first);
  }
  external["next_output_zero"] = resident->InputHandle(next_layer, 1, "output_zero");
  std::unordered_map<std::string, std::string> aliases;
  if (full) {
    for (const char *name : {"rope_cos", "rope_sin", "kvcache_mask", "past_key", "past_value"}) {
      external["next_" + std::string(name)] = resident->InputHandle(next_layer, 1, name);
    }
    aliases["next_present_key"] = "next_past_key";
    aliases["next_present_value"] = "next_past_value";
  } else {
    external["next_past_conv_state"] = resident->InputHandle(next_layer, 1, "past_conv_state");
    external["next_past_recurrent_state_bits"] = resident->InputHandle(next_layer, 1, "past_recurrent_state_bits");
    aliases["next_present_conv_state"] = "next_past_conv_state";
    aliases["next_present_recurrent_state_bits"] = "next_past_recurrent_state_bits";
  }
  const auto missing =
    std::find_if(external.begin(), external.end(), [](const auto &item) { return item.second == nullptr; });
  if (missing != external.end()) {
    if (error) *error = "missing shared resident tensor: " + missing->first;
    return false;
  }
  return session->CreateTensorSet(SetName(current_layer), inputs, OutputShapes(full), aliases, shared, external, files,
                                  error);
}

bool Qwen35FusedDecodeRuntime::RunBoundary(int32_t current_layer, const std::array<int32_t, 8> &experts,
                                           const std::array<uint16_t, 8> &route_weights, const TensorView &expert_input,
                                           const TensorView &attention_residual, const TensorView &shared_expert_output,
                                           const uint16_t *rope_cos, const uint16_t *rope_sin,
                                           const uint16_t *kvcache_mask, Qwen35ExpertRuntime *expert_runtime,
                                           Qwen35ResidentOutputs *outputs, double *elapsed_ms, std::string *error) {
  if (!enabled_ || current_layer < 0 || current_layer >= 39 || expert_runtime == nullptr || outputs == nullptr) {
    if (error) *error = "invalid fused decoder boundary";
    return false;
  }
  const bool full = Qwen35ResidentRuntime::IsFullLayer(current_layer + 1);
  NnrtSession *session = full ? &full_ : &linear_;
  const std::string set = SetName(current_layer);
  const bool perf_stats = std::getenv("QWEN35_PERF_STATS") != nullptr;
  const auto bind_start = std::chrono::steady_clock::now();
  std::array<int32_t, 8> route_to_slot{};
  const bool bound = BindExperts(session, set, current_layer, experts, &route_to_slot, expert_runtime, error);
  if (!bound || !CopyActivations(session, set, {expert_input, attention_residual, shared_expert_output}, error)) {
    return false;
  }
  const auto bind_end = std::chrono::steady_clock::now();
  if (!CopyRouteWeights(session, set, route_weights, route_to_slot, error)) return false;
  if (full && !CopyNextAttention(session, set, rope_cos, rope_sin, kvcache_mask, error)) return false;
  const auto copy_end = std::chrono::steady_clock::now();
  double run_ms = 0.0;
  if (!session->Run(set, &run_ms, error)) return false;
  const auto run_end = std::chrono::steady_clock::now();
  if (elapsed_ms != nullptr) *elapsed_ms = run_ms;
  if (perf_stats) {
    const double bind_ms = std::chrono::duration<double, std::milli>(bind_end - bind_start).count();
    const double copy_ms = std::chrono::duration<double, std::milli>(copy_end - bind_end).count();
    const double run_wall_ms = std::chrono::duration<double, std::milli>(run_end - copy_end).count();
    std::cerr << "qwen35_boundary_ms layer=" << current_layer << " bind=" << bind_ms << " copy=" << copy_ms
              << " run=" << run_wall_ms << '\n';
  }
  outputs->attention_residual = session->Output(set, "next_attention_residual");
  outputs->expert_input = session->Output(set, "next_expert_input");
  outputs->router_logits = session->Output(set, "next_router_logits");
  outputs->shared_expert_output = session->Output(set, "next_shared_expert_output");
  return outputs->attention_residual.data != nullptr && outputs->expert_input.data != nullptr &&
         outputs->router_logits.data != nullptr && outputs->shared_expert_output.data != nullptr;
}

std::string Qwen35FusedDecodeRuntime::SetName(int32_t current_layer) {
  return "boundary_" + std::to_string(current_layer);
}

Qwen35FusedDecodeRuntime::ModelChoice Qwen35FusedDecodeRuntime::ChooseModels(const std::string &artifact_root,
                                                                             Qwen35ExpertRuntime *expert_runtime) {
  ModelChoice choice;
  const fs::path root = fs::path(artifact_root) / "omc/fused_decode";
  const fs::path linear = root / "qwen35_fused_expert_resident_linear_s1.omc";
  const fs::path full = root / "qwen35_fused_expert_resident_full_s1.omc";
  const fs::path indexed_root = fs::path(artifact_root) / "omc/fused_decode_indexed";
  const fs::path indexed_linear = indexed_root / "qwen35_fused_expert_resident_indexed_linear_s1.omc";
  const fs::path indexed_full = indexed_root / "qwen35_fused_expert_resident_indexed_full_s1.omc";
  choice.packed_slots = static_cast<int32_t>(expert_runtime == nullptr ? 0 : expert_runtime->PackedExpertSlots());
  const fs::path packed_root =
    fs::path(artifact_root) / ("omc/fused_decode_packed" + std::to_string(choice.packed_slots));
  const fs::path packed_linear = packed_root / "qwen35_fused_expert_resident_packed_linear_s1.omc";
  const fs::path packed_full = packed_root / "qwen35_fused_expert_resident_packed_full_s1.omc";
  const fs::path zero_copy_root = fs::path(artifact_root) / "omc/fused_decode_zero_copy";
  const fs::path zero_copy_linear = zero_copy_root / "qwen35_fused_expert_resident_zero_copy_linear_s1.omc";
  const fs::path zero_copy_full = zero_copy_root / "qwen35_fused_expert_resident_zero_copy_full_s1.omc";
  const fs::path banked_root = fs::path(artifact_root) / "omc/fused_decode_banked";
  const fs::path banked_linear = banked_root / "qwen35_fused_expert_resident_banked_linear_s1.omc";
  const fs::path banked_full = banked_root / "qwen35_fused_expert_resident_banked_full_s1.omc";
  const bool indexed_requested = std::getenv("QWEN35_DISABLE_INDEXED_EXPERTS") == nullptr;
  choice.indexed_available = indexed_requested && ModelPairExists(indexed_linear, indexed_full);
  const char *packed_value = std::getenv("QWEN35_ENABLE_PACKED_EXPERTS");
  const bool packed_requested = packed_value == nullptr || std::string(packed_value) != "0";
  choice.packed_available = packed_requested && expert_runtime != nullptr && expert_runtime->PackedExpertsEnabled() &&
                            ModelPairExists(packed_linear, packed_full);
  choice.zero_copy_available = false;
  choice.banked_available = EnvironmentIsOne("QWEN35_ENABLE_BANKED_EXPERTS") && expert_runtime != nullptr &&
                            expert_runtime->BankedExpertsEnabled() && ModelPairExists(banked_linear, banked_full);
  choice.linear_model = linear.string();
  choice.full_model = full.string();
  if (choice.indexed_available) {
    choice.linear_model = indexed_linear.string();
    choice.full_model = indexed_full.string();
  }
  if (choice.packed_available) {
    choice.linear_model = packed_linear.string();
    choice.full_model = packed_full.string();
  }
  if (choice.zero_copy_available) {
    choice.linear_model = zero_copy_linear.string();
    choice.full_model = zero_copy_full.string();
  }
  if (choice.banked_available) {
    choice.linear_model = banked_linear.string();
    choice.full_model = banked_full.string();
  }
  return choice;
}

bool Qwen35FusedDecodeRuntime::IsExpertWeight(const std::string &name) {
  const std::string prefix = "current_expert_";
  const bool has_prefix = name.rfind(prefix, 0) == 0;
  size_t digit = prefix.size();
  while (has_prefix && digit < name.size() && name[digit] >= '0' && name[digit] <= '9') {
    ++digit;
  }
  return has_prefix && digit > prefix.size() && digit < name.size() && name[digit] == '_';
}

bool Qwen35FusedDecodeRuntime::BindExperts(NnrtSession *session, const std::string &set, int32_t current_layer,
                                           const std::array<int32_t, 8> &experts, std::array<int32_t, 8> *route_to_slot,
                                           Qwen35ExpertRuntime *expert_runtime, std::string *error) {
  bool bound = false;
  if (banked_) {
    bound = expert_runtime->BindBankedInputs(session, set, current_layer, experts, route_to_slot, error);
    if (bound && !Copy(session->Input(set, "current_expert_ids"), route_to_slot->data(), sizeof(*route_to_slot),
                       "expert_ids", error))
      return false;
  } else if (zero_copy_) {
    bound =
      expert_runtime->BindZeroCopyInputs(session, set, "current_expert_", current_layer, experts, route_to_slot, error);
  } else if (packed_) {
    bound = expert_runtime->BindPackedInputs(session, set, current_layer, experts, route_to_slot, error);
    if (bound && !Copy(session->Input(set, "current_expert_ids"), route_to_slot->data(), sizeof(*route_to_slot),
                       "expert_ids", error)) {
      return false;
    }
  } else if (indexed_) {
    bound =
      expert_runtime->BindIndexedInputs(session, set, "current_expert_", current_layer, experts, route_to_slot, error);
    if (bound && !Copy(session->Input(set, "current_expert_ids"), route_to_slot->data(), sizeof(*route_to_slot),
                       "expert_ids", error)) {
      return false;
    }
  } else {
    bound =
      expert_runtime->BindGroupedInputs(session, set, "current_expert_", current_layer, experts, route_to_slot, error);
  }
  return bound;
}

void Qwen35FusedDecodeRuntime::ShareCurrentInputs(const TensorShapes &inputs, const FileInputBindings &files,
                                                  Qwen35ExpertRuntime *expert_runtime, SharedInputKeys *shared) {
  for (const auto &item : inputs) {
    if (item.first.rfind("current_", 0) != 0) continue;
    const bool expert_weight = IsExpertWeight(item.first);
    const bool private_nnrt_packed =
      packed_ && item.first == "current_packed_weights" && files.empty() && !expert_runtime->PackedNnrtSharedEnabled();
    if (!private_nnrt_packed && files.find(item.first) == files.end() && (!expert_weight || files.empty()))
      (*shared)[item.first] = item.first;
    if (zero_copy_ && expert_weight) (*shared)[item.first] = item.first;
    if (banked_ && item.first.rfind("current_packed_bank_", 0) == 0) (*shared)[item.first] = item.first;
  }
}

bool Qwen35FusedDecodeRuntime::CopyRouteWeights(NnrtSession *session, const std::string &set,
                                                const std::array<uint16_t, 8> &route_weights,
                                                const std::array<int32_t, 8> &route_to_slot, std::string *error) {
  // Indexed and packed GroupedTop8 write route-order outputs. The zero-copy
  // graph has eight independent physical shard branches, so its concat output
  // is physical-slot order and requires the matching weight permutation.
  std::array<uint16_t, 8 * 64> repeated_weights{};
  for (size_t route = 0; route < route_weights.size(); ++route) {
    const size_t output_slot = zero_copy_ ? static_cast<size_t>(route_to_slot[route]) : route;
    if (output_slot >= route_weights.size()) {
      if (error) *error = "invalid zero-copy expert route mapping";
      return false;
    }
    for (size_t lane = 0; lane < 64; ++lane) {
      repeated_weights[output_slot * 64 + lane] = route_weights[route];
    }
  }
  if (!Copy(session->Input(set, "current_route_weights"), repeated_weights.data(), sizeof(repeated_weights),
            "route_weights", error)) {
    return false;
  }
  return true;
}

bool Qwen35FusedDecodeRuntime::LoadSessions(const ModelChoice &choice, Qwen35ExpertRuntime *expert_runtime,
                                            std::string *error) {
  const auto &linear_model = choice.linear_model;
  const auto &full_model = choice.full_model;
  const auto packed_slots = choice.packed_slots;
  const bool indexed_available = choice.indexed_available;
  const bool packed_available = choice.packed_available;
  const bool zero_copy_available = choice.zero_copy_available;
  const bool banked_available = choice.banked_available;
  if (expert_runtime == nullptr ||
      (!zero_copy_available && !banked_available &&
       !expert_runtime->PrepareFusedDmaCache(
         (packed_available || indexed_available) ? 40 : 39,
         packed_available ? static_cast<size_t>(packed_slots) : (indexed_available ? 16 : 8), error)) ||
      !linear_.Load(linear_model, "", error) || !full_.Load(full_model, "", error)) {
    return false;
  }
  return true;
}

bool Qwen35FusedDecodeRuntime::CopyNextAttention(NnrtSession *session, const std::string &set, const uint16_t *rope_cos,
                                                 const uint16_t *rope_sin, const uint16_t *kvcache_mask,
                                                 std::string *error) {
  if ((!Copy(session->Input(set, "next_rope_cos"), rope_cos, 64 * sizeof(uint16_t), "rope_cos", error) ||
       !Copy(session->Input(set, "next_rope_sin"), rope_sin, 64 * sizeof(uint16_t), "rope_sin", error) ||
       !Copy(session->Input(set, "next_kvcache_mask"), kvcache_mask, kCache * sizeof(uint16_t), "kvcache_mask",
             error))) {
    return false;
  }
  return true;
}

bool Qwen35FusedDecodeRuntime::CopyActivations(NnrtSession *session, const std::string &set,
                                               const std::array<TensorView, 3> &activations, std::string *error) {
  if (!Copy(session->Input(set, "current_expert_input"), activations[0].data, activations[0].bytes, "expert_input",
            error) ||
      !Copy(session->Input(set, "current_attention_residual"), activations[1].data, activations[1].bytes,
            "attention_residual", error) ||
      !Copy(session->Input(set, "current_shared_expert_output"), activations[2].data, activations[2].bytes,
            "shared_expert_output", error)) {
    return false;
  }
  return true;
}

}  // namespace mslite_llm::qwen35
