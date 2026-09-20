// Copyright 2026 Huawei Technologies Co., Ltd
// Adapted from HMOS-Device-AI llm/engine/runtime; preserves its paged Qwen3.5 execution.
#include "backend/nnrt/qwen35/qwen35_resident_runtime.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17): lite_llm requires C++17.
#include <fstream>
#include <iomanip>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "manifest/artifact_file.h"

namespace fs = std::filesystem;

namespace mslite_llm::qwen35 {
namespace {

constexpr int32_t kLayers = 40;
constexpr int32_t kHidden = 2048;
constexpr int32_t kCache = 8192;

const TensorShapes kLinearWeightShapes = {
  {"in_norm", {2048}},
  {"post_norm", {2048}},
  {"qkv_w", {8192, 2048}},
  {"z_w", {4096, 2048}},
  {"a_w", {32, 2048}},
  {"b_w", {32, 2048}},
  {"conv_w", {8192, 1, 4}},
  {"dt_bias", {32}},
  {"a_log_bits", {64}},
  {"gated_norm", {128}},
  {"out_w", {2048, 4096}},
  {"router_w", {256, 2048}},
  {"shared_gate_w", {512, 2048}},
  {"shared_up_w", {512, 2048}},
  {"shared_down_w", {2048, 512}},
  {"shared_score_w", {16, 2048}},
};

const TensorShapes kFullWeightShapes = {
  {"in_norm", {2048}},
  {"post_norm", {2048}},
  {"q_w", {8192, 2048}},
  {"k_w", {512, 2048}},
  {"v_w", {512, 2048}},
  {"o_w", {2048, 4096}},
  {"q_norm", {256}},
  {"k_norm", {256}},
  {"router_w", {256, 2048}},
  {"shared_gate_w", {512, 2048}},
  {"shared_up_w", {512, 2048}},
  {"shared_down_w", {2048, 512}},
  {"shared_score_w", {16, 2048}},
};

const TensorShapes kLinearFusedWeightShapes = {
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

const TensorShapes kFullFusedWeightShapes = {
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

const std::unordered_map<std::string, std::string> kLinearAliases = {
  {"present_conv_state", "past_conv_state"},
  {"present_recurrent_state_bits", "past_recurrent_state_bits"},
};

const std::unordered_map<std::string, std::string> kFullAliases = {
  {"present_key", "past_key"},
  {"present_value", "past_value"},
};

const TensorShapes &WeightShapes(bool full, bool fused) {
  if (full) return fused ? kFullFusedWeightShapes : kFullWeightShapes;
  return fused ? kLinearFusedWeightShapes : kLinearWeightShapes;
}

TensorShapes InputShapes(bool full, int32_t sequence, bool fused) {
  TensorShapes shapes = WeightShapes(full, fused);
  shapes["hidden_states"] = {1, sequence, kHidden};
  shapes["output_zero"] = {1};
  if (full) {
    shapes["rope_cos"] = {1, sequence, 64};
    shapes["rope_sin"] = {1, sequence, 64};
    shapes["kvcache_mask"] = {1, 1, sequence, kCache};
    shapes["past_key"] = {1, 2, kCache, 256};
    shapes["past_value"] = {1, 2, kCache, 256};
  } else {
    shapes["past_conv_state"] = {1, 8192, 4};
    shapes["past_recurrent_state_bits"] = {1, 32, 128, 256};
  }
  return shapes;
}

TensorShapes OutputShapes(bool full, int32_t sequence) {
  TensorShapes shapes = {
    {"attention_residual", {1, sequence, kHidden}},
    {"expert_input", {1, sequence, kHidden}},
    {"router_logits", {1, sequence, 256}},
    {"shared_expert_output", {1, sequence, kHidden}},
  };
  if (full) {
    shapes["present_key"] = {1, 2, kCache, 256};
    shapes["present_value"] = {1, 2, kCache, 256};
  } else {
    shapes["present_conv_state"] = {1, 8192, 4};
    shapes["present_recurrent_state_bits"] = {1, 32, 128, 256};
  }
  return shapes;
}

std::string LayerPrefix(int32_t layer) {
  std::ostringstream stream;
  stream << "layer_" << std::setw(2) << std::setfill('0') << layer;
  return stream.str();
}

SharedInputKeys SharedKeys(bool full, int32_t layer, int32_t sequence, bool fused) {
  const TensorShapes &weights = WeightShapes(full, fused);
  SharedInputKeys keys;
  const std::string layer_prefix = LayerPrefix(layer) + "/";
  for (const auto &item : weights) keys[item.first] = layer_prefix + item.first;
  keys["hidden_states"] = std::string(full ? "full" : "linear") + "/s" + std::to_string(sequence) + "/hidden_states";
  keys["output_zero"] = std::string(full ? "full" : "linear") + "/output_zero";
  if (full) {
    keys["rope_cos"] = "full/s" + std::to_string(sequence) + "/rope_cos";
    keys["rope_sin"] = "full/s" + std::to_string(sequence) + "/rope_sin";
    keys["kvcache_mask"] = "full/s" + std::to_string(sequence) + "/kvcache_mask";
    keys["past_key"] = layer_prefix + "past_key";
    keys["past_value"] = layer_prefix + "past_value";
  } else {
    keys["past_conv_state"] = layer_prefix + "past_conv_state";
    keys["past_recurrent_state_bits"] = layer_prefix + "past_recurrent_state_bits";
  }
  return keys;
}

bool CopyFile(const fs::path &path, TensorView view, std::string *error) {
  return view.data != nullptr && ReadArtifact(path, view.data, view.bytes, error);
}

bool CopyMemory(TensorView target, const void *source, size_t bytes, const char *name, std::string *error) {
  if (target.data == nullptr || source == nullptr || target.bytes != bytes) {
    if (error) {
      *error = std::string("resident input size mismatch: ") + name;
    }
    return false;
  }
  std::memcpy(target.data, source, bytes);
  return true;
}

}  // namespace

bool Qwen35ResidentRuntime::Load(const std::string &artifact_root, std::string *error) {
  if (loaded_) {
    if (error) *error = "Qwen35 resident runtime is already loaded";
    return false;
  }
  const fs::path root(artifact_root);
  // Candidate builds are explicitly marked and live beside the accepted OMC.
  // The default path and its input ABI remain unchanged.
  projection_fused_ = ArtifactExists(root / "omc/resident_projection_fused/READY");
  const fs::path resident_root = root / (projection_fused_ ? "omc/resident_projection_fused" : "omc/resident");
  const fs::path linear_dir = resident_root / "qwen35_resident_linear_s1_s64";
  const fs::path full_dir = resident_root / "qwen35_resident_full_s1_s64";
  if (!linear_.Load((linear_dir / "qwen35_resident_linear_s1_s64.omc").string(),
                    (linear_dir / "SubGraph_0.weight").string(), error) ||
      !full_.Load((full_dir / "qwen35_resident_full_s1_s64.omc").string(), (full_dir / "SubGraph_0.weight").string(),
                  error)) {
    return false;
  }
  for (int32_t layer = 0; layer < kLayers; ++layer) {
    if (!CreateLayer(layer, IsFull(layer), (root / "weights" / LayerPrefix(layer)).string(), error)) {
      return false;
    }
  }
  loaded_ = true;
  return ResetCaches(error);
}

bool Qwen35ResidentRuntime::CreateLayer(int32_t layer, bool full, const std::string &weight_dir, std::string *error) {
  NnrtSession *session = Session(full);
  const auto &aliases = full ? kFullAliases : kLinearAliases;
  const std::array<int32_t, 2> sequences = {1, 64};
  const bool tensor_failure =
    std::any_of(sequences.begin(), sequences.end(), [this, session, layer, full, &aliases, error](int32_t sequence) {
      return !session->CreateTensorSet(SetName(layer, sequence), InputShapes(full, sequence, projection_fused_),
                                       OutputShapes(full, sequence), aliases,
                                       SharedKeys(full, layer, sequence, projection_fused_), error);
    });
  if (tensor_failure) return false;
  const std::string set = SetName(layer, 1);
  const TensorShapes &weights = WeightShapes(full, projection_fused_);
  const bool weight_failure =
    std::any_of(weights.begin(), weights.end(), [session, &set, &weight_dir, error](const auto &item) {
      return !CopyFile(fs::path(weight_dir) / ("weight_" + item.first + ".bin"), session->Input(set, item.first),
                       error);
    });
  if (weight_failure) return false;
  TensorView zero = session->Input(set, "output_zero");
  if (zero.data == nullptr || zero.bytes != sizeof(uint16_t)) {
    if (error) *error = "invalid output_zero tensor";
    return false;
  }
  std::memset(zero.data, 0, zero.bytes);
  return true;
}

bool Qwen35ResidentRuntime::RunLayer(int32_t layer, int32_t sequence, const uint16_t *hidden, const uint16_t *rope_cos,
                                     const uint16_t *rope_sin, const uint16_t *kvcache_mask,
                                     Qwen35ResidentOutputs *outputs, double *elapsed_ms, std::string *error) {
  if (!loaded_ || layer < 0 || layer >= kLayers || (sequence != 1 && sequence != 64) || outputs == nullptr) {
    if (error) *error = "invalid Qwen35 resident layer request";
    return false;
  }
  const bool full = IsFull(layer);
  NnrtSession *session = Session(full);
  const std::string set = SetName(layer, sequence);
  if (!CopyMemory(session->Input(set, "hidden_states"), hidden,
                  static_cast<size_t>(sequence) * kHidden * sizeof(uint16_t), "hidden_states", error)) {
    return false;
  }
  if (full && (!CopyMemory(session->Input(set, "rope_cos"), rope_cos,
                           static_cast<size_t>(sequence) * 64 * sizeof(uint16_t), "rope_cos", error) ||
               !CopyMemory(session->Input(set, "rope_sin"), rope_sin,
                           static_cast<size_t>(sequence) * 64 * sizeof(uint16_t), "rope_sin", error) ||
               !CopyMemory(session->Input(set, "kvcache_mask"), kvcache_mask,
                           static_cast<size_t>(sequence) * kCache * sizeof(uint16_t), "kvcache_mask", error))) {
    return false;
  }
  if (!session->Run(set, elapsed_ms, error)) return false;
  outputs->attention_residual = session->Output(set, "attention_residual");
  outputs->expert_input = session->Output(set, "expert_input");
  outputs->router_logits = session->Output(set, "router_logits");
  outputs->shared_expert_output = session->Output(set, "shared_expert_output");
  return outputs->attention_residual.data != nullptr && outputs->expert_input.data != nullptr &&
         outputs->router_logits.data != nullptr && outputs->shared_expert_output.data != nullptr;
}

bool Qwen35ResidentRuntime::ResetCaches(std::string *error) {
  if (!loaded_) return true;
  for (int32_t layer = 0; layer < kLayers; ++layer) {
    NnrtSession *session = Session(IsFull(layer));
    const std::string set = SetName(layer, 1);
    for (const char *name : IsFull(layer) ? std::vector<const char *>{"past_key", "past_value"}
                                          : std::vector<const char *>{"past_conv_state", "past_recurrent_state_bits"}) {
      TensorView view = session->Input(set, name);
      if (view.data == nullptr) {
        if (error) *error = std::string("missing resident cache: ") + name;
        return false;
      }
      std::memset(view.data, 0, view.bytes);
    }
  }
  return true;
}

bool Qwen35ResidentRuntime::SaveCaches(std::vector<std::vector<uint8_t>> *buffers, std::string *error) const {
  if (!loaded_ || buffers == nullptr) {
    if (error) *error = "resident cache snapshot is unavailable";
    return false;
  }
  buffers->clear();
  buffers->reserve(kLayers * 2);
  for (int32_t layer = 0; layer < kLayers; ++layer) {
    const NnrtSession *session = Session(IsFull(layer));
    const std::string set = SetName(layer, 1);
    for (const char *name : IsFull(layer) ? std::vector<const char *>{"past_key", "past_value"}
                                          : std::vector<const char *>{"past_conv_state", "past_recurrent_state_bits"}) {
      TensorView view = session->Input(set, name);
      if (view.data == nullptr) return false;
      const auto *begin = static_cast<const uint8_t *>(view.data);
      buffers->emplace_back(begin, begin + view.bytes);
    }
  }
  return true;
}

bool Qwen35ResidentRuntime::RestoreCaches(const std::vector<std::vector<uint8_t>> &buffers, std::string *error) {
  if (!loaded_ || buffers.size() != kLayers * 2) {
    if (error) *error = "resident cache snapshot count mismatch";
    return false;
  }
  size_t index = 0;
  for (int32_t layer = 0; layer < kLayers; ++layer) {
    NnrtSession *session = Session(IsFull(layer));
    const std::string set = SetName(layer, 1);
    for (const char *name : IsFull(layer) ? std::vector<const char *>{"past_key", "past_value"}
                                          : std::vector<const char *>{"past_conv_state", "past_recurrent_state_bits"}) {
      TensorView view = session->Input(set, name);
      if (view.data == nullptr || view.bytes != buffers[index].size()) {
        if (error) *error = "resident cache snapshot size mismatch";
        return false;
      }
      std::memcpy(view.data, buffers[index].data(), view.bytes);
      ++index;
    }
  }
  return true;
}

TensorView Qwen35ResidentRuntime::InputView(int32_t layer, int32_t sequence, const std::string &name) const {
  if (!loaded_ || layer < 0 || layer >= kLayers || (sequence != 1 && sequence != 64)) {
    return {};
  }
  return Session(IsFull(layer))->Input(SetName(layer, sequence), name);
}

void *Qwen35ResidentRuntime::InputHandle(int32_t layer, int32_t sequence, const std::string &name) const {
  if (!loaded_ || layer < 0 || layer >= kLayers || (sequence != 1 && sequence != 64)) {
    return nullptr;
  }
  return Session(IsFull(layer))->InputHandle(SetName(layer, sequence), name);
}

NnrtSession *Qwen35ResidentRuntime::Session(bool full) { return full ? &full_ : &linear_; }

const NnrtSession *Qwen35ResidentRuntime::Session(bool full) const { return full ? &full_ : &linear_; }

std::string Qwen35ResidentRuntime::SetName(int32_t layer, int32_t sequence) {
  return LayerPrefix(layer) + "_s" + std::to_string(sequence);
}

}  // namespace mslite_llm::qwen35
