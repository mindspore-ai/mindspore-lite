// Copyright 2026 Huawei Technologies Co., Ltd
// Adapted from HMOS-Device-AI llm/engine/runtime; preserves its paged Qwen3.5 execution.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "backend/nnrt/qwen35/nnrt_session.h"

namespace mslite_llm::qwen35 {

struct Qwen35ResidentOutputs {
  TensorView attention_residual;
  TensorView expert_input;
  TensorView router_logits;
  TensorView shared_expert_output;
};

class Qwen35ResidentRuntime {
 public:
  Qwen35ResidentRuntime() = default;
  ~Qwen35ResidentRuntime() = default;
  Qwen35ResidentRuntime(const Qwen35ResidentRuntime &) = delete;
  Qwen35ResidentRuntime &operator=(const Qwen35ResidentRuntime &) = delete;

  bool Load(const std::string &artifact_root, std::string *error);
  bool RunLayer(int32_t layer, int32_t sequence, const uint16_t *hidden, const uint16_t *rope_cos,
                const uint16_t *rope_sin, const uint16_t *kvcache_mask, Qwen35ResidentOutputs *outputs,
                double *elapsed_ms, std::string *error);
  bool ResetCaches(std::string *error);
  bool SaveCaches(std::vector<std::vector<uint8_t>> *buffers, std::string *error) const;
  bool RestoreCaches(const std::vector<std::vector<uint8_t>> &buffers, std::string *error);
  TensorView InputView(int32_t layer, int32_t sequence, const std::string &name) const;
  void *InputHandle(int32_t layer, int32_t sequence, const std::string &name) const;
  static bool IsFullLayer(int32_t layer) { return layer % 4 == 3; }

 private:
  bool CreateLayer(int32_t layer, bool full, const std::string &weight_dir, std::string *error);
  NnrtSession *Session(bool full);
  const NnrtSession *Session(bool full) const;
  static bool IsFull(int32_t layer) { return IsFullLayer(layer); }
  static std::string SetName(int32_t layer, int32_t sequence);

  NnrtSession linear_;
  NnrtSession full_;
  bool loaded_ = false;
  bool projection_fused_ = false;
};

}  // namespace mslite_llm::qwen35
