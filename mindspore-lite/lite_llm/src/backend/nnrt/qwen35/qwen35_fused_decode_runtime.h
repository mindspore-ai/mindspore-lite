// Copyright 2026 Huawei Technologies Co., Ltd
// Adapted from HMOS-Device-AI llm/engine/runtime; preserves its paged Qwen3.5 execution.
#pragma once

#include <array>
#include <cstdint>
#include <string>

#include "backend/nnrt/qwen35/nnrt_session.h"
#include "backend/nnrt/qwen35/qwen35_resident_runtime.h"

namespace mslite_llm::qwen35 {

class Qwen35ExpertRuntime;

class Qwen35FusedDecodeRuntime {
 public:
  bool Load(const std::string &artifact_root, Qwen35ResidentRuntime *resident, Qwen35ExpertRuntime *expert_runtime,
            std::string *error);
  bool Enabled() const { return enabled_; }
  bool RunBoundary(int32_t current_layer, const std::array<int32_t, 8> &experts,
                   const std::array<uint16_t, 8> &route_weights, const TensorView &expert_input,
                   const TensorView &attention_residual, const TensorView &shared_expert_output,
                   const uint16_t *rope_cos, const uint16_t *rope_sin, const uint16_t *kvcache_mask,
                   Qwen35ExpertRuntime *expert_runtime, Qwen35ResidentOutputs *outputs, double *elapsed_ms,
                   std::string *error);

 private:
  struct ModelChoice {
    std::string linear_model;
    std::string full_model;
    int32_t packed_slots = 0;
    bool indexed_available = false;
    bool packed_available = false;
    bool zero_copy_available = false;
    bool banked_available = false;
  };

  bool CopyActivations(NnrtSession *session, const std::string &set, const std::array<TensorView, 3> &activations,
                       std::string *error);

  bool CopyNextAttention(NnrtSession *session, const std::string &set, const uint16_t *rope_cos,
                         const uint16_t *rope_sin, const uint16_t *kvcache_mask, std::string *error);
  bool LoadSessions(const ModelChoice &choice, Qwen35ExpertRuntime *expert_runtime, std::string *error);
  bool CopyRouteWeights(NnrtSession *session, const std::string &set, const std::array<uint16_t, 8> &route_weights,
                        const std::array<int32_t, 8> &route_to_slot, std::string *error);
  void ShareCurrentInputs(const TensorShapes &inputs, const FileInputBindings &files,
                          Qwen35ExpertRuntime *expert_runtime, SharedInputKeys *shared);

  bool BindExperts(NnrtSession *session, const std::string &set, int32_t current_layer,
                   const std::array<int32_t, 8> &experts, std::array<int32_t, 8> *route_to_slot,
                   Qwen35ExpertRuntime *expert_runtime, std::string *error);
  bool IsExpertWeight(const std::string &name);
  Qwen35FusedDecodeRuntime::ModelChoice ChooseModels(const std::string &artifact_root,
                                                     Qwen35ExpertRuntime *expert_runtime);

  bool CreateBoundary(int32_t current_layer, Qwen35ResidentRuntime *resident, Qwen35ExpertRuntime *expert_runtime,
                      std::string *error);
  static std::string SetName(int32_t current_layer);

  NnrtSession linear_;
  NnrtSession full_;
  bool enabled_ = false;
  bool indexed_ = false;
  bool packed_ = false;
  bool zero_copy_ = false;
  bool banked_ = false;
  int32_t packed_slots_ = 0;
};

}  // namespace mslite_llm::qwen35
