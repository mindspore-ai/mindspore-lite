// Copyright 2026 Huawei Technologies Co., Ltd
// Adapted from HMOS-Device-AI llm/engine/runtime; preserves its paged Qwen3.5 execution.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "backend/nnrt/qwen35/model_runtime.h"
#include "backend/nnrt/qwen35/qwen35_expert_runtime.h"
#include "backend/nnrt/qwen35/qwen35_fused_decode_runtime.h"
#include "backend/nnrt/qwen35/qwen35_lm_head.h"
#include "backend/nnrt/qwen35/qwen35_resident_runtime.h"

namespace mslite_llm::qwen35 {

// Runtime adapter for the paged Qwen3.5-35B-A3B GPTQ model. Resident
// attention/shared-expert weights stay in the two resident OMC executors;
// routed expert shards are loaded into the reusable expert OMC inputs on
// demand.
class Qwen35MoeModel final : public ModelRuntime {
 public:
  Qwen35MoeModel() = default;
  ~Qwen35MoeModel() override = default;
  Qwen35MoeModel(const Qwen35MoeModel &) = delete;
  Qwen35MoeModel &operator=(const Qwen35MoeModel &) = delete;

  bool Load(const std::string &artifact_root, size_t expert_cache_shards, std::string *error);

  const ModelMetadata &Metadata() const override { return metadata_; }
  bool HasHiddenStates() const override { return false; }
  bool HasCheckpointStates() const override { return false; }
  int32_t CacheLength() const override { return metadata_.max_cache_length; }
  bool ResetCache(std::string *error) override;
  bool SaveCache(CacheSnapshot *snapshot, std::string *error) const override;
  bool RestoreCache(const CacheSnapshot &snapshot, std::string *error) override;
  bool PrepareGear(ModelGear gear, std::string *error) override;
  TensorBuffer Input(ModelGear gear, const std::string &name) const override;
  TensorBuffer Logits(ModelGear gear) const override;
  TensorBuffer TokenIds(ModelGear gear) const override;
  TensorBuffer HiddenStates(ModelGear gear) const override;
  bool Run(ModelGear gear, double *elapsed_ms, std::string *error, bool defer_cache_commit = false) override;
  bool CommitCache(std::string *error) override;
  bool CommitCheckpointCache(std::string *error) override;
  bool DiscardCache(std::string *error) override;

 private:
  struct Gear {
    int32_t sequence = 0;
    std::vector<int32_t> lmhead_idx;
    std::vector<uint16_t> kvcache_mask;
    std::vector<uint16_t> rope_cos;
    std::vector<uint16_t> rope_sin;
    std::vector<uint16_t> inputs_embeds;
    std::vector<uint16_t> attention_mask;
    std::vector<uint16_t> logits;
  };

  struct Route {
    int32_t token = 0;
    int32_t slot = 0;
    float weight = 0.0f;
  };

  struct ExpertExecution {
    ExpertExecution();
    double total = 0.0;
    double resident_total = 0.0;
    double expert_total = 0.0;
    std::vector<std::vector<Route>> routes;
    std::vector<uint16_t> expert_input;
    std::vector<uint16_t> gated;
    std::vector<uint16_t> expert_output;
  };

  struct FusedExecution {
    FusedExecution();
    double total = 0.0;
    double resident_total = 0.0;
    double fused_total = 0.0;
    double expert_total = 0.0;
    double resident_wall = 0.0;
    double route_build_wall = 0.0;
    double route_select_wall = 0.0;
    double boundary_wall = 0.0;
    double final_expert_wall = 0.0;
    double lm_head_wall = 0.0;
    std::vector<std::vector<Route>> routes;
    std::array<int32_t, 8> selected_experts{};
    std::array<uint16_t, 8> selected_weights{};
    std::vector<uint16_t> final_expert_output;
    Qwen35ResidentOutputs resident_outputs;
  };

  bool RunFinalExperts(FusedExecution &execution, std::string *error);
  bool SelectFusedRoutes(int32_t layer, const TensorView &router_logits, FusedExecution &execution, std::string *error);

  bool RunRoutedExperts(int32_t layer, const uint16_t *expert_source, ExpertExecution &execution, std::string *error);
  bool RunGroupedExperts(int32_t layer, const uint16_t *expert_source, ExpertExecution &execution, std::string *error);
  bool RunLayerExperts(Gear *gear, int32_t layer, ExpertExecution &execution, std::string *error);

  static bool BuildTokenRoutes(const uint16_t *values, int32_t token, std::vector<std::vector<Route>> *routes,
                               std::string *error);

  static Gear *SelectGear(ModelGear gear, Gear *decode, Gear *prefill);
  static const Gear *SelectGear(ModelGear gear, const Gear *decode, const Gear *prefill);
  static bool IsActiveGear(ModelGear gear);
  static bool BuildRoutes(const TensorView &router_logits, const std::vector<uint16_t> &attention_mask,
                          int32_t sequence, std::vector<std::vector<Route>> *routes, std::string *error);
  bool RunGear(Gear *gear, double *elapsed_ms, std::string *error);
  bool RunFusedDecode(Gear *gear, double *elapsed_ms, std::string *error);

  ModelMetadata metadata_;
  std::string artifact_root_;
  Qwen35ResidentRuntime resident_;
  Qwen35ExpertRuntime experts_;
  Qwen35FusedDecodeRuntime fused_decode_;
  Qwen35LmHead lm_head_;
  Gear decode_;
  Gear prefill_;
  Gear *active_ = nullptr;
  std::vector<uint16_t> hidden_;
  std::vector<float> combined_;
  std::array<std::vector<int32_t>, 40> route_cache_8_;
  std::array<std::vector<int32_t>, 40> route_cache_16_;
  std::array<std::vector<int32_t>, 40> route_cache_32_;
  std::array<std::vector<int32_t>, 40> previous_router_top16_;
  std::array<std::vector<int32_t>, 40> previous_router_top32_;
  bool loaded_ = false;
};

}  // namespace mslite_llm::qwen35
