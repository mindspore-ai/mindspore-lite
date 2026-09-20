// Copyright 2026 Huawei Technologies Co., Ltd
// Adapted from HMOS-Device-AI llm/engine/runtime; preserves its paged Qwen3.5 execution.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "backend/nnrt/qwen35/moe_expert_cache.h"
#include "backend/nnrt/qwen35/nnrt_session.h"

namespace mslite_llm::qwen35 {

class ExpertCopyPool;

class Qwen35ExpertRuntime {
 public:
  Qwen35ExpertRuntime();
  ~Qwen35ExpertRuntime();
  Qwen35ExpertRuntime(const Qwen35ExpertRuntime &) = delete;
  Qwen35ExpertRuntime &operator=(const Qwen35ExpertRuntime &) = delete;

  bool Load(const std::string &artifact_root, size_t max_shards, std::string *error);
  bool RunGateUp(int32_t layer, int32_t expert, int32_t rows, const uint16_t *hidden, uint16_t *gated,
                 double *elapsed_ms, std::string *error);
  bool RunDown(int32_t layer, int32_t expert, int32_t rows, const uint16_t *gated, uint16_t *output, double *elapsed_ms,
               std::string *error);
  bool GroupedDecodeEnabled() const { return grouped_decode_enabled_; }
  bool RunGroupedDecode(int32_t layer, const std::array<int32_t, 8> &experts, const uint16_t *hidden, uint16_t *outputs,
                        double *elapsed_ms, std::string *error);
  bool BindGroupedInputs(NnrtSession *session, const std::string &set, const std::string &prefix, int32_t layer,
                         const std::array<int32_t, 8> &experts, std::array<int32_t, 8> *route_to_slot,
                         std::string *error);
  bool BindIndexedInputs(NnrtSession *session, const std::string &set, const std::string &prefix, int32_t layer,
                         const std::array<int32_t, 8> &experts, std::array<int32_t, 8> *route_to_slot,
                         std::string *error);
  bool BindPackedInputs(NnrtSession *session, const std::string &set, int32_t layer,
                        const std::array<int32_t, 8> &experts, std::array<int32_t, 8> *route_to_slot,
                        std::string *error);
  bool BindBankedInputs(NnrtSession *session, const std::string &set, int32_t layer,
                        const std::array<int32_t, 8> &experts, std::array<int32_t, 8> *route_to_slot,
                        std::string *error);
  bool BindZeroCopyInputs(NnrtSession *session, const std::string &set, const std::string &prefix, int32_t layer,
                          const std::array<int32_t, 8> &experts, std::array<int32_t, 8> *route_to_slot,
                          std::string *error);
  bool PrepareFusedDmaCache(size_t layer_count, size_t slot_count, std::string *error);
  bool PrepareBankedDmaCache(std::string *error);
  bool HasFusedDmaCache() const { return !fused_dma_fds_.empty(); }
  FileInputBindings FusedDmaBindings(int32_t layer, const std::string &prefix, size_t slot_count = 8) const;
  FileInputBindings PackedDmaBindings(int32_t layer, const std::string &tensor_name) const;
  bool PackedExpertsEnabled() const { return grouped_decode_packed_; }
  size_t PackedExpertSlots() const { return grouped_decode_packed_slots_; }
  bool PackedNnrtOwnedEnabled() const { return packed_nnrt_owned_; }
  bool PackedNnrtSharedEnabled() const { return packed_nnrt_shared_; }
  bool ZeroCopyExpertsEnabled() const { return zero_copy_experts_; }
  bool BankedExpertsEnabled() const { return grouped_decode_banked_; }
  FileInputBindings BankedDmaBindings(const std::string &prefix) const;
  bool PrimeZeroCopyInputs(NnrtSession *session, const std::string &set, const std::string &prefix, int32_t layer,
                           FileInputBindings *bindings, std::string *error);

 private:
  struct Stage {
    std::array<std::unique_ptr<NnrtSession>, 3> sessions;
    std::array<int32_t, 3> capacities = {1, 8, 64};
    bool gate_up = false;
    bool s16s4 = false;
  };

  struct ZeroCopySlotCache {
    ZeroCopySlotCache() {
      experts.fill(-1);
      last_used.fill(0);
    }
    std::array<int32_t, 8> experts;
    std::array<uint64_t, 8> last_used;
    std::array<std::shared_ptr<const ExpertShard>, 8> shards;
    uint64_t clock = 0;
  };

  struct PackedSlotCache {
    PackedSlotCache() {
      experts.fill(-1);
      last_used.fill(0);
    }
    std::array<int32_t, 384> experts;
    std::array<uint64_t, 384> last_used;
    uint64_t clock = 0;
  };

  struct IndexedSlotCache {
    IndexedSlotCache() {
      experts.fill(-1);
      last_used.fill(0);
    }
    std::array<int32_t, 16> experts;
    std::array<uint64_t, 16> last_used;
    uint64_t clock = 0;
  };

  struct GroupedSlotCache {
    GroupedSlotCache() { experts.fill(-1); }
    std::array<int32_t, 8> experts;
    std::array<uint64_t, 8> last_used{};
    uint64_t clock = 0;
  };

  struct PackedCopy {
    int32_t layer;
    bool tensor_copy;
    TensorView tensor;
    const std::array<int32_t, 8> &routes;
    const std::array<bool, 8> &hit;
    const std::array<std::shared_ptr<const ExpertShard>, 8> &misses;
    int32_t cache_hits;
    size_t copied_bytes = 0;
    std::array<std::array<ExpertTensorView, 6>, 8> views{};
    std::array<const uint8_t *, 8> payloads{};
    std::vector<int32_t> copy_routes{};
  };

  struct GroupedModelChoice {
    std::string path;
    bool banked = false;
    bool packed = false;
    bool indexed = false;
    bool fused_output = false;
    size_t packed_slots = 0;
  };

  void SetGroupedChoice(const std::string &path, const std::string &fused_path, int32_t requested_packed_slots,
                        const std::array<bool, 3> &available, GroupedModelChoice *choice);
  bool CopyStageActivation(const Stage *stage, NnrtSession *session, int32_t rows, int32_t capacity,
                           const uint16_t *input, std::string *error);
  bool LayoutFusedSlot(std::string *error);
  bool PrepareBankInputs(NnrtSession *session, const std::string &set, int32_t layer, bool &tensor_copy,
                         std::array<TensorView, 6> &bank_tensors, std::string *error);
  bool ValidatePackedInput(const NnrtSession *session, int32_t layer, bool tensor_copy, const TensorView &packed_tensor,
                           std::string *error);
  bool CopyStageWeights(const Stage *stage, NnrtSession *session, const ExpertShard &shard, std::string *error);
  bool CopyGroupedOutputs(const std::array<int32_t, 8> &route_to_slot, uint16_t *outputs, std::string *error);
  bool BindDecodeExperts(int32_t layer, const std::array<int32_t, 8> &experts, std::array<int32_t, 8> *route_to_slot,
                         std::string *error);
  void GroupedShapes(bool indexed, TensorShapes *inputs, TensorShapes *outputs);
  bool PrepareGroupedCache(std::string *error);
  bool ChooseGroupedModel(const std::string &artifact_root, GroupedModelChoice *choice, std::string *error);

  bool CopyGroupedShard(NnrtSession *session, const std::string &set, const std::string &slot_prefix,
                        const ExpertShard &shard, size_t *copied_bytes, std::string *error);
  void CopyBankedPayloads(bool tensor_copy, const std::array<TensorView, 6> &bank_tensors,
                          const std::array<int32_t, 8> &route_to_slot,
                          const std::array<std::shared_ptr<const ExpertShard>, 8> &misses,
                          const std::vector<int32_t> &copy_routes);
  void CopyPackedPayloads(const PackedCopy &copy);
  bool PreparePackedCopy(PackedCopy &copy, std::string *error);

  void ReleaseFusedDmaCache();
  bool AllocateFusedDmaCache(std::string *error);
  bool ConfigureGroupedDecode(const std::string &artifact_root, std::string *error);

  bool LoadStage(Stage *stage, const std::string &model_root, const char *stem, bool gate_up, bool s16s4,
                 const std::array<bool, 3> &enabled_gears, std::string *error);
  bool RunStage(Stage *stage, ExpertCache *cache, int32_t layer, int32_t expert, int32_t rows, const uint16_t *input,
                uint16_t *output, double *elapsed_ms, std::string *error);
  bool LoadGroupedDecode(const std::string &model_path, bool indexed, std::string *error);

  ExpertCache cache_;
  ExpertCache s16s4_cache_;
  Stage gate_up_;
  Stage down_;
  Stage s16s4_gate_up_;
  Stage s16s4_down_;
  std::unique_ptr<NnrtSession> grouped_decode_;
  std::unordered_map<NnrtSession *, std::unordered_map<std::string, GroupedSlotCache>> grouped_slot_caches_;
  std::unordered_map<NnrtSession *, std::unordered_map<std::string, IndexedSlotCache>> indexed_slot_caches_;
  std::unordered_map<NnrtSession *, std::unordered_map<std::string, PackedSlotCache>> packed_slot_caches_;
  std::unordered_map<NnrtSession *, std::unordered_map<std::string, ZeroCopySlotCache>> zero_copy_slot_caches_;
  std::vector<int> fused_dma_fds_;
  std::vector<uint8_t *> fused_dma_mappings_;
  size_t fused_dma_layer_bytes_ = 0;
  size_t fused_dma_slot_bytes_ = 0;
  size_t fused_dma_slot_count_ = 0;
  size_t fused_dma_layers_ = 0;
  std::array<size_t, 6> fused_dma_offsets_{};
  std::array<size_t, 6> fused_dma_tensor_bytes_{};
  bool s16s4_enabled_ = false;
  bool grouped_decode_enabled_ = false;
  bool grouped_decode_fused_output_ = false;
  bool grouped_decode_indexed_ = false;
  bool grouped_decode_packed_ = false;
  bool grouped_decode_banked_ = false;
  size_t grouped_decode_packed_slots_ = 0;
  bool packed_nnrt_owned_ = false;
  bool packed_nnrt_shared_ = false;
  bool zero_copy_experts_ = false;
  std::array<int, 6> banked_dma_fds_ = {-1, -1, -1, -1, -1, -1};
  std::array<uint8_t *, 6> banked_dma_mappings_{};
  PackedSlotCache banked_slot_cache_;
  std::unique_ptr<ExpertCopyPool> copy_pool_;
  std::mutex banked_cache_mutex_;

  std::shared_ptr<const ExpertShard> AcquireS16S4(int32_t layer, int32_t expert, std::string *error);
};

}  // namespace mslite_llm::qwen35
