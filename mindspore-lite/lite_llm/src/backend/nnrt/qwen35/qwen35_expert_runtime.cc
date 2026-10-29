// Copyright 2026 Huawei Technologies Co., Ltd
// Adapted from HMOS-Device-AI llm/engine/runtime; preserves its paged Qwen3.5 execution.
#include "backend/nnrt/qwen35/qwen35_expert_runtime.h"

#include <fcntl.h>
#include <linux/dma-buf.h>
#include <linux/dma-heap.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17): lite_llm requires C++17.
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include "manifest/artifact_file.h"

namespace fs = std::filesystem;

namespace mslite_llm::qwen35 {
namespace {

constexpr int32_t kHidden = 2048;
constexpr int32_t kIntermediate = 512;
constexpr int32_t kTopK = 8;
constexpr int32_t kIndexedSlots = 16;
constexpr int32_t kMaxPackedSlots = 384;
constexpr int32_t kMinPackedSlots = 16;
constexpr int32_t kDefaultPackedSlots = 32;
constexpr int32_t kLayers = 40;
constexpr int32_t kBankCount = 6;
constexpr int32_t kBankSlots = 64;
constexpr size_t kDmaAlignment = 4096;
struct TensorBinding {
  const char *onnx_name;
  const char *shard_name;
};

constexpr TensorBinding kGateUpBindings[] = {
  {"gate_up_qweight", "gate_up.qweight"},
  {"gate_up_qzeros", "gate_up.qzeros"},
  {"gate_up_scales", "gate_up.scales"},
  {"gate_up_g_idx", "gate_up.g_idx"},
};

constexpr TensorBinding kDownBindings[] = {
  {"down_qweight", "down_proj.qweight"},
  {"down_qzeros", "down_proj.qzeros"},
  {"down_scales", "down_proj.scales"},
  {"down_g_idx", "down_proj.g_idx"},
};

constexpr TensorBinding kS16S4GateUpBindings[] = {
  {"w", "gate_up.w"},
  {"bias", "gate_up.bias"},
  {"w_scale", "gate_up.w_scale"},
};

constexpr TensorBinding kS16S4DownBindings[] = {
  {"w", "down.w"},
  {"bias", "down.bias"},
  {"w_scale", "down.w_scale"},
};

int GearIndex(int32_t rows) {
  if (rows <= 1) return 0;
  if (rows <= 8) return 1;
  if (rows <= 64) return 2;
  return -1;
}

bool CopyTensor(NnrtSession *session, const std::string &set, const char *name, const void *data, size_t bytes,
                std::string *error) {
  TensorView view = session->Input(set, name);
  if (view.data == nullptr || view.bytes != bytes) {
    if (error) {
      *error = std::string("expert tensor size mismatch: ") + name + " expected=" + std::to_string(bytes) +
               " actual=" + std::to_string(view.bytes);
    }
    return false;
  }
  std::memcpy(view.data, data, bytes);
  return true;
}

size_t AlignUp(size_t value, size_t alignment) { return (value + alignment - 1) / alignment * alignment; }

int AllocateDmaBuffer(size_t bytes, std::string *error) {
  std::array<const char *, 3> paths = {
    "/dev/dma_heap/npu_device_hugepage_heap",
    "/dev/dma_heap/system_heap",
    "/dev/dma_heap/system",
  };
  const char *heap_choice = std::getenv("QWEN35_EXPERT_DMA_HEAP");
  if (heap_choice != nullptr && std::string(heap_choice) == "system") {
    paths = {
      "/dev/dma_heap/system_heap",
      "/dev/dma_heap/system",
      "/dev/dma_heap/npu_device_hugepage_heap",
    };
  }
  for (const char *path : paths) {
    const int heap = open(path, O_RDWR | O_CLOEXEC);
    if (heap < 0) continue;
    dma_heap_allocation_data allocation{};
    allocation.len = bytes;
    allocation.fd_flags = O_RDWR | O_CLOEXEC;
    const int status = ioctl(heap, DMA_HEAP_IOCTL_ALLOC, &allocation);
    close(heap);
    if (status == 0 && allocation.fd >= 0) return allocation.fd;
  }
  if (error) *error = "cannot allocate fused expert DMA cache";
  return -1;
}

bool SyncDmaBuffer(int fd, uint64_t flags) {
  dma_buf_sync synchronization{};
  synchronization.flags = flags;
  return ioctl(fd, DMA_BUF_IOCTL_SYNC, &synchronization) == 0;
}

// Resolve all hits before choosing victims, so misses cannot evict this route set.
struct SlotSelection {
  size_t first = 0;
  size_t count = 0;
  int32_t key_base = 0;
  bool resident = true;
  bool validate_upper = false;
  bool refresh_hits = true;
  bool protect_claimed = false;
  bool use_lru = true;
  const char *mode = nullptr;
  bool first_unclaimed = false;
};

template <typename Cache, size_t Slots>
bool FindRouteHits(Cache &cache, const std::array<int32_t, 8> &experts, std::array<int32_t, 8> *routes,
                   std::array<bool, Slots> *claimed, const SlotSelection &selection) {
  routes->fill(-1);
  for (size_t route = 0; route < experts.size(); ++route) {
    if (experts[route] < 0 || (selection.validate_upper && experts[route] >= 256)) return false;
    if (!selection.resident) continue;
    for (size_t slot = selection.first; slot < selection.first + selection.count; ++slot) {
      if (selection.protect_claimed && (*claimed)[slot]) continue;
      if (cache.experts[slot] != selection.key_base + experts[route]) continue;
      (*routes)[route] = static_cast<int32_t>(slot);
      (*claimed)[slot] = true;
      if (selection.refresh_hits) cache.last_used[slot] = ++cache.clock;
      break;
    }
  }
  return true;
}

template <typename Cache, size_t Slots>
int32_t ReplacementSlot(const Cache &cache, const std::array<bool, Slots> &claimed, const SlotSelection &selection) {
  int32_t slot = -1;
  for (size_t candidate = selection.first; candidate < selection.first + selection.count; ++candidate) {
    if (!claimed[candidate] && (selection.first_unclaimed || cache.experts[candidate] < 0))
      return static_cast<int32_t>(candidate);
  }
  for (size_t candidate = selection.first; candidate < selection.first + selection.count; ++candidate) {
    if (claimed[candidate]) continue;
    if (selection.first_unclaimed) return static_cast<int32_t>(candidate);
    if (!selection.use_lru || slot < 0 || cache.last_used[candidate] < cache.last_used[slot]) {
      slot = static_cast<int32_t>(candidate);
    }
  }
  return slot;
}

template <typename Cache, size_t Slots>
bool AssignRouteSlots(Cache &cache, std::array<int32_t, 8> *routes, std::array<bool, Slots> *claimed,
                      const SlotSelection &selection) {
  for (int32_t &route : *routes) {
    if (route >= 0) continue;
    route = ReplacementSlot(cache, *claimed, selection);
    if (route < 0) return false;
    (*claimed)[route] = true;
  }
  return true;
}

template <typename Cache, size_t Slots>
bool SelectRouteSlots(Cache &cache, const std::array<int32_t, 8> &experts, std::array<int32_t, 8> *routes,
                      std::array<bool, 8> *hit, const SlotSelection &selection, std::string *error) {
  std::array<bool, Slots> claimed{};
  if (!FindRouteHits(cache, experts, routes, &claimed, selection)) {
    if (error) *error = std::string("invalid ") + selection.mode + " S16S4 expert id";
    return false;
  }
  for (size_t route = 0; route < routes->size(); ++route) (*hit)[route] = (*routes)[route] >= 0;
  if (!AssignRouteSlots(cache, routes, &claimed, selection)) {
    if (error) *error = std::string("cannot assign ") + selection.mode + " S16S4 cache slot";
    return false;
  }
  return true;
}

bool IsPackedSlotCount(size_t slots) {
  static constexpr std::array<size_t, 5> kPackedSlots = {kMinPackedSlots, kDefaultPackedSlots, 64, 128,
                                                         kMaxPackedSlots};
  return std::find(kPackedSlots.begin(), kPackedSlots.end(), slots) != kPackedSlots.end();
}

}  // namespace

class ExpertCopyPool {
 public:
  explicit ExpertCopyPool(size_t worker_count) : worker_count_(worker_count) {
    workers_.reserve(worker_count_);
    for (size_t index = 0; index < worker_count_; ++index) workers_.emplace_back([this] { WorkerLoop(); });
  }

  ~ExpertCopyPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
      ++generation_;
    }
    work_ready_.notify_all();
    for (std::thread &worker : workers_) worker.join();
  }

  ExpertCopyPool(const ExpertCopyPool &) = delete;
  ExpertCopyPool &operator=(const ExpertCopyPool &) = delete;

  size_t WorkerCount() const { return worker_count_; }

  void Run(size_t task_count, std::function<void(size_t)> task) {
    if (task_count == 0) return;
    std::unique_lock<std::mutex> lock(mutex_);
    task_ = std::move(task);
    task_count_ = task_count;
    next_task_.store(0, std::memory_order_relaxed);
    completed_workers_ = 0;
    ++generation_;
    work_ready_.notify_all();
    work_done_.wait(lock, [this] { return completed_workers_ == worker_count_; });
    task_ = nullptr;
    task_count_ = 0;
  }

 private:
  void WorkerLoop() {
    size_t observed_generation = 0;
    for (;;) {
      std::function<void(size_t)> task;
      size_t task_count = 0;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        work_ready_.wait(lock, [this, &observed_generation] { return stop_ || generation_ != observed_generation; });
        if (stop_) return;
        observed_generation = generation_;
        task = task_;
        task_count = task_count_;
      }
      for (;;) {
        const size_t index = next_task_.fetch_add(1, std::memory_order_relaxed);
        if (index >= task_count) break;
        task(index);
      }
      {
        std::lock_guard<std::mutex> lock(mutex_);
        ++completed_workers_;
        if (completed_workers_ == worker_count_) work_done_.notify_one();
      }
    }
  }

  const size_t worker_count_;
  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable work_ready_;
  std::condition_variable work_done_;
  std::atomic<size_t> next_task_{0};
  std::function<void(size_t)> task_;
  size_t task_count_ = 0;
  size_t completed_workers_ = 0;
  size_t generation_ = 0;
  bool stop_ = false;
};

Qwen35ExpertRuntime::Qwen35ExpertRuntime() = default;

Qwen35ExpertRuntime::~Qwen35ExpertRuntime() {
  for (size_t index = 0; index < fused_dma_mappings_.size(); ++index) {
    if (index < fused_dma_fds_.size() && fused_dma_mappings_[index] != nullptr && fused_dma_layer_bytes_ != 0) {
      munmap(fused_dma_mappings_[index], fused_dma_layer_bytes_);
    }
  }
  for (const int fd : fused_dma_fds_) {
    if (fd >= 0) close(fd);
  }
  const size_t bank_bytes = static_cast<size_t>(kBankSlots) * 1777664;
  for (size_t bank = 0; bank < banked_dma_fds_.size(); ++bank) {
    if (banked_dma_mappings_[bank] != nullptr) munmap(banked_dma_mappings_[bank], bank_bytes);
    if (banked_dma_fds_[bank] >= 0) close(banked_dma_fds_[bank]);
  }
}

std::shared_ptr<const ExpertShard> Qwen35ExpertRuntime::AcquireS16S4(int32_t layer, int32_t expert,
                                                                     std::string *error) {
  std::lock_guard<std::mutex> lock(banked_cache_mutex_);
  return s16s4_cache_.Acquire(layer, expert, error);
}

bool Qwen35ExpertRuntime::Load(const std::string &artifact_root, size_t max_shards, std::string *error) {
  const fs::path root(artifact_root);
  if (!cache_.Load((root / "experts").string(), max_shards, error)) return false;
  if (!LoadStage(&gate_up_, (root / "omc/expert").string(), "qwen35_moe_gate_up", true, false, {true, true, true},
                 error) ||
      !LoadStage(&down_, (root / "omc/expert").string(), "qwen35_moe_down", false, false, {true, true, true}, error)) {
    return false;
  }

  const fs::path s16s4_root = root / "experts_s16s4";
  const fs::path s16s4_models = root / "omc/expert_s16s4";
  const std::array<fs::path, 4> required_models = {
    s16s4_models / "qwen35_moe_s16s4_gate_up_s1.omc",
    s16s4_models / "qwen35_moe_s16s4_gate_up_s64.omc",
    s16s4_models / "qwen35_moe_s16s4_down_s1.omc",
    s16s4_models / "qwen35_moe_s16s4_down_s64.omc",
  };
  s16s4_enabled_ =
    ArtifactExists(s16s4_root / "READY") && std::all_of(required_models.begin(), required_models.end(),
                                                        [](const fs::path &path) { return ArtifactExists(path); });
  if (s16s4_enabled_ && (!s16s4_cache_.Load(s16s4_root.string(), max_shards, error, ".s4") ||
                         !LoadStage(&s16s4_gate_up_, s16s4_models.string(), "qwen35_moe_s16s4_gate_up", true, true,
                                    {true, false, true}, error) ||
                         !LoadStage(&s16s4_down_, s16s4_models.string(), "qwen35_moe_s16s4_down", false, true,
                                    {true, false, true}, error))) {
    return false;
  }
  return ConfigureGroupedDecode(artifact_root, error);
}

bool Qwen35ExpertRuntime::PrepareFusedDmaCache(size_t layer_count, size_t slot_count, std::string *error) {
  const char *enable = std::getenv("QWEN35_ENABLE_EXPERT_DMA_CACHE");
  const bool indexed = slot_count == kIndexedSlots;
  const bool packed = IsPackedSlotCount(slot_count);
  if ((!indexed && !packed && (enable == nullptr || std::string(enable) != "1")) ||
      std::getenv("QWEN35_DISABLE_EXPERT_DMA_CACHE") != nullptr)
    return true;
  if (!s16s4_enabled_ || layer_count == 0 || layer_count > kLayers ||
      (slot_count != kTopK && slot_count != kIndexedSlots && !packed)) {
    if (error) *error = "invalid fused expert DMA cache configuration";
    return false;
  }
  if (!fused_dma_fds_.empty()) return layer_count == fused_dma_layers_ && slot_count == fused_dma_slot_count_;

  if (!LayoutFusedSlot(error)) return false;
  fused_dma_slot_count_ = slot_count;
  fused_dma_layers_ = layer_count;
  fused_dma_layer_bytes_ = fused_dma_slot_bytes_ * fused_dma_slot_count_;
  if (packed && packed_nnrt_owned_) return true;
  return AllocateFusedDmaCache(error);
}

bool Qwen35ExpertRuntime::PrepareBankedDmaCache(std::string *error) {
  if (banked_dma_fds_[0] >= 0) return true;
  const std::shared_ptr<const ExpertShard> probe = s16s4_cache_.Acquire(0, 0, error);
  if (probe == nullptr) return false;
  const std::array<const char *, 6> names = {"gate_up.w", "gate_up.bias", "gate_up.w_scale",
                                             "down.w",    "down.bias",    "down.w_scale"};
  size_t slot_bytes = 0;
  for (size_t index = 0; index < names.size(); ++index) {
    const ExpertTensorView view = probe->Tensor(names[index]);
    if (view.data == nullptr || view.bytes == 0) {
      if (error) *error = "invalid banked expert shard layout";
      return false;
    }
    slot_bytes = AlignUp(slot_bytes, kDmaAlignment);
    fused_dma_offsets_[index] = slot_bytes;
    fused_dma_tensor_bytes_[index] = view.bytes;
    slot_bytes += view.bytes;
  }
  fused_dma_slot_bytes_ = AlignUp(slot_bytes, kDmaAlignment);
  if (fused_dma_slot_bytes_ != 1777664) {
    if (error) *error = "unexpected banked expert slot size";
    return false;
  }
  fused_dma_slot_count_ = kBankCount * kBankSlots;
  const size_t bank_bytes = static_cast<size_t>(kBankSlots) * fused_dma_slot_bytes_;
  for (int32_t bank = 0; bank < kBankCount; ++bank) {
    const int fd = AllocateDmaBuffer(bank_bytes, error);
    if (fd < 0) return false;
    void *mapping = mmap(nullptr, bank_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapping == MAP_FAILED) {
      close(fd);
      if (error) *error = "cannot map banked expert DMA cache";
      return false;
    }
    banked_dma_fds_[bank] = fd;
    banked_dma_mappings_[bank] = static_cast<uint8_t *>(mapping);
  }
  return true;
}

FileInputBindings Qwen35ExpertRuntime::BankedDmaBindings(const std::string &prefix) const {
  FileInputBindings bindings;
  const size_t bank_bytes = static_cast<size_t>(kBankSlots) * 1777664;
  for (int32_t bank = 0; bank < kBankCount; ++bank) {
    if (banked_dma_fds_[bank] < 0) return {};
    bindings[prefix + std::to_string(bank)] = {banked_dma_fds_[bank], bank_bytes, 0};
  }
  return bindings;
}

FileInputBindings Qwen35ExpertRuntime::PackedDmaBindings(int32_t layer, const std::string &tensor_name) const {
  FileInputBindings bindings;
  if (packed_nnrt_owned_ || fused_dma_fds_.empty() || layer < 0 || static_cast<size_t>(layer) >= fused_dma_layers_ ||
      fused_dma_slot_count_ < kMinPackedSlots) {
    return bindings;
  }
  bindings[tensor_name] = {fused_dma_fds_[layer], fused_dma_layer_bytes_, 0};
  return bindings;
}

FileInputBindings Qwen35ExpertRuntime::FusedDmaBindings(int32_t layer, const std::string &prefix,
                                                        size_t slot_count) const {
  FileInputBindings bindings;
  if (fused_dma_fds_.empty() || layer < 0 || static_cast<size_t>(layer) >= fused_dma_layers_ ||
      slot_count > fused_dma_slot_count_) {
    return bindings;
  }
  const std::array<const char *, 6> names = {
    "_gate_up_w", "_gate_up_bias", "_gate_up_w_scale", "_down_w", "_down_bias", "_down_w_scale",
  };
  for (size_t slot = 0; slot < slot_count; ++slot) {
    const size_t slot_base = static_cast<size_t>(slot) * fused_dma_slot_bytes_;
    for (size_t index = 0; index < names.size(); ++index) {
      bindings[prefix + std::to_string(slot) + names[index]] = {fused_dma_fds_[layer], fused_dma_layer_bytes_,
                                                                slot_base + fused_dma_offsets_[index]};
    }
  }
  return bindings;
}

bool Qwen35ExpertRuntime::PrimeZeroCopyInputs(NnrtSession *session, const std::string &set, const std::string &prefix,
                                              int32_t layer, FileInputBindings *bindings, std::string *error) {
  if (!zero_copy_experts_ || session == nullptr || bindings == nullptr || layer < 0 || layer >= kLayers) {
    if (error) *error = "zero-copy S16S4 inputs are unavailable";
    return false;
  }
  // All fused boundaries of one executor share these physical shard tensors.
  // Keep one slot map per executor rather than one map per boundary set.
  ZeroCopySlotCache &cache = zero_copy_slot_caches_[session]["__shared_zero_copy_shards"];
  bindings->clear();
  const bool initialized =
    std::all_of(cache.shards.begin(), cache.shards.end(), [](const auto &shard) { return shard != nullptr; });
  // NnrtSession shares a tensor by input name across its boundary sets. Only
  // the first set needs file bindings; later sets inherit those same tensors.
  if (initialized) return true;
  for (int32_t slot = 0; slot < kTopK; ++slot) {
    const std::shared_ptr<const ExpertShard> shard = s16s4_cache_.Acquire(layer, slot, error);
    if (shard == nullptr || shard->Fd() < 0 || shard->FileSize() != 1781760) {
      if (error && error->empty()) *error = "invalid zero-copy S16S4 expert shard";
      return false;
    }
    cache.experts[slot] = layer * 256 + slot;
    cache.shards[slot] = shard;
    cache.last_used[slot] = ++cache.clock;
    (*bindings)[prefix + std::to_string(slot) + "_shard"] = {shard->Fd(), shard->BindingSize(), shard->FileOffset()};
  }
  return true;
}

bool Qwen35ExpertRuntime::LoadGroupedDecode(const std::string &model_path, bool indexed, std::string *error) {
  auto session = std::make_unique<NnrtSession>();
  if (!session->Load(model_path, "", error)) return false;
  TensorShapes inputs = {{"hidden", {1, kHidden}}};
  TensorShapes outputs;
  GroupedShapes(indexed, &inputs, &outputs);
  FileInputBindings files;
  if (grouped_decode_banked_) {
    files = BankedDmaBindings("packed_bank_");
  } else if (zero_copy_experts_) {
    if (!PrimeZeroCopyInputs(session.get(), "main", "expert_", kLayers - 1, &files, error)) {
      return false;
    }
  } else if (grouped_decode_packed_) {
    files = PackedDmaBindings(kLayers - 1, "packed_weights");
  } else if (indexed) {
    files = FusedDmaBindings(kLayers - 1, "expert_", kIndexedSlots);
  }
  if (!session->CreateTensorSet("main", inputs, outputs, std::unordered_map<std::string, std::string>{},
                                SharedInputKeys{}, ExternalInputHandles{}, files, error)) {
    return false;
  }
  grouped_decode_ = std::move(session);
  return true;
}

bool Qwen35ExpertRuntime::LoadStage(Stage *stage, const std::string &model_root, const char *stem, bool gate_up,
                                    bool s16s4, const std::array<bool, 3> &enabled_gears, std::string *error) {
  stage->gate_up = gate_up;
  stage->s16s4 = s16s4;
  for (size_t index = 0; index < stage->sessions.size(); ++index) {
    if (!enabled_gears[index]) continue;
    auto session = std::make_unique<NnrtSession>();
    const int32_t rows = stage->capacities[index];
    const std::string path = std::string(model_root) + "/" + stem + "_s" + std::to_string(rows) + ".omc";
    if (!session->Load(path, "", error)) return false;
    TensorShapes inputs;
    inputs[gate_up ? "hidden" : "gated"] = {rows, gate_up ? kHidden : kIntermediate};
    if (s16s4) {
      const int32_t k = gate_up ? kHidden : kIntermediate;
      const int32_t n = gate_up ? 2 * kIntermediate : kHidden;
      inputs["w"] = {n, k / 2};
      inputs["bias"] = {n};
      inputs["w_scale"] = {n / 128, k / 128, 128, 8};
    } else if (gate_up) {
      inputs["gate_up_qweight"] = {256, 1024};
      inputs["gate_up_qzeros"] = {16, 128};
      inputs["gate_up_scales"] = {16, 1024};
      inputs["gate_up_g_idx"] = {2048};
    } else {
      inputs["down_qweight"] = {512, 256};
      inputs["down_qzeros"] = {4, 256};
      inputs["down_scales"] = {4, 2048};
      inputs["down_g_idx"] = {512};
    }
    TensorShapes outputs;
    outputs[gate_up ? "gated" : "output"] = {rows, gate_up ? kIntermediate : kHidden};
    if (!session->CreateTensorSet("main", inputs, outputs, std::unordered_map<std::string, std::string>{},
                                  std::unordered_set<std::string>{}, error)) {
      return false;
    }
    stage->sessions[index] = std::move(session);
  }
  return true;
}

bool Qwen35ExpertRuntime::RunGateUp(int32_t layer, int32_t expert, int32_t rows, const uint16_t *hidden,
                                    uint16_t *gated, double *elapsed_ms, std::string *error) {
  if (s16s4_enabled_ && (rows == 1 || rows > 8)) {
    return RunStage(&s16s4_gate_up_, &s16s4_cache_, layer, expert, rows, hidden, gated, elapsed_ms, error);
  }
  return RunStage(&gate_up_, &cache_, layer, expert, rows, hidden, gated, elapsed_ms, error);
}

bool Qwen35ExpertRuntime::RunDown(int32_t layer, int32_t expert, int32_t rows, const uint16_t *gated, uint16_t *output,
                                  double *elapsed_ms, std::string *error) {
  if (s16s4_enabled_ && (rows == 1 || rows > 8)) {
    return RunStage(&s16s4_down_, &s16s4_cache_, layer, expert, rows, gated, output, elapsed_ms, error);
  }
  return RunStage(&down_, &cache_, layer, expert, rows, gated, output, elapsed_ms, error);
}

bool Qwen35ExpertRuntime::RunGroupedDecode(int32_t layer, const std::array<int32_t, 8> &experts, const uint16_t *hidden,
                                           uint16_t *outputs, double *elapsed_ms, std::string *error) {
  if (!grouped_decode_enabled_ || grouped_decode_ == nullptr || layer < 0 || layer >= kLayers || hidden == nullptr ||
      outputs == nullptr) {
    if (error) *error = "grouped S16S4 decode is unavailable";
    return false;
  }
  if (!CopyTensor(grouped_decode_.get(), "main", "hidden", hidden, static_cast<size_t>(kHidden) * sizeof(uint16_t),
                  error)) {
    return false;
  }
  std::array<int32_t, kTopK> route_to_slot{};
  if (!BindDecodeExperts(layer, experts, &route_to_slot, error)) return false;
  double elapsed = 0.0;
  if (!grouped_decode_->Run("main", &elapsed, error)) return false;
  if (elapsed_ms != nullptr) *elapsed_ms = elapsed;
  return CopyGroupedOutputs(route_to_slot, outputs, error);
}

bool Qwen35ExpertRuntime::BindZeroCopyInputs(NnrtSession *session, const std::string &set, const std::string &prefix,
                                             int32_t layer, const std::array<int32_t, 8> &experts,
                                             std::array<int32_t, 8> *route_to_slot, std::string *error) {
  if (!zero_copy_experts_ || session == nullptr || route_to_slot == nullptr || layer < 0 || layer >= kLayers) {
    if (error) *error = "zero-copy S16S4 inputs are unavailable";
    return false;
  }
  ZeroCopySlotCache &cache = zero_copy_slot_caches_[session]["__shared_zero_copy_shards"];
  std::array<bool, kTopK> hit{};
  const SlotSelection selection{0, kTopK, layer * 256, true, true, true, false, true, "zero-copy"};
  if (!SelectRouteSlots<decltype(cache), kTopK>(cache, experts, route_to_slot, &hit, selection, error)) {
    return false;
  }

  for (int32_t route = 0; route < kTopK; ++route) {
    if (hit[route]) continue;
    const std::shared_ptr<const ExpertShard> shard = s16s4_cache_.Acquire(layer, experts[route], error);
    if (shard == nullptr || shard->Fd() < 0 || shard->FileSize() != 1781760) {
      if (error && error->empty()) *error = "invalid zero-copy S16S4 expert shard";
      return false;
    }
    const int32_t slot = (*route_to_slot)[route];
    if (!session->ReplaceFileInput(set, prefix + std::to_string(slot) + "_shard",
                                   {shard->Fd(), shard->BindingSize(), shard->FileOffset()}, error)) {
      return false;
    }
    cache.experts[slot] = layer * 256 + experts[route];
    cache.shards[slot] = shard;
    cache.last_used[slot] = ++cache.clock;
  }
  return true;
}

bool Qwen35ExpertRuntime::BindPackedInputs(NnrtSession *session, const std::string &set, int32_t layer,
                                           const std::array<int32_t, 8> &experts, std::array<int32_t, 8> *route_to_slot,
                                           std::string *error) {
  const char *tensor_copy_value = std::getenv("QWEN35_PACKED_EXPERT_TENSOR_COPY");
  // NNRT exposes the file-backed tensor buffer as coherent zero-copy memory.
  // Directly writing it avoids an extra userspace mapping and DMA-BUF sync per
  // boundary. Set to 0 only for the legacy explicit-sync diagnostic path.
  const bool tensor_copy = tensor_copy_value == nullptr || std::string(tensor_copy_value) != "0";
  TensorView packed_tensor;
  if (tensor_copy && session != nullptr) {
    packed_tensor = session->Input(set, "current_packed_weights");
    if (packed_tensor.data == nullptr) packed_tensor = session->Input(set, "packed_weights");
  }
  if (route_to_slot == nullptr || !ValidatePackedInput(session, layer, tensor_copy, packed_tensor, error)) return false;

  const std::string cache_set = packed_nnrt_shared_ ? "__shared_packed_weights" : set;
  PackedSlotCache &cache = packed_slot_caches_[session][cache_set];
  std::array<bool, kTopK> hit{};
  const SlotSelection selection{0, fused_dma_slot_count_, layer * 256, true, false, true, false, true, "packed"};
  if (!SelectRouteSlots<decltype(cache), kMaxPackedSlots>(cache, experts, route_to_slot, &hit, selection, error)) {
    return false;
  }

  std::array<std::shared_ptr<const ExpertShard>, kTopK> misses{};
  int32_t cache_hits = 0;
  const auto acquire_start = std::chrono::steady_clock::now();
  for (int32_t route = 0; route < kTopK; ++route) {
    if (hit[route]) {
      ++cache_hits;
      continue;
    }
    misses[route] = s16s4_cache_.Acquire(layer, experts[route], error);
    if (misses[route] == nullptr) return false;
  }
  const auto acquire_end = std::chrono::steady_clock::now();

  PackedCopy copy{layer, tensor_copy, packed_tensor, *route_to_slot, hit, misses, cache_hits};
  if (!PreparePackedCopy(copy, error)) return false;
  const auto copy_start = std::chrono::steady_clock::now();
  CopyPackedPayloads(copy);
  const auto copy_end = std::chrono::steady_clock::now();

  for (int32_t route : copy.copy_routes) {
    const int32_t slot = (*route_to_slot)[route];
    cache.experts[slot] = layer * 256 + experts[route];
    cache.last_used[slot] = ++cache.clock;
  }
  if (!tensor_copy && cache_hits != kTopK &&
      !SyncDmaBuffer(fused_dma_fds_[layer], DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE)) {
    if (error) *error = "cannot finish packed expert DMA write";
    return false;
  }
  if (std::getenv("QWEN35_EXPERT_DMA_CACHE_STATS") != nullptr) {
    const double acquire_ms = std::chrono::duration<double, std::milli>(acquire_end - acquire_start).count();
    const double copy_ms = std::chrono::duration<double, std::milli>(copy_end - copy_start).count();
    std::cerr << "qwen35_packed_expert_cache layer=" << layer << " hits=" << cache_hits
              << " misses=" << kTopK - cache_hits << " copied_bytes=" << copy.copied_bytes
              << " acquire_ms=" << acquire_ms << " copy_ms=" << copy_ms << '\n';
  }
  return true;
}

bool Qwen35ExpertRuntime::BindBankedInputs(NnrtSession *session, const std::string &set, int32_t layer,
                                           const std::array<int32_t, 8> &experts, std::array<int32_t, 8> *route_to_slot,
                                           std::string *error) {
  bool tensor_copy = false;
  std::array<TensorView, kBankCount> bank_tensors{};
  if (route_to_slot == nullptr || !PrepareBankInputs(session, set, layer, tensor_copy, bank_tensors, error))
    return false;
  PackedSlotCache &cache = banked_slot_cache_;
  std::array<bool, kTopK> hit{};
  const SlotSelection selection{
    static_cast<size_t>(layer * kTopK), kTopK, layer * 256, true, true, false, false, false, "banked"};
  if (!SelectRouteSlots<decltype(cache), kMaxPackedSlots>(cache, experts, route_to_slot, &hit, selection, error)) {
    return false;
  }

  std::array<std::shared_ptr<const ExpertShard>, kTopK> misses{};
  std::vector<int32_t> copy_routes;
  std::array<bool, kBankCount> touched_banks{};
  size_t copied_bytes = 0;
  for (int32_t route = 0; route < kTopK; ++route) {
    if (hit[route]) continue;
    misses[route] = AcquireS16S4(layer, experts[route], error);
    if (misses[route] == nullptr) return false;
    copy_routes.push_back(route);
    touched_banks[static_cast<size_t>((*route_to_slot)[route] / kBankSlots)] = true;
    copied_bytes += fused_dma_slot_bytes_;
  }
  for (int32_t bank = 0; !tensor_copy && bank < kBankCount; ++bank) {
    if (touched_banks[bank] && !SyncDmaBuffer(banked_dma_fds_[bank], DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE)) {
      if (error) *error = "cannot begin banked expert DMA write";
      return false;
    }
  }
  CopyBankedPayloads(tensor_copy, bank_tensors, *route_to_slot, misses, copy_routes);
  for (int32_t bank = 0; !tensor_copy && bank < kBankCount; ++bank) {
    if (touched_banks[bank] && !SyncDmaBuffer(banked_dma_fds_[bank], DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE)) {
      if (error) *error = "cannot finish banked expert DMA write";
      return false;
    }
  }
  for (int32_t route : copy_routes) {
    const int32_t slot = (*route_to_slot)[route];
    cache.experts[slot] = layer * 256 + experts[route];
  }
  if (std::getenv("QWEN35_EXPERT_DMA_CACHE_STATS") != nullptr) {
    std::cerr << "qwen35_banked_expert_cache layer=" << layer << " hits=" << kTopK - copy_routes.size()
              << " misses=" << copy_routes.size() << " copied_bytes=" << copied_bytes << '\n';
  }
  return true;
}

bool Qwen35ExpertRuntime::BindGroupedInputs(NnrtSession *session, const std::string &set, const std::string &prefix,
                                            int32_t layer, const std::array<int32_t, 8> &experts,
                                            std::array<int32_t, 8> *route_to_slot, std::string *error) {
  if (!s16s4_enabled_ || session == nullptr || route_to_slot == nullptr || layer < 0 || layer >= kLayers) {
    if (error) *error = "grouped S16S4 inputs are unavailable";
    return false;
  }
  const bool resident_cache = HasFusedDmaCache() && set.rfind("boundary_", 0) == 0;
  GroupedSlotCache &cache = grouped_slot_caches_[session][set];
  std::array<bool, kTopK> hit{};
  const SlotSelection selection{0, kTopK, 0, resident_cache, false, false, true, false, "grouped", true};
  if (!SelectRouteSlots<decltype(cache), kTopK>(cache, experts, route_to_slot, &hit, selection, error)) return false;

  size_t copied_bytes = 0;
  int32_t cache_hits = 0;
  for (int32_t route = 0; route < kTopK; ++route) {
    const int32_t slot = (*route_to_slot)[route];
    if (resident_cache && hit[route]) {
      ++cache_hits;
      continue;
    }
    const std::shared_ptr<const ExpertShard> shard = s16s4_cache_.Acquire(layer, experts[route], error);
    if (shard == nullptr) return false;
    const std::string slot_prefix = prefix + std::to_string(slot);
    if (!CopyGroupedShard(session, set, slot_prefix, *shard, &copied_bytes, error)) return false;
    if (resident_cache) cache.experts[slot] = experts[route];
  }
  if (std::getenv("QWEN35_EXPERT_DMA_CACHE_STATS") != nullptr) {
    std::cerr << "qwen35_expert_dma_cache layer=" << layer << " hits=" << cache_hits << " misses=" << kTopK - cache_hits
              << " copied_bytes=" << copied_bytes << '\n';
  }
  return true;
}

bool Qwen35ExpertRuntime::BindIndexedInputs(NnrtSession *session, const std::string &set, const std::string &prefix,
                                            int32_t layer, const std::array<int32_t, 8> &experts,
                                            std::array<int32_t, 8> *route_to_slot, std::string *error) {
  if (!s16s4_enabled_ || session == nullptr || route_to_slot == nullptr || layer < 0 || layer >= kLayers) {
    if (error) *error = "indexed S16S4 inputs are unavailable";
    return false;
  }

  const bool resident_cache = HasFusedDmaCache();
  IndexedSlotCache &cache = indexed_slot_caches_[session][set];
  std::array<bool, kTopK> hit{};
  const SlotSelection selection{0, kIndexedSlots, 0, resident_cache, false, true, false, true, "indexed"};
  if (!SelectRouteSlots<decltype(cache), kIndexedSlots>(cache, experts, route_to_slot, &hit, selection, error)) {
    return false;
  }

  size_t copied_bytes = 0;
  int32_t cache_hits = 0;
  for (int32_t route = 0; route < kTopK; ++route) {
    const int32_t slot = (*route_to_slot)[route];
    if (resident_cache && hit[route]) {
      ++cache_hits;
      continue;
    }
    const auto shard = s16s4_cache_.Acquire(layer, experts[route], error);
    if (shard == nullptr) return false;
    const std::string slot_prefix = prefix + std::to_string(slot);
    if (!CopyGroupedShard(session, set, slot_prefix, *shard, &copied_bytes, error)) return false;
    if (resident_cache) {
      cache.experts[slot] = experts[route];
      cache.last_used[slot] = ++cache.clock;
    }
  }
  if (std::getenv("QWEN35_EXPERT_DMA_CACHE_STATS") != nullptr) {
    std::cerr << "qwen35_indexed_expert_cache layer=" << layer << " hits=" << cache_hits
              << " misses=" << kTopK - cache_hits << " copied_bytes=" << copied_bytes << '\n';
  }
  return true;
}

bool Qwen35ExpertRuntime::RunStage(Stage *stage, ExpertCache *cache, int32_t layer, int32_t expert, int32_t rows,
                                   const uint16_t *input, uint16_t *output, double *elapsed_ms, std::string *error) {
  const int index = GearIndex(rows);
  if (index < 0 || input == nullptr || output == nullptr || cache == nullptr || stage->sessions[index] == nullptr) {
    if (error) *error = "expert stage requires 1, 8, or 64 rows";
    return false;
  }
  const int32_t capacity = stage->capacities[index];
  NnrtSession *session = stage->sessions[index].get();
  if (!CopyStageActivation(stage, session, rows, capacity, input, error)) return false;
  const auto shard = cache->Acquire(layer, expert, error);
  if (shard == nullptr) return false;
  if (!CopyStageWeights(stage, session, *shard, error)) return false;
  double elapsed = 0.0;
  if (!session->Run("main", &elapsed, error)) return false;
  if (elapsed_ms != nullptr) *elapsed_ms = elapsed;
  const char *output_name = stage->gate_up ? "gated" : "output";
  const TensorView result = session->Output("main", output_name);
  const size_t output_width = stage->gate_up ? kIntermediate : kHidden;
  if (result.data == nullptr || result.bytes != static_cast<size_t>(capacity) * output_width * sizeof(uint16_t)) {
    if (error) *error = "expert output tensor size mismatch";
    return false;
  }
  std::memcpy(output, result.data, static_cast<size_t>(rows) * output_width * sizeof(uint16_t));
  return true;
}

bool Qwen35ExpertRuntime::ConfigureGroupedDecode(const std::string &artifact_root, std::string *error) {
  GroupedModelChoice choice;
  if (!ChooseGroupedModel(artifact_root, &choice, error)) return false;
  if (s16s4_enabled_ && ArtifactExists(choice.path)) {
    grouped_decode_banked_ = choice.banked;
    grouped_decode_packed_ = choice.packed;
    zero_copy_experts_ = false;
    grouped_decode_packed_slots_ = choice.packed_slots;
    const char *owned_value = std::getenv("QWEN35_PACKED_NNRT_OWNED");
    const char *shared_value = std::getenv("QWEN35_PACKED_NNRT_SHARED");
    packed_nnrt_shared_ = grouped_decode_packed_ && shared_value != nullptr && std::string(shared_value) == "1";
    packed_nnrt_owned_ =
      grouped_decode_packed_ && ((owned_value != nullptr && std::string(owned_value) == "1") || packed_nnrt_shared_);
    grouped_decode_indexed_ = choice.indexed;
    grouped_decode_fused_output_ = choice.fused_output;
    if (!PrepareGroupedCache(error)) return false;
    if (!LoadGroupedDecode(choice.path, grouped_decode_indexed_, error)) return false;
    grouped_decode_enabled_ = true;
  }
  return true;
}

bool Qwen35ExpertRuntime::AllocateFusedDmaCache(std::string *error) {
  fused_dma_fds_.reserve(fused_dma_layers_);
  fused_dma_mappings_.reserve(fused_dma_layers_);
  for (size_t layer = 0; layer < fused_dma_layers_; ++layer) {
    const int fd = AllocateDmaBuffer(fused_dma_layer_bytes_, error);
    if (fd < 0) {
      ReleaseFusedDmaCache();
      // Keep the accepted grouped path usable on firmware that cannot expose
      // enough DMA-heap pages. The caller will observe an empty binding map
      // and use its original NNRT-owned input tensors.
      if (error) error->clear();
      return true;
    }
    void *mapping = mmap(nullptr, fused_dma_layer_bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapping == MAP_FAILED) {
      close(fd);
      ReleaseFusedDmaCache();
      if (error) error->clear();
      return true;
    }
    fused_dma_fds_.push_back(fd);
    fused_dma_mappings_.push_back(static_cast<uint8_t *>(mapping));
  }
  return true;
}

void Qwen35ExpertRuntime::ReleaseFusedDmaCache() {
  for (size_t index = 0; index < fused_dma_mappings_.size(); ++index) {
    if (fused_dma_mappings_[index] != nullptr) munmap(fused_dma_mappings_[index], fused_dma_layer_bytes_);
  }
  for (const int allocated : fused_dma_fds_) close(allocated);
  fused_dma_fds_.clear();
  fused_dma_mappings_.clear();
  fused_dma_layer_bytes_ = 0;
  fused_dma_slot_bytes_ = 0;
  fused_dma_slot_count_ = 0;
  fused_dma_layers_ = 0;
}

bool Qwen35ExpertRuntime::PreparePackedCopy(PackedCopy &copy, std::string *error) {
  if (!copy.tensor_copy && copy.cache_hits != kTopK &&
      !SyncDmaBuffer(fused_dma_fds_[copy.layer], DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE)) {
    if (error) *error = "cannot begin packed expert DMA write";
    return false;
  }
  copy.copy_routes.reserve(kTopK - copy.cache_hits);
  for (int32_t route = 0; route < kTopK; ++route) {
    if (copy.hit[route]) continue;
    for (size_t index = 0; index < fused_dma_offsets_.size(); ++index) {
      const char *name = index < 3 ? kS16S4GateUpBindings[index].shard_name : kS16S4DownBindings[index - 3].shard_name;
      const ExpertTensorView view = copy.misses[route]->Tensor(name);
      if (view.data == nullptr || view.bytes != fused_dma_tensor_bytes_[index]) {
        if (!copy.tensor_copy && copy.cache_hits != kTopK)
          SyncDmaBuffer(fused_dma_fds_[copy.layer], DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE);
        if (error) *error = std::string("packed expert tensor mismatch: ") + name;
        return false;
      }
      copy.views[route][index] = view;
      copy.copied_bytes += view.bytes;
    }
    const auto *payload = static_cast<const uint8_t *>(copy.misses[route]->PayloadData(fused_dma_slot_bytes_));
    if (payload != nullptr) {
      bool matches_packed_layout = true;
      for (size_t index = 0; index < fused_dma_offsets_.size(); ++index) {
        if (copy.views[route][index].data != payload + fused_dma_offsets_[index]) {
          matches_packed_layout = false;
          break;
        }
      }
      if (matches_packed_layout) copy.payloads[route] = payload;
    }
    copy.copy_routes.push_back(route);
  }

  return true;
}

void Qwen35ExpertRuntime::CopyPackedPayloads(const PackedCopy &copy) {
  auto *packed_data = copy.tensor_copy ? static_cast<uint8_t *>(copy.tensor.data) : fused_dma_mappings_[copy.layer];
  auto copy_route = [this, &copy, packed_data](int32_t route) {
    const size_t slot_base = static_cast<size_t>(copy.routes[route]) * fused_dma_slot_bytes_;
    // The shard payload already matches the packed NNRT ABI. Read it straight
    // into the NNRT-owned tensor to avoid faulting it into a CPU mapping first.
    // Set QWEN35_PACKED_DIRECT_READ=0 for the memcpy diagnostic fallback.
    const char *direct_read_value = std::getenv("QWEN35_PACKED_DIRECT_READ");
    const bool direct_read = direct_read_value == nullptr || std::string(direct_read_value) != "0";
    if (direct_read && copy.misses[route]->ReadPayload(packed_data + slot_base, fused_dma_slot_bytes_)) {
      return;
    }
    if (copy.payloads[route] != nullptr) {
      std::memcpy(packed_data + slot_base, copy.payloads[route], fused_dma_slot_bytes_);
      return;
    }
    for (size_t index = 0; index < fused_dma_offsets_.size(); ++index) {
      const ExpertTensorView &view = copy.views[route][index];
      std::memcpy(packed_data + slot_base + fused_dma_offsets_[index], view.data, view.bytes);
    }
  };
  int32_t copy_thread_count = 1;
  if (const char *value = std::getenv("QWEN35_EXPERT_COPY_THREADS"))
    copy_thread_count = std::clamp(std::atoi(value), 1, kTopK);
  if (copy_thread_count > 1 && copy.copy_routes.size() > 1) {
    if (copy_pool_ == nullptr || copy_pool_->WorkerCount() != static_cast<size_t>(copy_thread_count)) {
      copy_pool_ = std::make_unique<ExpertCopyPool>(copy_thread_count);
    }
    copy_pool_->Run(copy.copy_routes.size(),
                    [&copy_route, &copy](size_t index) { copy_route(copy.copy_routes[index]); });
  } else {
    for (int32_t route : copy.copy_routes) copy_route(route);
  }
}

void Qwen35ExpertRuntime::CopyBankedPayloads(bool tensor_copy, const std::array<TensorView, 6> &bank_tensors,
                                             const std::array<int32_t, 8> &route_to_slot,
                                             const std::array<std::shared_ptr<const ExpertShard>, 8> &misses,
                                             const std::vector<int32_t> &copy_routes) {
  auto copy_route = [this, tensor_copy, &bank_tensors, &route_to_slot, &misses](int32_t route) {
    const int32_t slot = route_to_slot[route];
    const int32_t bank = slot / kBankSlots;
    const int32_t bank_slot = slot % kBankSlots;
    auto *bank_base = tensor_copy ? static_cast<uint8_t *>(bank_tensors[bank].data) : banked_dma_mappings_[bank];
    uint8_t *destination = bank_base + static_cast<size_t>(bank_slot) * fused_dma_slot_bytes_;
    const char *direct_read_value = std::getenv("QWEN35_BANKED_DIRECT_READ");
    if (direct_read_value != nullptr && std::string(direct_read_value) == "1" &&
        misses[route]->ReadPayload(destination, fused_dma_slot_bytes_)) {
      return;
    }
    const void *payload = misses[route]->PayloadData(fused_dma_slot_bytes_);
    if (payload != nullptr) {
      std::memcpy(destination, payload, fused_dma_slot_bytes_);
      return;
    }
    for (size_t index = 0; index < fused_dma_offsets_.size(); ++index) {
      const char *name = index < 3 ? kS16S4GateUpBindings[index].shard_name : kS16S4DownBindings[index - 3].shard_name;
      const ExpertTensorView view = misses[route]->Tensor(name);
      std::memcpy(destination + fused_dma_offsets_[index], view.data, view.bytes);
    }
  };
  int32_t threads = 1;
  if (const char *value = std::getenv("QWEN35_EXPERT_COPY_THREADS")) threads = std::clamp(std::atoi(value), 1, kTopK);
  if (threads > 1 && copy_routes.size() > 1) {
    if (copy_pool_ == nullptr || copy_pool_->WorkerCount() != static_cast<size_t>(threads))
      copy_pool_ = std::make_unique<ExpertCopyPool>(threads);
    copy_pool_->Run(copy_routes.size(), [&copy_route, &copy_routes](size_t index) { copy_route(copy_routes[index]); });
  } else {
    for (int32_t route : copy_routes) copy_route(route);
  }
}

bool Qwen35ExpertRuntime::CopyGroupedShard(NnrtSession *session, const std::string &set, const std::string &slot_prefix,
                                           const ExpertShard &shard, size_t *copied_bytes, std::string *error) {
  for (const TensorBinding &binding : kS16S4GateUpBindings) {
    const ExpertTensorView view = shard.Tensor(binding.shard_name);
    if (view.data == nullptr || !CopyTensor(session, set, (slot_prefix + "_gate_up_" + binding.onnx_name).c_str(),
                                            view.data, view.bytes, error)) {
      return false;
    }
    *copied_bytes += view.bytes;
  }
  for (const TensorBinding &binding : kS16S4DownBindings) {
    const ExpertTensorView view = shard.Tensor(binding.shard_name);
    if (view.data == nullptr ||
        !CopyTensor(session, set, (slot_prefix + "_down_" + binding.onnx_name).c_str(), view.data, view.bytes, error)) {
      return false;
    }
    *copied_bytes += view.bytes;
  }
  return true;
}

bool Qwen35ExpertRuntime::ChooseGroupedModel(const std::string &artifact_root, GroupedModelChoice *choice,
                                             std::string *error) {
  const fs::path root(artifact_root);
  const fs::path s16s4_models = root / "omc/expert_s16s4";
  const fs::path indexed_grouped_model = s16s4_models / "qwen35_moe_s16s4_grouped_top8_indexed_s1.omc";
  const fs::path banked_grouped_model = s16s4_models / "qwen35_moe_s16s4_grouped_top8_packed_banked_s1.omc";
  // Packed32 has the best verified decoder latency on the target device. It
  // also uses half the DMA cache of Packed64 while retaining a useful LRU.
  int32_t requested_packed_slots = kDefaultPackedSlots;
  if (const char *value = std::getenv("QWEN35_PACKED_EXPERT_SLOTS")) {
    requested_packed_slots = std::atoi(value);
  }
  if (!IsPackedSlotCount(static_cast<size_t>(requested_packed_slots))) {
    if (error) *error = "QWEN35_PACKED_EXPERT_SLOTS must be 16, 32, 64, 128, or 384";
    return false;
  }
  const fs::path packed_grouped_model =
    s16s4_models / ("qwen35_moe_s16s4_grouped_top8_packed" + std::to_string(requested_packed_slots) + "_s1.omc");
  const fs::path fused_grouped_model = s16s4_models / "qwen35_moe_s16s4_grouped_top8_fused_s1.omc";
  const fs::path legacy_grouped_model = s16s4_models / "qwen35_moe_s16s4_grouped_top8_s1.omc";
  const bool indexed_requested = std::getenv("QWEN35_DISABLE_INDEXED_EXPERTS") == nullptr;
  const char *packed_value = std::getenv("QWEN35_ENABLE_PACKED_EXPERTS");
  // Make the verified packed ABI the normal path. Set this to 0 to diagnose
  // the indexed ABI on devices with tighter DMA-heap limits.
  const bool packed_requested = packed_value == nullptr || std::string(packed_value) != "0";
  const bool packed_available = packed_requested && ArtifactExists(packed_grouped_model);
  // Packaged experts use bounded DMA staging; standalone-shard FD binding
  // is unsupported.
  const char *banked_value = std::getenv("QWEN35_ENABLE_BANKED_EXPERTS");
  const bool banked_available =
    banked_value != nullptr && std::string(banked_value) == "1" && ArtifactExists(banked_grouped_model);
  const bool indexed_available = !packed_available && indexed_requested && ArtifactExists(indexed_grouped_model);
  fs::path grouped_model = ArtifactExists(fused_grouped_model) ? fused_grouped_model : legacy_grouped_model;
  if (indexed_available) grouped_model = indexed_grouped_model;
  if (packed_available) grouped_model = packed_grouped_model;
  if (banked_available) grouped_model = banked_grouped_model;
  SetGroupedChoice(grouped_model.string(), fused_grouped_model.string(), requested_packed_slots,
                   {banked_available, packed_available, indexed_available}, choice);
  return true;
}

bool Qwen35ExpertRuntime::PrepareGroupedCache(std::string *error) {
  if ((grouped_decode_banked_ && !PrepareBankedDmaCache(error)) ||
      (grouped_decode_packed_ && !PrepareFusedDmaCache(kLayers, grouped_decode_packed_slots_, error)) ||
      (grouped_decode_indexed_ && !PrepareFusedDmaCache(kLayers, kIndexedSlots, error))) {
    return false;
  }
  if (grouped_decode_packed_ && !HasFusedDmaCache() && !packed_nnrt_owned_) {
    // Preserve the packed expert ABI on firmware that does not expose
    // DMA heaps to the app. Reuse one NNRT-owned cache across layers.
    packed_nnrt_owned_ = true;
    packed_nnrt_shared_ = true;
    if (!PrepareFusedDmaCache(kLayers, grouped_decode_packed_slots_, error)) return false;
    std::cerr << "Expert DMA unavailable; using shared NNRT-owned packed cache\n";
  }
  return true;
}

void Qwen35ExpertRuntime::GroupedShapes(bool indexed, TensorShapes *inputs, TensorShapes *outputs) {
  if (grouped_decode_banked_) {
    (*inputs)["expert_ids"] = {kTopK};
    for (int32_t bank = 0; bank < kBankCount; ++bank)
      (*inputs)["packed_bank_" + std::to_string(bank)] = {kBankSlots, 1777664};
  } else if (zero_copy_experts_) {
    for (int32_t slot = 0; slot < kTopK; ++slot) (*inputs)["expert_" + std::to_string(slot) + "_shard"] = {1777664};
  } else if (grouped_decode_packed_) {
    (*inputs)["expert_ids"] = {kTopK};
    (*inputs)["packed_weights"] = {static_cast<int32_t>(grouped_decode_packed_slots_), 1777664};
  } else if (indexed) {
    (*inputs)["expert_ids"] = {kTopK};
  }
  const int32_t slot_count = indexed ? kIndexedSlots : kTopK;
  for (int32_t slot = 0; !grouped_decode_packed_ && !zero_copy_experts_ && !grouped_decode_banked_ && slot < slot_count;
       ++slot) {
    const std::string prefix = "expert_" + std::to_string(slot);
    (*inputs)[prefix + "_gate_up_w"] = {1024, 1024};
    (*inputs)[prefix + "_gate_up_bias"] = {1024};
    (*inputs)[prefix + "_gate_up_w_scale"] = {8, 16, 128, 8};
    (*inputs)[prefix + "_down_w"] = {2048, 256};
    (*inputs)[prefix + "_down_bias"] = {2048};
    (*inputs)[prefix + "_down_w_scale"] = {16, 4, 128, 8};
  }
  if (!grouped_decode_fused_output_) {
    const int32_t output_slots = zero_copy_experts_ ? kTopK : slot_count;
    for (int32_t slot = 0; slot < output_slots; ++slot) (*outputs)["output_" + std::to_string(slot)] = {1, kHidden};
  }
  if (grouped_decode_fused_output_) (*outputs)["experts"] = {kTopK, kHidden};
}

bool Qwen35ExpertRuntime::BindDecodeExperts(int32_t layer, const std::array<int32_t, 8> &experts,
                                            std::array<int32_t, 8> *route_to_slot, std::string *error) {
  bool bound = false;
  if (grouped_decode_banked_) {
    bound = BindBankedInputs(grouped_decode_.get(), "main", layer, experts, route_to_slot, error);
    if (bound &&
        !CopyTensor(grouped_decode_.get(), "main", "expert_ids", route_to_slot->data(), sizeof(*route_to_slot), error))
      return false;
  } else if (zero_copy_experts_) {
    bound = BindZeroCopyInputs(grouped_decode_.get(), "main", "expert_", layer, experts, route_to_slot, error);
  } else if (grouped_decode_packed_) {
    bound = BindPackedInputs(grouped_decode_.get(), "main", layer, experts, route_to_slot, error);
    if (bound && !CopyTensor(grouped_decode_.get(), "main", "expert_ids", route_to_slot->data(), sizeof(*route_to_slot),
                             error)) {
      return false;
    }
  } else if (grouped_decode_indexed_) {
    bound = BindIndexedInputs(grouped_decode_.get(), "main", "expert_", layer, experts, route_to_slot, error);
    if (bound && !CopyTensor(grouped_decode_.get(), "main", "expert_ids", route_to_slot->data(), sizeof(*route_to_slot),
                             error)) {
      return false;
    }
  } else {
    bound = BindGroupedInputs(grouped_decode_.get(), "main", "expert_", layer, experts, route_to_slot, error);
  }
  if (!bound) {
    return false;
  }
  return true;
}

bool Qwen35ExpertRuntime::CopyGroupedOutputs(const std::array<int32_t, 8> &route_to_slot, uint16_t *outputs,
                                             std::string *error) {
  if (grouped_decode_fused_output_) {
    const TensorView result = grouped_decode_->Output("main", "experts");
    const size_t bytes = static_cast<size_t>(kTopK) * kHidden * sizeof(uint16_t);
    if (result.data == nullptr || result.bytes != bytes) {
      if (error) *error = "fused grouped S16S4 output tensor size mismatch";
      return false;
    }
    const auto *source = static_cast<const uint16_t *>(result.data);
    if (grouped_decode_banked_ || grouped_decode_indexed_ || grouped_decode_packed_) {
      std::memcpy(outputs, source, bytes);
      return true;
    }
    for (int32_t route = 0; route < kTopK; ++route) {
      std::memcpy(outputs + static_cast<size_t>(route) * kHidden,
                  source + static_cast<size_t>(route_to_slot[route]) * kHidden,
                  static_cast<size_t>(kHidden) * sizeof(uint16_t));
    }
    return true;
  }
  for (int32_t route = 0; route < kTopK; ++route) {
    const int32_t slot = route_to_slot[route];
    const TensorView result = grouped_decode_->Output("main", "output_" + std::to_string(slot));
    const size_t bytes = static_cast<size_t>(kHidden) * sizeof(uint16_t);
    if (result.data == nullptr || result.bytes != bytes) {
      if (error) *error = "grouped S16S4 output tensor size mismatch";
      return false;
    }
    std::memcpy(outputs + static_cast<size_t>(route) * kHidden, result.data, bytes);
  }
  return true;
}

bool Qwen35ExpertRuntime::CopyStageWeights(const Stage *stage, NnrtSession *session, const ExpertShard &shard,
                                           std::string *error) {
  const TensorBinding *bindings = nullptr;
  size_t binding_count = 0;
  if (stage->s16s4) {
    bindings = stage->gate_up ? kS16S4GateUpBindings : kS16S4DownBindings;
    binding_count = 3;
  } else if (stage->gate_up) {
    bindings = kGateUpBindings;
    binding_count = 4;
  } else {
    bindings = kDownBindings;
    binding_count = 4;
  }
  for (size_t index = 0; index < binding_count; ++index) {
    const TensorBinding &binding = bindings[index];
    const ExpertTensorView view = shard.Tensor(binding.shard_name);
    if (view.data == nullptr) {
      if (error) {
        *error = std::string("missing expert shard tensor: ") + binding.shard_name;
      }
      return false;
    }
    if (!CopyTensor(session, "main", binding.onnx_name, view.data, view.bytes, error)) {
      return false;
    }
  }
  return true;
}

bool Qwen35ExpertRuntime::ValidatePackedInput(const NnrtSession *session, int32_t layer, bool tensor_copy,
                                              const TensorView &packed_tensor, std::string *error) {
  if (!s16s4_enabled_ || session == nullptr || layer < 0 || layer >= kLayers ||
      fused_dma_slot_count_ < kMinPackedSlots || static_cast<size_t>(layer) >= fused_dma_layers_ ||
      (!tensor_copy &&
       (static_cast<size_t>(layer) >= fused_dma_mappings_.size() || fused_dma_mappings_[layer] == nullptr)) ||
      (tensor_copy && (packed_tensor.data == nullptr || packed_tensor.bytes != fused_dma_layer_bytes_))) {
    if (error) *error = "packed S16S4 expert cache is unavailable";
    return false;
  }

  return true;
}

bool Qwen35ExpertRuntime::PrepareBankInputs(NnrtSession *session, const std::string &set, int32_t layer,
                                            bool &tensor_copy, std::array<TensorView, 6> &bank_tensors,
                                            std::string *error) {
  const char *tensor_copy_value = std::getenv("QWEN35_BANKED_TENSOR_COPY");
  tensor_copy = tensor_copy_value == nullptr || std::string(tensor_copy_value) != "0";
  if (tensor_copy && session != nullptr) {
    for (int32_t bank = 0; bank < kBankCount; ++bank) {
      bank_tensors[bank] = session->Input(set, "current_packed_bank_" + std::to_string(bank));
      if (bank_tensors[bank].data == nullptr) {
        bank_tensors[bank] = session->Input(set, "packed_bank_" + std::to_string(bank));
      }
    }
  }
  const size_t bank_bytes = static_cast<size_t>(kBankSlots) * fused_dma_slot_bytes_;
  if (!grouped_decode_banked_ || session == nullptr || layer < 0 || layer >= kLayers || banked_dma_fds_[0] < 0 ||
      (tensor_copy && std::any_of(bank_tensors.begin(), bank_tensors.end(), [bank_bytes](const TensorView &view) {
         return view.data == nullptr || view.bytes != bank_bytes;
       }))) {
    if (error) *error = "banked S16S4 expert cache is unavailable";
    return false;
  }
  return true;
}

bool Qwen35ExpertRuntime::LayoutFusedSlot(std::string *error) {
  const std::shared_ptr<const ExpertShard> probe = s16s4_cache_.Acquire(0, 0, error);
  if (probe == nullptr) return false;
  const std::array<const char *, 6> names = {
    "gate_up.w", "gate_up.bias", "gate_up.w_scale", "down.w", "down.bias", "down.w_scale",
  };
  size_t slot_bytes = 0;
  for (size_t index = 0; index < names.size(); ++index) {
    const ExpertTensorView view = probe->Tensor(names[index]);
    if (view.data == nullptr || view.bytes == 0) {
      if (error) *error = std::string("missing fused DMA cache tensor: ") + names[index];
      return false;
    }
    slot_bytes = AlignUp(slot_bytes, kDmaAlignment);
    fused_dma_offsets_[index] = slot_bytes;
    fused_dma_tensor_bytes_[index] = view.bytes;
    slot_bytes += view.bytes;
  }
  fused_dma_slot_bytes_ = AlignUp(slot_bytes, kDmaAlignment);
  return true;
}

bool Qwen35ExpertRuntime::CopyStageActivation(const Stage *stage, NnrtSession *session, int32_t rows, int32_t capacity,
                                              const uint16_t *input, std::string *error) {
  const std::string input_name = stage->gate_up ? "hidden" : "gated";
  TensorView activation = session->Input("main", input_name);
  const size_t row_bytes = static_cast<size_t>(stage->gate_up ? kHidden : kIntermediate) * sizeof(uint16_t);
  const size_t input_bytes = static_cast<size_t>(capacity) * row_bytes;
  if (activation.data == nullptr || activation.bytes != input_bytes) {
    if (error) *error = "expert activation tensor size mismatch";
    return false;
  }
  std::memcpy(activation.data, input, static_cast<size_t>(rows) * row_bytes);
  if (rows < capacity) {
    std::memset(static_cast<uint8_t *>(activation.data) + static_cast<size_t>(rows) * row_bytes, 0,
                static_cast<size_t>(capacity - rows) * row_bytes);
  }
  return true;
}

void Qwen35ExpertRuntime::SetGroupedChoice(const std::string &path, const std::string &fused_path,
                                           int32_t requested_packed_slots, const std::array<bool, 3> &available,
                                           GroupedModelChoice *choice) {
  choice->path = path;
  choice->banked = available[0];
  choice->packed = available[1] && !available[0];
  choice->indexed = available[2] && !available[0];
  choice->packed_slots = available[1] ? static_cast<size_t>(requested_packed_slots) : 0;
  choice->fused_output = choice->banked || choice->packed || choice->indexed || path == fused_path;
}

}  // namespace mslite_llm::qwen35
