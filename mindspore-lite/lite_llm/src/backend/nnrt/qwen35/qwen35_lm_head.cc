// Copyright 2026 Huawei Technologies Co., Ltd
// Adapted from HMOS-Device-AI llm/engine/runtime; preserves its paged Qwen3.5 execution.
#include "backend/nnrt/qwen35/qwen35_lm_head.h"

#include <fcntl.h>
#include <linux/dma-buf.h>
#include <linux/dma-heap.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17): lite_llm requires C++17.
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

#include "manifest/artifact_file.h"

namespace fs = std::filesystem;

namespace mslite_llm::qwen35 {
namespace {

constexpr int32_t kHidden = 2048;
constexpr int32_t kVocab = 248320;
constexpr int32_t kDefaultTile = 8192;
constexpr int32_t kLargeTile = 16384;
constexpr int32_t kExtraLargeTile = 31040;
constexpr int32_t kMaximumTile = 62080;

bool Copy(TensorView view, const void *data, size_t bytes, const char *name, std::string *error) {
  if (view.data == nullptr || data == nullptr || view.bytes != bytes) {
    if (error) *error = std::string("LM Head tensor mismatch: ") + name;
    return false;
  }
  std::memcpy(view.data, data, bytes);
  return true;
}

bool ReadFile(const fs::path &path, TensorView view, std::string *error) {
  return view.data != nullptr && ReadArtifact(path, view.data, view.bytes, error);
}

bool ResidentEnabled() {
  const char *value = std::getenv("QWEN35_DISABLE_RESIDENT_LM_HEAD");
  return value == nullptr || std::string(value) != "1";
}

bool Top1Enabled() {
  const char *value = std::getenv("QWEN35_DISABLE_TOP1");
  return value == nullptr || std::string(value) != "1";
}

int AllocateDmaBuffer(size_t bytes, std::string *error) {
  for (const char *path :
       {"/dev/dma_heap/npu_device_hugepage_heap", "/dev/dma_heap/system_heap", "/dev/dma_heap/system"}) {
    const int heap = open(path, O_RDWR | O_CLOEXEC);
    if (heap < 0) continue;
    dma_heap_allocation_data allocation{};
    allocation.len = bytes;
    allocation.fd_flags = O_RDWR | O_CLOEXEC;
    const int status = ioctl(heap, DMA_HEAP_IOCTL_ALLOC, &allocation);
    close(heap);
    if (status == 0 && allocation.fd >= 0) return allocation.fd;
  }
  if (error) *error = "cannot allocate resident LM Head DMA buffer";
  return -1;
}

bool SyncDmaBuffer(int fd, uint64_t flags) {
  dma_buf_sync synchronization{};
  synchronization.flags = flags;
  return ioctl(fd, DMA_BUF_IOCTL_SYNC, &synchronization) == 0;
}

}  // namespace

Qwen35LmHead::~Qwen35LmHead() {
  if (mapping_ != nullptr) {
    munmap(const_cast<uint8_t *>(mapping_), mapping_bytes_);
  }
  if (fd_ >= 0) close(fd_);
}

bool Qwen35LmHead::Load(const std::string &artifact_root, std::string *error) {
  const fs::path root(artifact_root);
  const fs::path omc = root / "omc/lm_head";
  if (ResidentEnabled() && ArtifactExists(omc / "lm_head_62080.omc")) {
    bool dma_unavailable = false;
    file_backed_mode_ = LoadFileBacked(artifact_root, &dma_unavailable, error);
    if (file_backed_mode_ || !dma_unavailable) return file_backed_mode_;
    // Use the same complete vocabulary through a reused tile buffer when the
    // application cannot allocate or map the full resident DMA weight buffer.
    std::cerr << "Resident LM Head DMA unavailable; using tiled LM Head\n";
    if (error) error->clear();
  }
  const fs::path main_tile = SelectTile(omc);
  if (!norm_.Load((omc / "final_norm.omc").string(), "", error) ||
      !norm_.CreateTensorSet("main", {{"hidden", {1, kHidden}}, {"weight", {kHidden}}}, {{"normalized", {1, kHidden}}},
                             {}, std::unordered_set<std::string>{}, error) ||
      !ReadFile(root / "global/final_norm_fp16.bin", norm_.Input("main", "weight"), error) ||
      !LoadSession(&tile8192_, main_tile.string(), tile_rows_, error) ||
      (kVocab % tile_rows_ != 0 &&
       !LoadSession(&tile2560_, (omc / "lm_head_2560.omc").string(), kVocab % tile_rows_, error))) {
    return false;
  }
  const fs::path weights = root / "global/lm_head_nz_fp16.bin";
  size_t resource_offset = 0;
  fd_ = OpenArtifact(weights.string(), &mapping_bytes_, &resource_offset, error);
  if (fd_ < 0 || mapping_bytes_ != static_cast<size_t>(kVocab) * kHidden * sizeof(uint16_t)) {
    if (error) *error = "cannot open tiled LM Head: " + weights.string();
    return false;
  }
  void *mapping = mmap(nullptr, mapping_bytes_, PROT_READ, MAP_PRIVATE, fd_, static_cast<off_t>(resource_offset));
  if (mapping == MAP_FAILED) {
    mapping_ = nullptr;
    if (error) *error = "cannot mmap tiled LM Head";
    return false;
  }
  mapping_ = static_cast<const uint8_t *>(mapping);
  return true;
}

bool Qwen35LmHead::LoadFileBacked(const std::string &artifact_root, bool *dma_unavailable, std::string *error) {
  const fs::path root(artifact_root);
  const fs::path omc = root / "omc/lm_head";
  const fs::path weights = root / "global/lm_head_nz_fp16.bin";
  size_t source_size = 0, source_offset = 0;
  const int source_fd = OpenArtifact(weights.string(), &source_size, &source_offset, error);
  const size_t expected = static_cast<size_t>(kVocab) * kHidden * sizeof(uint16_t);
  if (source_fd < 0 || source_size != expected) {
    if (error) *error = "cannot open resident LM Head: " + weights.string();
    if (source_fd >= 0) close(source_fd);
    return false;
  }
  mapping_bytes_ = expected;
  const void *source =
    mmap(nullptr, mapping_bytes_, PROT_READ, MAP_PRIVATE, source_fd, static_cast<off_t>(source_offset));
  fd_ = AllocateDmaBuffer(mapping_bytes_, error);
  void *mapping = fd_ < 0 ? MAP_FAILED : mmap(nullptr, mapping_bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
  if (source == MAP_FAILED || mapping == MAP_FAILED) {
    *dma_unavailable = source != MAP_FAILED && mapping == MAP_FAILED;
    if (source != MAP_FAILED) munmap(const_cast<void *>(source), mapping_bytes_);
    if (mapping != MAP_FAILED) munmap(mapping, mapping_bytes_);
    if (fd_ >= 0) close(fd_);
    fd_ = -1;
    close(source_fd);
    mapping_ = nullptr;
    if (error && error->empty()) *error = "cannot mmap resident LM Head DMA buffer";
    return false;
  }
  mapping_ = static_cast<const uint8_t *>(mapping);
  SyncDmaBuffer(fd_, DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE);
  std::memcpy(mapping, source, mapping_bytes_);
  SyncDmaBuffer(fd_, DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE);
  munmap(const_cast<void *>(source), mapping_bytes_);
  close(source_fd);

  return CreateResidentSets(artifact_root, error);
}

bool Qwen35LmHead::LoadSession(NnrtSession *session, const std::string &path, int32_t rows, std::string *error) {
  return session->Load(path, "", error) &&
         session->CreateTensorSet("main", {{"hidden", {1, kHidden}}, {"weight", {rows, kHidden}}},
                                  {{"logits", {1, rows}}}, {}, std::unordered_set<std::string>{}, error);
}

bool Qwen35LmHead::Run(const uint16_t *hidden, uint16_t *logits, double *elapsed_ms, std::string *error) {
  if ((!file_backed_mode_ && mapping_ == nullptr) || hidden == nullptr || (!top1_mode_ && logits == nullptr) ||
      !Copy(norm_.Input("main", "hidden"), hidden, kHidden * sizeof(uint16_t), "final hidden", error)) {
    return false;
  }
  double total = 0.0;
  double elapsed = 0.0;
  if (!norm_.Run("main", &elapsed, error)) {
    if (error) *error = "resident LM Head norm: " + *error;
    return false;
  }
  total += elapsed;
  const TensorView normalized = norm_.Output("main", "normalized");
  if (top1_mode_) return RunTop1(normalized, total, elapsed_ms, error);
  if (file_backed_mode_) {
    for (int32_t tile = 0; tile < kVocab / kMaximumTile; ++tile) {
      if (!RunFileTile(tile, static_cast<const uint16_t *>(normalized.data), logits, &elapsed, error)) {
        if (error) *error = "resident LM Head tile " + std::to_string(tile) + ": " + *error;
        return false;
      }
      total += elapsed;
    }
    if (elapsed_ms != nullptr) *elapsed_ms = total;
    return true;
  }
  for (int32_t first = 0; first < kVocab; first += tile_rows_) {
    const int32_t rows = std::min(tile_rows_, kVocab - first);
    NnrtSession *session = rows == tile_rows_ ? &tile8192_ : &tile2560_;
    if (!RunTile(session, first, rows, static_cast<const uint16_t *>(normalized.data), logits, &elapsed, error)) {
      return false;
    }
    total += elapsed;
  }
  if (elapsed_ms != nullptr) *elapsed_ms = total;
  return true;
}

bool Qwen35LmHead::RunFileTile(int32_t tile, const uint16_t *normalized, uint16_t *logits, double *elapsed_ms,
                               std::string *error) {
  const std::string set = "tile_" + std::to_string(tile);
  if (!Copy(tile8192_.Input(set, "hidden"), normalized, kHidden * sizeof(uint16_t), "resident normalized hidden",
            error) ||
      !tile8192_.Run(set, elapsed_ms, error)) {
    return false;
  }
  const TensorView output = tile8192_.Output(set, "logits");
  const size_t bytes = static_cast<size_t>(kMaximumTile) * sizeof(uint16_t);
  if (output.data == nullptr || output.bytes != bytes) {
    if (error) *error = "file-backed LM Head output size mismatch";
    return false;
  }
  std::memcpy(logits + static_cast<size_t>(tile) * kMaximumTile, output.data, bytes);
  return true;
}

bool Qwen35LmHead::RunTile(NnrtSession *session, int32_t first, int32_t rows, const uint16_t *normalized,
                           uint16_t *logits, double *elapsed_ms, std::string *error) {
  const size_t offset = static_cast<size_t>(first) * kHidden * sizeof(uint16_t);
  const size_t bytes = static_cast<size_t>(rows) * kHidden * sizeof(uint16_t);
  if (offset > mapping_bytes_ || bytes > mapping_bytes_ - offset ||
      !Copy(session->Input("main", "hidden"), normalized, kHidden * sizeof(uint16_t), "normalized hidden", error) ||
      !Copy(session->Input("main", "weight"), mapping_ + offset, bytes, "LM Head tile", error) ||
      !session->Run("main", elapsed_ms, error)) {
    return false;
  }
  TensorView output = session->Output("main", "logits");
  if (output.data == nullptr || output.bytes != rows * sizeof(uint16_t)) {
    if (error) *error = "LM Head output size mismatch";
    return false;
  }
  std::memcpy(logits + first, output.data, output.bytes);
  return true;
}

std::filesystem::path Qwen35LmHead::SelectTile(const std::filesystem::path &omc) {
  const fs::path maximum_tile = omc / "lm_head_62080.omc";
  const fs::path extra_large_tile = omc / "lm_head_31040.omc";
  const fs::path large_tile = omc / "lm_head_16384.omc";
  tile_rows_ =
    ArtifactExists(maximum_tile)
      ? kMaximumTile
      : (ArtifactExists(extra_large_tile) ? kExtraLargeTile : (ArtifactExists(large_tile) ? kLargeTile : kDefaultTile));
  return omc /
         (tile_rows_ == kMaximumTile
            ? "lm_head_62080.omc"
            : (tile_rows_ == kExtraLargeTile ? "lm_head_31040.omc"
                                             : (tile_rows_ == kLargeTile ? "lm_head_16384.omc" : "lm_head_8192.omc")));
}

bool Qwen35LmHead::CreateResidentSets(const std::string &artifact_root, std::string *error) {
  const fs::path root(artifact_root);
  const fs::path omc = root / "omc/lm_head";
  const fs::path top1_path = omc / "lm_head_top1.omc";
  top1_mode_ = Top1Enabled() && ArtifactExists(top1_path);
  if (!norm_.Load((omc / "final_norm.omc").string(), "", error) ||
      !norm_.CreateTensorSet("main", {{"hidden", {1, kHidden}}, {"weight", {kHidden}}}, {{"normalized", {1, kHidden}}},
                             {}, std::unordered_set<std::string>{}, error) ||
      !ReadFile(root / "global/final_norm_fp16.bin", norm_.Input("main", "weight"), error) ||
      (top1_mode_ ? !top1_.Load(top1_path.string(), "", error)
                  : !tile8192_.Load((omc / "lm_head_62080.omc").string(), "", error))) {
    return false;
  }
  if (top1_mode_) {
    const FileInputBindings files = {{"weight", {fd_, mapping_bytes_, 0}}};
    if (!top1_.CreateTensorSet("main", {{"hidden", {1, kHidden}}, {"weight", {kVocab, kHidden}}}, {{"token_ids", {1}}},
                               {}, SharedInputKeys{}, {}, files, error)) {
      return false;
    }
    munmap(const_cast<uint8_t *>(mapping_), mapping_bytes_);
    mapping_ = nullptr;
    tile_rows_ = kMaximumTile;
    return true;
  }
  const size_t tile_bytes = static_cast<size_t>(kMaximumTile) * kHidden * sizeof(uint16_t);
  for (int32_t tile = 0; tile < kVocab / kMaximumTile; ++tile) {
    const std::string set = "tile_" + std::to_string(tile);
    const FileInputBindings files = {{"weight", {fd_, mapping_bytes_, static_cast<size_t>(tile) * tile_bytes}}};
    if (!tile8192_.CreateTensorSet(set, {{"hidden", {1, kHidden}}, {"weight", {kMaximumTile, kHidden}}},
                                   {{"logits", {1, kMaximumTile}}}, {}, SharedInputKeys{{"hidden", "lm_head/hidden"}},
                                   {}, files, error)) {
      return false;
    }
  }
  munmap(const_cast<uint8_t *>(mapping_), mapping_bytes_);
  mapping_ = nullptr;
  tile_rows_ = kMaximumTile;
  return true;
}

bool Qwen35LmHead::RunTop1(const TensorView &normalized, double total, double *elapsed_ms, std::string *error) {
  double elapsed = 0.0;
  if (!Copy(top1_.Input("main", "hidden"), normalized.data, kHidden * sizeof(uint16_t), "Top1 normalized hidden",
            error) ||
      !top1_.Run("main", &elapsed, error)) {
    return false;
  }
  const TensorView output = top1_.Output("main", "token_ids");
  if (output.data == nullptr || output.bytes != sizeof(int32_t)) {
    if (error) *error = "Top1 token output size mismatch";
    return false;
  }
  token_id_ = *static_cast<const int32_t *>(output.data);
  if (token_id_ < 0 || token_id_ >= kVocab) {
    if (error) *error = "Top1 token output is outside vocabulary";
    return false;
  }
  if (elapsed_ms != nullptr) *elapsed_ms = total + elapsed;
  return true;
}

}  // namespace mslite_llm::qwen35
