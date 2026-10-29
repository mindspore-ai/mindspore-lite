// Copyright 2026 Huawei Technologies Co., Ltd
// Adapted from HMOS-Device-AI llm/engine/runtime; preserves its paged Qwen3.5 execution.
#include "backend/nnrt/qwen35/moe_expert_cache.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstring>
#include <limits>

#include "manifest/artifact_file.h"

namespace mslite_llm::qwen35 {
namespace {

constexpr char kGptqMagic[] = "Q35GPTQ\0";
constexpr char kS16S4Magic[] = "Q35S16\0\0";
constexpr size_t kMagicBytes = 8;
constexpr uint32_t kVersion = 1;
constexpr size_t kHeaderBytes = 64;
constexpr size_t kEntryBytes = 128;

bool ReadU32(const uint8_t *data, size_t size, size_t offset, uint32_t *value) {
  if (offset > size || size - offset < sizeof(uint32_t)) return false;
  std::memcpy(value, data + offset, sizeof(uint32_t));
  return true;
}

bool ReadU64(const uint8_t *data, size_t size, size_t offset, uint64_t *value) {
  if (offset > size || size - offset < sizeof(uint64_t)) return false;
  std::memcpy(value, data + offset, sizeof(uint64_t));
  return true;
}

bool ValidRange(size_t offset, size_t bytes, size_t total) { return offset <= total && bytes <= total - offset; }

std::string ShardPath(const std::string &root, int32_t layer, int32_t expert, const std::string &extension) {
  return root + "/layer_" + (layer < 10 ? "0" : "") + std::to_string(layer) + "/expert_" +
         (expert < 100 ? (expert < 10 ? "00" : "0") : "") + std::to_string(expert) + extension;
}

}  // namespace

ExpertShard::~ExpertShard() {
  if (mapping_ != nullptr) munmap(const_cast<uint8_t *>(mapping_), mapping_size_);
  if (fd_ >= 0) close(fd_);
}

bool ExpertShard::Load(const std::string &path, std::string *error) {
  if (mapping_ != nullptr || fd_ >= 0) {
    if (error) *error = "expert shard is already loaded";
    return false;
  }
  fd_ = OpenArtifact(path, &mapping_size_, &file_offset_, error);
  if (fd_ < 0) {
    if (error) *error = "cannot open expert shard: " + path;
    return false;
  }
  if (mapping_size_ < kHeaderBytes) {
    if (error) *error = "invalid expert shard size: " + path;
    close(fd_);
    fd_ = -1;
    return false;
  }
  void *mapping = mmap(nullptr, mapping_size_, PROT_READ, MAP_PRIVATE, fd_, static_cast<off_t>(file_offset_));
  if (mapping == MAP_FAILED) {
    mapping_ = nullptr;
    if (error) *error = "mmap failed for expert shard: " + path;
    close(fd_);
    fd_ = -1;
    mapping_size_ = 0;
    return false;
  }
  mapping_ = static_cast<const uint8_t *>(mapping);
  uint32_t count = 0;
  if (!ValidateHeader(path, count, error)) return false;
  if (!LoadEntries(count, error)) return false;
  path_ = path;
  return true;
}

ExpertTensorView ExpertShard::Tensor(const std::string &name) const {
  const auto found = entries_.find(name);
  return found == entries_.end() ? ExpertTensorView{} : found->second.view;
}

const void *ExpertShard::PayloadData(size_t bytes) const {
  if (mapping_ == nullptr || payload_offset_ > mapping_size_ || bytes > mapping_size_ - payload_offset_) {
    return nullptr;
  }
  return mapping_ + payload_offset_;
}

bool ExpertShard::ReadPayload(void *destination, size_t bytes) const {
  if (destination == nullptr || fd_ < 0 || payload_offset_ > mapping_size_ || bytes > mapping_size_ - payload_offset_) {
    return false;
  }
  uint8_t *output = static_cast<uint8_t *>(destination);
  size_t done = 0;
  while (done < bytes) {
    const ssize_t count =
      pread(fd_, output + done, bytes - done, static_cast<off_t>(file_offset_ + payload_offset_ + done));
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) return false;
    done += static_cast<size_t>(count);
  }
  return true;
}

bool ExpertCache::Load(const std::string &shard_root, size_t max_shards, std::string *error,
                       const std::string &extension) {
  if (shard_root.empty() || max_shards == 0 || extension.empty() || extension.front() != '.') {
    if (error) *error = "expert cache requires a shard root and max_shards > 0";
    return false;
  }
  Clear();
  shard_root_ = shard_root;
  extension_ = extension;
  max_shards_ = max_shards;
  return true;
}

std::shared_ptr<const ExpertShard> ExpertCache::Acquire(int32_t layer, int32_t expert, std::string *error) {
  if (layer < 0 || expert < 0 || max_shards_ == 0) {
    if (error) *error = "invalid expert cache request";
    return nullptr;
  }
  const Key key{layer, expert};
  auto found = entries_.find(key);
  if (found != entries_.end()) {
    lru_.erase(found->second.lru);
    lru_.push_front(key);
    found->second.lru = lru_.begin();
    return found->second.shard;
  }
  auto shard = std::make_shared<ExpertShard>();
  if (!shard->Load(ShardPath(shard_root_, layer, expert, extension_), error)) {
    return nullptr;
  }
  while (entries_.size() >= max_shards_ && !lru_.empty()) {
    const Key evicted = lru_.back();
    lru_.pop_back();
    entries_.erase(evicted);
  }
  lru_.push_front(key);
  entries_.emplace(key, Entry{shard, lru_.begin()});
  return shard;
}

void ExpertCache::Clear() {
  entries_.clear();
  lru_.clear();
}

bool ExpertShard::LoadEntries(uint32_t count, std::string *error) {
  entries_.reserve(count);
  for (uint32_t index = 0; index < count; ++index) {
    const size_t base = kHeaderBytes + static_cast<size_t>(index) * kEntryBytes;
    if (!LoadEntry(base, error)) return false;
  }
  return true;
}

bool ExpertShard::LoadEntry(size_t base, std::string *error) {
  char name_buffer[65] = {};
  std::memcpy(name_buffer, mapping_ + base, 64);
  uint32_t dtype = 0;
  uint32_t rank = 0;
  if (!ReadU32(mapping_, mapping_size_, base + 64, &dtype) || !ReadU32(mapping_, mapping_size_, base + 68, &rank) ||
      rank > 4U) {
    if (error) *error = "invalid expert tensor entry";
    return false;
  }
  ExpertTensorView view;
  view.dtype_code = dtype;
  view.rank = rank;
  for (uint32_t dim = 0; dim < 4U; ++dim) {
    if (!ReadU64(mapping_, mapping_size_, base + 72 + dim * 8, &view.shape[dim])) {
      if (error) *error = "invalid expert tensor shape";
      return false;
    }
  }
  uint64_t offset = 0;
  uint64_t bytes = 0;
  if (!ReadU64(mapping_, mapping_size_, base + 104, &offset) || !ReadU64(mapping_, mapping_size_, base + 112, &bytes) ||
      offset > std::numeric_limits<size_t>::max() || bytes > std::numeric_limits<size_t>::max() ||
      !ValidRange(static_cast<size_t>(offset), static_cast<size_t>(bytes), mapping_size_)) {
    if (error) *error = "invalid expert tensor byte range";
    return false;
  }
  view.data = mapping_ + static_cast<size_t>(offset);
  view.bytes = static_cast<size_t>(bytes);
  Entry entry;
  entry.name = name_buffer;
  entry.view = view;
  entry.offset = static_cast<size_t>(offset);
  entries_.emplace(entry.name, std::move(entry));
  return true;
}

bool ExpertShard::ValidateHeader(const std::string &path, uint32_t &count, std::string *error) {
  if (std::memcmp(mapping_, kGptqMagic, kMagicBytes) != 0 && std::memcmp(mapping_, kS16S4Magic, kMagicBytes) != 0) {
    if (error) *error = "invalid expert shard magic: " + path;
    return false;
  }
  uint32_t version = 0;
  uint64_t header_bytes = 0;
  if (!ReadU32(mapping_, mapping_size_, 8, &version) || !ReadU32(mapping_, mapping_size_, 12, &count) ||
      !ReadU64(mapping_, mapping_size_, 24, &header_bytes) || version != kVersion || header_bytes < kHeaderBytes ||
      header_bytes > mapping_size_ || count > 64U || count > (header_bytes - kHeaderBytes) / kEntryBytes) {
    if (error) *error = "invalid expert shard header: " + path;
    return false;
  }
  payload_offset_ = static_cast<size_t>(header_bytes);
  return true;
}

}  // namespace mslite_llm::qwen35
