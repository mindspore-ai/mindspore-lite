// Copyright 2026 Huawei Technologies Co., Ltd
// Adapted from HMOS-Device-AI llm/engine/runtime; preserves its paged Qwen3.5 execution.
#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>

namespace mslite_llm::qwen35 {

struct ExpertTensorView {
  const void *data = nullptr;
  size_t bytes = 0;
  uint32_t dtype_code = 0;
  uint32_t rank = 0;
  uint64_t shape[4] = {0, 0, 0, 0};
};

class ExpertShard {
 public:
  ExpertShard() = default;
  ~ExpertShard();
  ExpertShard(const ExpertShard &) = delete;
  ExpertShard &operator=(const ExpertShard &) = delete;

  bool Load(const std::string &path, std::string *error);
  ExpertTensorView Tensor(const std::string &name) const;
  // Returns the data region immediately following the aligned shard header.
  // S16S4 paged shards preserve the NPU packed-slot layout in this region.
  const void *PayloadData(size_t bytes) const;
  // Reads the packed payload directly into a caller-owned DMA mapping. This
  // avoids faulting the mmap into userspace before copying it again.
  bool ReadPayload(void *destination, size_t bytes) const;
  // The file remains open while this shard is cached. This makes it usable as
  // an NNRT file-backed tensor without copying its packed payload.
  int Fd() const { return fd_; }
  size_t FileSize() const { return mapping_size_; }
  size_t FileOffset() const { return file_offset_; }
  size_t BindingSize() const { return file_offset_ + mapping_size_; }
  const std::string &Path() const { return path_; }

 private:
  bool ValidateHeader(const std::string &path, uint32_t &count, std::string *error);
  bool LoadEntry(size_t base, std::string *error);
  bool LoadEntries(uint32_t count, std::string *error);
  struct Entry {
    std::string name;
    ExpertTensorView view;
    size_t offset = 0;
  };

  int fd_ = -1;
  const uint8_t *mapping_ = nullptr;
  size_t mapping_size_ = 0;
  size_t payload_offset_ = 0;
  size_t file_offset_ = 0;
  std::string path_;
  std::unordered_map<std::string, Entry> entries_;
};

class ExpertCache {
 public:
  ExpertCache() = default;
  ~ExpertCache() = default;
  ExpertCache(const ExpertCache &) = delete;
  ExpertCache &operator=(const ExpertCache &) = delete;

  bool Load(const std::string &shard_root, size_t max_shards, std::string *error, const std::string &extension = ".q4");
  std::shared_ptr<const ExpertShard> Acquire(int32_t layer, int32_t expert, std::string *error);
  void Clear();
  size_t Size() const { return entries_.size(); }
  size_t MaxShards() const { return max_shards_; }

 private:
  struct Key {
    int32_t layer = 0;
    int32_t expert = 0;
    bool operator==(const Key &other) const { return layer == other.layer && expert == other.expert; }
  };
  struct KeyHash {
    size_t operator()(const Key &key) const {
      return (static_cast<size_t>(static_cast<uint32_t>(key.layer)) << 32U) ^
             static_cast<size_t>(static_cast<uint32_t>(key.expert));
    }
  };
  struct Entry {
    std::shared_ptr<ExpertShard> shard;
    std::list<Key>::iterator lru;
  };

  std::string shard_root_;
  std::string extension_ = ".q4";
  size_t max_shards_ = 0;
  std::list<Key> lru_;
  std::unordered_map<Key, Entry, KeyHash> entries_;
};

}  // namespace mslite_llm::qwen35
