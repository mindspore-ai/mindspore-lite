// Copyright 2026 Huawei Technologies Co., Ltd
// Adapted from HMOS-Device-AI llm/engine/runtime; preserves its paged Qwen3.5 execution.
#include "backend/nnrt/qwen35/embedding.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <limits>

#include "manifest/artifact_file.h"

namespace mslite_llm::qwen35 {

EmbeddingTable::~EmbeddingTable() {
  if (mapping_ != nullptr) munmap(mapping_, byte_size_);
  if (fd_ >= 0) close(fd_);
}

bool EmbeddingTable::Load(const std::string &path, int32_t vocab_size, int32_t hidden_size, std::string *error) {
  if (mapping_ != nullptr || fd_ >= 0) {
    if (error) *error = "embedding table is already loaded";
    return false;
  }
  if (vocab_size <= 0 || hidden_size <= 0) {
    if (error) *error = "invalid embedding dimensions";
    return false;
  }
  const uint64_t expected = static_cast<uint64_t>(vocab_size) * static_cast<uint64_t>(hidden_size) * sizeof(uint16_t);
  if (expected > std::numeric_limits<size_t>::max()) {
    if (error) *error = "embedding table is too large";
    return false;
  }

  size_t resource_size = 0, resource_offset = 0;
  fd_ = OpenArtifact(path, &resource_size, &resource_offset, error);
  if (fd_ < 0) {
    if (error) *error = "cannot open embedding file: " + path;
    return false;
  }
  if (resource_size != expected) {
    if (error) {
      *error = "embedding file size mismatch: expected " + std::to_string(expected) + " bytes";
    }
    close(fd_);
    fd_ = -1;
    return false;
  }
  byte_size_ = static_cast<size_t>(expected);
  mapping_ = mmap(nullptr, byte_size_, PROT_READ, MAP_PRIVATE, fd_, static_cast<off_t>(resource_offset));
  if (mapping_ == MAP_FAILED) {
    mapping_ = nullptr;
    if (error) *error = "mmap failed for embedding file: " + path;
    close(fd_);
    fd_ = -1;
    byte_size_ = 0;
    return false;
  }
  vocab_size_ = vocab_size;
  hidden_size_ = hidden_size;
  return true;
}

const uint16_t *EmbeddingTable::Lookup(int32_t token_id) const {
  if (mapping_ == nullptr || token_id < 0 || token_id >= vocab_size_) return nullptr;
  return static_cast<const uint16_t *>(mapping_) + static_cast<size_t>(token_id) * hidden_size_;
}

}  // namespace mslite_llm::qwen35
