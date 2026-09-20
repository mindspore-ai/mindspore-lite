// Copyright 2026 Huawei Technologies Co., Ltd
// Adapted from HMOS-Device-AI llm/engine/runtime; preserves its paged Qwen3.5 execution.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace mslite_llm::qwen35 {

class EmbeddingTable {
 public:
  EmbeddingTable() = default;
  ~EmbeddingTable();

  EmbeddingTable(const EmbeddingTable &) = delete;
  EmbeddingTable &operator=(const EmbeddingTable &) = delete;

  bool Load(const std::string &path, int32_t vocab_size, int32_t hidden_size, std::string *error);
  const uint16_t *Lookup(int32_t token_id) const;

  int32_t VocabSize() const { return vocab_size_; }
  int32_t HiddenSize() const { return hidden_size_; }
  size_t ByteSize() const { return byte_size_; }

 private:
  int fd_ = -1;
  void *mapping_ = nullptr;
  size_t byte_size_ = 0;
  int32_t vocab_size_ = 0;
  int32_t hidden_size_ = 0;
};

}  // namespace mslite_llm::qwen35
