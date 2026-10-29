// Copyright 2026 Huawei Technologies Co., Ltd
// Adapted from HMOS-Device-AI llm/engine/runtime; preserves its paged Qwen3.5 execution.
#pragma once

#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17): lite_llm requires C++17.
#include <string>

#include "backend/nnrt/qwen35/nnrt_session.h"

namespace mslite_llm::qwen35 {

class Qwen35LmHead {
 public:
  Qwen35LmHead() = default;
  ~Qwen35LmHead();
  Qwen35LmHead(const Qwen35LmHead &) = delete;
  Qwen35LmHead &operator=(const Qwen35LmHead &) = delete;

  bool Load(const std::string &artifact_root, std::string *error);
  bool Run(const uint16_t *hidden, uint16_t *logits, double *elapsed_ms, std::string *error);
  bool HasTop1() const { return top1_mode_; }
  int32_t TokenId() const { return token_id_; }
  const int32_t *TokenIdData() const { return &token_id_; }

 private:
  bool RunTop1(const TensorView &normalized, double total, double *elapsed_ms, std::string *error);
  bool CreateResidentSets(const std::string &artifact_root, std::string *error);
  std::filesystem::path SelectTile(const std::filesystem::path &omc);
  bool LoadSession(NnrtSession *session, const std::string &path, int32_t rows, std::string *error);
  bool LoadFileBacked(const std::string &artifact_root, bool *dma_unavailable, std::string *error);
  bool RunFileTile(int32_t tile, const uint16_t *normalized, uint16_t *logits, double *elapsed_ms, std::string *error);
  bool RunTile(NnrtSession *session, int32_t first, int32_t rows, const uint16_t *normalized, uint16_t *logits,
               double *elapsed_ms, std::string *error);

  int fd_ = -1;
  const uint8_t *mapping_ = nullptr;
  size_t mapping_bytes_ = 0;
  bool file_backed_mode_ = false;
  NnrtSession norm_;
  NnrtSession tile8192_;
  NnrtSession tile2560_;
  NnrtSession top1_;
  int32_t tile_rows_ = 8192;
  bool top1_mode_ = false;
  int32_t token_id_ = -1;
};

}  // namespace mslite_llm::qwen35
