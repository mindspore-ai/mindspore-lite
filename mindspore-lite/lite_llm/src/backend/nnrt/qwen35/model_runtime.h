// Copyright 2026 Huawei Technologies Co., Ltd
// Adapted from HMOS-Device-AI llm/engine/runtime; preserves its paged Qwen3.5 execution.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mslite_llm::qwen35 {

enum class ModelGear {
  kDecode1,
  kVerify2,
  kVerify3,
  kVerify4,
  kPrefill8,
  kPrefill64,
};

struct CacheSnapshot {
  std::vector<std::vector<uint8_t>> buffers;
};

struct TensorBuffer {
  void *data = nullptr;
  size_t bytes = 0;
};

struct ModelMetadata {
  std::string model_family = "qwen3.5";
  int32_t hidden_size = 1024;
  int32_t vocab_size = 248320;
  int32_t rotary_dim = 64;
  int32_t prefill_chunk = 64;
  int32_t max_cache_length = 8192;
  int32_t pad_token_id = 248044;
  int32_t eos_token_id = 248046;
  float rope_theta = 10000000.0f;
};

class ModelRuntime {
 public:
  virtual ~ModelRuntime() = default;

  virtual const ModelMetadata &Metadata() const = 0;
  virtual bool HasHiddenStates() const = 0;
  virtual bool HasCheckpointStates() const = 0;
  virtual int32_t CacheLength() const = 0;
  virtual bool ResetCache(std::string *error) = 0;
  virtual bool SaveCache(CacheSnapshot *snapshot, std::string *error) const = 0;
  virtual bool RestoreCache(const CacheSnapshot &snapshot, std::string *error) = 0;
  virtual bool PrepareGear(ModelGear gear, std::string *error) = 0;
  virtual TensorBuffer Input(ModelGear gear, const std::string &name) const = 0;
  virtual TensorBuffer Logits(ModelGear gear) const = 0;
  virtual TensorBuffer TokenIds(ModelGear gear) const = 0;
  virtual TensorBuffer HiddenStates(ModelGear gear) const = 0;
  virtual bool Run(ModelGear gear, double *elapsed_ms, std::string *error, bool defer_cache_commit = false) = 0;
  virtual bool CommitCache(std::string *error) = 0;
  virtual bool CommitCheckpointCache(std::string *error) = 0;
  virtual bool DiscardCache(std::string *error) = 0;
};

}  // namespace mslite_llm::qwen35
