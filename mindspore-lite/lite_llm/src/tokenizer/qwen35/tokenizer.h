// Copyright 2026 Huawei Technologies Co., Ltd
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace mslite_llm::qwen35_tokenizer {

struct EncodeOptions {
  // Apply the model tokenizer.json post-processor. Qwen has no automatic
  // BOS/EOS post-processor; MiniCPM adds BOS by default.
  bool add_special_tokens = true;
  // Explicitly force BOS/EOS in addition to the model defaults.
  bool add_bos = false;
  bool add_eos = false;
};

struct DecodeOptions {
  bool skip_special_tokens = false;
};

enum class TokenizerFamily {
  kUnknown = 0,
  kQwenByteLevelBpe = 1,
  kMiniCpmSentencePieceBpe = 2,
};

class Tokenizer {
 public:
  virtual ~Tokenizer() = default;

  // Auto-detects the exported format and creates the corresponding backend.
  static std::unique_ptr<Tokenizer> Load(const std::string &path, std::string *error = nullptr);

  virtual std::vector<int32_t> Encode(std::string_view text, const EncodeOptions &options = {}) const = 0;
  virtual std::string Decode(const std::vector<int32_t> &ids, const DecodeOptions &options = {}) const = 0;

  virtual TokenizerFamily Family() const = 0;
  virtual const char *FamilyName() const = 0;
  virtual int32_t VocabSize() const = 0;
  virtual int32_t UnknownId() const = 0;
  virtual int32_t BosId() const = 0;
  virtual int32_t EosId() const = 0;
  virtual bool SupportsDecode() const = 0;
};

}  // namespace mslite_llm::qwen35_tokenizer
