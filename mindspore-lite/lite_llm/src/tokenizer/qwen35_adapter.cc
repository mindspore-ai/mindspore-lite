// Copyright 2026 Huawei Technologies Co., Ltd
// Licensed under the Apache License, Version 2.0.

#include <utility>

#include "tokenizer/qwen35/qwen_tokenizer.h"
#include "tokenizer/tokenizer.h"

namespace mslite_llm {
namespace {
// Encoding requires Qwen's Unicode pre-tokenization rules. The native vocabulary
// continues to own chat policy and buffered UTF-8 decoding for the public API.
class Qwen35Tokenizer final : public TokenizerBase {
 public:
  explicit Qwen35Tokenizer(std::unique_ptr<TokenizerBase> native) : native_(std::move(native)) {}
  bool Load(const std::string &path) { return encoder_.Load(path); }
  std::vector<int32_t> Encode(const std::string &text) override { return encoder_.Encode(text); }
  std::string Decode(const std::vector<int32_t> &ids) override { return native_->Decode(ids); }
  std::string DecodeIncremental(int32_t id) override { return native_->DecodeIncremental(id); }
  std::string FlushDecode() override { return native_->FlushDecode(); }
  bool HasChatTemplate() const override { return native_->HasChatTemplate(); }
  std::string ApplyChatTemplate(const std::vector<MSLlmChatMessage> &messages, bool prompt) override {
    return native_->ApplyChatTemplate(messages, prompt);
  }
  const std::vector<int32_t> &SuppressedTokenIds() const override { return native_->SuppressedTokenIds(); }
  bool IsStopTokenId(int32_t id) const override { return native_->IsStopTokenId(id); }

 private:
  std::unique_ptr<TokenizerBase> native_;
  qwen35_tokenizer::QwenTokenizer encoder_;
};
}  // namespace

std::unique_ptr<TokenizerBase> CreateQwen35Tokenizer(const std::string &vocab_path, const std::string &encoding_path) {
  auto native = CreateTokenizer(vocab_path);
  if (!native) return nullptr;
  return CreateQwen35Tokenizer(std::move(native), encoding_path);
}

std::unique_ptr<TokenizerBase> CreateQwen35Tokenizer(std::unique_ptr<TokenizerBase> native,
                                                     const std::string &encoding_path) {
  if (!native) return nullptr;
  auto tokenizer = std::make_unique<Qwen35Tokenizer>(std::move(native));
  if (!tokenizer->Load(encoding_path)) return nullptr;
  return tokenizer;
}
}  // namespace mslite_llm
