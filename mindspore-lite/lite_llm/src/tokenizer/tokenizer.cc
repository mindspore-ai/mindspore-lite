/**
 * Copyright 2026 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "tokenizer/tokenizer.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tokenizer/bpe_codec.h"
#include "tokenizer/sentencepiece_codec.h"
#include "tokenizer/chat_template.h"
namespace mslite_llm {

namespace {

constexpr uint32_t kMagic = 0x4D534C54;
// v2: custom template payload is restricted IR; v1 packages (raw
// Jinja / builtin types) are rejected at load and must be re-exported.
constexpr uint32_t kVersion = 2;
constexpr uint32_t kCodecBPE = 0;
constexpr uint32_t kCodecSentencePiece = 1;

uint32_t ReadU32(const uint8_t *data, size_t &offset, size_t size) {
  if (offset + sizeof(uint32_t) > size) return 0;
  uint32_t val;
  std::memcpy(&val, data + offset, sizeof(val));
  offset += sizeof(val);
  return val;
}

int32_t ReadI32(const uint8_t *data, size_t &offset, size_t size) {
  if (offset + sizeof(int32_t) > size) return -1;
  int32_t val;
  std::memcpy(&val, data + offset, sizeof(val));
  offset += sizeof(val);
  return val;
}

std::string ReadStr(const uint8_t *data, size_t &offset, size_t size) {
  uint32_t len = ReadU32(data, offset, size);
  if (offset + len > size) return "";
  std::string s(reinterpret_cast<const char *>(data + offset), len);
  offset += len;
  return s;
}

struct TextSegment {
  std::string text;
  bool is_special;
};

bool IsDefaultStopToken(const std::string &token) {
  return token == "<|endoftext|>" || token == "<|im_end|>" || token == "</s>";
}

bool IsDefaultSuppressedToken(const std::string &token) { return token == "<|im_start|>"; }

bool IsDefaultSpecialToken(const std::string &token) {
  return IsDefaultStopToken(token) || IsDefaultSuppressedToken(token);
}

std::vector<TextSegment> SplitOnSpecialTokens(const std::string &text,
                                              const std::unordered_map<std::string, int32_t> &special_tokens) {
  std::vector<TextSegment> segments;
  size_t i = 0;

  while (i < text.size()) {
    bool found_special = false;
    for (const auto &entry : special_tokens) {
      const std::string &token_str = entry.first;
      if (text.compare(i, token_str.size(), token_str) == 0) {
        segments.push_back({token_str, true});
        i += token_str.size();
        found_special = true;
        break;
      }
    }

    if (found_special) continue;

    size_t start = i;
    while (i < text.size()) {
      bool is_special_boundary = false;
      for (const auto &entry : special_tokens) {
        const std::string &token_str = entry.first;
        if (text.compare(i, token_str.size(), token_str) == 0) {
          is_special_boundary = true;
          break;
        }
      }
      if (is_special_boundary) break;
      i++;
    }

    if (i > start) {
      segments.push_back({text.substr(start, i - start), false});
    }
  }

  return segments;
}

}  // namespace

class TokenizerImpl : public Tokenizer {
 public:
  TokenizerImpl() : bos_token_id_(-1), eos_token_id_(-1), pad_token_id_(-1), unk_token_id_(-1) {}

  bool Load(const std::string &vocab_path) {
    std::ifstream ifs(vocab_path, std::ios::binary | std::ios::ate);
    if (!ifs.good()) return false;

    auto file_size = ifs.tellg();
    if (file_size <= 0) return false;

    size_t data_size = static_cast<size_t>(file_size);
    ifs.seekg(0, std::ios::beg);

    std::vector<uint8_t> data(data_size);
    if (!ifs.read(reinterpret_cast<char *>(data.data()), data_size)) return false;

    return LoadFromBuffer(data.data(), data_size);
  }

  bool LoadFromBuffer(const uint8_t *data, size_t data_size) {
    if (data == nullptr || data_size == 0) return false;

    size_t offset = 0;

    uint32_t magic = ReadU32(data, offset, data_size);
    if (magic != kMagic) return false;

    uint32_t version = ReadU32(data, offset, data_size);
    if (version != kVersion) return false;

    uint32_t codec_type = ReadU32(data, offset, data_size);
    uint32_t vocab_size = ReadU32(data, offset, data_size);

    bos_token_id_ = ReadI32(data, offset, data_size);
    eos_token_id_ = ReadI32(data, offset, data_size);
    pad_token_id_ = ReadI32(data, offset, data_size);
    unk_token_id_ = ReadI32(data, offset, data_size);

    // Legacy chat template type (builtin families removed): read
    // only to advance past the header field, value carries no meaning.
    (void)ReadU32(data, offset, data_size);

    token_to_id_.clear();
    id_to_token_.clear();
    special_tokens_.clear();
    special_token_ids_.clear();
    suppressed_token_ids_.clear();
    stop_token_ids_.clear();

    for (uint32_t i = 0; i < vocab_size; ++i) {
      std::string token = ReadStr(data, offset, data_size);
      int32_t id = ReadI32(data, offset, data_size);
      token_to_id_[token] = id;
      id_to_token_[id] = token;
    }

    if (codec_type == kCodecBPE) {
      bpe_codec_ = std::make_unique<BPECodec>();
      bpe_codec_->SetVocab(token_to_id_, id_to_token_);
      if (!bpe_codec_->Load(data, data_size, offset)) return false;
    } else if (codec_type == kCodecSentencePiece) {
      sp_codec_ = std::make_unique<SentencePieceCodec>();
      sp_codec_->SetVocab(token_to_id_, id_to_token_);
      if (!sp_codec_->Load(data, data_size, offset)) return false;
    } else {
      return false;
    }

    // Legacy type field: kept for byte layout, carries no meaning anymore
    // (only the custom template below matters).
    chat_template_ = std::make_unique<ChatTemplate>();

    if (offset + sizeof(uint32_t) <= data_size) {
      uint32_t tmpl_len = ReadU32(data, offset, data_size);
      if (tmpl_len > 0 && offset + tmpl_len <= data_size) {
        std::string custom_tmpl(reinterpret_cast<const char *>(data + offset), tmpl_len);
        chat_template_->SetCustomTemplate(custom_tmpl);
        offset += tmpl_len;
      }
    }

    if (!LoadSpecialTokenPolicy(data, offset, data_size)) {
      MarkDefaultSpecialTokenPolicy();
    }
    MarkSpecialTokenId(bos_token_id_, false);
    MarkSpecialTokenId(eos_token_id_, true);
    MarkSpecialTokenId(pad_token_id_, true);
    MarkSpecialTokenId(unk_token_id_, false);

    return true;
  }

  std::vector<int32_t> Encode(const std::string &text) override {
    std::vector<int32_t> ids;

    if (bos_token_id_ >= 0) {
      ids.push_back(bos_token_id_);
    }

    if (!special_tokens_.empty()) {
      auto segments = SplitOnSpecialTokens(text, special_tokens_);

      for (const auto &segment : segments) {
        if (segment.is_special) {
          auto it = token_to_id_.find(segment.text);
          if (it != token_to_id_.end()) {
            ids.push_back(it->second);
          } else if (unk_token_id_ >= 0) {
            ids.push_back(unk_token_id_);
          }
        } else {
          EncodeRegularText(segment.text, ids);
        }
      }
    } else {
      EncodeRegularText(text, ids);
    }

    return ids;
  }

  std::string Decode(const std::vector<int32_t> &token_ids) override {
    std::vector<std::string> tokens;
    for (auto id : token_ids) {
      if (special_token_ids_.count(id) > 0) {
        continue;
      }
      auto it = id_to_token_.find(id);
      if (it != id_to_token_.end()) {
        tokens.push_back(it->second);
      }
    }

    if (bpe_codec_) {
      return bpe_codec_->Decode(tokens);
    } else if (sp_codec_) {
      return sp_codec_->Decode(tokens);
    }
    return "";
  }

  std::string DecodeIncremental(int32_t token_id) override {
    if (special_token_ids_.count(token_id) > 0) return "";
    auto it = id_to_token_.find(token_id);
    if (it == id_to_token_.end()) return "";

    if (bpe_codec_) {
      pending_bytes_ += bpe_codec_->Decode({it->second});
    } else if (sp_codec_) {
      pending_bytes_ += sp_codec_->Decode({it->second});
    }
    return DrainCompleteUtf8();
  }

  std::string FlushDecode() override {
    std::string out = std::move(pending_bytes_);
    pending_bytes_.clear();
    return out;
  }

  std::string ApplyChatTemplate(const std::vector<MSLlmChatMessage> &messages, bool add_generation_prompt) override {
    if (chat_template_ && chat_template_->HasTemplate()) {
      return chat_template_->Apply(messages, add_generation_prompt);
    }
    return {};
  }

  bool HasChatTemplate() const override { return chat_template_ != nullptr && chat_template_->HasTemplate(); }

  const std::vector<int32_t> &SuppressedTokenIds() const override { return suppressed_token_ids_; }

  bool IsStopTokenId(int32_t token_id) const override { return stop_token_ids_.count(token_id) > 0; }

 private:
  void MarkSpecialTokenId(int32_t id, bool stop) {
    if (id < 0) {
      return;
    }
    special_token_ids_.insert(id);
    if (stop) {
      stop_token_ids_.insert(id);
    } else if (std::find(suppressed_token_ids_.begin(), suppressed_token_ids_.end(), id) ==
               suppressed_token_ids_.end()) {
      suppressed_token_ids_.push_back(id);
    }
  }

  void MarkSpecialTokenString(const std::string &token, bool stop) {
    auto it = token_to_id_.find(token);
    if (it == token_to_id_.end()) {
      return;
    }
    special_tokens_[token] = it->second;
    MarkSpecialTokenId(it->second, stop);
  }

  void MarkDefaultSpecialTokenPolicy() {
    for (const auto &entry : token_to_id_) {
      if (!IsDefaultSpecialToken(entry.first)) {
        continue;
      }
      MarkSpecialTokenString(entry.first, IsDefaultStopToken(entry.first));
    }
  }

  bool LoadSpecialTokenPolicy(const uint8_t *data, size_t &offset, size_t size) {
    if (offset + sizeof(uint32_t) > size) {
      return false;
    }

    const uint32_t stop_count = ReadU32(data, offset, size);
    std::vector<std::string> stop_tokens;
    stop_tokens.reserve(stop_count);
    for (uint32_t i = 0; i < stop_count; ++i) {
      if (offset + sizeof(uint32_t) > size) {
        return false;
      }
      std::string token = ReadStr(data, offset, size);
      if (token.empty()) {
        return false;
      }
      stop_tokens.push_back(token);
    }

    if (offset + sizeof(uint32_t) > size) {
      return false;
    }
    const uint32_t suppress_count = ReadU32(data, offset, size);
    std::vector<std::string> suppress_tokens;
    suppress_tokens.reserve(suppress_count);
    for (uint32_t i = 0; i < suppress_count; ++i) {
      if (offset + sizeof(uint32_t) > size) {
        return false;
      }
      std::string token = ReadStr(data, offset, size);
      if (token.empty()) {
        return false;
      }
      suppress_tokens.push_back(token);
    }

    for (const auto &token : stop_tokens) {
      MarkSpecialTokenString(token, true);
    }
    for (const auto &token : suppress_tokens) {
      MarkSpecialTokenString(token, false);
    }
    return true;
  }

  /// Emit complete UTF-8 characters from pending_bytes_, keeping any
  /// incomplete trailing sequence buffered.
  std::string DrainCompleteUtf8() {
    std::string out;
    size_t i = 0;
    const size_t n = pending_bytes_.size();
    while (i < n) {
      size_t len = Utf8SeqLen(static_cast<uint8_t>(pending_bytes_[i]));
      if (i + len > n) break;  // incomplete tail
      out.append(pending_bytes_, i, len);
      i += len;
    }
    pending_bytes_.erase(0, i);
    return out;
  }

  static size_t Utf8SeqLen(uint8_t lead) {
    if ((lead & 0x80u) == 0) return 1;
    if ((lead & 0xE0u) == 0xC0u) return 2;
    if ((lead & 0xF0u) == 0xE0u) return 3;
    if ((lead & 0xF8u) == 0xF0u) return 4;
    return 1;  // invalid leading byte: pass through as a single byte
  }

  void EncodeRegularText(const std::string &text, std::vector<int32_t> &ids) {
    if (text.empty()) return;

    std::vector<std::string> tokens;
    if (bpe_codec_) {
      tokens = bpe_codec_->Encode(text);
    } else if (sp_codec_) {
      tokens = sp_codec_->Encode(text);
    }

    for (const auto &tok : tokens) {
      auto it = token_to_id_.find(tok);
      if (it != token_to_id_.end()) {
        ids.push_back(it->second);
      } else if (unk_token_id_ >= 0) {
        ids.push_back(unk_token_id_);
      }
    }
  }

  std::unique_ptr<BPECodec> bpe_codec_;
  std::unique_ptr<SentencePieceCodec> sp_codec_;
  std::unique_ptr<ChatTemplate> chat_template_;
  std::string pending_bytes_;

  std::unordered_map<std::string, int32_t> token_to_id_;
  std::unordered_map<int32_t, std::string> id_to_token_;
  std::unordered_map<std::string, int32_t> special_tokens_;
  std::unordered_set<int32_t> special_token_ids_;
  std::unordered_set<int32_t> stop_token_ids_;
  std::vector<int32_t> suppressed_token_ids_;

  int32_t bos_token_id_;
  int32_t eos_token_id_;
  int32_t pad_token_id_;
  int32_t unk_token_id_;
};

std::unique_ptr<Tokenizer> CreateTokenizer(const std::string &vocab_path) {
  auto impl = std::make_unique<TokenizerImpl>();
  if (!impl->Load(vocab_path)) {
    return nullptr;
  }
  return impl;
}

std::unique_ptr<Tokenizer> CreateTokenizerFromBuffer(const uint8_t *data, size_t size) {
  auto impl = std::make_unique<TokenizerImpl>();
  if (!impl->LoadFromBuffer(data, size)) {
    return nullptr;
  }
  return impl;
}

}  // namespace mslite_llm
