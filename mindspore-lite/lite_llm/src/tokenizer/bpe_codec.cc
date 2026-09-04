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
#include "tokenizer/bpe_codec.h"

#include <algorithm>
#include <iterator>
#include <numeric>
#include <cctype>
#include <cstring>
#include <limits>

namespace mslite_llm {

namespace {

uint32_t ReadU32(const uint8_t *data, size_t &offset, size_t size) {
  if (offset + sizeof(uint32_t) > size) return 0;
  uint32_t val;
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

}  // namespace

BPECodec::BPECodec() { InitByteEncoder(); }

BPECodec::~BPECodec() = default;

void BPECodec::InitByteEncoder() {
  std::vector<int> bs;
  std::vector<int> cs;

  for (int b = 33; b <= 126; ++b) {
    bs.push_back(b);
    cs.push_back(b);
  }
  for (int b = 161; b <= 172; ++b) {
    bs.push_back(b);
    cs.push_back(b);
  }
  for (int b = 174; b <= 255; ++b) {
    bs.push_back(b);
    cs.push_back(b);
  }

  int n = 0;
  for (int b = 0; b < 256; ++b) {
    bool found = std::any_of(bs.begin(), bs.end(), [b](int j) { return j == b; });
    if (!found) {
      bs.push_back(b);
      cs.push_back(256 + n);
      n++;
    }
  }

  for (size_t i = 0; i < bs.size(); ++i) {
    uint8_t byte_val = static_cast<uint8_t>(bs[i]);
    std::string utf8_char = CodePointToUTF8(static_cast<uint32_t>(cs[i]));
    byte_encoder_[byte_val] = utf8_char;
    byte_decoder_cp_[static_cast<uint32_t>(cs[i])] = byte_val;
  }
}

std::string BPECodec::CodePointToUTF8(uint32_t cp) {
  std::string result;
  if (cp <= 0x7F) {
    result += static_cast<char>(cp);
  } else if (cp <= 0x7FF) {
    result += static_cast<char>(0xC0 | (cp >> 6));
    result += static_cast<char>(0x80 | (cp & 0x3F));
  } else if (cp <= 0xFFFF) {
    result += static_cast<char>(0xE0 | (cp >> 12));
    result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    result += static_cast<char>(0x80 | (cp & 0x3F));
  } else if (cp <= 0x10FFFF) {
    result += static_cast<char>(0xF0 | (cp >> 18));
    result += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
    result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    result += static_cast<char>(0x80 | (cp & 0x3F));
  }
  return result;
}

uint32_t BPECodec::UTF8ToCodePoint(const std::string &s, size_t &pos) {
  if (pos >= s.size()) return 0;
  uint8_t c = static_cast<uint8_t>(s[pos]);
  if (c < 0x80) {
    pos++;
    return c;
  } else if ((c & 0xE0) == 0xC0) {
    if (pos + 1 >= s.size()) {
      pos++;
      return c;
    }
    uint32_t cp = (c & 0x1F) << 6;
    cp |= (static_cast<uint8_t>(s[pos + 1]) & 0x3F);
    pos += 2;
    return cp;
  } else if ((c & 0xF0) == 0xE0) {
    if (pos + 2 >= s.size()) {
      pos++;
      return c;
    }
    uint32_t cp = (c & 0x0F) << 12;
    cp |= (static_cast<uint8_t>(s[pos + 1]) & 0x3F) << 6;
    cp |= (static_cast<uint8_t>(s[pos + 2]) & 0x3F);
    pos += 3;
    return cp;
  } else if ((c & 0xF8) == 0xF0) {
    if (pos + 3 >= s.size()) {
      pos++;
      return c;
    }
    uint32_t cp = (c & 0x07) << 18;
    cp |= (static_cast<uint8_t>(s[pos + 1]) & 0x3F) << 12;
    cp |= (static_cast<uint8_t>(s[pos + 2]) & 0x3F) << 6;
    cp |= (static_cast<uint8_t>(s[pos + 3]) & 0x3F);
    pos += 4;
    return cp;
  }
  pos++;
  return c;
}

bool BPECodec::IsAlpha(unsigned char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); }

bool BPECodec::IsDigit(unsigned char c) { return c >= '0' && c <= '9'; }

void BPECodec::SetVocab(const Vocabulary &vocabulary) { token_to_id_ = &vocabulary.token_to_id; }

uint64_t BPECodec::MergeKey(int32_t left, int32_t right) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(left)) << 32U) | static_cast<uint32_t>(right);
}

bool BPECodec::Load(const uint8_t *data, size_t size, size_t &offset) {
  if (token_to_id_ == nullptr) {
    return false;
  }
  const uint32_t num_merges = ReadU32(data, offset, size);
  merge_rank_.clear();
  merge_rank_.reserve(num_merges);

  for (uint32_t i = 0; i < num_merges; ++i) {
    std::string merge_str = ReadStr(data, offset, size);
    const size_t space_pos = merge_str.find(' ');
    if (space_pos == std::string::npos) {
      continue;
    }
    const auto left = token_to_id_->find(merge_str.substr(0, space_pos));
    const auto right = token_to_id_->find(merge_str.substr(space_pos + 1));
    if (left == token_to_id_->end() || right == token_to_id_->end()) {
      return false;
    }
    merge_rank_[MergeKey(left->second, right->second)] = static_cast<int32_t>(i);
  }
  return !merge_rank_.empty() || num_merges == 0;
}

std::vector<std::string> BPECodec::PreTokenize(const std::string &text) {
  std::vector<std::string> chunks;
  size_t i = 0;
  size_t n = text.size();

  while (i < n) {
    if (i + 1 < n && text[i] == '\'') {
      char next = static_cast<char>(text[i + 1]);
      if (next == 's' || next == 't' || next == 'm' || next == 'd') {
        chunks.push_back(text.substr(i, 2));
        i += 2;
        continue;
      }
      if (i + 2 < n) {
        if ((next == 'l' && text[i + 2] == 'l') || (next == 'v' && text[i + 2] == 'e') ||
            (next == 'r' && text[i + 2] == 'e')) {
          chunks.push_back(text.substr(i, 3));
          i += 3;
          continue;
        }
      }
    }

    if (static_cast<unsigned char>(text[i]) == ' ') {
      size_t start = i;
      i++;
      if (i < n && IsAlpha(static_cast<unsigned char>(text[i]))) {
        while (i < n && IsAlpha(static_cast<unsigned char>(text[i]))) i++;
        chunks.push_back(text.substr(start, i - start));
      } else if (i < n && IsDigit(static_cast<unsigned char>(text[i]))) {
        while (i < n && IsDigit(static_cast<unsigned char>(text[i]))) i++;
        chunks.push_back(text.substr(start, i - start));
      } else if (i < n && static_cast<unsigned char>(text[i]) > 127) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        size_t char_len = 1;
        if ((c & 0xE0) == 0xC0)
          char_len = 2;
        else if ((c & 0xF0) == 0xE0)
          char_len = 3;
        else if ((c & 0xF8) == 0xF0)
          char_len = 4;
        i += char_len;
        chunks.push_back(text.substr(start, i - start));
      } else if (i < n && !std::isspace(static_cast<unsigned char>(text[i]))) {
        while (i < n && !std::isspace(static_cast<unsigned char>(text[i])) &&
               !IsAlpha(static_cast<unsigned char>(text[i])) && !IsDigit(static_cast<unsigned char>(text[i])))
          i++;
        chunks.push_back(text.substr(start, i - start));
      } else {
        chunks.push_back(" ");
      }
      continue;
    }

    if (IsAlpha(static_cast<unsigned char>(text[i]))) {
      size_t start = i;
      while (i < n && IsAlpha(static_cast<unsigned char>(text[i]))) i++;
      chunks.push_back(text.substr(start, i - start));
      continue;
    }

    if (IsDigit(static_cast<unsigned char>(text[i]))) {
      size_t start = i;
      while (i < n && IsDigit(static_cast<unsigned char>(text[i]))) i++;
      chunks.push_back(text.substr(start, i - start));
      continue;
    }

    if (static_cast<unsigned char>(text[i]) > 127) {
      size_t start = i;
      unsigned char c = static_cast<unsigned char>(text[i]);
      size_t char_len = 1;
      if ((c & 0xE0) == 0xC0)
        char_len = 2;
      else if ((c & 0xF0) == 0xE0)
        char_len = 3;
      else if ((c & 0xF8) == 0xF0)
        char_len = 4;
      i += char_len;
      chunks.push_back(text.substr(start, char_len));
      continue;
    }

    if (!std::isspace(static_cast<unsigned char>(text[i]))) {
      size_t start = i;
      while (i < n && !std::isspace(static_cast<unsigned char>(text[i])) &&
             !IsAlpha(static_cast<unsigned char>(text[i])) && !IsDigit(static_cast<unsigned char>(text[i])) &&
             static_cast<unsigned char>(text[i]) <= 127)
        i++;
      if (i > start) {
        chunks.push_back(text.substr(start, i - start));
      } else {
        chunks.push_back(text.substr(i, 1));
        i++;
      }
      continue;
    }

    i++;
  }

  return chunks;
}

std::vector<std::string> BPECodec::ApplyBPE(const std::string &token) {
  if (token.size() <= 1) {
    return {token};
  }

  if (token_to_id_ == nullptr) {
    return {};
  }
  auto it = token_to_id_->find(token);
  if (it != token_to_id_->end()) {
    return {token};
  }

  std::vector<std::string> word;
  size_t pos = 0;
  while (pos < token.size()) {
    uint32_t cp = UTF8ToCodePoint(token, pos);
    word.push_back(CodePointToUTF8(cp));
  }

  while (word.size() > 1) {
    int32_t min_rank = std::numeric_limits<int32_t>::max();
    std::pair<std::string, std::string> min_pair;

    for (size_t i = 0; i + 1 < word.size(); ++i) {
      const auto left = token_to_id_->find(word[i]);
      const auto right = token_to_id_->find(word[i + 1]);
      if (left == token_to_id_->end() || right == token_to_id_->end()) {
        continue;
      }
      auto merge_it = merge_rank_.find(MergeKey(left->second, right->second));
      if (merge_it != merge_rank_.end() && merge_it->second < min_rank) {
        min_rank = merge_it->second;
        min_pair = {word[i], word[i + 1]};
      }
    }

    if (min_rank == std::numeric_limits<int32_t>::max()) {
      break;
    }

    std::vector<std::string> new_word;
    size_t i = 0;
    while (i < word.size()) {
      if (i + 1 < word.size() && word[i] == min_pair.first && word[i + 1] == min_pair.second) {
        new_word.push_back(min_pair.first + min_pair.second);
        i += 2;
      } else {
        new_word.push_back(word[i]);
        i++;
      }
    }
    word = std::move(new_word);
  }

  return word;
}

std::vector<std::string> BPECodec::Encode(const std::string &text) {
  if (text.empty()) return {};

  std::vector<std::string> chunks = PreTokenize(text);
  std::vector<std::string> result;

  for (const auto &chunk : chunks) {
    std::string byte_encoded;
    for (size_t i = 0; i < chunk.size();) {
      unsigned char byte_val = static_cast<unsigned char>(chunk[i]);
      auto enc_it = byte_encoder_.find(byte_val);
      if (enc_it != byte_encoder_.end()) {
        byte_encoded += enc_it->second;
      } else {
        byte_encoded += chunk[i];
      }
      i++;
    }

    std::vector<std::string> bpe_tokens = ApplyBPE(byte_encoded);
    std::move(bpe_tokens.begin(), bpe_tokens.end(), std::back_inserter(result));
  }

  return result;
}

std::string BPECodec::Decode(const std::vector<std::string> &tokens) {
  std::string combined = std::accumulate(tokens.begin(), tokens.end(), std::string());

  std::vector<uint8_t> bytes;
  size_t pos = 0;
  while (pos < combined.size()) {
    uint32_t cp = UTF8ToCodePoint(combined, pos);
    auto dec_it = byte_decoder_cp_.find(cp);
    if (dec_it != byte_decoder_cp_.end()) {
      bytes.push_back(dec_it->second);
    } else {
      std::string utf8 = CodePointToUTF8(cp);
      std::copy(utf8.begin(), utf8.end(), std::back_inserter(bytes));
    }
  }

  return std::string(reinterpret_cast<const char *>(bytes.data()), bytes.size());
}

}  // namespace mslite_llm
