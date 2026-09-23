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

// UTF-8 encoding constants.
constexpr size_t kUtf8SeqLen2B = 2;              // length of a 2-byte sequence (U+0080..U+07FF)
constexpr size_t kUtf8SeqLen3B = 3;              // length of a 3-byte sequence (U+0800..U+FFFF)
constexpr size_t kUtf8SeqLen4B = 4;              // length of a 4-byte sequence (U+10000..U+10FFFF)
constexpr int kUtf8ContBits = 6;                 // payload bits carried by a UTF-8 continuation byte
constexpr int kUtf8ContMask = 0x3F;              // payload mask of a UTF-8 continuation byte
constexpr int kUtf8Shift2B = 2 * kUtf8ContBits;  // 12: shift of the byte two positions from the end
constexpr int kUtf8Shift3B = 3 * kUtf8ContBits;  // 18: shift of the byte three positions from the end
constexpr unsigned int kMaxAscii = 0x7F;         // upper bound of the UTF-8 ASCII range

// GPT-2 pretokenizer contraction lengths, a port of the 's|'t|'re|'ve|'m|'ll|'d
// part of the GPT-2 pretokenizer regex.
constexpr size_t kShortContractionLen = 2;  // 's/'t/'m/'d two-char contractions
constexpr size_t kLongContractionLen = 3;   // 'll/'ve/'re three-char contractions

// GPT-2 bytes_to_unicode() code point ranges, a value-by-value port of its
// range() calls: printable ASCII and printable Latin-1 code points map to
// themselves, every other byte value is remapped to 256 + n.
constexpr int kGpt2PrintableMin = 33;     // ord('!')
constexpr int kGpt2PrintableMax = 126;    // ord('~')
constexpr int kGpt2Latin1Min = 161;       // ord('¡')
constexpr int kGpt2Latin1Max = 172;       // ord('¬')
constexpr int kGpt2Latin1UpperMin = 174;  // ord('®')
constexpr int kGpt2Latin1UpperMax = 255;  // ord('ÿ')
constexpr int kGpt2ByteDomainSize = 256;  // 2**8, remap base (256 + n) for non-printable bytes

// Length in bytes of the UTF-8 sequence introduced by the given lead byte.
// An invalid leading byte is passed through as a single byte.
size_t Utf8CharLen(unsigned char lead) {
  if ((lead & 0xE0) == 0xC0) {
    return kUtf8SeqLen2B;
  }
  if ((lead & 0xF0) == 0xE0) {
    return kUtf8SeqLen3B;
  }
  if ((lead & 0xF8) == 0xF0) {
    return kUtf8SeqLen4B;
  }
  return 1;
}

uint32_t ReadU32(const uint8_t *data, size_t &offset, size_t size) {
  if (offset + sizeof(uint32_t) > size) {
    return 0;
  }
  uint32_t val;
  std::memcpy(&val, data + offset, sizeof(val));
  offset += sizeof(val);
  return val;
}

std::string ReadStr(const uint8_t *data, size_t &offset, size_t size) {
  uint32_t len = ReadU32(data, offset, size);
  if (offset + len > size) {
    return "";
  }
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

  for (int b = kGpt2PrintableMin; b <= kGpt2PrintableMax; ++b) {
    bs.push_back(b);
    cs.push_back(b);
  }
  for (int b = kGpt2Latin1Min; b <= kGpt2Latin1Max; ++b) {
    bs.push_back(b);
    cs.push_back(b);
  }
  for (int b = kGpt2Latin1UpperMin; b <= kGpt2Latin1UpperMax; ++b) {
    bs.push_back(b);
    cs.push_back(b);
  }

  int n = 0;
  for (int b = 0; b < kGpt2ByteDomainSize; ++b) {
    bool found = std::any_of(bs.begin(), bs.end(), [b](int j) { return j == b; });
    if (!found) {
      bs.push_back(b);
      cs.push_back(kGpt2ByteDomainSize + n);
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
    result += static_cast<char>(0xC0 | (cp >> kUtf8ContBits));
    result += static_cast<char>(0x80 | (cp & 0x3F));
  } else if (cp <= 0xFFFF) {
    result += static_cast<char>(0xE0 | (cp >> kUtf8Shift2B));
    result += static_cast<char>(0x80 | ((cp >> kUtf8ContBits) & kUtf8ContMask));
    result += static_cast<char>(0x80 | (cp & 0x3F));
  } else if (cp <= 0x10FFFF) {
    result += static_cast<char>(0xF0 | (cp >> kUtf8Shift3B));
    result += static_cast<char>(0x80 | ((cp >> kUtf8Shift2B) & 0x3F));
    result += static_cast<char>(0x80 | ((cp >> kUtf8ContBits) & kUtf8ContMask));
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
    constexpr size_t seq_len = kUtf8SeqLen2B;
    if (pos + 1 >= s.size()) {
      pos++;
      return c;
    }
    uint32_t cp = (c & 0x1F) << kUtf8ContBits;
    cp |= (static_cast<uint8_t>(s[pos + 1]) & 0x3F);
    pos += seq_len;
    return cp;
  } else if ((c & 0xF0) == 0xE0) {
    constexpr size_t seq_len = kUtf8SeqLen3B;
    if (pos + seq_len - 1 >= s.size()) {
      pos++;
      return c;
    }
    uint32_t cp = (c & 0x0F);
    for (size_t k = 1; k < seq_len; ++k) {
      cp = (cp << kUtf8ContBits) | (static_cast<uint8_t>(s[pos + k]) & 0x3F);
    }
    pos += seq_len;
    return cp;
  } else if ((c & 0xF8) == 0xF0) {
    constexpr size_t seq_len = kUtf8SeqLen4B;
    if (pos + seq_len - 1 >= s.size()) {
      pos++;
      return c;
    }
    uint32_t cp = (c & 0x07);
    for (size_t k = 1; k < seq_len; ++k) {
      cp = (cp << kUtf8ContBits) | (static_cast<uint8_t>(s[pos + k]) & 0x3F);
    }
    pos += seq_len;
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
        chunks.push_back(text.substr(i, kShortContractionLen));
        i += kShortContractionLen;
        continue;
      }
      if (i + kLongContractionLen - 1 < n) {
        if ((next == 'l' && text[i + kLongContractionLen - 1] == 'l') ||
            (next == 'v' && text[i + kLongContractionLen - 1] == 'e') ||
            (next == 'r' && text[i + kLongContractionLen - 1] == 'e')) {
          chunks.push_back(text.substr(i, kLongContractionLen));
          i += kLongContractionLen;
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
      } else if (i < n && (static_cast<unsigned char>(text[i]) > kMaxAscii)) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        size_t char_len = Utf8CharLen(c);
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

    if (static_cast<unsigned char>(text[i]) > kMaxAscii) {
      size_t start = i;
      unsigned char c = static_cast<unsigned char>(text[i]);
      size_t char_len = Utf8CharLen(c);
      i += char_len;
      chunks.push_back(text.substr(start, char_len));
      continue;
    }

    if (!std::isspace(static_cast<unsigned char>(text[i]))) {
      size_t start = i;
      while (i < n && !std::isspace(static_cast<unsigned char>(text[i])) &&
             !IsAlpha(static_cast<unsigned char>(text[i])) && !IsDigit(static_cast<unsigned char>(text[i])) &&
             static_cast<unsigned char>(text[i]) <= kMaxAscii)
        i++;
      if (i > start) {
        chunks.push_back(text.substr(start, i - start));
      } else {
        chunks.push_back(text.substr(i, 1));
        i++;
      }
      continue;
    }

    // Byte-level BPE represents every input byte, including non-space
    // whitespace.  Keep it as a chunk so ChatML line breaks and tabs reach
    // the byte encoder instead of being silently discarded.
    chunks.push_back(text.substr(i, 1));
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

    // One merge consumes the two adjacent pieces of min_pair.
    constexpr size_t kMergePairLen = 2;
    std::vector<std::string> new_word;
    size_t i = 0;
    while (i < word.size()) {
      if (i + 1 < word.size() && word[i] == min_pair.first && word[i + 1] == min_pair.second) {
        new_word.push_back(min_pair.first + min_pair.second);
        i += kMergePairLen;
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
