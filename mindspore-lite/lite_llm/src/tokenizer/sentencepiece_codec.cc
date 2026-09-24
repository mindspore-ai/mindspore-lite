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
#include "tokenizer/sentencepiece_codec.h"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>

namespace mslite_llm {

namespace {

// UTF-8 encoding constants.
constexpr size_t kUtf8SeqLen2B = 2;  // length of a 2-byte sequence (U+0080..U+07FF)
constexpr size_t kUtf8SeqLen3B = 3;  // length of a 3-byte sequence (U+0800..U+FFFF)
constexpr size_t kUtf8SeqLen4B = 4;  // length of a 4-byte sequence (U+10000..U+10FFFF)

// Protobuf wire-format constants.
constexpr uint32_t kWtVarint = 0;
constexpr uint32_t kWtFixed64 = 1;
constexpr uint32_t kWtLenDelim = 2;
constexpr uint32_t kWtFixed32 = 5;
constexpr size_t kFixed64Bytes = 8;
constexpr size_t kFixed32Bytes = 4;

// Varint encoding constants.
constexpr int kVarintPayloadBits = 7;  // payload bits carried by one varint byte
constexpr uint8_t kVarintPayloadMask = 0x7F;

// SentencePiece ModelProto field numbers (proto2 schema).
constexpr uint32_t kSpFieldPieces = 1;      // ModelProto.pieces
constexpr uint32_t kPieceFieldText = 1;     // SentencePiece.text
constexpr uint32_t kPieceFieldScore = 2;    // SentencePiece.score
constexpr uint32_t kPieceFieldType = 3;     // SentencePiece.type
constexpr int32_t kSpPieceTypeUnknown = 2;  // SentencePiece.Type::UNKNOWN

constexpr size_t kMaxViterbiPieceLen = 64;    // max byte length of a Viterbi candidate piece
constexpr int kHexDigitsPerByte = 2;          // two hex digits in a byte-fallback <0xN> token
constexpr unsigned int kMaxByteValue = 255;   // upper bound of a single byte value
constexpr size_t kByteTokenMinLen = 5;        // min length of a byte-fallback token (<0xN>)
constexpr size_t kSpByteTokenPrefixLen = 3;   // length of the "<0x" token prefix
constexpr char kSpMarker[] = "\xe2\x96\x81";  // U+2581 sentencepiece whitespace marker
constexpr size_t kSpMarkerLen = 3;

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

uint64_t ReadVarint(const uint8_t *data, size_t &offset, size_t size) {
  uint64_t result = 0;
  int shift = 0;
  while (offset < size) {
    uint8_t b = data[offset++];
    result |= static_cast<uint64_t>(b & kVarintPayloadMask) << shift;
    if ((b & 0x80) == 0) {
      break;
    }
    shift += kVarintPayloadBits;
  }
  return result;
}

uint32_t ReadTag(const uint8_t *data, size_t &offset, size_t size) {
  return static_cast<uint32_t>(ReadVarint(data, offset, size));
}

float ReadFloat32(const uint8_t *data, size_t &offset, size_t size) {
  if (offset + sizeof(float) > size) {
    return 0.0f;
  }
  float val;
  std::memcpy(&val, data + offset, sizeof(val));
  offset += sizeof(val);
  return val;
}

std::string ReadBytes(const uint8_t *data, size_t &offset, size_t size) {
  uint64_t len = ReadVarint(data, offset, size);
  if (offset + len > size) {
    return "";
  }
  std::string s(reinterpret_cast<const char *>(data + offset), len);
  offset += len;
  return s;
}

void SkipField(uint32_t wire_type, const uint8_t *data, size_t &offset, size_t size) {
  switch (wire_type) {
    case kWtVarint:
      ReadVarint(data, offset, size);
      break;
    case kWtFixed64:
      offset += kFixed64Bytes;
      break;
    case kWtLenDelim: {
      uint64_t len = ReadVarint(data, offset, size);
      offset += len;
      break;
    }
    case kWtFixed32:
      offset += kFixed32Bytes;
      break;
    default:
      break;
  }
}

}  // namespace

SentencePieceCodec::SentencePieceCodec() : unk_id_(-1), byte_fallback_(true) {}

SentencePieceCodec::~SentencePieceCodec() = default;

void SentencePieceCodec::SetVocab(const Vocabulary &vocabulary) {
  token_to_id_ = &vocabulary.token_to_id;
  const auto unk_it = token_to_id_->find("<unk>");
  if (unk_it != token_to_id_->end()) {
    unk_id_ = unk_it->second;
  }
}

bool SentencePieceCodec::Load(const uint8_t *data, size_t size, size_t &offset) {
  uint32_t sp_model_size = ReadU32(data, offset, size);
  if (offset + sp_model_size > size) return false;

  bool parsed = ParseModel(data + offset, sp_model_size);
  offset += sp_model_size;

  if (!parsed) {
    if (token_to_id_ == nullptr) {
      return false;
    }
    for (const auto &entry : *token_to_id_) {
      SPPiece piece;
      piece.piece = std::string(entry.first);
      piece.score = 0.0f;
      piece.type = 1;
      pieces_.push_back(piece);
      piece_score_[piece.piece] = 0.0f;
    }
  }

  return true;
}

bool SentencePieceCodec::ParseModel(const uint8_t *data, size_t size) {
  pieces_.clear();
  piece_score_.clear();

  size_t pos = 0;
  while (pos < size) {
    uint32_t tag = ReadTag(data, pos, size);
    uint32_t field_number = tag >> 3;
    uint32_t wire_type = tag & 0x07;

    if (field_number == kSpFieldPieces && wire_type == kWtLenDelim) {
      uint64_t msg_len = ReadVarint(data, pos, size);
      if (pos + msg_len > size) break;

      size_t msg_end = pos + msg_len;

      SPPiece piece;
      piece.score = 0.0f;
      piece.type = 1;

      while (pos < msg_end) {
        uint32_t inner_tag = ReadTag(data, pos, size);
        uint32_t inner_field = inner_tag >> 3;
        uint32_t inner_wire = inner_tag & 0x07;

        if (inner_field == kPieceFieldText && inner_wire == kWtLenDelim) {
          piece.piece = ReadBytes(data, pos, size);
        } else if (inner_field == kPieceFieldScore && inner_wire == kWtFixed32) {
          piece.score = ReadFloat32(data, pos, size);
        } else if (inner_field == kPieceFieldType && inner_wire == kWtVarint) {
          piece.type = static_cast<int32_t>(ReadVarint(data, pos, size));
        } else {
          SkipField(inner_wire, data, pos, size);
        }
      }

      pos = msg_end;

      pieces_.push_back(piece);
      piece_score_[piece.piece] = piece.score;

      if (piece.type == kSpPieceTypeUnknown) {
        unk_id_ = static_cast<int32_t>(pieces_.size() - 1);
      }
    } else {
      SkipField(wire_type, data, pos, size);
    }
  }

  return !pieces_.empty();
}

std::vector<std::string> SentencePieceCodec::ViterbiEncode(const std::string &text) {
  size_t n = text.size();
  if (n == 0) return {};

  const float kNegInf = -1e10f;
  std::vector<float> best_score(n + 1, kNegInf);
  std::vector<int> best_edge(n + 1, -1);
  best_score[0] = 0.0f;

  for (size_t i = 0; i < n; ++i) {
    if (best_score[i] <= kNegInf + 1.0f) continue;

    for (size_t len = 1; len <= n - i && len <= kMaxViterbiPieceLen; ++len) {
      std::string piece = text.substr(i, len);
      auto score_it = piece_score_.find(piece);
      if (score_it == piece_score_.end()) continue;

      float score = best_score[i] + score_it->second;
      if (score > best_score[i + len]) {
        best_score[i + len] = score;
        best_edge[i + len] = static_cast<int>(i);
      }
    }
  }

  if (best_score[n] <= kNegInf + 1.0f) {
    return ByteFallbackEncode(text);
  }

  std::vector<std::string> result;
  size_t i = n;
  while (i > 0) {
    size_t j = static_cast<size_t>(best_edge[i]);
    result.insert(result.begin(), text.substr(j, i - j));
    i = j;
  }
  return result;
}

std::vector<std::string> SentencePieceCodec::ByteFallbackEncode(const std::string &text) {
  if (token_to_id_ == nullptr) {
    return {};
  }
  std::vector<std::string> result;
  size_t i = 0;
  while (i < text.size()) {
    unsigned char c = static_cast<unsigned char>(text[i]);
    size_t char_len = Utf8CharLen(c);

    if (char_len == 1) {
      std::ostringstream oss;
      oss << "<0x" << std::uppercase << std::hex << std::setw(kHexDigitsPerByte) << std::setfill('0')
          << static_cast<int>(c) << ">";
      std::string byte_token = oss.str();

      auto it = token_to_id_->find(byte_token);
      if (it != token_to_id_->end()) {
        result.push_back(byte_token);
      } else {
        std::string single(1, static_cast<char>(c));
        auto single_it = token_to_id_->find(single);
        if (single_it != token_to_id_->end()) {
          result.push_back(single);
        } else if (unk_id_ >= 0) {
          result.push_back("<unk>");
        }
      }
      i++;
    } else {
      if (i + char_len > text.size()) {
        i++;
        continue;
      }

      std::string utf8_char = text.substr(i, char_len);
      auto it = token_to_id_->find(utf8_char);
      if (it != token_to_id_->end()) {
        result.push_back(utf8_char);
      } else if (byte_fallback_) {
        for (size_t b = 0; b < char_len; ++b) {
          std::ostringstream oss;
          oss << "<0x" << std::uppercase << std::hex << std::setw(kHexDigitsPerByte) << std::setfill('0')
              << static_cast<int>(static_cast<unsigned char>(text[i + b])) << ">";
          std::string byte_token = oss.str();
          auto byte_it = token_to_id_->find(byte_token);
          if (byte_it != token_to_id_->end()) {
            result.push_back(byte_token);
          }
        }
      } else if (unk_id_ >= 0) {
        result.push_back("<unk>");
      }
      i += char_len;
    }
  }
  return result;
}

std::vector<std::string> SentencePieceCodec::Encode(const std::string &text) {
  if (text.empty()) return {};

  std::string normalized;
  for (size_t i = 0; i < text.size();) {
    unsigned char c = static_cast<unsigned char>(text[i]);
    if (c == ' ') {
      normalized += "\xe2\x96\x81";
      i++;
    } else {
      size_t char_len = Utf8CharLen(c);
      for (size_t j = 0; j < char_len && i + j < text.size(); ++j) {
        normalized += text[i + j];
      }
      i += char_len;
    }
  }

  if (normalized.empty()) return {};

  if (static_cast<unsigned char>(normalized[0]) != 0xe2) {
    bool starts_with_marker = normalized.compare(0, kSpMarkerLen, kSpMarker) == 0;
    if (!starts_with_marker) {
      normalized = "\xe2\x96\x81" + normalized;
    }
  }

  return ViterbiEncode(normalized);
}

std::string SentencePieceCodec::Decode(const std::vector<std::string> &tokens) {
  std::string result;
  for (const auto &token : tokens) {
    if (token.size() >= kByteTokenMinLen && token.substr(0, 3) == "<0x" && token.back() == '>') {
      std::string hex_str = token.substr(kSpByteTokenPrefixLen, token.size() - kSpByteTokenPrefixLen - 1);
      try {
        unsigned int byte_val = 0;
        std::istringstream iss(hex_str);
        iss >> std::hex >> byte_val;
        if (byte_val <= kMaxByteValue) {
          result += static_cast<char>(byte_val);
        } else {
          result += token;
        }
      } catch (...) {
        result += token;
      }
    } else {
      std::string decoded = token;
      std::string marker = kSpMarker;
      size_t pos = 0;
      while ((pos = decoded.find(marker, pos)) != std::string::npos) {
        decoded.replace(pos, kSpMarkerLen, " ");
        pos++;
      }
      result += decoded;
    }
  }
  return result;
}

}  // namespace mslite_llm
