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
#ifndef MSLLM_SENTENCEPIECE_CODEC_H
#define MSLLM_SENTENCEPIECE_CODEC_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace mslite_llm {

struct SPPiece {
  std::string piece;
  float score;
  int32_t type;
};

class SentencePieceCodec {
 public:
  SentencePieceCodec();
  ~SentencePieceCodec();

  bool Load(const uint8_t *data, size_t size, size_t &offset);
  std::vector<std::string> Encode(const std::string &text);
  std::string Decode(const std::vector<std::string> &tokens);

  void SetVocab(const std::unordered_map<std::string, int32_t> &token_to_id,
                const std::unordered_map<int32_t, std::string> &id_to_token);

 private:
  bool ParseModel(const uint8_t *data, size_t size);
  std::vector<std::string> ViterbiEncode(const std::string &text);
  std::vector<std::string> ByteFallbackEncode(const std::string &text);

  std::vector<SPPiece> pieces_;
  std::unordered_map<std::string, float> piece_score_;
  std::unordered_map<std::string, int32_t> token_to_id_;
  std::unordered_map<int32_t, std::string> id_to_token_;
  int32_t unk_id_;
  bool byte_fallback_;
};

}  // namespace mslite_llm

#endif
