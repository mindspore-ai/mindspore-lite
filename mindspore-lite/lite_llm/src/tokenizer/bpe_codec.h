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
#ifndef MSLLM_BPE_CODEC_H
#define MSLLM_BPE_CODEC_H

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mslite_llm {

class BPECodec {
 public:
  BPECodec();
  ~BPECodec();

  bool Load(const uint8_t *data, size_t size, size_t &offset);
  std::vector<std::string> Encode(const std::string &text);
  std::string Decode(const std::vector<std::string> &tokens);

  void SetVocab(const std::unordered_map<std::string, int32_t> &token_to_id,
                const std::unordered_map<int32_t, std::string> &id_to_token);

 private:
  void InitByteEncoder();
  std::string CodePointToUTF8(uint32_t cp);
  uint32_t UTF8ToCodePoint(const std::string &s, size_t &pos);
  std::vector<std::string> PreTokenize(const std::string &text);
  std::vector<std::string> ApplyBPE(const std::string &token);

  static bool IsAlpha(unsigned char c);
  static bool IsDigit(unsigned char c);

  std::vector<std::pair<std::string, std::string>> merges_;
  std::map<std::pair<std::string, std::string>, int32_t> merge_rank_;
  std::unordered_map<std::string, int32_t> token_to_id_;
  std::unordered_map<int32_t, std::string> id_to_token_;

  std::unordered_map<uint8_t, std::string> byte_encoder_;
  std::unordered_map<uint32_t, uint8_t> byte_decoder_cp_;
};

}  // namespace mslite_llm

#endif
