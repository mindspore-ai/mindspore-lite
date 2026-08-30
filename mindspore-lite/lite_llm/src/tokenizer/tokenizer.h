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
#ifndef MSLLM_TOKENIZER_H
#define MSLLM_TOKENIZER_H

#include <memory>
#include <string>
#include <vector>
#include "../llm_types_internal.h"

namespace mslite_llm {

class Tokenizer {
 public:
  virtual ~Tokenizer() = default;

  virtual std::vector<int32_t> Encode(const std::string &text) = 0;
  virtual std::string Decode(const std::vector<int32_t> &token_ids) = 0;
  /// Streaming decode: returns only complete UTF-8 characters for a single
  /// token, buffering any trailing incomplete byte sequence internally.
  virtual std::string DecodeIncremental(int32_t token_id) = 0;
  /// Emit any buffered incomplete bytes (call once after the last token).
  virtual std::string FlushDecode() = 0;
  /// Whether the tokenizer carries a pinned chat template (export-time IR).
  virtual bool HasChatTemplate() const = 0;
  virtual std::string ApplyChatTemplate(const std::vector<MSLlmChatMessage> &messages, bool add_generation_prompt) = 0;
  virtual const std::vector<int32_t> &SuppressedTokenIds() const = 0;
  virtual bool IsStopTokenId(int32_t token_id) const = 0;
};

std::unique_ptr<Tokenizer> CreateTokenizer(const std::string &vocab_path);

/// Create a tokenizer from an in-memory vocab image (single-file .msl entry).
std::unique_ptr<Tokenizer> CreateTokenizerFromBuffer(const uint8_t *data, size_t size);

}  // namespace mslite_llm

#endif
