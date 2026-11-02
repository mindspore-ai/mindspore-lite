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

#include <future>
#include <memory>
#include <string>
#include <vector>
#include "../llm_types_internal.h"

#include "tokenizer/tokenizer_base.h"

namespace mslite_llm {

std::unique_ptr<TokenizerBase> CreateTokenizer(const std::string &vocab_path);

/// Async variant: parse on a background thread; the future yields nullptr on
/// failure. Falls back to synchronous parsing if no thread can be created.
std::future<std::unique_ptr<TokenizerBase>> CreateTokenizerAsync(const std::string &vocab_path);

/// Create a tokenizer from an in-memory vocab image (single-file .msl entry).
std::unique_ptr<TokenizerBase> CreateTokenizerFromBuffer(const uint8_t *data, size_t size);

/// Async buffer variant: the buffer moves into the background task.
std::future<std::unique_ptr<TokenizerBase>> CreateTokenizerFromBufferAsync(std::vector<uint8_t> vocab_buffer);

}  // namespace mslite_llm

#endif
