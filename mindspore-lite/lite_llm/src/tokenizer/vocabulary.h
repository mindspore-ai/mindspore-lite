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
#ifndef MSLLM_VOCABULARY_H
#define MSLLM_VOCABULARY_H

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mslite_llm {

using TokenToIdMap = std::unordered_map<std::string_view, int32_t>;

// Token strings are owned once by id_to_token. The string_view index is built
// only after that vector is complete and remains valid while Vocabulary is
// immutable.
struct Vocabulary {
  std::vector<std::string> id_to_token;
  TokenToIdMap token_to_id;
};

}  // namespace mslite_llm

#endif  // MSLLM_VOCABULARY_H
