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
// Command-line driver for offline Hugging Face/C++ tokenizer parity tests.
// Emits encoded IDs, suppressed IDs, and decoded text for the Python assertions.

#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include "tokenizer/tokenizer.h"

int main(int argc, char **argv) {
  if (argc != 3) return 2;
  auto tokenizer = mslite_llm::CreateTokenizer(argv[1]);
  if (tokenizer == nullptr) return 3;
  std::ifstream input(argv[2]);
  if (!input) return 4;
  std::string text((std::istreambuf_iterator<char>(input)), {});
  auto ids = tokenizer->Encode(text);
  for (auto id : ids) std::cout << id << " ";
  std::cout << "\n";
  for (auto id : tokenizer->SuppressedTokenIds()) std::cout << id << " ";
  std::cout << "\n" << tokenizer->Decode(ids);
  return 0;
}
