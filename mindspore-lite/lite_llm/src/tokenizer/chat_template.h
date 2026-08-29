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
#ifndef MSLLM_CHAT_TEMPLATE_H
#define MSLLM_CHAT_TEMPLATE_H

#include <string>
#include <vector>
#include "../llm_types_internal.h"

namespace mslite_llm {

// Restricted IR interpreter: the custom template stored in the
// package is a compiled instruction stream, rendered with no Jinja dependency.
// Built-in families (kChatML/Vicuna/Llama/Alpaca) were removed: the runtime no
// longer ships incomplete default renderers.
class ChatTemplate {
 public:
  ChatTemplate() = default;
  ~ChatTemplate() = default;

  void SetCustomTemplate(const std::string &tmpl);

  /// True when the stored bytes are a valid v1 IR program.
  bool HasTemplate() const;
  /// Render messages through the IR program.
  std::string Apply(const std::vector<MSLlmChatMessage> &messages, bool add_generation_prompt) const;

 private:
  std::string custom_template_;
};

}  // namespace mslite_llm

#endif
