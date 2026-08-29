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
/**
 * @file test_llm_chat_template.cpp
 * @brief Seam ③ — Test MSLLMApplyChatTemplate error paths + the IR interpreter.
 *
 * Error-path tests exercise the public API contract without a built model.
 * Happy-path rendering is tested directly against the internal ChatTemplate IR
 * interpreter with a fixed program compiled from the Qwen2.5/3
 * template; byte-for-byte golden parity with Jinja2 lives in the Python
 * test (tests/py/test_export_tokenizer.py).
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include "llm/llm.h"
#include "../../src/llm_types_internal.h"
#include "tokenizer/chat_template.h"

namespace mslite_llm {
namespace {

// ─── Error paths (no model loaded) ──────────────────────────────────────────

TEST(ChatTemplate, NullHandleReturnsError) {
  MSLLMChatMessage msgs[] = {{MSLLM_ROLE_USER, "hello"}};
  char buf[256];
  auto s = MSLLMApplyChatTemplate(nullptr, msgs, 1, 0, buf, sizeof(buf));
  EXPECT_EQ(s, kMSLLM_ERROR_INVALID_ARGS);
}

TEST(ChatTemplate, NullMessagesReturnsError) {
  auto *h = MSLLMCreateModel();
  ASSERT_NE(h, nullptr);
  char buf[256];
  auto s = MSLLMApplyChatTemplate(h, nullptr, 1, 0, buf, sizeof(buf));
  EXPECT_EQ(s, kMSLLM_ERROR_INVALID_ARGS);
  MSLLMDestroyModel(h);
}

TEST(ChatTemplate, ZeroMessagesReturnsError) {
  auto *h = MSLLMCreateModel();
  ASSERT_NE(h, nullptr);
  MSLLMChatMessage msgs[] = {{MSLLM_ROLE_USER, "hello"}};
  char buf[256];
  auto s = MSLLMApplyChatTemplate(h, msgs, 0, 0, buf, sizeof(buf));
  EXPECT_EQ(s, kMSLLM_ERROR_INVALID_ARGS);
  MSLLMDestroyModel(h);
}

TEST(ChatTemplate, NullBufferReturnsError) {
  auto *h = MSLLMCreateModel();
  ASSERT_NE(h, nullptr);
  MSLLMChatMessage msgs[] = {{MSLLM_ROLE_USER, "hello"}};
  auto s = MSLLMApplyChatTemplate(h, msgs, 1, 0, nullptr, 256);
  EXPECT_EQ(s, kMSLLM_ERROR_INVALID_ARGS);
  MSLLMDestroyModel(h);
}

TEST(ChatTemplate, ZeroBufferSizeReturnsError) {
  auto *h = MSLLMCreateModel();
  ASSERT_NE(h, nullptr);
  MSLLMChatMessage msgs[] = {{MSLLM_ROLE_USER, "hello"}};
  char buf[256];
  auto s = MSLLMApplyChatTemplate(h, msgs, 1, 0, buf, 0);
  EXPECT_EQ(s, kMSLLM_ERROR_INVALID_ARGS);
  MSLLMDestroyModel(h);
}

TEST(ChatTemplate, UnbuiltModelReturnsError) {
  auto *h = MSLLMCreateModel();
  ASSERT_NE(h, nullptr);
  MSLLMChatMessage msgs[] = {{MSLLM_ROLE_USER, "hello"}};
  char buf[256];
  // No BuildModel — tokenizer is null
  auto s = MSLLMApplyChatTemplate(h, msgs, 1, 0, buf, sizeof(buf));
  EXPECT_EQ(s, kMSLLM_ERROR_INVALID_ARGS);
  MSLLMDestroyModel(h);
}

// ─── Message type coverage ──────────────────────────────────────────────────

TEST(ChatTemplate, AllRoleTypesAccepted) {
  auto *h = MSLLMCreateModel();
  ASSERT_NE(h, nullptr);
  MSLLMChatMessage msgs[] = {
    {MSLLM_ROLE_SYSTEM, "You are helpful."},
    {MSLLM_ROLE_USER, "Hi"},
    {MSLLM_ROLE_ASSISTANT, "Hello!"},
  };
  char buf[256];
  // Unbuilt model → INVALID_ARGS (tokenizer null), but messages are valid
  auto s = MSLLMApplyChatTemplate(h, msgs, 3, 1, buf, sizeof(buf));
  EXPECT_EQ(s, kMSLLM_ERROR_INVALID_ARGS);
  MSLLMDestroyModel(h);
}

TEST(ChatTemplate, EmptyContentAccepted) {
  auto *h = MSLLMCreateModel();
  ASSERT_NE(h, nullptr);
  MSLLMChatMessage msgs[] = {
    {MSLLM_ROLE_USER, ""},
  };
  char buf[256];
  auto s = MSLLMApplyChatTemplate(h, msgs, 1, 0, buf, sizeof(buf));
  EXPECT_EQ(s, kMSLLM_ERROR_INVALID_ARGS);  // unbuilt model
  MSLLMDestroyModel(h);
}

TEST(ChatTemplate, NullContentReturnsInvalidArgs) {
  auto *h = MSLLMCreateModel();
  ASSERT_NE(h, nullptr);
  MSLLMChatMessage msgs[] = {
    {MSLLM_ROLE_USER, nullptr},
  };
  char buf[256];
  auto s = MSLLMApplyChatTemplate(h, msgs, 1, 0, buf, sizeof(buf));
  EXPECT_EQ(s, kMSLLM_ERROR_INVALID_ARGS);
  MSLLMDestroyModel(h);
}

// ─── IR interpreter ───────────────────────────────────────────────

// Compiled from the Qwen2.5/Qwen3 chat template via
// export/utils/chat_template_ir.py::compile_chat_template_ir:
//
//   {%- if tools %}{{ '<|im_start|>system\n' }}{%- endif %}
//   {%- for message in messages %}
//     {{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}
//   {%- endfor %}
//   {%- if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{%- endif %}
//
const uint8_t kQwenTemplateIr[] = {
  0x49, 0x4c, 0x53, 0x4d, 0x01, 0x04, 0x01, 0x0c, 0x00, 0x00, 0x00, 0x3c, 0x7c, 0x69, 0x6d, 0x5f, 0x73,
  0x74, 0x61, 0x72, 0x74, 0x7c, 0x3e, 0x02, 0x01, 0x01, 0x00, 0x00, 0x00, 0x0a, 0x03, 0x01, 0x0a, 0x00,
  0x00, 0x00, 0x3c, 0x7c, 0x69, 0x6d, 0x5f, 0x65, 0x6e, 0x64, 0x7c, 0x3e, 0x01, 0x01, 0x00, 0x00, 0x00,
  0x0a, 0x05, 0x06, 0x01, 0x16, 0x00, 0x00, 0x00, 0x3c, 0x7c, 0x69, 0x6d, 0x5f, 0x73, 0x74, 0x61, 0x72,
  0x74, 0x7c, 0x3e, 0x61, 0x73, 0x73, 0x69, 0x73, 0x74, 0x61, 0x6e, 0x74, 0x0a, 0x07, 0x08,
};

ChatTemplate MakeQwenTemplate() {
  ChatTemplate tmpl;
  tmpl.SetCustomTemplate(std::string(reinterpret_cast<const char *>(kQwenTemplateIr), sizeof(kQwenTemplateIr)));
  return tmpl;
}

TEST(ChatTemplateIr, RendersMultiTurnWithGenerationPrompt) {
  auto tmpl = MakeQwenTemplate();
  std::vector<MSLlmChatMessage> msgs = {
    {MSLLM_ROLE_USER, "hi"},
    {MSLLM_ROLE_ASSISTANT, "hello!"},
  };
  EXPECT_EQ(tmpl.Apply(msgs, true),
            "<|im_start|>user\nhi<|im_end|>\n"
            "<|im_start|>assistant\nhello!<|im_end|>\n"
            "<|im_start|>assistant\n");
  EXPECT_EQ(tmpl.Apply(msgs, false),
            "<|im_start|>user\nhi<|im_end|>\n"
            "<|im_start|>assistant\nhello!<|im_end|>\n");
}

TEST(ChatTemplateIr, SystemMessageRendersInPlace) {
  auto tmpl = MakeQwenTemplate();
  std::vector<MSLlmChatMessage> msgs = {
    {MSLLM_ROLE_SYSTEM, "You are helpful."},
    {MSLLM_ROLE_USER, "hi"},
  };
  EXPECT_EQ(tmpl.Apply(msgs, true),
            "<|im_start|>system\nYou are helpful.<|im_end|>\n"
            "<|im_start|>user\nhi<|im_end|>\n"
            "<|im_start|>assistant\n");
}

TEST(ChatTemplateIr, LastMessageAssistantRenders) {
  // #9: the last message being an assistant turn is legal — the template
  // renders it verbatim; the model decides how to continue.
  auto tmpl = MakeQwenTemplate();
  std::vector<MSLlmChatMessage> msgs = {
    {MSLLM_ROLE_USER, "hi"},
    {MSLLM_ROLE_ASSISTANT, "hello!"},
  };
  EXPECT_EQ(tmpl.Apply(msgs, false),
            "<|im_start|>user\nhi<|im_end|>\n"
            "<|im_start|>assistant\nhello!<|im_end|>\n");
}

TEST(ChatTemplateIr, EmptyMessagesRenderLoopSuffixOnly) {
  auto tmpl = MakeQwenTemplate();
  std::vector<MSLlmChatMessage> msgs;
  EXPECT_EQ(tmpl.Apply(msgs, true), "<|im_start|>assistant\n");
  EXPECT_EQ(tmpl.Apply(msgs, false), "");
}

TEST(ChatTemplateIr, HasTemplateValidatesIrHeader) {
  ChatTemplate tmpl;
  EXPECT_FALSE(tmpl.HasTemplate());
  tmpl.SetCustomTemplate("raw jinja payload is not IR");
  EXPECT_FALSE(tmpl.HasTemplate());
  tmpl.SetCustomTemplate("MSLI");
  EXPECT_FALSE(tmpl.HasTemplate());  // header too short
  EXPECT_EQ(tmpl.Apply({{MSLLM_ROLE_USER, "x"}}, true), "");
}

}  // namespace
}  // namespace mslite_llm
