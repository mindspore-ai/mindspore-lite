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
#include "tokenizer/chat_template.h"

#include <cstdint>
#include <cstring>
#include <sstream>
#include <vector>

namespace mslite_llm {

namespace {

// IR byte schema (v1) — must match export/utils/chat_template_ir.py.
constexpr uint32_t kIrMagic = 0x4D534C49;
constexpr uint8_t kIrVersion = 1;
constexpr uint8_t kIrHeaderSize = 5;

constexpr uint8_t kOpEmitConst = 0x01;
constexpr uint8_t kOpEmitRole = 0x02;
constexpr uint8_t kOpEmitContent = 0x03;
constexpr uint8_t kOpLoopStart = 0x04;
constexpr uint8_t kOpLoopEnd = 0x05;
constexpr uint8_t kOpIfAddGenStart = 0x06;
constexpr uint8_t kOpIfEnd = 0x07;
constexpr uint8_t kOpEnd = 0x08;

uint32_t ReadU32(const uint8_t *data, size_t size, size_t &offset) {
  if (offset + 4 > size) return 0;
  uint32_t v;
  std::memcpy(&v, data + offset, 4);
  offset += 4;
  return v;
}

const char *RoleName(MSLlmChatRole role) {
  switch (role) {
    case MSLLM_ROLE_SYSTEM:
      return "system";
    case MSLLM_ROLE_USER:
      return "user";
    case MSLLM_ROLE_ASSISTANT:
      return "assistant";
  }
  return "";
}

struct Frame {
  bool is_loop;
  size_t body_start;  // LOOP: resume position for the next message
  size_t msg_index;   // LOOP: current message index
};

const Frame *CurrentLoop(const std::vector<Frame> &frames) {
  for (auto it = frames.rbegin(); it != frames.rend(); ++it) {
    if (it->is_loop) return &*it;
  }
  return nullptr;
}

/// Advance past a false IF block. Parses nested IFs and EMIT_CONST payloads so
/// constant bytes that happen to look like opcodes are not misread.
size_t SkipToIfEnd(const uint8_t *data, size_t size, size_t pos) {
  int depth = 1;
  while (pos < size) {
    uint8_t op = data[pos++];
    if (op == kOpEmitConst) {
      size_t len = ReadU32(data, size, pos);
      pos += len;
    } else if (op == kOpIfAddGenStart) {
      ++depth;
    } else if (op == kOpIfEnd) {
      --depth;
      if (depth == 0) return pos;
    }
  }
  return size;
}

/// Advance past a loop body with no messages to iterate. Parses nested loops
/// and EMIT_CONST payloads.
size_t SkipToLoopEnd(const uint8_t *data, size_t size, size_t pos) {
  int depth = 1;
  while (pos < size) {
    uint8_t op = data[pos++];
    if (op == kOpEmitConst) {
      size_t len = ReadU32(data, size, pos);
      pos += len;
    } else if (op == kOpLoopStart) {
      ++depth;
    } else if (op == kOpLoopEnd) {
      --depth;
      if (depth == 0) return pos;
    }
  }
  return size;
}

}  // namespace

void ChatTemplate::SetCustomTemplate(const std::string &tmpl) { custom_template_ = tmpl; }

bool ChatTemplate::HasTemplate() const {
  if (custom_template_.size() < kIrHeaderSize) return false;
  const auto *data = reinterpret_cast<const uint8_t *>(custom_template_.data());
  size_t offset = 0;
  return ReadU32(data, custom_template_.size(), offset) == kIrMagic && offset < custom_template_.size() &&
         data[offset] == kIrVersion;
}

std::string ChatTemplate::Apply(const std::vector<MSLlmChatMessage> &messages, bool add_generation_prompt) const {
  if (!HasTemplate()) return {};
  const auto *data = reinterpret_cast<const uint8_t *>(custom_template_.data());
  const size_t size = custom_template_.size();
  size_t pos = kIrHeaderSize;

  std::ostringstream oss;
  std::vector<Frame> frames;

  while (pos < size) {
    uint8_t op = data[pos++];
    switch (op) {
      case kOpEmitConst: {
        size_t len = ReadU32(data, size, pos);
        if (pos + len > size) return {};
        oss.write(reinterpret_cast<const char *>(data + pos), static_cast<std::streamsize>(len));
        pos += len;
        break;
      }
      case kOpEmitRole: {
        const Frame *frame = CurrentLoop(frames);
        if (frame == nullptr || frame->msg_index >= messages.size()) return {};
        oss << RoleName(messages[frame->msg_index].role);
        break;
      }
      case kOpEmitContent: {
        const Frame *frame = CurrentLoop(frames);
        if (frame == nullptr || frame->msg_index >= messages.size()) return {};
        const char *content = messages[frame->msg_index].content;
        if (content != nullptr) oss << content;
        break;
      }
      case kOpLoopStart: {
        if (messages.empty()) {
          pos = SkipToLoopEnd(data, size, pos);
        } else {
          frames.push_back({true, pos, 0});
        }
        break;
      }
      case kOpLoopEnd: {
        if (frames.empty() || !frames.back().is_loop) return {};
        Frame &frame = frames.back();
        ++frame.msg_index;
        if (frame.msg_index < messages.size()) {
          pos = frame.body_start;
        } else {
          frames.pop_back();
        }
        break;
      }
      case kOpIfAddGenStart: {
        if (!add_generation_prompt) {
          pos = SkipToIfEnd(data, size, pos);
        }
        break;
      }
      case kOpIfEnd:
        break;
      case kOpEnd:
        return oss.str();
      default:
        return {};
    }
  }
  return oss.str();
}

}  // namespace mslite_llm
