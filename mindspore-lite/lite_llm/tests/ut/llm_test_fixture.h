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
#ifndef LLM_TEST_FIXTURE_H
#define LLM_TEST_FIXTURE_H

#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <utility>
#include <vector>
namespace mslite_llm_test {

/// Builds the bytes of a minimal valid v2 BPE vocab.bin understood by
/// TokenizerImpl::LoadFromBuffer. Format (little-endian):
///
///   u32 magic(0x4D534C54) | u32 version(2) | u32 codec(0=BPE) | u32 vocab_size
///   i32 bos | i32 eos | i32 pad | i32 unk | u32 legacy_type(0)
///   vocab_size x (u32 len + token bytes + i32 id)
///   u32 num_merges(0) | u32 template_len(0)
///   u32 stop_count + stop tokens | u32 suppress_count + suppress tokens
inline std::vector<uint8_t> BuildMinimalVocabBin() {
  constexpr uint32_t kMagic = 0x4D534C54;
  constexpr uint32_t kVersion = 2;
  constexpr uint32_t kCodecBPE = 0;

  // Sorted by token id. Ids 5/6/7 are the byte-encoder tokens for the three
  // UTF-8 bytes of '你' (0xE4 0xBD 0xA0), used by the incremental-decode test.
  const std::vector<std::pair<std::string, int32_t>> vocab = {
    {"<unk>", 0}, {"<s>", 1}, {"</s>", 2}, {"a", 3}, {"b", 4}, {"ä", 5}, {"½", 6}, {"ł", 7},
  };

  std::vector<uint8_t> b;
  auto u32 = [&b](uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
  };
  auto i32 = [&u32](int32_t v) { u32(static_cast<uint32_t>(v)); };
  auto str = [&b, &u32](const std::string &s) {
    u32(static_cast<uint32_t>(s.size()));
    b.insert(b.end(), s.begin(), s.end());
  };

  u32(kMagic);
  u32(kVersion);
  u32(kCodecBPE);
  u32(static_cast<uint32_t>(vocab.size()));
  i32(1);   // bos
  i32(2);   // eos
  i32(-1);  // pad
  i32(0);   // unk
  u32(0);   // legacy chat template type

  for (const auto &[tok, id] : vocab) {
    str(tok);
    i32(id);
  }

  u32(0);  // num_merges
  // Minimal chat-template IR (v1): for each message emit "role:content\n",
  // then END.  Lets ApplyChatTemplate exercise the render path (buffer size,
  // arbitrary role sequences) instead of short-circuiting with MODEL_LOAD.
  // Bytes: magic(4) ver(1) | LOOP_START | EMIT_ROLE | EMIT_CONST(":") |
  //        EMIT_CONTENT | LOOP_END | EMIT_CONST("\n") | END
  const std::vector<uint8_t> tmpl = {0x49, 0x4c, 0x53, 0x4d, 0x01, 0x04, 0x02, 0x01, 0x01, 0x00, 0x00,
                                     0x00, 0x3a, 0x03, 0x05, 0x01, 0x01, 0x00, 0x00, 0x00, 0x0a, 0x08};
  u32(static_cast<uint32_t>(tmpl.size()));
  b.insert(b.end(), tmpl.begin(), tmpl.end());

  u32(1);  // stop_count
  str("</s>");
  u32(0);  // suppress_count

  return b;
}

inline const char *MinimalManifestJson() {
  return R"({
  "model_name": "test-llm",
  "version": "1.0.0",
  "dtype": "float16",
  "architecture": {
    "num_layers": 1,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_heads": 1,
    "num_kv_heads": 1,
    "head_dim": 64,
    "vocab_size": 8,
    "max_position_embeddings": 64,
    "rope_theta": 10000.0,
    "norm_eps": 1e-06
  },
  "generation": {
    "stop_token_ids": [2],
    "suppress_token_ids": []
  },
  "litert": {
    "precision": "float16",
    "prefill": "npu/prefill.omc",
    "decode": "npu/decode.omc"
  },
  "assets": {
    "tokenizer": "vocab.bin",
    "embedding": "embedding.bin",
    "rope_sin": "rope_sin.bin",
    "rope_cos": "rope_cos.bin",
    "attention_mask": "attention_mask.bin"
  },
  "npu": {
    "max_length": 64,
    "chunk_size": 16
  }
})";
}

/// A model-package directory written under a tempdir. All files are removed on
/// destruction. Nothing is committed to the repo.
struct ModelFixture {
  std::string dir;

  ~ModelFixture() {
    if (dir.empty()) return;
    RemoveFile(dir + "/manifest.json");
    RemoveFile(dir + "/vocab.bin");
    RemoveFile(dir + "/embedding.bin");
    RemoveFile(dir + "/rope_sin.bin");
    RemoveFile(dir + "/rope_cos.bin");
    RemoveFile(dir + "/attention_mask.bin");
    RemoveFile(dir + "/npu/prefill.omc");
    RemoveFile(dir + "/npu/decode.omc");
    ::rmdir((dir + "/npu").c_str());
    ::rmdir(dir.c_str());
  }

  ModelFixture(const ModelFixture &) = delete;
  ModelFixture &operator=(const ModelFixture &) = delete;
  ModelFixture(ModelFixture &&o) noexcept : dir(std::move(o.dir)) { o.dir.clear(); }
  ModelFixture() = default;

 private:
  static void RemoveFile(const std::string &p) { ::unlink(p.c_str()); }
};

inline bool WriteFile(const std::string &path, const void *data, size_t size) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f.is_open()) return false;
  f.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
  return f.good();
}

inline bool TouchFile(const std::string &path) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  return f.is_open();
}

/// Write a minimal, valid model package (manifest + vocab + dummy assets) into
/// a fresh tempdir and return it. Returns a fixture with empty `dir` on failure.
inline ModelFixture WriteMinimalModelDir() {
  char tmpl[] = "/tmp/msllm_fixture_XXXXXX";
  char *made = ::mkdtemp(tmpl);
  if (made == nullptr) return {};

  ModelFixture fx;
  fx.dir = made;

  ::mkdir((fx.dir + "/npu").c_str(), 0755);

  auto vocab = BuildMinimalVocabBin();
  bool ok = WriteFile(fx.dir + "/manifest.json", MinimalManifestJson(), std::string(MinimalManifestJson()).size()) &&
            WriteFile(fx.dir + "/vocab.bin", vocab.data(), vocab.size()) && TouchFile(fx.dir + "/embedding.bin") &&
            TouchFile(fx.dir + "/rope_sin.bin") && TouchFile(fx.dir + "/rope_cos.bin") &&
            TouchFile(fx.dir + "/attention_mask.bin") && TouchFile(fx.dir + "/npu/prefill.omc") &&
            TouchFile(fx.dir + "/npu/decode.omc");

  if (!ok) {
    fx.dir.clear();
  }
  return fx;
}

}  // namespace mslite_llm_test

#endif  // LLM_TEST_FIXTURE_H
