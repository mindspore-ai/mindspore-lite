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
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <string>
#include <vector>

#include "manifest/model_manifest.h"
#include "manifest/msl_package_reader.h"

namespace mslite_llm {
namespace {

using namespace std::chrono;  // NOLINT(build/namespaces)

std::string MakeStamp() { return std::to_string(steady_clock::now().time_since_epoch().count()); }

/// Create a minimal valid manifest.json in `root`.
void WriteManifest(const std::filesystem::path &root, const std::string &extra_json = "") {
  auto path = root / "manifest.json";
  std::ofstream out(path);
  out << R"({
  "model_name": "test-model",
  "version": "0.1.0",
  "format_version": "1.0",
  "dtype": "float32",
  "architecture": {
    "num_layers": 28,
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "num_heads": 16,
    "num_kv_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "max_position_embeddings": 40960,
    "rope_theta": 1000000.0,
    "norm_eps": 1e-6,
    "tie_word_embeddings": true
  },)" << extra_json
      << R"(
  "litert": {
    "precision": "float32",
    "capabilities": {
      "prefill": { "path": "prefill/graph.ms", "seq_len": 4096 },
      "decode": { "path": "decode/graph.ms", "dynamic_past_len": true, "max_past_len": 4096 }
    },
    "graph_io": {
      "prefill": {
        "inputs": [{"name": "input_ids", "role": "input_ids", "dtype": "int32", "shape": [1, -1]}],
        "outputs": [{"name": "logits", "role": "logits", "dtype": "float32", "shape": [1, -1, 151936], "token_axis": 1, "vocab_axis": 2}]
      },
      "decode": {
        "inputs": [{"name": "input_ids", "role": "input_ids", "dtype": "int32", "shape": [1, 1]}],
        "outputs": [{"name": "logits", "role": "logits", "dtype": "float32", "shape": [1, 1, 151936], "token_axis": 1, "vocab_axis": 2}]
      }
    }
  },
  "assets": {
    "tokenizer": "vocab/vocab.bin",
    "embedding": "embedding/embedding.bin",
    "embedding_fp16": "embedding/embedding_fp16.bin",
    "rope_sin": "rope/sin.bin",
    "rope_cos": "rope/cos.bin",
    "attention_mask": "mask/attention_mask.bin"
  },
  "npu": {
    "max_length": 4096,
    "chunk_size": 128
  }
})";
}

/// Create a minimal directory tree for a valid .msl package.
std::filesystem::path MakePackage(const std::string &suffix = "") {
  auto root = std::filesystem::temp_directory_path() / ("test_msl_" + MakeStamp() + suffix);
  std::filesystem::create_directories(root);
  std::filesystem::create_directories(root / "prefill");
  std::filesystem::create_directories(root / "decode");
  std::filesystem::create_directories(root / "vocab");
  std::filesystem::create_directories(root / "embedding");
  std::filesystem::create_directories(root / "rope");
  std::filesystem::create_directories(root / "mask");
  // Touch required asset files.
  std::ofstream(root / "prefill/graph.ms").close();
  std::ofstream(root / "decode/graph.ms").close();
  std::ofstream(root / "vocab/vocab.bin").close();
  std::ofstream(root / "embedding/embedding.bin").close();
  std::ofstream(root / "embedding/embedding_fp16.bin").close();
  std::ofstream(root / "rope/sin.bin").close();
  std::ofstream(root / "rope/cos.bin").close();
  std::ofstream(root / "mask/attention_mask.bin").close();
  return root;
}

// Path Safety Tests

TEST(PathSafety, RejectsEmptyPath) { EXPECT_FALSE(IsPackageRelativePath("")); }

TEST(PathSafety, RejectsAbsolutePath) {
  EXPECT_FALSE(IsPackageRelativePath("/etc/passwd"));
  EXPECT_FALSE(IsPackageRelativePath("C:\\windows\\system32"));
}

TEST(PathSafety, RejectsParentTraversal) {
  EXPECT_FALSE(IsPackageRelativePath("../outside/vocab.bin"));
  EXPECT_FALSE(IsPackageRelativePath("subdir/../../../escape.bin"));
  EXPECT_FALSE(IsPackageRelativePath(".."));
}

TEST(PathSafety, AcceptsSimpleRelative) {
  EXPECT_TRUE(IsPackageRelativePath("vocab/vocab.bin"));
  EXPECT_TRUE(IsPackageRelativePath("prefill/graph.ms"));
  EXPECT_TRUE(IsPackageRelativePath("a/b/c/d.bin"));
}

TEST(PathSafety, RejectsDriveColon) { EXPECT_FALSE(IsPackageRelativePath("D:file.bin")); }

// Manifest Parsing with Assets

TEST(ManifestAssets, ParsesAssetsSection) {
  auto root = MakePackage("_assets");
  WriteManifest(root);
  ModelManifest manifest;
  auto status = LoadModelManifest((root / "manifest.json").string(), &manifest);
  ASSERT_EQ(status, MSLLM_SUCCESS);
  EXPECT_TRUE(manifest.assets.present);
  EXPECT_EQ(manifest.assets.tokenizer, "vocab/vocab.bin");
  EXPECT_EQ(manifest.assets.embedding, "embedding/embedding.bin");
  EXPECT_EQ(manifest.assets.rope_sin, "rope/sin.bin");
  EXPECT_EQ(manifest.assets.rope_cos, "rope/cos.bin");
  std::filesystem::remove_all(root);
}

TEST(ManifestAssets, HandlesMissingAssetsSection) {
  auto root = MakePackage("_no_assets");
  // Write manifest without assets field
  {
    auto path = root / "manifest.json";
    std::ofstream out(path);
    out << R"({"model_name":"test","version":"1","format_version":"1.0","dtype":"float32"})";
  }
  ModelManifest manifest;
  auto status = LoadModelManifest((root / "manifest.json").string(), &manifest);
  ASSERT_EQ(status, MSLLM_SUCCESS);
  EXPECT_FALSE(manifest.assets.present);
  EXPECT_TRUE(manifest.assets.tokenizer.empty());
  std::filesystem::remove_all(root);
}

// Resource Loader Tests

TEST(ModelResourceLoader, LoadsValidNpuPackage) {
  auto root = MakePackage("_npu");
  WriteManifest(root);
  ModelResources resources;
  auto status = LoadModelResources(root.string(), &resources, MSLLM_BACKEND_NNRT);
  ASSERT_EQ(status, MSLLM_SUCCESS);
  EXPECT_FALSE(resources.prefill_path.empty());
  EXPECT_FALSE(resources.decode_path.empty());
  EXPECT_FALSE(resources.tokenizer_path.empty());
  std::filesystem::remove_all(root);
}

TEST(ModelResourceLoader, RejectsMissingNpuPrefill) {
  auto root = MakePackage("_npu_no_prefill");
  // Write manifest without prefill graph.
  auto path = root / "manifest.json";
  std::ofstream out(path);
  out << R"({
  "model_name":"test","version":"1","format_version":"1.0","dtype":"float32",
  "architecture":{"num_layers":1,"hidden_size":128,"intermediate_size":512,"num_heads":2,"num_kv_heads":2,"head_dim":64,"vocab_size":100,"max_position_embeddings":512,"rope_theta":10000,"norm_eps":1e-6,"tie_word_embeddings":false},
  "litert":{"precision":"float32","capabilities":{"decode":{"path":"decode/graph.ms"}},"graph_io":{"prefill":{"inputs":[],"outputs":[]}}},
  "assets":{"tokenizer":"vocab/vocab.bin"}
})";
  out.close();  // flush before the loader reads the manifest
  ModelResources resources;
  std::string error;
  auto status = LoadModelResources(root.string(), &resources, MSLLM_BACKEND_NNRT, &error);
  EXPECT_EQ(status, MSLLM_ERROR_MODEL_LOAD);
  EXPECT_FALSE(error.empty());
  std::filesystem::remove_all(root);
}

TEST(ModelResourceLoader, RejectsMissingManifest) {
  auto root = MakePackage("_no_manifest");
  std::filesystem::remove(root / "manifest.json");
  ModelResources resources;
  auto status = LoadModelResources(root.string(), &resources, MSLLM_BACKEND_NNRT);
  EXPECT_EQ(status, MSLLM_ERROR_IO);
  std::filesystem::remove_all(root);
}

TEST(ModelResourceLoader, NullResourcesIsInvalidArg) {
  std::string error;
  auto status = LoadModelResources("/nonexistent", nullptr, MSLLM_BACKEND_NNRT, &error);
  EXPECT_EQ(status, MSLLM_ERROR_INVALID_ARGS);
  EXPECT_FALSE(error.empty());
}

// Single-file .msl (v1) Tests

void WriteU32(std::vector<uint8_t> *buf, uint32_t v) {
  for (int i = 0; i < 4; ++i) {
    buf->push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
  }
}

void WriteU64(std::vector<uint8_t> *buf, uint64_t v) {
  for (int i = 0; i < 8; ++i) {
    buf->push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
  }
}

std::vector<uint8_t> StrVal(const std::string &s) { return std::vector<uint8_t>(s.begin(), s.end()); }
std::vector<uint8_t> U32Val(uint32_t v) {
  std::vector<uint8_t> b;
  WriteU32(&b, v);
  return b;
}
std::vector<uint8_t> BoolVal(bool v) { return std::vector<uint8_t>{static_cast<uint8_t>(v ? 1u : 0u)}; }

void WriteKvEntry(std::vector<uint8_t> *buf, const std::string &key, uint32_t type, const std::vector<uint8_t> &value) {
  WriteU32(buf, static_cast<uint32_t>(key.size()));
  buf->insert(buf->end(), key.begin(), key.end());
  WriteU32(buf, type);
  WriteU32(buf, static_cast<uint32_t>(value.size()));
  buf->insert(buf->end(), value.begin(), value.end());
}

/// Build a single-file .msl (v1: .MSL header + KV region + resource table +
/// data) from KV pairs and (name, content) resource pairs.
void WriteSingleFileMslV1(const std::filesystem::path &path,
                          const std::vector<std::pair<std::string, std::pair<uint32_t, std::vector<uint8_t>>>> &kv,
                          const std::vector<std::pair<std::string, std::string>> &resources,
                          uint32_t alignment = 4096) {
  std::vector<uint8_t> buf;
  const char kMagic[4] = {'.', 'M', 'S', 'L'};
  buf.insert(buf.end(), kMagic, kMagic + 4);
  WriteU32(&buf, 1);                                        // version
  WriteU32(&buf, static_cast<uint32_t>(kv.size()));         // kv_count
  WriteU32(&buf, static_cast<uint32_t>(resources.size()));  // resource_count
  WriteU32(&buf, alignment);                                // alignment
  WriteU32(&buf, 0);                                        // reserved

  for (const auto &[key, typed] : kv) {
    WriteKvEntry(&buf, key, typed.first, typed.second);
  }

  // Layout: header | KV region | resource table | aligned payloads.
  uint64_t cursor = 24 + static_cast<uint64_t>(buf.size()) + 88 * resources.size();
  std::vector<uint64_t> offsets;
  for (const auto &[name, content] : resources) {
    const uint64_t aligned = ((cursor + alignment - 1) / alignment) * alignment;
    offsets.push_back(aligned);
    for (size_t j = 0; j < 64; ++j) {
      buf.push_back(j < name.size() ? static_cast<uint8_t>(name[j]) : 0);
    }
    WriteU64(&buf, aligned);
    WriteU64(&buf, content.size());
    WriteU32(&buf, 1);  // access (read)
    WriteU32(&buf, 0);  // reserved
    cursor = aligned + content.size();
  }

  // Append payloads, padding each to its aligned offset from the table.
  for (size_t i = 0; i < resources.size(); ++i) {
    while (buf.size() < offsets[i]) {
      buf.push_back(0);
    }
    buf.insert(buf.end(), resources[i].second.begin(), resources[i].second.end());
  }

  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char *>(buf.data()), static_cast<std::streamsize>(buf.size()));
}

TEST(MslPackageReader, ParsesAndReadsEntries) {
  auto path = std::filesystem::temp_directory_path() / ("test_msl_single_" + MakeStamp() + ".msl");
  const uint32_t kTypeString = mslite_llm::msl_format::kTypeString;
  WriteSingleFileMslV1(path, {{"model.name", {kTypeString, StrVal("t")}}},
                       {{"npu_offline/x.omc", "omcdata"}, {"assets/embedding_quant.bin", "embdata"}});

  MslPackageReader reader;
  std::string err;
  ASSERT_TRUE(reader.Open(path.string(), &err)) << err;
  EXPECT_EQ(reader.entry_count(), 2u);
  EXPECT_EQ(reader.kv_count(), 1u);
  EXPECT_NE(reader.Lookup("npu_offline/x.omc"), nullptr);
  EXPECT_EQ(reader.Lookup("missing.bin"), nullptr);

  std::string name;
  EXPECT_TRUE(reader.GetKvString("model.name", &name));
  EXPECT_EQ(name, "t");

  std::vector<uint8_t> out;
  ASSERT_TRUE(reader.Read("npu_offline/x.omc", &out));
  EXPECT_EQ(std::string(out.begin(), out.end()), "omcdata");

  const uint8_t *data = nullptr;
  size_t size = 0;
  ASSERT_TRUE(reader.Mmap("assets/embedding_quant.bin", &data, &size));
  EXPECT_EQ(size, 7u);
  EXPECT_EQ(std::string(reinterpret_cast<const char *>(data), size), "embdata");
  std::filesystem::remove(path);
}

TEST(MslPackageReader, SkipsUnknownKvAndRejectsUnknownType) {
  const uint32_t kTypeString = mslite_llm::msl_format::kTypeString;

  auto path = std::filesystem::temp_directory_path() / ("test_msl_kv_" + MakeStamp() + ".msl");
  WriteSingleFileMslV1(path,
                       {{"model.name", {kTypeString, StrVal("t")}}, {"future.key", {kTypeString, StrVal("skipped")}}},
                       {{"a.bin", "x"}});
  MslPackageReader reader;
  std::string err;
  ASSERT_TRUE(reader.Open(path.string(), &err)) << err;
  EXPECT_EQ(reader.kv_count(), 2u);
  // Unknown keys are tolerated (forward compatible): stored but never
  // consumed by the typed getters of this reader version.
  std::string name;
  EXPECT_TRUE(reader.GetKvString("model.name", &name));
  EXPECT_EQ(name, "t");
  EXPECT_TRUE(reader.GetKvString("future.key", &name));  // stored, no error
  EXPECT_EQ(name, "skipped");
  std::filesystem::remove(path);

  auto bad_path = std::filesystem::temp_directory_path() / ("test_msl_badtype_" + MakeStamp() + ".msl");
  WriteSingleFileMslV1(bad_path, {{"bad", {99, StrVal("x")}}}, {{"a.bin", "x"}});
  MslPackageReader bad_reader;
  EXPECT_FALSE(bad_reader.Open(bad_path.string(), &err));  // unknown type: rejected
  EXPECT_FALSE(err.empty());
  std::filesystem::remove(bad_path);
}

TEST(MslPackageReader, RejectsBadMagic) {
  auto path = std::filesystem::temp_directory_path() / ("test_msl_bad_" + MakeStamp() + ".msl");
  std::ofstream out(path, std::ios::binary);
  out.write("NOTMSL", 6);
  out.close();

  MslPackageReader reader;
  std::string err;
  EXPECT_FALSE(reader.Open(path.string(), &err));
  EXPECT_FALSE(err.empty());
  std::filesystem::remove(path);
}

TEST(ModelResourceLoader, LoadsSingleFileNpuPackage) {
  auto path = std::filesystem::temp_directory_path() / ("test_msl_pkg_" + MakeStamp() + ".msl");
  const uint32_t kTypeString = mslite_llm::msl_format::kTypeString;
  const uint32_t kTypeUint32 = mslite_llm::msl_format::kTypeUint32;
  const uint32_t kTypeBool = mslite_llm::msl_format::kTypeBool;
  WriteSingleFileMslV1(path,
                       {{"model.name", {kTypeString, StrVal("test")}},
                        {"model.format_version", {kTypeString, StrVal("1.0")}},
                        {"litert.prefill.path", {kTypeString, StrVal("npu_offline/x.omc")}},
                        {"asset.tokenizer", {kTypeString, StrVal("vocab/vocab.bin")}},
                        {"asset.embedding", {kTypeString, StrVal("assets/embedding_quant.bin")}},
                        {"asset.rope_cos", {kTypeString, StrVal("assets/rope_cos.bin")}},
                        {"asset.rope_sin", {kTypeString, StrVal("assets/rope_sin.bin")}},
                        {"asset.attention_mask", {kTypeString, StrVal("assets/attention_mask.bin")}},
                        {"npu.max_length", {kTypeUint32, U32Val(1024)}},
                        {"npu.chunk_size", {kTypeUint32, U32Val(128)}},
                        {"npu.embedding_quant", {kTypeBool, BoolVal(true)}},
                        {"npu.scale_gp_size", {kTypeUint32, U32Val(32)}}},
                       {{"npu_offline/x.omc", "omc"},
                        {"assets/embedding_quant.bin", "e"},
                        {"assets/rope_cos.bin", "c"},
                        {"assets/rope_sin.bin", "s"},
                        {"assets/attention_mask.bin", "m"},
                        {"vocab/vocab.bin", "v"}});

  ModelResources resources;
  std::string err;
  auto status = LoadModelResources(path.string(), &resources, MSLLM_BACKEND_NNRT, &err);
  ASSERT_EQ(status, MSLLM_SUCCESS) << err;
  EXPECT_TRUE(resources.single_file);
  EXPECT_NE(resources.package_reader, nullptr);
  EXPECT_EQ(resources.prefill_path, "npu_offline/x.omc");
  EXPECT_EQ(resources.embedding_path, "assets/embedding_quant.bin");
  EXPECT_EQ(resources.rope_cos_path, "assets/rope_cos.bin");
  EXPECT_EQ(resources.rope_sin_path, "assets/rope_sin.bin");
  EXPECT_EQ(resources.attention_mask_path, "assets/attention_mask.bin");
  EXPECT_EQ(resources.tokenizer_path, "vocab/vocab.bin");
  std::filesystem::remove(path);
}

TEST(ModelResourceLoader, RejectsSingleFileMissingNpuConfig) {
  auto path = std::filesystem::temp_directory_path() / ("test_msl_nonpu_" + MakeStamp() + ".msl");
  const uint32_t kTypeString = mslite_llm::msl_format::kTypeString;
  // Only a resource table, no KV metadata: NPU validation must reject it.
  WriteSingleFileMslV1(path, {}, {{"npu_offline/x.omc", "omc"}});

  ModelResources resources;
  std::string err;
  auto status = LoadModelResources(path.string(), &resources, MSLLM_BACKEND_NNRT, &err);
  EXPECT_EQ(status, MSLLM_ERROR_MODEL_LOAD);
  EXPECT_FALSE(err.empty());
  std::filesystem::remove(path);
}

}  // namespace
}  // namespace mslite_llm
