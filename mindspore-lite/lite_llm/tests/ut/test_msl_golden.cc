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
// Golden test for the .msl v1 format contract.
//
// Validates tests/data/golden_v1.msl (produced by msl_pack.py via
// tests/data/gen_golden.py) byte-for-byte against the v1 layout:
//
//   MslHeader(24B): ".MSL" | version=1 | kv_count | resource_count
//                   | alignment=4096 | reserved
//   KV region:      key_len(u32) | key | type(u32) | value_len(u32) | value
//   Resource table: name[64] | offset(u64) | size(u64) | access(u32) | reserved(u32)
//   Data region:    payloads, offsets aligned to `alignment`
//
// This test deliberately parses raw bytes and does NOT depend on the
// runtime reader (MslPackageReader): it pins the format contract itself,
// so a layout change on either side breaks it before the reader exists.

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "manifest/msl_package_reader.h"

#ifndef MSL_GOLDEN_FILE
#error "MSL_GOLDEN_FILE must be set by the build (absolute path to golden_v1.msl)"
#endif

namespace {

constexpr uint32_t kVersion = 1;
constexpr size_t kHeaderSize = 24;
constexpr size_t kEntrySize = 88;
constexpr size_t kNameSize = 64;
constexpr uint32_t kAlignment = 4096;

// KV value types (v1 closed set).
constexpr uint32_t kTypeBool = 0;
constexpr uint32_t kTypeUint32 = 1;
constexpr uint32_t kTypeUint64 = 2;
constexpr uint32_t kTypeFloat32 = 3;
constexpr uint32_t kTypeString = 4;
constexpr uint32_t kTypeStringArray = 5;

struct ExpectedKv {
  const char *key;
  uint32_t type;
  const std::vector<uint8_t> value;  // raw value bytes
};

struct ExpectedResource {
  const char *name;
  uint64_t offset;
  uint64_t size;
  uint32_t access;
};

const std::vector<uint8_t> Bytes(const char *s) { return std::vector<uint8_t>(s, s + std::strlen(s)); }
std::vector<uint8_t> Hex(const char *hex) {
  std::vector<uint8_t> out;
  while (*hex != '\0') {
    uint8_t byte = 0;
    for (int i = 0; i < 2; ++i) {
      char c = *hex++;
      byte <<= 4;
      if (c >= '0' && c <= '9')
        byte |= static_cast<uint8_t>(c - '0');
      else
        byte |= static_cast<uint8_t>(c - 'a' + 10);
    }
    out.push_back(byte);
  }
  return out;
}

// Expected KV entries, in file order (see gen_golden.py KV dict).
const std::vector<ExpectedKv> &ExpectedKvs() {
  static const std::vector<ExpectedKv> kvs = {
    {"model.name", kTypeString, Bytes("qwen2.5-0.5b")},
    {"model.dtype", kTypeString, Bytes("fp16")},
    {"arch.num_layers", kTypeUint32, Hex("18000000")},
    {"arch.rope_theta", kTypeFloat32, Hex("00401c46")},
    {"arch.norm_eps", kTypeFloat32, Hex("bd378635")},
    {"npu.max_length", kTypeUint32, Hex("00040000")},
    {"npu.embedding_quant", kTypeBool, Hex("01")},
    {"gen.eos_token_id", kTypeUint32, Hex("5b500200")},
    {"litert.decode_variants", kTypeString, Bytes("[{\"past_len\":1,\"path\":\"npu_offline/x.omc\"}]")},
    {"string.array", kTypeStringArray, Hex("03000000010000006102000000626203000000636363")},
  };
  return kvs;
}

const std::vector<ExpectedResource> &ExpectedResources() {
  static const std::vector<ExpectedResource> res = {
    {"npu_offline/x.omc", 4096, 70000, 0},
    {"assets/embedding_quant.bin", 77824, 12345, 0},
    {"vocab/vocab.bin", 94208, 3000, 1},
    {"a.bin", 98304, 1, 1},
  };
  return res;
}

// Deterministic payload bytes used by gen_golden.py.
uint8_t PayloadByte(size_t i) { return static_cast<uint8_t>((i * 7 + 3) % 256); }

uint32_t ReadU32(const uint8_t *p) {
  uint32_t v = 0;
  std::memcpy(&v, p, sizeof(v));
  return v;
}
uint64_t ReadU64(const uint8_t *p) {
  uint64_t v = 0;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

std::vector<uint8_t> ReadWholeFile(const std::string &path) {
  int fd = ::open(path.c_str(), O_RDONLY);
  EXPECT_GE(fd, 0) << "cannot open " << path;
  if (fd < 0) return {};
  struct stat st {};
  EXPECT_EQ(::fstat(fd, &st), 0);
  std::vector<uint8_t> data(static_cast<size_t>(st.st_size));
  size_t off = 0;
  while (off < data.size()) {
    ssize_t n = ::read(fd, data.data() + off, data.size() - off);
    if (n <= 0) break;
    off += static_cast<size_t>(n);
  }
  ::close(fd);
  return data;
}

class MslGoldenTest : public ::testing::Test {
 protected:
  void SetUp() override {
    struct stat st {};
    if (::stat(MSL_GOLDEN_FILE, &st) != 0) {
      // golden_v1.msl is a generated binary kept out of the repo; it is
      // provisioned by the CI data repo.  Skip (not fail) when absent.
      GTEST_SKIP() << "golden_v1.msl not provisioned; see tests/data/gen_golden.py";
    }
    data_ = ReadWholeFile(MSL_GOLDEN_FILE);
  }

  std::vector<uint8_t> data_;
};

TEST_F(MslGoldenTest, Header) {
  ASSERT_GE(data_.size(), kHeaderSize);
  EXPECT_EQ(std::memcmp(data_.data(), ".MSL", 4), 0);
  EXPECT_EQ(ReadU32(data_.data() + 4), kVersion);
  EXPECT_EQ(ReadU32(data_.data() + 8), ExpectedKvs().size());
  EXPECT_EQ(ReadU32(data_.data() + 12), ExpectedResources().size());
  EXPECT_EQ(ReadU32(data_.data() + 16), kAlignment);
  EXPECT_EQ(ReadU32(data_.data() + 20), 0u);  // reserved
}

TEST_F(MslGoldenTest, KvRegion) {
  const auto &expected = ExpectedKvs();
  size_t pos = kHeaderSize;
  for (size_t i = 0; i < expected.size(); ++i) {
    ASSERT_LE(pos + 4, data_.size());
    uint32_t key_len = ReadU32(data_.data() + pos);
    pos += 4;
    ASSERT_LE(pos + key_len + 8, data_.size());
    std::string key(reinterpret_cast<const char *>(data_.data() + pos), key_len);
    EXPECT_EQ(key, expected[i].key) << "KV key mismatch at index " << i;
    pos += key_len;
    uint32_t type = ReadU32(data_.data() + pos);
    uint32_t value_len = ReadU32(data_.data() + pos + 4);
    pos += 8;
    ASSERT_LE(pos + value_len, data_.size());
    EXPECT_EQ(type, expected[i].type) << "KV type mismatch for " << expected[i].key;
    ASSERT_EQ(value_len, expected[i].value.size()) << "KV value length for " << expected[i].key;
    EXPECT_EQ(std::memcmp(data_.data() + pos, expected[i].value.data(), value_len), 0)
      << "KV value bytes for " << expected[i].key;
    pos += value_len;
  }
  // The resource table starts right after the last KV entry.
  EXPECT_EQ(pos + kEntrySize * ExpectedResources().size() <= data_.size(), true);
}

TEST_F(MslGoldenTest, ResourceTable) {
  // Locate the table: header + all KV entries.
  const auto &kvs = ExpectedKvs();
  size_t pos = kHeaderSize;
  for (const auto &kv : kvs) {
    pos += 4 + std::strlen(kv.key);
    pos += 8 + kv.value.size();
  }
  const auto &expected = ExpectedResources();
  for (size_t i = 0; i < expected.size(); ++i) {
    size_t base = pos + kEntrySize * i;
    ASSERT_LE(base + kEntrySize, data_.size());
    std::string name(reinterpret_cast<const char *>(data_.data() + base), kNameSize);
    name = name.substr(0, name.find('\0'));
    EXPECT_EQ(name, expected[i].name) << "resource name mismatch at index " << i;
    uint64_t offset = ReadU64(data_.data() + base + kNameSize);
    uint64_t size = ReadU64(data_.data() + base + kNameSize + 8);
    uint32_t access = ReadU32(data_.data() + base + kNameSize + 16);
    EXPECT_EQ(offset, expected[i].offset) << "offset for " << expected[i].name;
    EXPECT_EQ(size, expected[i].size) << "size for " << expected[i].name;
    EXPECT_EQ(access, expected[i].access) << "access for " << expected[i].name;
    EXPECT_EQ(offset % kAlignment, 0u) << "payload not aligned: " << expected[i].name;
    EXPECT_LE(offset + size, data_.size());
  }
}

TEST_F(MslGoldenTest, PayloadBytes) {
  const auto &expected = ExpectedResources();
  for (const auto &res : expected) {
    ASSERT_LE(res.offset + res.size, data_.size());
    for (uint64_t j = 0; j < res.size; ++j) {
      EXPECT_EQ(data_[res.offset + j], PayloadByte(static_cast<size_t>(j)))
        << "payload byte " << j << " of " << res.name;
    }
  }
}

TEST_F(MslGoldenTest, RuntimeReader) {
  // The runtime reader must accept the golden file and surface the same
  // metadata/resources the raw-byte checks above pin down.
  mslite_llm::MslPackageReader reader;
  ASSERT_TRUE(reader.Open(MSL_GOLDEN_FILE));
  EXPECT_EQ(reader.kv_count(), ExpectedKvs().size());
  EXPECT_EQ(reader.entry_count(), ExpectedResources().size());

  std::string str;
  EXPECT_TRUE(reader.GetKvString("model.name", &str));
  EXPECT_EQ(str, "qwen2.5-0.5b");
  EXPECT_TRUE(reader.GetKvString("litert.decode_variants", &str));
  EXPECT_EQ(str, "[{\"past_len\":1,\"path\":\"npu_offline/x.omc\"}]");

  uint32_t u32 = 0;
  EXPECT_TRUE(reader.GetKvUint32("arch.num_layers", &u32));
  EXPECT_EQ(u32, 24u);
  EXPECT_TRUE(reader.GetKvUint32("gen.eos_token_id", &u32));
  EXPECT_EQ(u32, 151643u);

  bool flag = false;
  EXPECT_TRUE(reader.GetKvBool("npu.embedding_quant", &flag));
  EXPECT_TRUE(flag);

  float f32 = 0.0f;
  EXPECT_TRUE(reader.GetKvFloat32("arch.rope_theta", &f32));
  EXPECT_FLOAT_EQ(f32, 10000.0f);

  std::vector<std::string> arr;
  EXPECT_TRUE(reader.GetKvStringArray("string.array", &arr));
  ASSERT_EQ(arr.size(), 3u);
  EXPECT_EQ(arr[0], "a");
  EXPECT_EQ(arr[1], "bb");
  EXPECT_EQ(arr[2], "ccc");

  // Unknown keys are skipped (absent), unknown access fails type check.
  EXPECT_FALSE(reader.GetKvString("no.such.key", &str));
  EXPECT_FALSE(reader.GetKvUint32("model.name", &u32));

  // Resource access via the entry table.
  const auto *entry = reader.Lookup("npu_offline/x.omc");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->size, 70000u);
  const uint8_t *data = nullptr;
  size_t size = 0;
  ASSERT_TRUE(reader.Mmap("npu_offline/x.omc", &data, &size));
  EXPECT_EQ(size, 70000u);
  EXPECT_EQ(data[0], PayloadByte(0));
  EXPECT_EQ(data[69999], PayloadByte(69999));
  EXPECT_TRUE(reader.Reclaim("npu_offline/x.omc"));
  EXPECT_FALSE(reader.Reclaim("missing.bin"));
  EXPECT_EQ(data[0], PayloadByte(0));

  std::vector<uint8_t> out;
  EXPECT_TRUE(reader.Read("a.bin", &out));
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0], PayloadByte(0));
  EXPECT_EQ(reader.Lookup("missing.bin"), nullptr);
}

}  // namespace
