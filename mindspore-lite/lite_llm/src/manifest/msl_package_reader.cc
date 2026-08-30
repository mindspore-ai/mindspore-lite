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
#include "manifest/msl_package_reader.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
namespace mslite_llm {

namespace {

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

// Decode a KV value of `type` from `raw`. Unknown types are rejected.
bool DecodeKv(const msl_format::KvType type, const uint8_t *raw, size_t len, MslKvValue *out) {
  switch (type) {
    case msl_format::kTypeBool:
      out->value.assign(raw, raw + len);
      return len == 1;
    case msl_format::kTypeUint32:
      return len == 4 && (out->value.assign(raw, raw + len), true);
    case msl_format::kTypeUint64:
      return len == 8 && (out->value.assign(raw, raw + len), true);
    case msl_format::kTypeFloat32:
      return len == 4 && (out->value.assign(raw, raw + len), true);
    case msl_format::kTypeString:
      out->value.assign(raw, raw + len);
      return true;
    case msl_format::kTypeStringArray: {
      if (len < 4) {
        return false;
      }
      const uint32_t count = ReadU32(raw);
      size_t pos = 4;
      for (uint32_t i = 0; i < count; ++i) {
        if (pos + 4 > len) {
          return false;
        }
        const uint32_t item_len = ReadU32(raw + pos);
        pos += 4;
        if (pos + item_len > len) {
          return false;
        }
        pos += item_len;
      }
      if (pos != len) {
        return false;
      }
      out->value.assign(raw, raw + len);
      return true;
    }
    default:
      return false;
  }
}

}  // namespace

MslPackageReader::~MslPackageReader() {
  if (mapped_ != nullptr) {
    ::munmap(mapped_, mapped_size_);
    mapped_ = nullptr;
  }
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

bool MslPackageReader::Open(const std::string &path, std::string *error_message) {
  fd_ = ::open(path.c_str(), O_RDONLY);
  if (fd_ < 0) {
    if (error_message != nullptr) {
      *error_message = "cannot open .msl file: " + path;
    }
    return false;
  }

  struct stat st {};
  if (::fstat(fd_, &st) != 0 || st.st_size < static_cast<off_t>(msl_format::kHeaderSize)) {
    if (error_message != nullptr) {
      *error_message = "invalid .msl file (too small): " + path;
    }
    return false;
  }

  mapped_size_ = static_cast<size_t>(st.st_size);
  void *mapped = ::mmap(nullptr, mapped_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (mapped == MAP_FAILED) {
    if (error_message != nullptr) {
      *error_message = "mmap .msl failed: " + path;
    }
    return false;
  }
  mapped_ = static_cast<uint8_t *>(mapped);

  if (std::memcmp(mapped_, msl_format::kMagic, sizeof(msl_format::kMagic)) != 0) {
    if (error_message != nullptr) {
      *error_message = "bad .msl magic (expected .MSL): " + path;
    }
    return false;
  }
  const uint32_t version = ReadU32(mapped_ + 4);
  if (version != msl_format::kVersion) {
    if (error_message != nullptr) {
      *error_message = "unsupported .msl version: " + std::to_string(version);
    }
    return false;
  }
  const uint32_t kv_count = ReadU32(mapped_ + 8);
  const uint32_t resource_count = ReadU32(mapped_ + 12);
  const uint32_t alignment = ReadU32(mapped_ + 16);
  if (alignment == 0) {
    if (error_message != nullptr) {
      *error_message = "invalid .msl alignment: 0";
    }
    return false;
  }

  // ── KV region: unknown keys are skipped, unknown types rejected ───────
  kv_.clear();
  kv_.reserve(kv_count);
  size_t pos = msl_format::kHeaderSize;
  for (uint32_t i = 0; i < kv_count; ++i) {
    if (pos + 4 > mapped_size_) {
      if (error_message != nullptr) {
        *error_message = "KV region truncated at entry " + std::to_string(i);
      }
      return false;
    }
    const uint32_t key_len = ReadU32(mapped_ + pos);
    pos += 4;
    if (pos + key_len + 8 > mapped_size_) {
      if (error_message != nullptr) {
        *error_message = "KV key truncated at entry " + std::to_string(i);
      }
      return false;
    }
    std::string key(reinterpret_cast<const char *>(mapped_ + pos), key_len);
    pos += key_len;
    const uint32_t type = ReadU32(mapped_ + pos);
    const uint32_t value_len = ReadU32(mapped_ + pos + 4);
    pos += 8;
    if (pos + value_len > mapped_size_) {
      if (error_message != nullptr) {
        *error_message = "KV value truncated for key \"" + key + "\"";
      }
      return false;
    }
    MslKvValue kv;
    kv.key = std::move(key);
    kv.type = type;
    if (!DecodeKv(static_cast<msl_format::KvType>(type), mapped_ + pos, value_len, &kv)) {
      if (error_message != nullptr) {
        *error_message = "unknown or malformed KV value type " + std::to_string(type) + " for key \"" + kv.key + "\"";
      }
      return false;
    }
    kv_.push_back(std::move(kv));
    pos += value_len;
  }

  // ── Resource table ─────────────────────────────────────────────────────
  entries_.clear();
  entries_.reserve(resource_count);
  for (uint32_t i = 0; i < resource_count; ++i) {
    if (pos + msl_format::kEntrySize > mapped_size_) {
      if (error_message != nullptr) {
        *error_message = "resource table overflow";
      }
      return false;
    }
    const uint8_t *raw = mapped_ + pos;

    MslEntry entry;
    size_t name_len = 0;
    while (name_len < msl_format::kNameSize && raw[name_len] != '\0') {
      ++name_len;
    }
    entry.name.assign(reinterpret_cast<const char *>(raw), name_len);
    entry.offset = ReadU64(raw + msl_format::kNameSize);
    entry.size = ReadU64(raw + msl_format::kNameSize + 8);
    entry.access = ReadU32(raw + msl_format::kNameSize + 16);

    if (entry.name.empty()) {
      if (error_message != nullptr) {
        *error_message = "empty resource name at index " + std::to_string(i);
      }
      return false;
    }
    if (entry.access != msl_format::kAccessMmap && entry.access != msl_format::kAccessRead) {
      if (error_message != nullptr) {
        *error_message = "invalid access mode " + std::to_string(entry.access) + " for " + entry.name;
      }
      return false;
    }
    if (entry.offset % alignment != 0) {
      if (error_message != nullptr) {
        *error_message = "resource payload not aligned (" + std::to_string(entry.offset) + " % " +
                         std::to_string(alignment) + "): " + entry.name;
      }
      return false;
    }
    if (entry.offset > mapped_size_ || entry.size > mapped_size_ - entry.offset) {
      if (error_message != nullptr) {
        *error_message = "resource range overflow: " + entry.name;
      }
      return false;
    }
    entries_.push_back(std::move(entry));
    pos += msl_format::kEntrySize;
  }
  return true;
}

const MslEntry *MslPackageReader::Lookup(const std::string &name) const {
  auto it = std::find_if(entries_.begin(), entries_.end(), [&name](const auto &entry) { return entry.name == name; });
  return it == entries_.end() ? nullptr : &*it;
}

bool MslPackageReader::Read(const std::string &name, std::vector<uint8_t> *out) const {
  if (out == nullptr || mapped_ == nullptr) {
    return false;
  }
  const MslEntry *entry = Lookup(name);
  if (entry == nullptr) {
    return false;
  }
  out->assign(mapped_ + entry->offset, mapped_ + entry->offset + entry->size);
  return true;
}

bool MslPackageReader::Mmap(const std::string &name, const uint8_t **data, size_t *size) const {
  if (data == nullptr || size == nullptr || mapped_ == nullptr) {
    return false;
  }
  const MslEntry *entry = Lookup(name);
  if (entry == nullptr) {
    return false;
  }
  *data = mapped_ + entry->offset;
  *size = entry->size;
  return true;
}

// ─── KV typed getters ─────────────────────────────────────────────────────

const MslKvValue *FindKv(const MslPackageReader *reader, const std::string &key) {
  const auto &kvs = reader->kv();
  auto it = std::find_if(kvs.begin(), kvs.end(), [&key](const auto &kv) { return kv.key == key; });
  return it == kvs.end() ? nullptr : &*it;
}

bool MslPackageReader::GetKvString(const std::string &key, std::string *out) const {
  if (out == nullptr) {
    return false;
  }
  const MslKvValue *kv = FindKv(this, key);
  if (kv == nullptr || kv->type != msl_format::kTypeString) {
    return false;
  }
  out->assign(reinterpret_cast<const char *>(kv->value.data()), kv->value.size());
  return true;
}

bool MslPackageReader::GetKvUint32(const std::string &key, uint32_t *out) const {
  if (out == nullptr) {
    return false;
  }
  const MslKvValue *kv = FindKv(this, key);
  if (kv == nullptr || kv->type != msl_format::kTypeUint32 || kv->value.size() != sizeof(uint32_t)) {
    return false;
  }
  *out = ReadU32(kv->value.data());
  return true;
}

bool MslPackageReader::GetKvUint64(const std::string &key, uint64_t *out) const {
  if (out == nullptr) {
    return false;
  }
  const MslKvValue *kv = FindKv(this, key);
  if (kv == nullptr || kv->type != msl_format::kTypeUint64 || kv->value.size() != sizeof(uint64_t)) {
    return false;
  }
  *out = ReadU64(kv->value.data());
  return true;
}

bool MslPackageReader::GetKvFloat32(const std::string &key, float *out) const {
  if (out == nullptr) {
    return false;
  }
  const MslKvValue *kv = FindKv(this, key);
  if (kv == nullptr || kv->type != msl_format::kTypeFloat32 || kv->value.size() != sizeof(float)) {
    return false;
  }
  std::memcpy(out, kv->value.data(), sizeof(float));
  return true;
}

bool MslPackageReader::GetKvBool(const std::string &key, bool *out) const {
  if (out == nullptr) {
    return false;
  }
  const MslKvValue *kv = FindKv(this, key);
  if (kv == nullptr || kv->type != msl_format::kTypeBool || kv->value.size() != 1) {
    return false;
  }
  *out = (kv->value[0] != 0);
  return true;
}

bool MslPackageReader::GetKvStringArray(const std::string &key, std::vector<std::string> *out) const {
  if (out == nullptr) {
    return false;
  }
  const MslKvValue *kv = FindKv(this, key);
  if (kv == nullptr || kv->type != msl_format::kTypeStringArray) {
    return false;
  }
  const uint8_t *raw = kv->value.data();
  const size_t len = kv->value.size();
  const uint32_t count = ReadU32(raw);
  size_t pos = 4;
  out->clear();
  out->reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    if (pos + 4 > len) {
      return false;
    }
    const uint32_t item_len = ReadU32(raw + pos);
    pos += 4;
    if (pos + item_len > len) {
      return false;
    }
    out->push_back(std::string(reinterpret_cast<const char *>(raw + pos), item_len));
    pos += item_len;
  }
  return true;
}

}  // namespace mslite_llm
