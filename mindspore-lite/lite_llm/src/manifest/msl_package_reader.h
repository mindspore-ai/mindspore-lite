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
#ifndef MSLLM_MSL_PACKAGE_READER_H
#define MSLLM_MSL_PACKAGE_READER_H

#include <cstdint>
#include <string>
#include <vector>

#include "manifest/msl_format.h"

namespace mslite_llm {

/// One entry in a single-file .msl (v1) container.
struct MslEntry {
  std::string name;     // relative path, e.g. "npu_offline/x.omc"
  uint64_t offset = 0;  // byte offset into the data region
  uint64_t size = 0;    // byte length
  uint32_t access = 1;  // 0 = mmap, 1 = read
};

/// One metadata key in the v1 KV region (raw value bytes + type tag).
struct MslKvValue {
  std::string key;
  uint32_t type = 0;
  std::vector<uint8_t> value;
};

/// Read a single-file .msl (v1) container:
/// ``.MSL header(24B) + KV region + MslResourceEntry[N](88B) + data``.
///
/// The whole file is mmap'd once; ``Read`` copies an entry out and
/// ``Mmap`` returns a pointer into the mapping (valid until the reader
/// is destroyed).  Unknown KV keys are skipped on Open (forward
/// compatible); unknown KV value types and unknown versions are
/// rejected.
class MslPackageReader {
 public:
  MslPackageReader() = default;
  ~MslPackageReader();
  MslPackageReader(const MslPackageReader &) = delete;
  MslPackageReader &operator=(const MslPackageReader &) = delete;

  bool Open(const std::string &path, std::string *error_message = nullptr);
  bool IsOpen() const { return mapped_ != nullptr; }

  const MslEntry *Lookup(const std::string &name) const;

  /// Copy an entry's bytes into ``out`` (access=1 semantics). Returns false on
  /// miss.
  bool Read(const std::string &name, std::vector<uint8_t> *out) const;

  /// Point at an entry's bytes inside the mapping (access=0 semantics). Returns
  /// false on miss.
  bool Mmap(const std::string &name, const uint8_t **data, size_t *size) const;

  /// Discard the caller's resident pages for an entry that has been fully
  /// consumed. The entry remains addressable and pages fault back in on a later
  /// access.
  bool Reclaim(const std::string &name) const;

  size_t entry_count() const { return entries_.size(); }

  /// Read-only access to the entry table (msl package content listing).
  const std::vector<MslEntry> &entries() const { return entries_; }

  // ── KV metadata ───────────────────────────────────────────────────────
  size_t kv_count() const { return kv_.size(); }
  const std::vector<MslKvValue> &kv() const { return kv_; }

  /// Typed KV getters; return false when the key is absent or the stored
  /// type does not match.  Unknown keys are simply absent (skipped).
  bool GetKvString(const std::string &key, std::string *out) const;
  bool GetKvUint32(const std::string &key, uint32_t *out) const;
  bool GetKvUint64(const std::string &key, uint64_t *out) const;
  bool GetKvFloat32(const std::string &key, float *out) const;
  bool GetKvBool(const std::string &key, bool *out) const;
  bool GetKvStringArray(const std::string &key, std::vector<std::string> *out) const;

 private:
  int fd_ = -1;
  uint8_t *mapped_ = nullptr;  // nullptr == not mapped
  size_t mapped_size_ = 0;
  std::vector<MslEntry> entries_;
  std::vector<MslKvValue> kv_;
};

}  // namespace mslite_llm

#endif  // MSLLM_MSL_PACKAGE_READER_H
