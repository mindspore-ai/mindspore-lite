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
// Exercise external weight export with synthetic packages, without an NPU runtime.

#include <gtest/gtest.h>
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include "backend/nnrt/nnrt_executor.h"
#include "manifest/msl_format.h"
#include "manifest/msl_package_reader.h"

namespace mslite {
namespace backend {
namespace nnrt {

class ExternalWeightExportTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::string pattern = (std::filesystem::temp_directory_path() / "msl_export_XXXXXX").string();
    std::vector<char> name(pattern.begin(), pattern.end());
    name.push_back('\0');
    const char *created = ::mkdtemp(name.data());
    ASSERT_NE(created, nullptr);
    root_ = created;
    executor_ = std::make_unique<NnrtExecutor>();
  }

  void TearDown() override {
    executor_.reset();
    if (!root_.empty()) {
      std::error_code error;
      std::filesystem::remove_all(root_, error);
      EXPECT_FALSE(error);
    }
  }

  void Prepare(size_t size) {
    const auto page_size = ::sysconf(_SC_PAGESIZE);
    ASSERT_GT(page_size, 0);
    payload_.resize(size);
    for (size_t i = 0; i < size; ++i) {
      payload_[i] = static_cast<char>(i % 251);
    }
    using mslite_llm::msl_format::MslHeader;
    using mslite_llm::msl_format::MslResourceEntry;
    MslHeader header{{'.', 'M', 'S', 'L'}, 1, 0, 1, static_cast<uint32_t>(page_size), 0};
    MslResourceEntry entry{};
    const std::string entry_name = "SubGraph_0.weight";
    std::copy(entry_name.begin(), entry_name.end(), entry.name);
    entry.offset = static_cast<uint64_t>(page_size);
    entry.size = size;
    std::ofstream file(root_ + "/model.msl", std::ios::binary);
    file.write(reinterpret_cast<const char *>(&header), sizeof(header));
    file.write(reinterpret_cast<const char *>(&entry), sizeof(entry));
    file.seekp(page_size - 1);
    file.put(0);
    file.write(payload_.data(), static_cast<std::streamsize>(size));
    file.close();
    ASSERT_TRUE(file.good());
    executor_->package_reader_ = std::make_shared<mslite_llm::MslPackageReader>();
    ASSERT_TRUE(executor_->package_reader_->Open(root_ + "/model.msl"));
    executor_->package_root_ = root_ + "/model.msl";
    executor_->om_weight_dir_ = "weights";
    ASSERT_TRUE(executor_->package_reader_->Mmap(entry_name, &data_, &size_));
  }

  bool Write(const std::string &path) { return executor_->WriteExternalWeightFile(path, data_, size_); }
  bool Extract() { return executor_->ExtractExternalWeightFile(); }
  bool Reclaim() { return executor_->package_reader_->Reclaim(executor_->external_weight_entry_); }
  void MissingEntry() { executor_->external_weight_entry_ = "missing"; }
  void MissingReader() { executor_->package_reader_.reset(); }
  std::string ExportDirectory() const { return executor_->temp_weight_dir_; }

  void CheckBytes(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    ASSERT_TRUE(file.is_open());
    const std::vector<char> actual((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    EXPECT_EQ(actual, payload_);
  }

  // Run in an ASSERT_EXIT child: never alter the test runner's resource limits.
  void RejectWriteWithFileLimit() {
    struct rlimit limit {};
    if (::getrlimit(RLIMIT_FSIZE, &limit) != 0) {
      ::_exit(2);
    }
    const struct rlimit original = limit;
    limit.rlim_cur = 0;
    if (std::signal(SIGXFSZ, SIG_IGN) == SIG_ERR || ::setrlimit(RLIMIT_FSIZE, &limit) != 0) {
      ::_exit(3);
    }
    // gtest captures stderr in a file, also affected by RLIMIT_FSIZE.
    std::ostringstream errors;
    auto *original_buffer = std::cerr.rdbuf(errors.rdbuf());
    const bool written = Write(root_ + "/limited.bin");
    if (::setrlimit(RLIMIT_FSIZE, &original) != 0) {
      ::_exit(4);
    }
    std::cerr.rdbuf(original_buffer);
    std::cerr << errors.str();
    ::_exit(written ? 1 : 0);
  }

  std::string root_;
  std::unique_ptr<NnrtExecutor> executor_;
  std::vector<char> payload_;
  const uint8_t *data_{nullptr};
  size_t size_{0};
};

TEST_F(ExternalWeightExportTest, PreservesBytesAcrossChunkBoundaries) {
  for (size_t bytes : {1UL, 4194303UL, 4194304UL, 4194305UL, 8388625UL}) {
    SCOPED_TRACE(bytes);
    ASSERT_NO_FATAL_FAILURE(Prepare(bytes));
    ASSERT_TRUE(Reclaim());
    ASSERT_TRUE(Write(root_ + "/output.bin"));
    CheckBytes(root_ + "/output.bin");
    // Reclaim leaves the source address usable, including the final partial page.
    EXPECT_TRUE(std::equal(payload_.begin(), payload_.end(), reinterpret_cast<const char *>(data_)));
  }
}

TEST_F(ExternalWeightExportTest, ReclaimFailureStillExportsAllChunks) {
  ASSERT_NO_FATAL_FAILURE(Prepare(8388625));
  MissingEntry();
  ASSERT_FALSE(Reclaim());
  ASSERT_TRUE(Write(root_ + "/output.bin"));
  CheckBytes(root_ + "/output.bin");
}

TEST_F(ExternalWeightExportTest, OpenFailureIsReported) {
  ASSERT_NO_FATAL_FAILURE(Prepare(1));
  EXPECT_FALSE(Write(root_ + "/missing/output.bin"));
}

TEST_F(ExternalWeightExportTest, BufferedFlushFailureIsReported) {
  ASSERT_NO_FATAL_FAILURE(Prepare(1));
  ASSERT_EXIT(RejectWriteWithFileLimit(), ::testing::ExitedWithCode(0), "Failed to close extracted external weight");
}

TEST_F(ExternalWeightExportTest, LargeWriteFailureIsReported) {
  ASSERT_NO_FATAL_FAILURE(Prepare(4194305));
  ASSERT_EXIT(RejectWriteWithFileLimit(), ::testing::ExitedWithCode(0), "Failed to extract external weight");
}

TEST_F(ExternalWeightExportTest, ExtractsAndRemovesTemporaryWeightFile) {
  ASSERT_NO_FATAL_FAILURE(Prepare(4194305));
  ASSERT_TRUE(Extract());
  const std::string directory = ExportDirectory();
  CheckBytes(directory + "/weights/SubGraph_0.weight");
  executor_.reset();
  EXPECT_FALSE(std::filesystem::exists(directory));
}

TEST_F(ExternalWeightExportTest, MissingEntryAndReaderAreRejected) {
  ASSERT_NO_FATAL_FAILURE(Prepare(1));
  MissingEntry();
  EXPECT_FALSE(Extract());
  MissingReader();
  EXPECT_FALSE(Extract());
}

TEST_F(ExternalWeightExportTest, EmptyWeightIsRejected) {
  ASSERT_NO_FATAL_FAILURE(Prepare(0));
  EXPECT_FALSE(Extract());
}

}  // namespace nnrt
}  // namespace backend
}  // namespace mslite
