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
// Exercise the real Build -> LoadExternalWeights path with a fake NNRT API.

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
#include "backend/nnrt/nnrt_wrapper.h"
#include "manifest/msl_format.h"
#include "manifest/msl_package_reader.h"

namespace mslite {
namespace backend {
namespace nnrt {
namespace {

struct WeightRuntime {
  int compilation = 0;
  int executor = 0;
  int init_calls = 0;
  int init_result = 0;
  bool contract_queried = false;
  std::string directory;
  std::vector<char> bytes;
};
WeightRuntime runtime;

NNRTFunctions WeightApi() {
  NNRTFunctions api{};
  api.Compilation_ConstructWithOfflineModelBuffer = [](const void *, size_t) {
    return reinterpret_cast<OH_NNCompilation *>(&runtime.compilation);
  };
  api.Compilation_SetDevice = [](OH_NNCompilation *, size_t) { return 0; };
  api.Compilation_SetPerformanceMode = [](OH_NNCompilation *, NnrtPerformanceMode) { return 0; };
  api.HIAIOptions_SetAsyncModeEnable = [](OH_NNCompilation *, bool) { return 0; };
  api.Compilation_Build = [](OH_NNCompilation *) { return 0; };
  api.Compilation_Destroy = [](OH_NNCompilation **handle) { *handle = nullptr; };
  api.Executor_Construct = [](OH_NNCompilation *) { return reinterpret_cast<OH_NNExecutor *>(&runtime.executor); };
  api.Executor_Destroy = [](OH_NNExecutor **handle) { *handle = nullptr; };
  api.HIAIExecutor_InitWeights = [](OH_NNExecutor *, const char *directory) {
    ++runtime.init_calls;
    runtime.directory = directory;
    std::ifstream file(runtime.directory + "/SubGraph_0.weight", std::ios::binary);
    EXPECT_TRUE(file.is_open());
    runtime.bytes.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    return runtime.init_result;
  };
  // Stop after successful weight initialization, before unrelated tensor setup.
  // Build returns false deliberately; tests also assert this boundary was reached.
  api.Executor_GetInputCount = [](const OH_NNExecutor *, size_t *) {
    runtime.contract_queried = true;
    return -1;
  };
  api.Executor_GetOutputCount = [](const OH_NNExecutor *, size_t *) { return -1; };
  return api;
}

class ExternalWeightExportTest : public ::testing::Test {
 protected:
  void SetUp() override {
    saved_api_ = NNRTWrapper::GetApi();
    runtime = {};
    NNRTWrapper::SetApiForTesting(WeightApi());
    std::string pattern = (std::filesystem::temp_directory_path() / "msl_export_XXXXXX").string();
    std::vector<char> name(pattern.begin(), pattern.end());
    name.push_back('\0');
    const char *created = ::mkdtemp(name.data());
    ASSERT_NE(created, nullptr);
    root_ = created;
    config_.vocab_size = 128;
    config_.hidden_size = 128;
    config_.num_layers = 1;
    config_.num_key_value_heads = 1;
    config_.head_dim = 128;
    config_.max_length = 64;
    config_.chunk_size = 64;
    config_.single_file = true;
    config_.has_external_weights = true;
    config_.prefill_path = "npu_offline/x.omc";
    config_.package_root = root_ + "/model.msl";
    config_.om_weight_dir = "weights";
  }

  void TearDown() override {
    executor_.reset();
    config_.package_reader.reset();
    NNRTWrapper::SetApiForTesting(saved_api_);
    if (!root_.empty()) {
      std::error_code error;
      std::filesystem::remove_all(root_, error);
      EXPECT_FALSE(error);
    }
  }

  void Prepare(size_t size, bool aligned = true) {
    const auto page_size = ::sysconf(_SC_PAGESIZE);
    ASSERT_GT(page_size, 0);
    payload_.resize(size);
    for (size_t i = 0; i < size; ++i) {
      payload_[i] = static_cast<char>(i % 251);
    }
    using mslite_llm::msl_format::MslHeader;
    using mslite_llm::msl_format::MslResourceEntry;
    const uint32_t alignment = aligned ? static_cast<uint32_t>(page_size) : 1;
    MslHeader header{{'.', 'M', 'S', 'L'}, 1, 0, 2, alignment, 0};
    MslResourceEntry omc{};
    std::copy(config_.prefill_path.begin(), config_.prefill_path.end(), omc.name);
    omc.offset = page_size;
    omc.size = 1;
    MslResourceEntry weight{};
    std::copy(config_.external_weight_entry.begin(), config_.external_weight_entry.end(), weight.name);
    weight.offset = 2 * page_size + (aligned ? 0 : 1);
    weight.size = size;
    std::ofstream file(config_.package_root, std::ios::binary);
    file.write(reinterpret_cast<const char *>(&header), sizeof(header));
    file.write(reinterpret_cast<const char *>(&omc), sizeof(omc));
    file.write(reinterpret_cast<const char *>(&weight), sizeof(weight));
    file.seekp(omc.offset);
    file.put(0);
    file.seekp(weight.offset - 1);
    file.put(0);
    file.write(payload_.data(), static_cast<std::streamsize>(size));
    file.close();
    ASSERT_TRUE(file.good());
    config_.package_reader = std::make_shared<mslite_llm::MslPackageReader>();
    ASSERT_TRUE(config_.package_reader->Open(config_.package_root));
    ASSERT_TRUE(config_.package_reader->Mmap(config_.external_weight_entry, &data_, &size_));
  }

  void BuildAndCheckExport() {
    executor_ = std::make_unique<NnrtExecutor>();
    EXPECT_FALSE(executor_->Build(config_));
    ASSERT_EQ(runtime.init_calls, 1);
    EXPECT_TRUE(runtime.contract_queried);
    EXPECT_EQ(runtime.bytes, payload_);
    EXPECT_TRUE(std::equal(payload_.begin(), payload_.end(), reinterpret_cast<const char *>(data_)));
  }

  void BuildAndCheckRejected() {
    executor_ = std::make_unique<NnrtExecutor>();
    EXPECT_FALSE(executor_->Build(config_));
    EXPECT_EQ(runtime.init_calls, 0);
    EXPECT_FALSE(runtime.contract_queried);
  }

  // Run in a child: never alter the test runner's resource limits or signals.
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
    std::ostringstream errors;
    auto *original_buffer = std::cerr.rdbuf(errors.rdbuf());
    bool built = false;
    {
      NnrtExecutor executor;
      built = executor.Build(config_);
    }
    if (::setrlimit(RLIMIT_FSIZE, &original) != 0) {
      ::_exit(4);
    }
    std::cerr.rdbuf(original_buffer);
    std::cerr << errors.str();
    ::_exit(!built && runtime.init_calls == 0 && !runtime.contract_queried ? 0 : 1);
  }

  std::string root_;
  NnrtConfig config_;
  NNRTFunctions saved_api_{};
  std::unique_ptr<NnrtExecutor> executor_;
  std::vector<char> payload_;
  const uint8_t *data_{nullptr};
  size_t size_{0};
};

TEST_F(ExternalWeightExportTest, PreservesBytesAcrossChunkBoundaries) {
  for (size_t bytes : {1UL, 4194303UL, 4194304UL, 4194305UL, 8388625UL}) {
    SCOPED_TRACE(bytes);
    executor_.reset();
    runtime = {};
    ASSERT_NO_FATAL_FAILURE(Prepare(bytes));
    ASSERT_NO_FATAL_FAILURE(BuildAndCheckExport());
  }
}

TEST_F(ExternalWeightExportTest, ReclaimFailureStillExportsAllChunks) {
  ASSERT_NO_FATAL_FAILURE(Prepare(8388625, false));
  ASSERT_FALSE(config_.package_reader->Reclaim(config_.external_weight_entry));
  ASSERT_NO_FATAL_FAILURE(BuildAndCheckExport());
}

TEST_F(ExternalWeightExportTest, OpenFailureIsReported) {
  config_.external_weight_entry = "missing/SubGraph_0.weight";
  ASSERT_NO_FATAL_FAILURE(Prepare(1));
  BuildAndCheckRejected();
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
  ASSERT_NO_FATAL_FAILURE(BuildAndCheckExport());
  const auto directory = std::filesystem::path(runtime.directory).parent_path();
  ASSERT_TRUE(std::filesystem::exists(directory / "weights/SubGraph_0.weight"));
  executor_.reset();
  EXPECT_FALSE(std::filesystem::exists(directory));
}

TEST_F(ExternalWeightExportTest, MissingEntryAndReaderAreRejected) {
  ASSERT_NO_FATAL_FAILURE(Prepare(1));
  config_.external_weight_entry = "missing";
  BuildAndCheckRejected();
  config_.package_reader.reset();
  BuildAndCheckRejected();
}

TEST_F(ExternalWeightExportTest, EmptyWeightIsRejected) {
  ASSERT_NO_FATAL_FAILURE(Prepare(0));
  BuildAndCheckRejected();
}

TEST_F(ExternalWeightExportTest, InitWeightsFailureStopsBuildAndCleansExport) {
  ASSERT_NO_FATAL_FAILURE(Prepare(4194305));
  runtime.init_result = -1;
  executor_ = std::make_unique<NnrtExecutor>();
  EXPECT_FALSE(executor_->Build(config_));
  EXPECT_EQ(runtime.init_calls, 1);
  EXPECT_EQ(runtime.bytes, payload_);
  EXPECT_FALSE(runtime.contract_queried);
  const auto directory = std::filesystem::path(runtime.directory).parent_path();
  ASSERT_FALSE(directory.empty());
  executor_.reset();
  EXPECT_FALSE(std::filesystem::exists(directory));
}

}  // namespace
}  // namespace nnrt
}  // namespace backend
}  // namespace mslite
