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
#ifndef LLM_TEST_FAKE_BACKEND_H
#define LLM_TEST_FAKE_BACKEND_H

#include <condition_variable>
#include <map>
#include <mutex>
#include <vector>

#include "backend/common/backend.h"

namespace mslite_llm_test {

/// Test double for the NNRT/NPU system boundary.
///
/// Scripted behavior:
///  - QueueLogits(logits, count): serve `logits` for the next `count` backend
///    calls (prefill + each decode; default: repeat the last scripted vector
///    forever).
///  - QueueError(status, at_call): make the `at_call`-th (1-based) backend
///    call return `status`.
///  - ScriptInit(status): make Init() return `status`.
///  - BlockExecute()/UnblockExecute(): hold backend calls until unblocked, so a
///    test can keep a generation in-flight deterministically.
class FakeBackend : public mslite_llm::Backend {
 public:
  void QueueLogits(std::vector<float> logits) { script_.push_back(std::move(logits)); }

  void QueueError(MSLlmStatus status, int at_call) { error_script_[at_call] = status; }

  void ScriptInit(MSLlmStatus status) { init_status_ = status; }

  void BlockExecute() { block_ = true; }

  void UnblockExecute() {
    {
      std::lock_guard<std::mutex> lk(block_mu_);
      block_ = false;
    }
    block_cv_.notify_all();
  }

  int execute_calls() const { return execute_calls_; }

  MSLlmStatus Init(const mslite_llm::BackendConfig &) override { return init_status_; }

  MSLlmStatus Prefill(const mslite_llm::BackendInput &input, mslite_llm::BackendOutput *output) override {
    return RunForward(input, output);
  }

  MSLlmStatus Decode(const mslite_llm::BackendInput &input, mslite_llm::BackendOutput *output) override {
    return RunForward(input, output);
  }

  MSLlmStatus Reset() override { return MSLLM_SUCCESS; }

 private:
  MSLlmStatus RunForward(const mslite_llm::BackendInput &input, mslite_llm::BackendOutput *output) {
    {
      std::unique_lock<std::mutex> lk(block_mu_);
      block_cv_.wait(lk, [this] { return !block_; });
    }
    ++execute_calls_;
    auto it = error_script_.find(execute_calls_);
    if (it != error_script_.end()) {
      return it->second;
    }
    if (script_.empty()) {
      output->logits.clear();
      return MSLLM_SUCCESS;
    }
    size_t idx = static_cast<size_t>(execute_calls_ - 1);
    if (idx >= script_.size()) {
      idx = script_.size() - 1;  // repeat last
    }
    output->logits = script_[idx];
    return MSLLM_SUCCESS;
  }

  std::vector<std::vector<float>> script_;
  std::map<int, MSLlmStatus> error_script_;
  MSLlmStatus init_status_ = MSLLM_SUCCESS;
  int execute_calls_ = 0;
  std::mutex block_mu_;
  std::condition_variable block_cv_;
  bool block_ = false;
};

}  // namespace mslite_llm_test

#endif  // LLM_TEST_FAKE_BACKEND_H
