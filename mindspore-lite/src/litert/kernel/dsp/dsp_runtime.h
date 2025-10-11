/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_DSP_RUNTIME_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_DSP_RUNTIME_H_

#include <vector>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <type_traits>
#include "ir/dtype/type_id.h"
#include "src/common/log_adapter.h"
#include "src/litert/kernel/dsp/dsp_runtime_wrapper.h"
#include "src/litert/kernel/dsp/dsp_allocator.h"

namespace mindspore::lite::dsp {
enum InitState { UnInit = 0, InitSuccess = 1, InitFailed = 2 };

class DSPRuntimeInnerWrapper;
class DSPRuntimeWrapper;
class DSPRuntime {
 public:
  friend DSPRuntimeInnerWrapper;
  friend DSPRuntimeWrapper;
  ~DSPRuntime();
  DSPRuntime(const DSPRuntime &) = delete;
  DSPRuntime &operator=(const DSPRuntime &) = delete;

  int Init();
  int Uninit();

  std::shared_ptr<DSPAllocator> GetAllocator() { return allocator_; }
  uint64_t GetMaxAllocSize();
  int32_t GetDeviceID() { return device_id_; }

  int RunKernel(const std::string &kernel_name, const std::vector<uint64_t> &kernel_args, const int core_mask);

  int CopyDeviceMemToHost(void *dst, const void *src, size_t size) const;
  int CopyHostMemToDevice(void *dst, const void *src, size_t size) const;

 private:
  static DSPRuntime *GetInstance();
  static void DeleteInstance();
  DSPRuntime() = default;

 private:
  static InitState init_state_;
  static size_t instance_count_;
  static DSPRuntime *dsp_runtime_instance_;
  int32_t device_id_{0};
  std::shared_ptr<DSPAllocator> allocator_{nullptr};
};

class DSPRuntimeInnerWrapper {
 public:
  DSPRuntimeInnerWrapper() { dsp_runtime_ = DSPRuntime::GetInstance(); }
  ~DSPRuntimeInnerWrapper() { DSPRuntime::DeleteInstance(); }
  DSPRuntimeInnerWrapper(const DSPRuntimeInnerWrapper &) = delete;
  DSPRuntimeInnerWrapper &operator=(const DSPRuntimeInnerWrapper &) = delete;
  DSPRuntime *GetInstance() { return dsp_runtime_; }

 private:
  DSPRuntime *dsp_runtime_{nullptr};
};
}  // namespace mindspore::lite::dsp
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_DSP_RUNTIME_H_
