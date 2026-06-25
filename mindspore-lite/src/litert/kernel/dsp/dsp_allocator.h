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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_DSP_ALLOCATOR_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_DSP_ALLOCATOR_H_

#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include "src/litert/inner_allocator.h"

namespace mindspore::lite::dsp {
enum class MemType : char { SMC, DDR, DDR36BIT, L2 };
#define UNLOCK_AND_RETURN_NULL(condition, ptr) \
  do {                                         \
    if (condition) {                           \
      UnLock();                                \
      return (ptr);                            \
    }                                          \
  } while (0)

class DSPRuntime;
class DSPAllocator : public mindspore::Allocator {
 public:
  explicit DSPAllocator(DSPRuntime *dsp_runtime);
  ~DSPAllocator() override;

  using Allocator::Malloc;
  void *Malloc(size_t size, MemType type) { return _Malloc(type, size); }
  void *Malloc(size_t size) override { return _Malloc(MemType::DDR, size); }

  void Free(void *ptr) override;
  int RefCount(void *ptr) override;
  int SetRefCount(void *ptr, int ref_count) override;
  int DecRefCount(void *ptr, int ref_count) override;
  int IncRefCount(void *ptr, int ref_count) override;
  size_t TotalSize();

  void Clear();
  MemType GetMemType(void *host_ptr);
  bool HasDeviceMemPtr(void *buffer);
  uint64_t GetDeviceMemPtr(void *buffer);
  void *Prepare(void *ptr) override { return ptr; }

 private:
  void Lock();
  void UnLock();
  void *MinimumFit(MemType mem_type, size_t size);
  void *_Malloc(MemType mem_type, size_t size = 0);
  template <typename T>
  void ClearMemList(T *list);

 private:
  DSPRuntime *dsp_runtime_{nullptr};
  int32_t device_id_{0};
  int32_t core_id_{0};
  std::mutex lock;
  struct MemBuf {
    std::atomic_int ref_count_ = 0;
    size_t size_{0};
    uint64_t device_ptr_{0};
    void *host_ptr_{nullptr};
    MemType mem_type_{MemType::DDR};
  };

  // <membuf->buf, membuf>
  std::unordered_map<void *, MemBuf *> allocated_list_;
  std::multimap<size_t, MemBuf *> free_list_;
  uint64_t total_size_{0};
  // 6 is empirical value
  int shift_factor_ = 6;
  bool lock_flag_ = true;
};
}  // namespace mindspore::lite::dsp

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_DSP_ALLOCATOR_H_
