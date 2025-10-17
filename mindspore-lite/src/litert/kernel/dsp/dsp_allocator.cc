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

#include "src/litert/kernel/dsp/dsp_allocator.h"
#include <utility>
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "src/litert/kernel/dsp/dsp_runtime.h"
#include "hthread/include/hthread_host.h"

namespace mindspore::lite::dsp {
DSPAllocator::DSPAllocator(DSPRuntime *dsp_runtime) : dsp_runtime_(dsp_runtime) {
  device_id_ = dsp_runtime->GetDeviceID();
}

DSPAllocator::~DSPAllocator() { Clear(); }

void DSPAllocator::Lock() {
  if (lock_flag_) {
    lock.lock();
  }
}

void DSPAllocator::UnLock() {
  if (lock_flag_) {
    lock.unlock();
  }
}

void *DSPAllocator::MinimumFit(MemType mem_type, size_t size) {
  auto iter = free_list_.lower_bound(size);
  while (iter != free_list_.end() && (iter->second->size_ >= size) && (iter->second->size_ < (size << shift_factor_))) {
    auto mem_buf = iter->second;
    bool is_match = mem_buf->mem_type_ == mem_type;
    if (is_match) {
      free_list_.erase(iter);
      allocated_list_[mem_buf->host_ptr_] = mem_buf;
      mem_buf->ref_count_ = 0;
      MS_LOG(DEBUG) << "Find Mem from free list. size: " << mem_buf->size_
                    << ", type: " << static_cast<int>(mem_buf->mem_type_);
      return mem_buf->host_ptr_;
    }
    ++iter;
  }
  return nullptr;
}

void *DSPAllocator::_Malloc(MemType mem_type, size_t size) {
  if (size > dsp_runtime_->GetMaxAllocSize()) {
    MS_LOG(ERROR) << "MallocData out of max_size, size: " << size;
    return nullptr;
  }
  Lock();
  void *host_ptr = MinimumFit(mem_type, size);
  UNLOCK_AND_RETURN_NULL(host_ptr != nullptr, host_ptr);

  total_size_ += size;

  MemBuf *mem_buf = new (std::nothrow) MemBuf;
  if (mem_buf == nullptr) {
    UnLock();
    return nullptr;
  }
  mem_buf->device_ptr_ = 0;
  auto ret = HostTlsfMalloc(device_id_, core_id_, static_cast<int>(mem_type), size, &mem_buf->device_ptr_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "HostTlsfMalloc failed, size: " << size << ", type: " << static_cast<int>(mem_type);
    delete mem_buf;
    UnLock();
    return nullptr;
  }
  host_ptr = reinterpret_cast<void *>(GetViraddr(mem_buf->device_ptr_, size));
  mem_buf->ref_count_ = 0;
  mem_buf->size_ = size;
  mem_buf->host_ptr_ = host_ptr;
  mem_buf->mem_type_ = mem_type;
  allocated_list_[host_ptr] = mem_buf;
  UnLock();

  MS_LOG(DEBUG) << "Malloc a new buffer. memory type: " << static_cast<int>(mem_buf->mem_type_)
                << ", size: " << std::dec << mem_buf->size_;
  return host_ptr;
}

void DSPAllocator::Free(void *buf) {
  if (buf == nullptr) {
    return;
  }
  Lock();
  auto iter = allocated_list_.find(buf);
  if (iter != allocated_list_.end()) {
    auto mem_buf = iter->second;
    mem_buf->ref_count_ = 0;
    allocated_list_.erase(iter);
    free_list_.insert(std::make_pair(mem_buf->size_, mem_buf));
    UnLock();
    return;
  }
  UnLock();
  MS_LOG(WARNING) << "Host ptr has freed";
}

int DSPAllocator::RefCount(void *buf) {
  if (buf == nullptr) {
    return -1;
  }
  Lock();
  auto iter = allocated_list_.find(buf);
  if (iter != allocated_list_.end()) {
    auto mem_buf = iter->second;
    int ref_count = std::atomic_load(&mem_buf->ref_count_);
    UnLock();
    return ref_count;
  }
  UnLock();
  return -1;
}

int DSPAllocator::SetRefCount(void *buf, int ref_count) {
  if (buf == nullptr) {
    return -1;
  }
  Lock();
  auto iter = allocated_list_.find(buf);
  if (iter != allocated_list_.end()) {
    auto mem_buf = iter->second;
    std::atomic_store(&mem_buf->ref_count_, ref_count);
    UnLock();
    return ref_count;
  }
  UnLock();
  return -1;
}

int DSPAllocator::IncRefCount(void *buf, int ref_count) {
  if (buf == nullptr) {
    return -1;
  }
  Lock();
  auto iter = allocated_list_.find(buf);
  if (iter != allocated_list_.end()) {
    auto membuf = iter->second;
    auto ref = std::atomic_fetch_add(&membuf->ref_count_, ref_count);
    UnLock();
    return (ref + ref_count);
  }
  UnLock();
  return -1;
}

int DSPAllocator::DecRefCount(void *buf, int ref_count) {
  if (buf == nullptr) {
    return -1;
  }
  Lock();
  auto iter = allocated_list_.find(buf);
  if (iter != allocated_list_.end()) {
    auto mem_buf = iter->second;
    auto ref = std::atomic_fetch_sub(&mem_buf->ref_count_, ref_count);
    UnLock();
    return (ref - ref_count);
  }
  UnLock();
  return -1;
}

size_t DSPAllocator::TotalSize() {
  Lock();
  size_t total_size = 0;
  for (auto it = allocated_list_.begin(); it != allocated_list_.end(); it++) {
    total_size += it->second->size_;
  }
  for (auto it = free_list_.begin(); it != free_list_.end(); it++) {
    total_size += it->second->size_;
  }
  UnLock();
  return total_size;
}

uint64_t DSPAllocator::GetDeviceMemPtr(void *buffer) {
  auto it = allocated_list_.find(buffer);
  if (it != allocated_list_.end()) {
    return it->second->device_ptr_;
  }
  MS_LOG(ERROR) << "Can not found device ptr!";
  return 0;
}

template <typename T>
void DSPAllocator::ClearMemList(T *list) {
  for (auto it = list->begin(); it != list->end(); it++) {
    if (it->second->host_ptr_ != nullptr) {
      MS_LOG(DEBUG) << "ReleaseViraddr host ptr.";
      ReleaseViraddr(reinterpret_cast<uint32_t>(it->second->host_ptr_), it->second->device_ptr_, it->second->size_);
      it->second->host_ptr_ = nullptr;
    }
    if (it->second->device_ptr_ != 0) {
      MS_LOG(DEBUG) << "HostTlsfFree device ptr.";
      HostTlsfFree(device_id_, core_id_, static_cast<int>(it->second->mem_type_), &it->second->device_ptr_);
    }
    delete it->second;
  }
  list->clear();
}

void DSPAllocator::Clear() {
  Lock();
  ClearMemList<std::unordered_map<void *, MemBuf *>>(&allocated_list_);
  ClearMemList<std::multimap<size_t, MemBuf *>>(&free_list_);
  UnLock();
}

MemType DSPAllocator::GetMemType(void *host_ptr) {
  MemType mem_type{MemType::DDR};
  Lock();
  auto it = allocated_list_.find(host_ptr);
  if (it == allocated_list_.end()) {
    UnLock();
    MS_LOG(ERROR) << "Can not found buffer!";
    return mem_type;
  }
  MemBuf *mem_buf = it->second;
  if (mem_buf == nullptr) {
    UnLock();
    MS_LOG(ERROR) << "MemBuf is nullptr for host_ptr!";
    return mem_type;
  }
  mem_type = mem_buf->mem_type_;
  UnLock();
  return mem_type;
}
}  // namespace mindspore::lite::dsp
