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

#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include "include/errorcode.h"
#include "src/common/file_utils.h"
#include "src/common/log_adapter.h"
#include "src/litert/kernel/dsp/dsp_runtime.h"
#include "src/litert/kernel/dsp/dsp_allocator.h"
#include "hthread/include/hthread_host.h"

namespace mindspore::lite::dsp {
static std::mutex g_mtx;
static std::mutex g_init_mtx;

InitState DSPRuntime::init_state_ = UnInit;
DSPRuntime *DSPRuntime::dsp_runtime_instance_ = nullptr;
size_t DSPRuntime::instance_count_ = 0;

DSPRuntime *DSPRuntime::GetInstance() {
  std::unique_lock<std::mutex> lck(g_mtx);
  static DSPRuntime dsp_runtime;
  if (instance_count_ == 0) {
    dsp_runtime_instance_ = &dsp_runtime;
  }
  instance_count_++;
  return dsp_runtime_instance_;
}

void DSPRuntime::DeleteInstance() {
  std::unique_lock<std::mutex> lck(g_mtx);
  if (instance_count_ == 0) {
    MS_LOG(ERROR) << "No DSPRuntime instance could delete!";
    return;
  }
  instance_count_--;
  if (instance_count_ == 0) {
    dsp_runtime_instance_->Uninit();
  }
}

// Init will get devices info, load dsp ops library.
int DSPRuntime::Init() {
  std::unique_lock<std::mutex> lck(g_init_mtx);
  if (init_state_ == InitSuccess) {
    return RET_OK;
  } else if (init_state_ == InitFailed) {
    return RET_ERROR;
  }
  init_state_ = InitFailed;

  if (IsPrintDebug()) {
    MT_INFO_LOG = 1;
  }
  GetHthreadVersion();
  auto device_status = DeviceOpen(device_id_);
  if (device_status < 0) {
    MS_LOG(ERROR) << "Open DSP Device failed!";
    return RET_ERROR;
  }
  std::string library_path = "/usr/lib/dsp_lib.dat";
  std::ifstream ifs(library_path);
  if (!ifs.good()) {
    MS_LOG(ERROR) << "DSP Lib: " << library_path << " is not exist.";
    return RET_ERROR;
  }
  if (ImportLib(library_path.data()) != RET_OK) {
    MS_LOG(ERROR) << "Load DSP OPS Library failed!";
    return RET_ERROR;
  }

  allocator_ = std::make_shared<DSPAllocator>(this);
  if (allocator_ == nullptr) {
    MS_LOG(ERROR) << "DSP allocator failed!";
    return RET_ERROR;
  }
  init_state_ = InitSuccess;
  MS_LOG(INFO) << "DSPRuntime init done!";
  return RET_OK;
}

int DSPRuntime::Uninit() {
  std::unique_lock<std::mutex> lck(g_init_mtx);
  if (init_state_ != InitSuccess) {
    return RET_OK;
  }
  auto ret = DeviceClose(device_id_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Close DSP Device failed!";
    return RET_ERROR;
  }
  allocator_ = nullptr;
  init_state_ = UnInit;
  MS_LOG(INFO) << "DSPRuntime uninit done!";
  return RET_OK;
}

DSPRuntime::~DSPRuntime() { Uninit(); }

int DSPRuntime::RunKernel(const std::string &kernel_name, const std::vector<uint64_t> &kernel_args,
                          const int core_mask) {
  int ret = -1;
  int thread_id = -1;
  ret = LaunchGroup(device_id_, core_mask, &thread_id, const_cast<char *>(kernel_name.c_str()), kernel_args.size(),
                    const_cast<uint64_t *>(kernel_args.data()));
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LaunchGroup failed! kernel name: " << kernel_name;
    return ret;
  }
  ret = WaitGroup(thread_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "WaitGroup failed! kernel name: " << kernel_name;
    return ret;
  }
  ret = DestroyGroup(thread_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DestroyGroup failed! kernel name: " << kernel_name;
    return ret;
  }
  return ret;
}

uint64_t DSPRuntime::GetMaxAllocSize() { return GetSysMemorySize(); }

int DSPRuntime::CopyDeviceMemToHost(void *dst, const void *src, size_t size) const {
  auto ret = HostMemCopy(dst, reinterpret_cast<uint64_t>(src), size, 1);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "CopyDeviceMemToHost failed!";
  }
  return ret;
}

int DSPRuntime::CopyHostMemToDevice(void *dst, const void *src, size_t size) const {
  auto ret = HostMemCopy(const_cast<void *>(src), reinterpret_cast<uint64_t>(dst), size, 0);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "CopyHostMemToDevice failed!";
  }
  return ret;
}
}  // namespace mindspore::lite::dsp
