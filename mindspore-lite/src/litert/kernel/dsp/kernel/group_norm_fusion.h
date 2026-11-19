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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_KERNEL_GROUP_NORM_FUSION_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_KERNEL_GROUP_NORM_FUSION_H_

#include <memory>
#include "src/litert/kernel/dsp/dsp_kernel.h"
#include "src/litert/kernel/dsp/dsp_allocator.h"
#include "nnacl_c/group_norm_parameter.h"

namespace mindspore::kernel {

class GroupNormFusionDSPKernel : public DSPKernel {
 public:
  using DSPKernel::DSPKernel;

  ~GroupNormFusionDSPKernel() override;

  int Prepare() override;
  int CheckSpecs() override;
  int Run() override;

 private:
  // The SDK writes mean/variance into caller-provided buffers (batch*num_groups
  // elements). Lite exposes only one output tensor, so allocate them here.
  int EnsureStatBuffers(int64_t count, TypeId dtype);

  GroupNormParameter *param_{nullptr};
  std::shared_ptr<lite::dsp::DSPAllocator> allocator_;
  void *mean_buffer_{nullptr};
  void *variance_buffer_{nullptr};
  size_t stat_bytes_{0};  // bytes currently allocated; reallocate if shape grows
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_KERNEL_GROUP_NORM_FUSION_H_
