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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_DSP_SUBGRAPH_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_DSP_SUBGRAPH_H_

#include <memory>
#include <set>
#include <vector>
#include "src/litert/kernel/dsp/dsp_kernel.h"
#include "src/executor/sub_graph_kernel.h"

namespace mindspore::kernel {
class DspSubGraph : public SubGraphKernel {
 public:
  DspSubGraph(const std::vector<kernel::KernelExec *> &inKernels, const std::vector<kernel::KernelExec *> &outKernels,
              const std::vector<kernel::KernelExec *> &nodes, MSKernel *kernel)
      : SubGraphKernel(inKernels, outKernels, nodes, kernel) {
    dsp_runtime_ = dsp_runtime_wrapper_.GetInstance();
    allocator_ = dsp_runtime_->GetAllocator();
    subgraph_type_ = kDspSubGraph;
    if (nodes.front()->desc().data_type == kNumberTypeFloat16) {
      desc_.data_type = kNumberTypeFloat16;
    } else {
      desc_.data_type = kNumberTypeFloat32;
    }
    desc_.arch = kernel::KERNEL_ARCH::kDSP;
    static std::atomic_int index = 0;
    this->set_name("DspSubGraph" + std::to_string(index++));
  }
  ~DspSubGraph() override;

  int Prepare() override;
  int ReSize() override;
  int Execute() override { return Execute(nullptr, nullptr); }
  int Execute(const KernelCallBack &before, const KernelCallBack &after) override;

 private:
  void UnInit();
  void GetInOutNodes();
  int UploadConstInputs();
  int UploadConstTensor(lite::Tensor *tensor);

 private:
  std::shared_ptr<lite::dsp::DSPAllocator> allocator_{nullptr};
  lite::dsp::DSPRuntimeInnerWrapper dsp_runtime_wrapper_;
  lite::dsp::DSPRuntime *dsp_runtime_{nullptr};
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_DSP_SUBGRAPH_H_
