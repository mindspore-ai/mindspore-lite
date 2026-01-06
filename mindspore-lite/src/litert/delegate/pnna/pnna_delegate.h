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

#ifndef MINDSPORE_LITE_SRC_LITERT_DELEGATE_PNNA_PNNA_DELEGATE_H_
#define MINDSPORE_LITE_SRC_LITERT_DELEGATE_PNNA_PNNA_DELEGATE_H_

#include <string>
#include <memory>
#include <vector>
#include <queue>
#include <unordered_map>
#include <utility>
#include "include/api/delegate.h"
#include "src/litert/delegate/pnna/pnna_subgraph.h"
#include "src/litert/delegate/pnna/pass/pnna_pass_manager.h"
#include "src/litert/delegate/pnna/op/pnna_op.h"
#include "src/litert/delegate/pnna/op/transpose_pnna.h"
#include "src/litert/delegate/pnna/op/abs_pnna.h"
#include "src/litert/delegate/pnna/op/addn_pnna.h"
#include "src/litert/delegate/pnna/op/arg_pnna.h"
#include "src/litert/delegate/pnna/op/pooling_pnna.h"

namespace mindspore {
namespace lite {
class PNNADelegate : public Delegate {
 public:
  PNNADelegate() = default;
  ~PNNADelegate() override;

  Status Init() override;
  Status Build(DelegateModel<schema::Primitive> *model) override;
  void ReplaceNodes(const std::shared_ptr<LiteDelegateGraph> &graph) override;

 private:
  template <typename T>
  STATUS FindReadyKernels(std::vector<T *> *kernels, std::vector<T *> *ready_kernels) {
    MS_ASSERT(kernels != nullptr && ready_kernels != nullptr);
    std::queue<T *> tmp_kernels;
    std::vector<mindspore::MSTensor> visited_tensor;
    std::function<bool(T *)> is_kernel_ready = [&visited_tensor](T *kernel) {
      return std::all_of(kernel->inputs().begin(), kernel->inputs().end(), [&visited_tensor](MSTensor input) {
        return input.IsConst() ||
               std::find(visited_tensor.begin(), visited_tensor.end(), input) != visited_tensor.end();
      });
    };

    // initialize the visited_tensor map.
    for (const auto &input : inputs_) {
      visited_tensor.push_back(input);
    }
    for (auto ready_kernel : sorted_kernels_) {
      for (const auto &output : ready_kernel->outputs()) {
        visited_tensor.push_back(output);
      }
    }
    for (auto kernel : *kernels) {
      MS_CHECK_TRUE_RET(kernel != nullptr, RET_ERROR);
      if (is_kernel_ready(kernel)) {
        tmp_kernels.push(kernel);
        ready_kernels->push_back(kernel);
        kernels->erase(std::find(kernels->begin(), kernels->end(), kernel));
        break;
      }
    }
    while (!tmp_kernels.empty()) {
      auto tmp_kernel = tmp_kernels.front();
      tmp_kernels.pop();
      for (const auto &output : tmp_kernel->outputs()) {
        visited_tensor.push_back(output);
      }
      for (auto itr = kernels->begin(); itr != kernels->end();) {
        if (is_kernel_ready(*itr)) {
          tmp_kernels.push(*itr);
          ready_kernels->push_back(*itr);
          kernels->erase(itr);
        } else {
          itr++;
        }
      }
    }
    return RET_OK;
  }
  PNNASubGraph *CreatePNNASubGraph(DelegateModel<schema::Primitive> *model, std::vector<PNNAOp *> *candidate_ops);
  Status AddPasses();

 private:
  std::shared_ptr<pnna::Context> ctx_{nullptr};
  std::vector<mindspore::MSTensor> inputs_;
  std::vector<kernel::Kernel *> sorted_kernels_;
  std::vector<kernel::Kernel *> remained_kernels_;
  std::vector<PNNASubGraph *> pnna_kernels_;
  std::unordered_map<schema::PrimitiveType, PNNAGetOp> op_func_lists_;
  PNNAPassManager *pass_manager_ = nullptr;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_LITERT_DELEGATE_PNNA_PNNA_DELEGATE_H_
