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

#ifndef MINDSPORE_LITE_SRC_LITERT_DELEGATE_PNNA_PNNA_SUBGRAPH_H_
#define MINDSPORE_LITE_SRC_LITERT_DELEGATE_PNNA_PNNA_SUBGRAPH_H_

#include <atomic>
#include <vector>
#include <utility>
#include <memory>
#include <map>
#include "include/cxx_api/kernel.h"
#include "src/common/log_adapter.h"
#include "src/litert/delegate/pnna/op/pnna_op.h"
#include "src/litert/delegate/pnna/pnna_utils.h"

namespace mindspore::lite {
class PNNAOp;
class PNNASubGraph : public kernel::Kernel {
 public:
  PNNASubGraph(const std::shared_ptr<pnna::Context> ctx, std::vector<PNNAOp *> ops,
               const std::vector<mindspore::MSTensor> &inputs, const std::vector<mindspore::MSTensor> &outputs)
      : kernel::Kernel(inputs, outputs, nullptr, nullptr), ctx_(ctx), ops_(std::move(ops)) {
    static std::atomic_int index = {0};
    origin_inputs_ = std::move(inputs);
    origin_outputs_ = std::move(outputs);
    this->set_name("PNNASubGraph" + std::to_string(index++));
  }
  ~PNNASubGraph() override;

  int Init();
  int CreatePNNAModel();
  int CompilePNNAModel();
  int Prepare() override;
  int ReSize() override {
    MS_LOG(ERROR) << "PNNA does not support the resize function temporarily.";
    return RET_ERROR;
  }
  int Execute() override;
  std::vector<PNNAOp *> *GetOps() { return &ops_; }
  std::vector<mindspore::MSTensor *> *GetInsertTensors() { return &insert_tensors_; }
  std::shared_ptr<pnna::Graph> graph() { return graph_; }
  std::shared_ptr<pnna::Tensor> GetMappedTensor(MSTensor *operand);
  void UpdateTensorMap(MSTensor *operand, std::shared_ptr<pnna::Tensor> tensor);
  std::shared_ptr<pnna::Tensor> AddTensor(MSTensor *operand);
  std::shared_ptr<pnna::Tensor> ConvertOperand(MSTensor *operand);

 private:
  int PreProcess();
  int PostProcess();

  std::shared_ptr<pnna::Context> ctx_{nullptr};
  std::vector<PNNAOp *> ops_;
  std::vector<mindspore::MSTensor> origin_inputs_;
  std::vector<mindspore::MSTensor> origin_outputs_;
  std::shared_ptr<pnna::Graph> graph_{nullptr};
  std::map<MSTensor *, std::vector<std::shared_ptr<pnna::Tensor>>> tensors_;
  std::vector<mindspore::MSTensor *> insert_tensors_;
};
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_SRC_LITERT_DELEGATE_PNNA_PNNA_SUBGRAPH_H_
