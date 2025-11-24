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

#include <string>
#include <vector>
#include <memory>
#include "include/errorcode.h"
#include "src/litert/delegate/delegate_utils.h"
#include "src/litert/delegate/pnna/pnna_delegate.h"

namespace mindspore {
namespace lite {
Status PNNADelegate::Init() {
  ctx_ = pnna::Context::Create();
  return mindspore::kSuccess;
}

Status PNNADelegate::Build(DelegateModel<schema::Primitive> *model) {
  MS_CHECK_TRUE_RET(model != nullptr, mindspore::kLiteNullptr);
  std::vector<PNNAOp *> candidate_ops;
  auto begin_iter = model->BeginKernelIterator();
  for (auto iter = begin_iter; iter != model->EndKernelIterator(); iter++) {
    auto kernel = *iter;
    MS_CHECK_TRUE_RET(kernel != nullptr, mindspore::kLiteNullptr);
    auto primitive = model->GetPrimitive(kernel);
    MS_ASSERT(primitive != nullptr);
    auto prim_type = primitive->value_type();
    if (op_func_lists_.find(prim_type) == op_func_lists_.end()) {
      MS_LOG(WARNING) << "Unsupported to get pnna op with type of " << prim_type;
      remained_kernels_.push_back(kernel);
      continue;
    }
    auto get_op_func = op_func_lists_.at(prim_type);
    MS_CHECK_TRUE_RET(get_op_func != nullptr, mindspore::kLiteNullptr);
    auto pnna_op = get_op_func(kernel->name(), primitive, kernel->inputs(), kernel->outputs(), kernel->quant_type());
    if (pnna_op == nullptr) {
      MS_LOG(WARNING) << "Get pnna op failed for " << prim_type;
      remained_kernels_.push_back(kernel);
      continue;
    }
    candidate_ops.push_back(pnna_op);
  }
  if (candidate_ops.empty()) {
    return mindspore::kSuccess;
  }
  inputs_ = model->inputs();
  std::vector<kernel::Kernel *> ready_kernels;
  auto ret = FindReadyKernels<kernel::Kernel>(&remained_kernels_, &ready_kernels);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "FindReadyKernels failed.";
    for (auto op : candidate_ops) {
      delete op;
      op = nullptr;
    }
    return mindspore::kLiteError;
  }
  sorted_kernels_.insert(sorted_kernels_.end(), ready_kernels.begin(), ready_kernels.end());
  ready_kernels.clear();
  // for every op, find pre and next ops
  FindPreNextOps<PNNAOp>(candidate_ops);
  while (!candidate_ops.empty()) {
    auto pnna_kernel = CreatePNNASubGraph(model, &candidate_ops);
    if (pnna_kernel != nullptr) {
      sorted_kernels_.push_back(reinterpret_cast<kernel::Kernel *>(pnna_kernel));
      pnna_kernels_.push_back(pnna_kernel);
    } else {
      MS_LOG(WARNING) << "Create pnna graph failed.";
      for (auto pnna_op : candidate_ops) {
        delete pnna_op;
        pnna_op = nullptr;
      }
      return mindspore::kLiteError;
    }
    ret = FindReadyKernels<kernel::Kernel>(&remained_kernels_, &ready_kernels);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "FindReadyKernels failed.";
      for (auto pnna_op : candidate_ops) {
        delete pnna_op;
      }
      return mindspore::kLiteError;
    }
    sorted_kernels_.insert(sorted_kernels_.end(), ready_kernels.begin(), ready_kernels.end());
    ready_kernels.clear();
  }
  if (!remained_kernels_.empty() || sorted_kernels_.empty()) {
    MS_LOG(ERROR) << "PNNA delegate build failed.";
    return mindspore::kLiteError;
  }
  for (auto pnna_kernel : pnna_kernels_) {
    ret = pnna_kernel->CompilePNNAModel();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Compile pnna model failed.";
      return mindspore::kLiteError;
    }
  }
  // Update the kernels of delegate model.
  ReplaceNodes(std::make_shared<LiteDelegateGraph>(*model));
  return mindspore::kSuccess;
}

void PNNADelegate::ReplaceNodes(const std::shared_ptr<LiteDelegateGraph> &graph) {
  MS_ASSERT(graph != nullptr);
  auto nodes = graph->nodes();
  MS_CHECK_TRUE_RET_VOID(nodes != nullptr);
  nodes->erase(nodes->begin(), nodes->end());
  nodes->insert(nodes->begin(), sorted_kernels_.begin(), sorted_kernels_.end());
}

PNNASubGraph *PNNADelegate::CreatePNNASubGraph(DelegateModel<schema::Primitive> *model,
                                               std::vector<PNNAOp *> *candidate_ops) {
  // find kernels that in the same subgraph
  std::vector<PNNAOp *> chosen_ops;
  auto ret = FindReadyKernels<PNNAOp>(candidate_ops, &chosen_ops);
  if (ret != RET_OK || chosen_ops.empty()) {
    MS_LOG(ERROR) << "Find ready pnna ops failed.";
    return nullptr;
  }
  // find inputs and outputs
  auto inputs = GetGraphInTensors<PNNAOp>(chosen_ops, nullptr);
  if (inputs.empty()) {
    MS_LOG(ERROR) << "Find inputs of subgraph failed.";
    return nullptr;
  }
  auto outputs = GetGraphOutTensors<PNNAOp>(chosen_ops);
  // find the output tensor which is an input of other kernel.
  for (auto pnna_op : chosen_ops) {
    for (auto kernel : remained_kernels_) {
      std::for_each(kernel->inputs().begin(), kernel->inputs().end(), [&pnna_op, &outputs](const MSTensor &tensor) {
        if (std::find(outputs.begin(), outputs.end(), tensor) == outputs.end() &&
            std::find(pnna_op->outputs().begin(), pnna_op->outputs().end(), tensor) != pnna_op->outputs().end()) {
          outputs.push_back(tensor);
        }
      });
    }
  }
  if (outputs.empty()) {
    MS_LOG(ERROR) << "Find outputs of subgraph failed.";
    return nullptr;
  }
  auto pnna_kernel = new (std::nothrow) PNNASubGraph(ctx_, chosen_ops, inputs, outputs);
  if (pnna_kernel == nullptr) {
    MS_LOG(ERROR) << "New pnna subgraph kernel failed.";
    return nullptr;
  }
  ret = pnna_kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init pnna subgraph failed.";
    delete pnna_kernel;
    return nullptr;
  }
  ret = pnna_kernel->CreatePNNAModel();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Create pnna model failed.";
    delete pnna_kernel;
    return nullptr;
  }
  return pnna_kernel;
}
}  // namespace lite
}  // namespace mindspore
