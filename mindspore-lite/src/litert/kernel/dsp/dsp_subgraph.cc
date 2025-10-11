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

#include "src/litert/kernel/dsp/dsp_subgraph.h"
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "include/errorcode.h"
#include "src/common/utils.h"

namespace mindspore::kernel {
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::dsp::MemType;

DspSubGraph::~DspSubGraph() { UnInit(); }

void DspSubGraph::GetInOutNodes() {
  this->in_nodes_.clear();
  this->out_nodes_.clear();
  auto in_tensors = this->in_tensors();
  auto out_tensors = this->out_tensors();
  for (auto *node : nodes_) {
    for (auto *tensor : node->in_tensors()) {
      if (std::find(in_tensors.begin(), in_tensors.end(), tensor) != in_tensors.end()) {
        in_nodes_.emplace_back(node);
        break;
      }
    }
    for (auto *tensor : node->out_tensors()) {
      if (std::find(out_tensors.begin(), out_tensors.end(), tensor) != out_tensors.end()) {
        out_nodes_.emplace_back(node);
        break;
      }
    }
  }
}

int DspSubGraph::Prepare() {
  for (const auto tensor : in_tensors()) {
    MS_ASSERT(tensor);
    tensor->set_allocator(allocator_);
  }
  for (const auto tensor : out_tensors()) {
    MS_ASSERT(tensor);
    tensor->set_allocator(allocator_);
  }
  for (auto node : this->nodes_) {
    if (node == nullptr) {
      MS_LOG(ERROR) << "node in Subgraph is nullptr";
      return mindspore::lite::RET_NULL_PTR;
    }
    for (const auto tensor : node->out_tensors()) {
      CHECK_NULL_RETURN(tensor);
      MS_CHECK_TRUE_RET(tensor->data() == nullptr, RET_ERROR);
      tensor->set_allocator(allocator_);
    }
  }
  return RET_OK;
}

void DspSubGraph::UnInit() {
  for (const auto &op : nodes_) {
    delete op;
  }
  nodes_.clear();
  delete this->executor_;
}

int DspSubGraph::ReSize() {
  for (auto kernel : nodes_) {
    if (kernel == nullptr) {
      MS_LOG(ERROR) << "input kernel is nullptr!";
      return RET_ERROR;
    }
    if (kernel->subgraph_type() != kernel::kNotSubGraph) {
      MS_LOG(ERROR) << "all nodes in should be kernel";
      return RET_ERROR;
    }
    std::vector<lite::Tensor *> outputs = kernel->out_tensors();
    for (auto &output : outputs) {
      output->FreeData();
      output->set_shape({-1});
    }
  }
  for (auto kernel : nodes_) {
    auto ret = kernel->ReSize();
    if (ret != RET_OK) {
      MS_LOG(WARNING) << "ReSize " << kernel->name() << "failed!, ret:" << ret;
      return ret;
    }
  }
  return RET_OK;
}

int DspSubGraph::Execute(const KernelCallBack &before, const KernelCallBack &after) {
  MS_ASSERT(allocator_ != nullptr);
  for (auto &tensor : in_tensors()) {
    MS_ASSERT(tensor);
    if (tensor->data() == nullptr) {
      MS_LOG(ERROR) << "Dsp subgraph input tensor data is null";
      return RET_ERROR;
    }
  }
  for (auto *kernel : nodes_) {
    MS_ASSERT(kernel != nullptr);
    auto ret = kernel->Execute(before, after);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "run kernel failed, name: " << kernel->name();
      return ret;
    }
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
