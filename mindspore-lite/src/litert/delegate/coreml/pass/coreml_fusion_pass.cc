/**
 * Copyright 2022-2026 Huawei Technologies Co., Ltd
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

#include "src/litert/delegate/coreml/pass/coreml_fusion_pass.h"
#include <vector>
#include "src/litert/delegate/coreml/pass/coreml_pass_utils.h"
#include "src/litert/delegate/fusion_pass_utils.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::lite {
bool CheckFusion(CoreMLOp *cur_op, const std::vector<mindspore::MSTensor> &graph_outputs) {
  if (cur_op->in_ops().empty() || cur_op->out_ops().empty()) {
    return false;
  }
  auto pre_flag = std::all_of(cur_op->in_ops().begin(), cur_op->in_ops().end(), [](CoreMLOp *in_op) {
    return CoreMLPassUtils::IsNchw2Nhwc(in_op) && in_op->out_ops().size() == 1;
  });
  if (!pre_flag) {
    return false;
  }
  auto post_flag = std::all_of(cur_op->out_ops().begin(), cur_op->out_ops().end(),
                               [](CoreMLOp *out_op) { return CoreMLPassUtils::IsNhwc2Nchw(out_op); });
  if (!post_flag) {
    return false;
  }
  for (auto out_op : cur_op->out_ops()) {
    // If the pattern is "nc2nh->cur_op->nh2nc" while the output tensors of "cur_op" and "nh2nc" are both graph output,
    // the trans ops can not be fused since it will cause the missing of graph output.
    if (out_op->out_ops().empty() &&
        std::find(graph_outputs.begin(), graph_outputs.end(), out_op->inputs().at(0)) != graph_outputs.end()) {
      return false;
    }
  }
  return true;
}

bool CheckFormatFusion(CoreMLOp *cur_op) {
  if (cur_op->out_ops().empty()) {
    return false;
  }
  if (CoreMLPassUtils::IsNhwc2Nchw(cur_op)) {
    return std::all_of(cur_op->out_ops().begin(), cur_op->out_ops().end(),
                       [](CoreMLOp *op) { return CoreMLPassUtils::IsNchw2Nhwc(op); });
  }
  if (CoreMLPassUtils::IsNchw2Nhwc(cur_op)) {
    return std::all_of(cur_op->out_ops().begin(), cur_op->out_ops().end(),
                       [](CoreMLOp *op) { return CoreMLPassUtils::IsNhwc2Nchw(op); });
  }
  return false;
}

void CoreMLFusionPass::RemoveAndFreeOp(CoreMLOp *cur_op) {
  auto itr = find(all_ops_->begin(), all_ops_->end(), cur_op);
  if (itr != all_ops_->end()) {
    all_ops_->erase(itr);
  }
  delete cur_op;
}

int CoreMLFusionPass::UpdatePreOps(CoreMLOp *cur_op) {
  return delegate::UpdatePreOps<CoreMLOp, false>(cur_op, all_ops_, [this](CoreMLOp *op) { this->RemoveAndFreeOp(op); });
}

int CoreMLFusionPass::UpdatePostOps(CoreMLOp *cur_op) {
  return delegate::UpdatePostOps<CoreMLOp, false>(cur_op, all_ops_,
                                                  [this](CoreMLOp *op) { this->RemoveAndFreeOp(op); });
}

int CoreMLFusionPass::UpdateOp(CoreMLOp *cur_op) {
  return delegate::UpdateOp<CoreMLOp, false>(cur_op, all_ops_, [this](CoreMLOp *op) { this->RemoveAndFreeOp(op); });
}

int CoreMLFusionPass::CommonFusion(CoreMLOp *cur_op) {
  if (cur_op == nullptr) {
    return RET_ERROR;
  }
  auto ret = UpdateOp(cur_op);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "UpdateOp failed.";
    return RET_ERROR;
  }
  ret = cur_op->HandleAxis();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "HandleAxis failed.";
    return ret;
  }
  return RET_OK;
}

void UpdateOutOpsOfPreOp(CoreMLOp *cur_op, bool found_graph_out_tensor, const mindspore::MSTensor &graph_out_tensor,
                         const std::vector<CoreMLOp *> &pre_insert_ops) {
  delegate::UpdateOutOpsOfPreOp<CoreMLOp>(cur_op, found_graph_out_tensor, graph_out_tensor, pre_insert_ops);
}

int CoreMLFusionPass::FormatFusion(CoreMLOp *cur_op) {
  return delegate::FormatFusion<CoreMLOp, CoreMLGraph>(cur_op, subgraph_, all_ops_, name_,
                                                       [this](CoreMLOp *op) { this->RemoveAndFreeOp(op); });
}

int CoreMLFusionPass::Run(CoreMLGraph *subgraph) {
  subgraph_ = subgraph;
  all_ops_ = subgraph->GetOps();
  for (size_t i = 0; i < all_ops_->size(); i++) {
    auto cur_op = (*all_ops_)[i];
    auto ret = RET_OK;
    if (CheckFusion(cur_op, subgraph->outputs())) {
      i -= cur_op->in_ops().size();
      ret = CommonFusion(cur_op);
    }
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Fusion failed.";
      return RET_ERROR;
    }
  }
  for (size_t i = 0; i < all_ops_->size(); ++i) {
    auto cur_op = (*all_ops_)[i];
    if (CheckFormatFusion(cur_op)) {
      i--;
      auto ret = FormatFusion(cur_op);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "FormatFusion failed.";
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite
