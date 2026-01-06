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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_PNNA_PASS_PNNA_PASS_UTILS_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_PNNA_PASS_PNNA_PASS_UTILS_H_

#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include "src/litert/delegate/pnna/op/pnna_op.h"
#include "src/litert/delegate/pnna/op/transpose_pnna.h"
namespace mindspore::lite {
PNNAOp *CreateNchw2NhwcOp(const std::vector<mindspore::MSTensor> &in_tensors,
                          const std::vector<mindspore::MSTensor> &out_tensors, const std::string &name);
PNNAOp *CreateNhwc2NchwOp(const std::vector<mindspore::MSTensor> &in_tensors,
                          const std::vector<mindspore::MSTensor> &out_tensors, const std::string &name);
void UpdateOp(PNNAOp *op, const std::vector<PNNAOp *> &in_ops, const std::vector<PNNAOp *> &out_ops,
              const std::vector<mindspore::MSTensor> &in_tensors, const std::vector<mindspore::MSTensor> &out_tensors);
void UpdateNH2NCTransNodePreOp(PNNAOp *pre_op, PNNAOp *trans_op, PNNAOp *op);
void UpdateNC2NHTransNodePreOp(PNNAOp *pre_op, const std::vector<PNNAOp *> &trans_ops,
                               const std::vector<PNNAOp *> &ops);
void UpdateNH2NCTransNodePostOp(PNNAOp *trans_op, PNNAOp *post_op);
void UpdateNC2NHTransNodePostOp(PNNAOp *op, PNNAOp *trans_op, PNNAOp *post_op,
                                const mindspore::MSTensor &org_in_tensor);
bool IsNhwc2Nchw(PNNAOp *op);
bool IsNchw2Nhwc(PNNAOp *op);
PNNAOp *OpInputFromOp(PNNAOp *op, mindspore::MSTensor in_tensor);
std::vector<mindspore::MSTensor> GetNonConstInputs(PNNAOp *op);
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_PNNA_PASS_PNNA_PASS_UTILS_H_
