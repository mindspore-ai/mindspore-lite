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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_PNNA_PASS_PNNA_FORMAT_TRANS_PASS_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_PNNA_PASS_PNNA_FORMAT_TRANS_PASS_H_

#include <set>
#include <vector>
#include "src/litert/delegate/pnna/op/pnna_op.h"
#include "src/litert/delegate/pnna/pass/pnna_base_pass.h"

namespace mindspore::lite {
class PNNAFormatTransPass : public PNNABasePass {
 public:
  PNNAFormatTransPass() { name_ = "PNNAFormatTransPass"; }
  int Run(PNNASubGraph *subgraph) override;

 private:
  int InsertPreNodes(PNNAOp *op, std::vector<PNNAOp *> *trans_ops);
  int InsertPostNodes(PNNAOp *op, std::vector<PNNAOp *> *trans_ops);

 private:
  int total = 0;
  std::vector<PNNAOp *> *all_ops_ = nullptr;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_PNNA_PASS_PNNA_FORMAT_TRANS_PASS_H_
