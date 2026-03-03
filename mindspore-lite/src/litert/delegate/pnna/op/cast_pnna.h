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
#ifndef MINDSPORE_LITE_SRC_LITERT_DELEGATE_PNNA_OP_CAST_PNNA_H_
#define MINDSPORE_LITE_SRC_LITERT_DELEGATE_PNNA_OP_CAST_PNNA_H_

#include <string>
#include <vector>
#include "src/litert/delegate/pnna/op/pnna_op.h"

namespace mindspore {
namespace lite {
class PNNACast : public PNNAOp {
 public:
  PNNACast(const std::string &name, const schema::Primitive *primitive,
           const std::vector<mindspore::MSTensor> &in_tensors, const std::vector<mindspore::MSTensor> &out_tensors,
           schema::QuantType quant_type)
      : PNNAOp(name, primitive, in_tensors, out_tensors, quant_type) {}

  ~PNNACast() override {}

  bool IsSupport() override { return true; };
  int InitParams() override { return RET_OK; };
  int AddOpToPNNAModel(PNNASubGraph *graph) override;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_LITERT_DELEGATE_PNNA_OP_CAST_PNNA_H_
