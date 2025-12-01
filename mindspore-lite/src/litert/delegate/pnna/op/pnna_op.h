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
#ifndef MINDSPORE_LITE_SRC_LITERT_DELEGATE_PNNA_OP_PNNA_OP_H_
#define MINDSPORE_LITE_SRC_LITERT_DELEGATE_PNNA_OP_PNNA_OP_H_

#include <string>
#include <vector>
#include <utility>
#include "include/api/kernel.h"
#include "include/api/data_type.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "schema/ops_generated.h"
#include "nnacl_c/op_base.h"
#include "src/litert/delegate/pnna/pnna_subgraph.h"
#include "src/litert/delegate/pnna/pnna_utils.h"

namespace mindspore {
namespace lite {
class PNNASubGraph;
class PNNAOp {
 public:
  explicit PNNAOp(const std::string &name, const schema::Primitive *primitive,
                  const std::vector<mindspore::MSTensor> &in_tensors,
                  const std::vector<mindspore::MSTensor> &out_tensors, schema::QuantType quant_type)
      : op_name_(name),
        op_primitive_(primitive),
        in_tensors_(std::move(in_tensors)),
        out_tensors_(std::move(out_tensors)),
        quant_type_(quant_type) {
    if (primitive != nullptr) {
      this->type_ = primitive->value_type();
    }
  }

  virtual ~PNNAOp() {}

  virtual bool IsSupport() = 0;
  virtual int InitParams() = 0;
  virtual int AddOpToPNNAModel(PNNASubGraph *graph) = 0;

  const std::vector<mindspore::MSTensor> &inputs() { return this->in_tensors_; }
  const std::vector<mindspore::MSTensor> &outputs() { return this->out_tensors_; }
  void set_inputs(const std::vector<mindspore::MSTensor> &inputs) { this->in_tensors_ = inputs; }
  void set_outputs(const std::vector<mindspore::MSTensor> &outputs) { this->out_tensors_ = outputs; }

  const std::vector<PNNAOp *> &in_ops() { return this->in_ops_; }
  const std::vector<PNNAOp *> &out_ops() { return this->out_ops_; }
  void set_in_ops(const std::vector<PNNAOp *> &in_ops) { this->in_ops_ = in_ops; }
  void set_out_ops(const std::vector<PNNAOp *> &out_ops) { this->out_ops_ = out_ops; }

  const std::string name() { return op_name_; }
  schema::QuantType get_quant_type() { return quant_type_; }
  schema::PrimitiveType type() const { return type_; }

 protected:
  std::string op_name_;
  const schema::Primitive *op_primitive_ = nullptr;
  std::vector<mindspore::MSTensor> in_tensors_;
  std::vector<mindspore::MSTensor> out_tensors_;
  schema::PrimitiveType type_ = schema::PrimitiveType_NONE;
  schema::QuantType quant_type_ = schema::QuantType_QUANT_NONE;

  std::vector<PNNAOp *> in_ops_;
  std::vector<PNNAOp *> out_ops_;
};

typedef PNNAOp *(*PNNAGetOp)(const std::string &name, const schema::Primitive *primitive,
                             const std::vector<mindspore::MSTensor> &in_tensors,
                             const std::vector<mindspore::MSTensor> &out_tensors, schema::QuantType quant_type);

template <class T>
PNNAOp *GetPNNAOp(const std::string &name, const schema::Primitive *primitive,
                  const std::vector<mindspore::MSTensor> &in_tensors,
                  const std::vector<mindspore::MSTensor> &out_tensors, schema::QuantType quant_type) {
  MS_ASSERT(primitive != nullptr);
  auto *op = new (std::nothrow) T(name, primitive, in_tensors, out_tensors, quant_type);
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is nullptr.";
    return nullptr;
  }
  auto ret = op->InitParams();
  if (ret != RET_OK) {
    MS_LOG(WARNING) << "PNNA op init failed.";
    delete op;
    return nullptr;
  }
  if (!op->IsSupport()) {
    MS_LOG(WARNING) << "PNNA op is not supported.";
    delete op;
    return nullptr;
  }
  return op;
}
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_LITERT_DELEGATE_PNNA_OP_PNNA_OP_H_
