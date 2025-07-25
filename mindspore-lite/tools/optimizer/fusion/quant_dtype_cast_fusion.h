/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_QUANT_DTYPE_CAST_FUSION_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_QUANT_DTYPE_CAST_FUSION_H_

#include <vector>
#include <memory>
#include <string>
#include "schema/inner/model_generated.h"
#include "tools/optimizer/common/pattern_process_pass_extends.h"
#include "include/common/utils/utils.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore {
namespace opt {
class QuantDtypeCastFusion : public LitePatternProcessPass {
 public:
  explicit QuantDtypeCastFusion(const std::string &name = "QuantDtypeCastFusion", bool multigraph = true)
      : LitePatternProcessPass(name, multigraph) {}

  ~QuantDtypeCastFusion() override = default;

 private:
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;
  bool CheckPattern(const EquivPtr &equiv, const AnfNodePtr &node) const;

 protected:
  mutable VarPtr input_ = nullptr;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_QUANT_DTYPE_CAST_FUSION_H_
