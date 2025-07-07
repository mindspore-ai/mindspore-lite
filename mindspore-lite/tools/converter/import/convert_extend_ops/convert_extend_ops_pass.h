/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_IMPORT_CONVERT_EXTEND_OPS_CONVERT_EXTEND_OPS_PASS_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_IMPORT_CONVERT_EXTEND_OPS_CONVERT_EXTEND_OPS_PASS_H_

#include <string>
#include <unordered_map>
#include "tools/optimizer/common/multiple_pattern_process_pass.h"

namespace mindspore {
namespace opt {
/**
 * ConvertExtendOpsPass will take effect when some extend operations are found in mindir.
 * Those extend operations can not be identified by lite, ConvertExtendOpsPass will convert them to normal operations.
 * ConvertExtendOpsPass will be removed when lite support those extend operations.
 */
class ConvertExtendOpsPass : public MultiplePatternProcessPass {
 public:
  explicit ConvertExtendOpsPass(const std::string &name = "ConvertExtendOpsPass", bool multigraph = true)
      : MultiplePatternProcessPass(name, multigraph) {}

  ~ConvertExtendOpsPass() override = default;

  AnfNodePtr Process(const std::string &pattern_name, const FuncGraphPtr &, const AnfNodePtr &,
                     const EquivPtr &) const override;
  std::unordered_map<std::string, VectorRef> DefinePatterns() const override;

 private:
  VectorRef DefineSumExtPattern() const;
  VectorRef DefineMatMulExtPattern() const;
  VectorRef DefineMaxPattern() const;
  VectorRef DefineMinPattern() const;
  VectorRef DefineDensePattern() const;
  VectorRef DefineOnesPattern() const;
  VectorRef DefineZerosPattern() const;
  VectorRef DefineMulsPattern() const;
};

AnfNodePtr ConvertSumExtPass(const FuncGraphPtr &func_graph, const mindspore::AnfNodePtr &node);
AnfNodePtr ConvertMatMulExtPass(const FuncGraphPtr &func_graph, const mindspore::AnfNodePtr &node);
AnfNodePtr ConvertMaxMinPass(const FuncGraphPtr &func_graph, const mindspore::AnfNodePtr &node);
AnfNodePtr ConvertDensePass(const FuncGraphPtr &func_graph, const mindspore::AnfNodePtr &node);
AnfNodePtr ConvertOnesPass(const FuncGraphPtr &func_graph, const mindspore::AnfNodePtr &node);
AnfNodePtr ConvertZerosPass(const FuncGraphPtr &func_graph, const mindspore::AnfNodePtr &node);
AnfNodePtr ConvertMulsPass(const FuncGraphPtr &func_graph, const mindspore::AnfNodePtr &node);
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_IMPORT_CONVERT_EXTEND_OPS_CONVERT_EXTEND_OPS_PASS_H_
