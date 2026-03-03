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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_CONST_FOLD_CONSTANT_TAG_UTILS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_CONST_FOLD_CONSTANT_TAG_UTILS_H_

#include <memory>
#include <vector>
#include "include/api/context.h"
#include "include/registry/converter_context.h"
#include "ir/anf.h"
#include "schema/inner/model_generated.h"
#include "src/litert/inner_context.h"
#include "tools/lite_exporter/fetch_content.h"

namespace mindspore {
namespace opt {
class ConstantTagUtils {
 public:
  ConstantTagUtils() = default;
  ~ConstantTagUtils() = default;

  static int GetCNodeInputTensors(const CNodePtr &cnode, std::vector<TensorPtr> *inputs, converter::FmkType fmk_type,
                                  bool train_flag, bool copy_data);

 private:
  static int FetchDataFromCNodeAttr(const CNodePtr &cnode, const AbstractBasePtr &abstract, lite::DataInfo *data_info);
  static int GetCNodeVarInput(const CNodePtr &cnode, const size_t &index, std::vector<TensorPtr> *var_ms_inputs);
  static int GetTensorDataNBytes(const tensor::TensorPtr &tensor);
  static TensorPtr CreateTensorFromData(const lite::DataInfo &data_info, const bool &has_inferred,
                                        const mindspore::Format &format);
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_CONST_FOLD_CONSTANT_TAG_UTILS_H_
