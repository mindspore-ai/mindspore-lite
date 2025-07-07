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
#include "tools/converter/parser/tf/tf_addn_parser.h"
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"
#include "op_def/auto_generate/gen_lite_ops.h"
#include "ops_utils/op_utils.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr TFAddNParser::Parse(const tensorflow::NodeDef &tf_op,
                                  const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                  std::vector<std::string> *inputs, int *output_size) {
  MS_CHECK_TRUE_RET(inputs != nullptr, nullptr);
  MS_CHECK_TRUE_RET(output_size != nullptr, nullptr);
  auto prim = std::make_unique<ops::AddN>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  tensorflow::AttrValue attr_value;
  if (!TensorFlowUtils::FindAttrValue(tf_op, "N", &attr_value)) {
    MS_LOG(ERROR) << "The N attr should be specified!";
    return nullptr;
  }
  std::vector<int64_t> param_n;
  param_n.push_back(static_cast<int64_t>(attr_value.i()));
  prim->AddAttr("dyn_input_sizes", api::MakeValue(param_n));
  *output_size = 1;
  for (int i = 0; i < tf_op.input_size(); i++) {
    inputs->emplace_back(tf_op.input(i));
  }
  return prim->GetPrim();
}
TFNodeRegistrar g_tfAddNParser("AddN", new TFAddNParser());
}  // namespace lite
}  // namespace mindspore
