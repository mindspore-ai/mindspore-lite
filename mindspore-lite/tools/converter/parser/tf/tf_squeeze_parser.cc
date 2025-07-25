/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "tools/converter/parser/tf/tf_squeeze_parser.h"
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"
#include "mindspore/ops/op_def/auto_generate/gen_lite_ops.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr TFSqueezeParser::Parse(const tensorflow::NodeDef &tf_op,
                                     const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                     std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::Squeeze>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  std::vector<int64_t> axis;
  tensorflow::AttrValue attr_value;
  if (!TensorFlowUtils::FindAttrValue(tf_op, "squeeze_dims", &attr_value)) {
    MS_LOG(ERROR) << "Find Squeeze input squeeze_dims attr failed";
    return nullptr;
  }
  auto dims = attr_value.list();
  for (int i = 0; i < dims.i_size(); ++i) {
    axis.push_back(dims.i(i));
  }
  prim->set_axis(axis);

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }

  return prim->GetPrim();
}

TFNodeRegistrar g_tfSqueezeParser("Squeeze", new TFSqueezeParser());
}  // namespace lite
}  // namespace mindspore
