/**
 * Copyright 2020-2026 Huawei Technologies Co., Ltd
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
#include "tools/converter/parser/tf/tf_pool_parser.h"
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "include/errorcode.h"
#include "tools/converter/parser/tf/tf_node_parser_registry.h"
#include "tools/converter/parser/tf/tf_util.h"
#include "src/common/ops/primitive/avg_pool_fusion.h"
#include "src/common/ops/primitive/max_pool_fusion.h"

namespace mindspore {
namespace lite {
namespace {
constexpr int kTfPoolStrideListSize = 4;
constexpr int kTfPoolKernelListSize = 4;

template <typename T>
STATUS ParseTfPoolCommonAttrs(const tensorflow::NodeDef &tf_op, T *prim) {
  MS_ASSERT(prim != nullptr);
  tensorflow::AttrValue attr_value;

  // Parse padding attribute
  if (TensorFlowUtils::FindAttrValue(tf_op, "padding", &attr_value)) {
    if (attr_value.s() == "VALID") {
      prim->set_pad_mode(mindspore::PadMode::VALID);
    } else if (attr_value.s() == "SAME") {
      prim->set_pad_mode(mindspore::PadMode::SAME);
    }
  }

  // Parse format and set as attribute
  auto format = TensorFlowUtils::ParseNodeFormat(tf_op);
  auto prim_c = prim->GetPrim();
  (void)prim_c->AddAttr(mindspore::ops::kOriginalFormat, MakeValue<int64_t>(format));

  // Parse strides attribute
  if (TensorFlowUtils::FindAttrValue(tf_op, "strides", &attr_value)) {
    const auto &stride_list = attr_value.list();
    MS_CHECK_TRUE_RET(stride_list.i_size() >= kTfPoolStrideListSize, lite::RET_ERROR);
    if (format == mindspore::Format::NCHW) {
      prim->set_strides({stride_list.i(2), stride_list.i(3)});
    } else {
      prim->set_strides({stride_list.i(1), stride_list.i(2)});
    }
  }

  // Parse kernel size attribute
  if (TensorFlowUtils::FindAttrValue(tf_op, "ksize", &attr_value)) {
    const auto &kernel_list = attr_value.list();
    MS_CHECK_TRUE_RET(kernel_list.i_size() >= kTfPoolKernelListSize, lite::RET_ERROR);
    if (format == mindspore::Format::NCHW) {
      prim->set_kernel_size({kernel_list.i(2), kernel_list.i(3)});
    } else {
      prim->set_kernel_size({kernel_list.i(1), kernel_list.i(2)});
    }
  }

  return lite::RET_OK;
}
}  // namespace
PrimitiveCPtr TFMaxPoolParser::Parse(const tensorflow::NodeDef &tf_op,
                                     const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                     std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::MaxPoolFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  MS_CHECK_TRUE_RET(ParseTfPoolCommonAttrs(tf_op, prim.get()) == lite::RET_OK, nullptr);

  *output_size = 1;
  for (int i = 0; i < tf_op.input_size(); i++) {
    inputs->emplace_back(tf_op.input(i));
  }

  return prim->GetPrim();
}

PrimitiveCPtr TFAvgPoolParser::Parse(const tensorflow::NodeDef &tf_op,
                                     const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                     std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::AvgPoolFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  MS_CHECK_TRUE_RET(ParseTfPoolCommonAttrs(tf_op, prim.get()) == lite::RET_OK, nullptr);

  *output_size = 1;
  for (int i = 0; i < tf_op.input_size(); i++) {
    inputs->emplace_back(tf_op.input(i));
  }

  return prim->GetPrim();
}

TFNodeRegistrar g_tfMaxPoolParser("MaxPool", new TFMaxPoolParser());
TFNodeRegistrar g_tfAvgPoolParser("AvgPool", new TFAvgPoolParser());
}  // namespace lite
}  // namespace mindspore
