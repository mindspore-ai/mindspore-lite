/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "parser/caffe/caffe_bias_parser.h"
#include <memory>
#include <vector>
#include "common/op_attr.h"
#include "infer/custom.h"
#include "mindspore/ops/op_def/op_name.h"

namespace mindspore {
namespace lite {
BaseOperatorPtr CaffeBiasParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_shared<ops::Custom>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "prim is nullptr.";
    return nullptr;
  }
  prim->set_type("Bias");
  const caffe::BiasParameter &biasParam = proto.bias_param();
  if (biasParam.has_axis()) {
    (void)prim->AddAttr(ops::kAxis, api::MakeValue<int64_t>(biasParam.axis()));
  }
  if (biasParam.has_num_axes()) {
    (void)prim->AddAttr(dpico::kNumAxes, api::MakeValue<int64_t>(biasParam.num_axes()));
  }

  return prim;
}

CaffeNodeRegistrar g_caffeBiasParser("Bias", new CaffeBiasParser());
}  // namespace lite
}  // namespace mindspore
