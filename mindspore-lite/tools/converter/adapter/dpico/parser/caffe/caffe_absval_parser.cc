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

#include "parser/caffe/caffe_absval_parser.h"
#include <memory>
#include <vector>
#include "mindspore/ops/op_def/auto_generate/gen_lite_ops.h"

namespace mindspore {
namespace lite {
BaseOperatorPtr CaffeAbsvalParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_shared<ops::Abs>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "prim is nullptr.";
    return nullptr;
  }
  return prim;
}

CaffeNodeRegistrar g_caffeAbsvalParser("AbsVal", new CaffeAbsvalParser());
}  // namespace lite
}  // namespace mindspore
