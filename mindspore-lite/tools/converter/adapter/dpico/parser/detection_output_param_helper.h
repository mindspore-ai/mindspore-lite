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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_PARSER_DETECTION_OUTPUT_PARAM_HELPER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_PARSER_DETECTION_OUTPUT_PARAM_HELPER_H_

#include <string>
#include <utility>
#include <memory>
#include <vector>
#include "op/detection_output_operator.h"
#include "infer/custom.h"
#include "./pico_caffe.pb.h"

namespace mindspore {
namespace dpico {
int SetAttrsByDetectionOutputParam(const std::shared_ptr<ops::Custom> &custom_prim, const caffe::LayerParameter &proto);
int SetAttrsByDecBboxParam(const std::shared_ptr<ops::Custom> &custom_prim, const caffe::LayerParameter &proto);
int GetDetectionOutputParamFromAttrs(std::vector<mapper::DetectionOutputParam> *detection_params,
                                     const api::SharedPtr<ops::Custom> &custom_prim);
}  // namespace dpico
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_PARSER_DETECTION_OUTPUT_PARAM_HELPER_H_
