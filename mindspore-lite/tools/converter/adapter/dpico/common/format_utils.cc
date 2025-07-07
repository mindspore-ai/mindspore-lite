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

#include "common/format_utils.h"
#include <set>
#include <string>
#include "mindspore/ops/op_def/auto_generate/gen_lite_ops.h"
#include "infer/tuple_get_item.h"
#include "infer/depend.h"
#include "infer/make_tuple.h"
#include "infer/return.h"
#include "infer/batch_to_space.h"
#include "infer/depth_to_space.h"
#include "infer/fused_batch_norm.h"
#include "infer/cxx_api/avg_pool_fusion.h"
#include "infer/cxx_api/conv2d_fusion.h"
#include "infer/cxx_api/conv2d_transpose_fusion.h"
#include "infer/cxx_api/max_pool_fusion.h"
#include "infer/cxx_api/prelu_fusion.h"
#include "infer/cxx_api/topk_fusion.h"
#include "infer/instance_norm.h"
#include "infer/lrn.h"
#include "infer/resize.h"
#include "infer/roi_pooling.h"
#include "infer/space_to_batch.h"
#include "infer/space_to_batch_nd.h"
#include "infer/space_to_depth.h"
#include "common/anf_util.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_name_a.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_name_b.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_name_c.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_name_d.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_name_f.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_name_i.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_name_l.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_name_m.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_name_p.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_name_r.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_name_s.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_name_t.h"

namespace mindspore {
namespace dpico {
namespace {
const std::set<std::string> kAssignedFormatOpSet = {
  mindspore::ops::kNameAvgPoolFusion, mindspore::ops::kNameBatchNorm,
  mindspore::ops::kNameBatchToSpace,  mindspore::ops::kNameBiasAdd,
  mindspore::ops::kNameConv2DFusion,  mindspore::ops::kNameConv2dTransposeFusion,
  mindspore::ops::kNameDepthToSpace,  mindspore::ops::kNameFusedBatchNorm,
  mindspore::ops::kNameInstanceNorm,  mindspore::ops::kNameLRN,
  mindspore::ops::kNameMaxPoolFusion, mindspore::ops::kNamePReLUFusion,
  mindspore::ops::kNameResize,        mindspore::ops::kNameROIPooling,
  mindspore::ops::kNameSpaceToBatch,  mindspore::ops::kNameSpaceToBatchND,
  mindspore::ops::kNameSpaceToDepth,  mindspore::ops::kNameTopKFusion};
}  // namespace

const std::set<std::string> &GetAssignedFormatOpSet() { return kAssignedFormatOpSet; }

bool IsSpecialType(const api::CNodePtr &cnode) {
  return CheckPrimitiveType(cnode, api::MakeShared<ops::TupleGetItem>()) ||
         CheckPrimitiveType(cnode, api::MakeShared<ops::Depend>()) ||
         CheckPrimitiveType(cnode, api::MakeShared<ops::MakeTuple>()) ||
         CheckPrimitiveType(cnode, api::MakeShared<ops::Return>());
}

std::string FormatEnumToString(mindspore::Format format) {
  static std::vector<std::string> names = {
    "NCHW", "NHWC", "NHWC4", "HWKC", "HWCK",   "KCHW",          "CKHW",  "KHWC", "CHWK",
    "HW",   "HW4",  "NC",    "NC4",  "NC4HW4", "NUM_OF_FORMAT", "NCDHW", "NWC",  "NCW",
  };
  if (format < mindspore::NCHW || format > mindspore::NCW) {
    return "";
  }
  return names[static_cast<size_t>(format)];
}
}  // namespace dpico
}  // namespace mindspore
