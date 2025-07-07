/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "tools/converter/quantizer/quant_params.h"

#include <set>
#include "mindspore/ops/op_def/lite_ops.h"
#include "mindspore/ops/op_def/math_ops.h"
#include "mindspore/ops/op_def/nn_ops.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"
namespace mindspore::lite::quant {

const std::set<PrimitivePtr> kHasBiasOperator = {prim::kPrimConv2DFusion,    prim::kPrimConv2dTransposeFusion,
                                                 prim::kPrimMatMulFusion,    prim::kPrimFullConnection,
                                                 prim::kPrimLayerNormFusion, prim::kPrimMatMul};
}  // namespace mindspore::lite::quant
