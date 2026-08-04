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

/**
 * @file quant_matmul_w4a8.h
 * @brief QuantMatmulW4a8 — INT4×INT8→BF16 matmul via CANN aclnn.
 */
#ifndef LITE_BOOST_OPS_PLUGIN_QUANT_MATMUL_W4A8_H_
#define LITE_BOOST_OPS_PLUGIN_QUANT_MATMUL_W4A8_H_

#include <ATen/Tensor.h>

at::Tensor QuantMatmulW4a8LiteBoostImplNPU(const at::Tensor &act, const at::Tensor &weight, const at::Tensor &scale,
                                           const at::Tensor &bias, const at::Tensor &x_scale,
                                           const at::Tensor &output_bias);

#endif  // LITE_BOOST_OPS_PLUGIN_QUANT_MATMUL_W4A8_H_
