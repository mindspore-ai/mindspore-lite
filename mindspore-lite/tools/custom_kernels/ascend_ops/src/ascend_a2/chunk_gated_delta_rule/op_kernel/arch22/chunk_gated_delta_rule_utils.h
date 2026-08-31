/** * copy from https://gitcode.com/cann/ops-transformer/tree/master/attention/chunk_gated_delta_rule
 *
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

/*!
 * \file chunk_gated_delta_rule_utils.h
 * \brief Common constants for chunk_gated_delta_rule kernels
 */

#ifndef CHUNK_GATED_DELTA_RULE_UTILS_H__
#define CHUNK_GATED_DELTA_RULE_UTILS_H__

#if __has_include("kernel_vec_intf.h")
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"

namespace ChunkGatedDeltaRule {
// sync signals
constexpr uint64_t V_MTE3_EVENT = 0;
constexpr uint64_t V_S_EVENT = 1;
constexpr uint64_t MTE2_V_EVENT = 2;
constexpr uint64_t S_V_EVENT = 3;
constexpr uint64_t MTE3_MTE2_EVENT = 4;
constexpr uint64_t MTE3_S_EVENT = 5;
constexpr uint64_t FIX_MTE2_EVENT = 6;
constexpr uint64_t S_MTE3_EVENT = 6;

constexpr uint64_t NUM_ONE = 1;
constexpr uint64_t BUFFER_NUM_ONE = 1;
constexpr uint64_t BUFFER_NUM_TWO = 2;
constexpr uint64_t TQUE_DEPTH_TWO = 2;
constexpr uint64_t AIC_AIV_1_1 = 2;
constexpr uint64_t BROADCAST_AXIS = 2;
constexpr uint64_t TASK_RATIO = 2;
constexpr uint64_t STAGE3_BUFFER_COUNT = 4;
constexpr uint32_t MAX_L0_SIZE = 64 * 1024;  // 64KB
constexpr uint32_t BLOCK_SIZE = 32;          // copypad aligned block size
constexpr uint32_t BLOCK_FLOAT_NUM = 8;
constexpr uint32_t BLOCK_BF16_NUM = 16;
constexpr uint32_t TILE_LEN = 1024;  // 1024 = 1KB; tested, 1KB and 10KB show negligible performance difference
}  // namespace ChunkGatedDeltaRule

#endif  // __CHUNK_GATED_DELTA_RULE_UTILS_H__
