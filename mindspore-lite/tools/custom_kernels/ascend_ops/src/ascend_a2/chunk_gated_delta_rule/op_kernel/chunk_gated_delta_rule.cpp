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
 * \file chunk_gated_delta_rule.cpp
 * \brief Device-kernel entry for ChunkGatedDeltaRule (ascend910b / arch22).
 *
 * Ported from ops-transformer attention/chunk_gated_delta_rule/op_kernel. The kernel is a
 * MIX_AIC_1_2 build (1 AIC : 2 AIV): the arch22 stages drive both the Cube (MMAD via
 * MatmulImpl) and the Vector units, synchronised with CrossCore flags. The op-def input
 * order is query, key, value, beta, initial_state, actual_seq_lengths, g(optional),
 * out, final_state. The entry is kept at global scope with NO `using namespace
 * ChunkGatedDeltaRule` — the CANN tiling macros below must expand against the globally
 * scoped ChunkGatedDeltaRuleTilingData, and an open user namespace around them makes the
 * autogen emit <ns>::-qualified tiling symbols that fail to resolve.
 */

#include "arch22/chunk_gated_delta_rule.h"
#include "chunk_gated_delta_rule_tiling_data.h"

using AscendC::GetUserWorkspace;
using AscendC::TPipe;

extern "C" __global__ __aicore__ void chunk_gated_delta_rule(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR beta,
                                                             GM_ADDR initialState, GM_ADDR seqlens, GM_ADDR gOptional,
                                                             GM_ADDR out, GM_ADDR finalState, GM_ADDR workspaceGM,
                                                             GM_ADDR tilingGM) {
  REGISTER_TILING_DEFAULT(ChunkGatedDeltaRuleTilingData);
  GET_TILING_DATA(tilingData, tilingGM);
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  TPipe pipe;

  __gm__ uint8_t *user = GetUserWorkspace(workspaceGM);

  ChunkGatedDeltaRule::CGDRInitParams initParams{query,   key,       value, beta,      initialState,
                                                 seqlens, gOptional, out,   finalState};
  // Dispatch on the low dtype: fp16 vs bf16 (both share highType=float).
  if (tilingData.isFp16 != 0) {
    ChunkGatedDeltaRule::CGDR<half, float> op(&pipe, &tilingData);
    op.Init(initParams, user);
    op.Process();
  } else {
    ChunkGatedDeltaRule::CGDR<bfloat16_t, float> op(&pipe, &tilingData);
    op.Init(initParams, user);
    op.Process();
  }
}
