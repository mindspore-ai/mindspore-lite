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

#include "chunk_gated_delta_rule.h"  // NOLINT(build/include_subdir)

using namespace AscendC;  // NOLINT(build/namespaces)
// CGDR class is in the global namespace (see chunk_gated_delta_rule.h) — no
// using-directive needed, and avoiding one keeps the CANN autogen from emitting
// tiling symbols under a user namespace.

// The kernel entry MUST stay at global scope. REGISTER_TILING_DEFAULT /
// GET_TILING_DATA expand inside this function, and if a user namespace (e.g.
// namespace cgdr) is open around it, the CANN autogen emits the tiling-data type
// and tiling-key as <ns>::-qualified symbols ("cgdr::ChunkGatedDeltaRuleTilingData",
// "cgdr::chunk_gated_delta_rule_0_tilingkey") that then fail to resolve —
// "unknown type" / "undeclared identifier" — and abort the device-kernel compile.
// (Reproduces on both CANN 8.5.0 and 9.1.0.) Keep the entry unwrapped.
extern "C" __global__ __aicore__ void chunk_gated_delta_rule(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR g,
                                                             GM_ADDR beta, GM_ADDR initialState, GM_ADDR cuSeqlens,
                                                             GM_ADDR ssmStateIndices, GM_ADDR out, GM_ADDR finalState,
                                                             GM_ADDR workspaceGM, GM_ADDR tilingGM) {
  REGISTER_TILING_DEFAULT(ChunkGatedDeltaRuleTilingData);
  GET_TILING_DATA(tilingData, tilingGM);
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  CGDRInitParams initParams{query, key, value, g, beta, initialState, cuSeqlens, ssmStateIndices, out, finalState};
  TPipe pipe;
  ChunkGatedDeltaRule<half, half> op(&tilingData);
  op.Init(initParams, &pipe);
  op.Process();
}
