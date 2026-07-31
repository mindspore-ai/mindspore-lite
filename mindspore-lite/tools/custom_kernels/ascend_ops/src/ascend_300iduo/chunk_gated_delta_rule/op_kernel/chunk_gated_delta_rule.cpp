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

// The kernel entry and tiling macros must remain in the global namespace.
extern "C" __global__ __aicore__ void chunk_gated_delta_rule(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR beta,
                                                             GM_ADDR initialState, GM_ADDR actualSeqLengths,
                                                             GM_ADDR gOptional, GM_ADDR out, GM_ADDR finalState,
                                                             GM_ADDR workspaceGM, GM_ADDR tilingGM) {
  REGISTER_TILING_DEFAULT(ChunkGatedDeltaRuleTilingData);
  GET_TILING_DATA(tilingData, tilingGM);
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  GM_ADDR userWorkspace = GetUserWorkspace(workspaceGM);
  CGDRInitParams initParams{query,     key, value,      beta,         initialState, actualSeqLengths,
                            gOptional, out, finalState, userWorkspace};
  TPipe pipe;
  ChunkGatedDeltaRule<half, half> op(&tilingData);
  op.Init(initParams, &pipe);
  op.Process();
}
