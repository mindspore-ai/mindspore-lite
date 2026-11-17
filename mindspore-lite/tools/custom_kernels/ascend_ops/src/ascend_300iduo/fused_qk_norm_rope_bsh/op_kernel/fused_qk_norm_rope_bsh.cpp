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

#include "fused_qk_norm_rope_bsh.h"  // NOLINT(build/include_subdir)

// The kernel entry and tiling macros must remain in the global namespace.
extern "C" __global__ __aicore__ void fused_qk_norm_rope_bsh(GM_ADDR q, GM_ADDR k, GM_ADDR gammaQ, GM_ADDR gammaK,
                                                             GM_ADDR cos, GM_ADDR sin, GM_ADDR qOut, GM_ADDR kOut,
                                                             GM_ADDR workspaceGM, GM_ADDR tilingGM) {
  REGISTER_TILING_DEFAULT(FusedQKNormRopeBSHTilingData);
  GET_TILING_DATA(tilingData, tilingGM);
  // Atlas 300I Duo (dav_m200) runs pure-vector kernels as unified AICore
  // tasks; AIV-only task metadata is not a supported inference launch mode.
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  (void)workspaceGM;  // no device workspace; ABI keeps the slot
  TPipe pipe;
  FusedQKNormRopeBSH::FusedQKNormRopeBSH op;
  op.Init(q, k, gammaQ, gammaK, cos, sin, qOut, kOut, &tilingData, &pipe);
  op.Process();
}
