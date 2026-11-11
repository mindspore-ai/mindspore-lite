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

#include "fused_swi_glu_act_quant_nz.h"  // NOLINT(build/include_subdir)

// The kernel entry and tiling macros must remain in the global namespace.
extern "C" __global__ __aicore__ void fused_swi_glu_act_quant_nz(GM_ADDR gate, GM_ADDR up, GM_ADDR smoothScale,
                                                                 GM_ADDR y, GM_ADDR tokenScale, GM_ADDR workspaceGM,
                                                                 GM_ADDR tilingGM) {
  REGISTER_TILING_DEFAULT(FusedSwiGLUActQuantNZTilingData);
  GET_TILING_DATA(tilingData, tilingGM);
  // Atlas 300I Duo (dav_m200) runs pure-vector kernels as unified AICore
  // tasks; AIV-only task metadata is not a supported inference launch mode.
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  TPipe pipe;
  FusedSwiGLUActQuantNZ::FusedSwiGLUActQuantNZ op;
  op.Init(gate, up, smoothScale, y, tokenScale, &tilingData, &pipe);
  op.Process();
}
