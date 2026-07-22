/**
 * modified from
 * https://gitcode.com/cann/ops-nn/blob/master/matmul/quant_batch_matmul_v4/op_kernel/quant_batch_matmul_v4.cpp
 * Copyright 2026 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file quant_matmul_w4a8.cpp
 * @brief w4a8 kernel entry — V5 MSD classes with w4a8 parameter mapping.
 *
 *   if ASCEND_IS_AIV {  split int8→int4b_t in-place on act  }
 *   pipe.Reset(); pipe.Destroy(); pipe.Init();
 *   SyncAll<false>();
 *   Matmul: Cube INT4×INT4 + VEC MSD combine + dequant + output_bias → BF16
 *
 * Parameter mapping (w4a8 → V5 MSD):
 *   act       → x1      (in-place split int8→int4b_t)
 *   weight    → x2      (int4b_t, bTrans=true, ND)
 *   scale     → x2_scale (SetQuantVector in Cube, uint64_t*)
 *   bias      → y_offset (host pre-multiplied bias × w_scale, float[N])
 *   x_scale   → x1_scale (BroadCast+Mul in VEC)
 *   output_bias → outputBias (added after ×x_scale, float[N])
 *   output    → y
 *   workspace → SYS_WS | Cube WS (V5 pattern — split in-place, no separate split buffer)
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "quant_matmul_w4a8_tiling_data.h"
#include "quant_matmul_w4a8_msd.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void quant_matmul_w4a8(
  GM_ADDR act,          // int8   [M, K]  (symmetric, split in-place to int4b_t)
  GM_ADDR weight,       // int4   [N, K]  (bTrans=true, ND)
  GM_ADDR scale,        // float  [N]     per-channel weight scale (→ uint64_t*)
  GM_ADDR bias,         // float  [N]     bias × w_scale (host pre-multiplied → y_offset)
  GM_ADDR x_scale,      // float  [M]     per-token activation scale (→ x1_scale)
  GM_ADDR output_bias,  // float  [N]     output bias (required; pass zeros to skip)
  GM_ADDR output,       // bf16   [M, N]
  GM_ADDR workspace,    // [SYS_WS | Cube workspace]
  GM_ADDR tiling) {     // QuantMatmulW4a8TilingData
  AscendC::SetSysWorkspace(workspace);
  GM_ADDR userWS = AscendC::GetUserWorkspace(workspace);
  if (userWS == nullptr) {
    return;
  }

  REGISTER_TILING_DEFAULT(QuantMatmulW4a8TilingData);
  GET_TILING_DATA(td, tiling);

  TPipe pipe;

  // ── Phase 1: AIV split (in-place, V5 exact pattern) ──
  if ASCEND_IS_AIV {
    QuantBatchMatmulV4MsdPre opPre;
    opPre.Init(act, act, userWS, &td, &pipe);
    opPre.Process();
    pipe.Reset();
    pipe.Destroy();
    pipe.Init();
  }

  SyncAll<false>();

  // ── Phase 2: Matmul + dequant (V5 exact pattern) ──
  using YType = bfloat16_t;
  QuantBatchMatmulV4Msd<int4b_t, int4b_t, float, YType, QuantType::K_C, true, false> op;

  // V5 Init: x1, x2, v5_bias, x1_scale, x2_scale, y_scale, x1_offset, x2_offset,
  //           y_offset, outputBias, y, workspace, tilingData, pipe
  op.Init(act, weight, nullptr, x_scale, scale, nullptr, nullptr, nullptr, bias, output_bias, output, userWS, &td,
          &pipe);
  op.Process();
}
