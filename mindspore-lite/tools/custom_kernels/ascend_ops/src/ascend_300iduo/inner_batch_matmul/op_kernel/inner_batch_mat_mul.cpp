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
 * ============================================================================
 */
// InnerBatchMatMul kernel entry (hand-written Cube channel).
#include "inner_batch_matmul_kernel.h"  // NOLINT(build/include_subdir)

using namespace AscendC;  // NOLINT(build/namespaces)

// The kernel entry and the tiling macros must remain in the global namespace (see the NOTE in
// inner_batch_matmul_kernel.h); the kernel class is qualified with its namespace.
extern "C" __global__ __aicore__ void inner_batch_mat_mul(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y,
                                                          GM_ADDR workspace, GM_ADDR tiling) {
  REGISTER_TILING_DEFAULT(InnerBatchMatMulTilingData);
  GET_TILING_DATA(tilingData, tiling);
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  TPipe pipe;
  InnerBatchMatMul::GmAddrs addrs{x1, x2, bias, y};
  InnerBatchMatMul::InnerBatchMatMulKernel op;
  // bias is OPTIONAL: GE passes nullptr when absent and tiling then sets biasMode=0, so the kernel
  // never dereferences it. When present it is consumed by the bias Mmad paths.
  op.Init(addrs, &tilingData.cube, &pipe);
  op.Process();
}
