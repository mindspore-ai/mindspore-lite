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
#include "kernel_operator.h"  // NOLINT(build/include_subdir)

extern "C" __aicore__ void ms_rms_norm_impl(GM_ADDR x, GM_ADDR w, GM_ADDR y, GM_ADDR workspace, uint64_t originM,
                                            uint64_t originK, float epsilon, float reciprocalOfHLength,
                                            uint32_t hasGamma, uint64_t blockM, uint32_t splitM, uint32_t splitK,
                                            uint32_t loopK, uint32_t tailK, uint32_t reduceSplitK, uint32_t reduceLoopK,
                                            uint32_t reduceTailK);

extern "C" __global__ __aicore__ void ms_rms_norm(GM_ADDR x, GM_ADDR w, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
  GET_TILING_DATA(tilingData, tiling);
  ms_rms_norm_impl(x, w, y, workspace, tilingData.originM, tilingData.originK, tilingData.epsilon,
                   tilingData.reciprocalOfHLength, tilingData.hasGamma, tilingData.tilingDataGm2Ub.blockM,
                   tilingData.tilingDataGm2Ub.splitM, tilingData.tilingDataGm2Ub.splitK,
                   tilingData.tilingDataGm2Ub.loopK, tilingData.tilingDataGm2Ub.tailK,
                   tilingData.tilingDataLargeReduce.reduceSplitK, tilingData.tilingDataLargeReduce.reduceLoopK,
                   tilingData.tilingDataLargeReduce.reduceTailK);
}
