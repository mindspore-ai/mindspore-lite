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
#ifndef RMSNORM_TILING_DATA_DEF_H
#define RMSNORM_TILING_DATA_DEF_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TilingDataGm2Ub)
TILING_DATA_FIELD_DEF(uint64_t, blockM);
TILING_DATA_FIELD_DEF(uint32_t, splitM);
TILING_DATA_FIELD_DEF(uint32_t, splitK);
TILING_DATA_FIELD_DEF(uint32_t, loopK);
TILING_DATA_FIELD_DEF(uint32_t, tailK);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(TilingDataLargeReduce)
TILING_DATA_FIELD_DEF(uint32_t, reduceSplitK);
TILING_DATA_FIELD_DEF(uint32_t, reduceLoopK);
TILING_DATA_FIELD_DEF(uint32_t, reduceTailK);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(TilingData4RmsNorm)
TILING_DATA_FIELD_DEF(uint64_t, originM);
TILING_DATA_FIELD_DEF(uint64_t, originK);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, reciprocalOfHLength);
TILING_DATA_FIELD_DEF(uint32_t, hasGamma);
TILING_DATA_FIELD_DEF_STRUCT(TilingDataGm2Ub, tilingDataGm2Ub);
TILING_DATA_FIELD_DEF_STRUCT(TilingDataLargeReduce, tilingDataLargeReduce);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TilingDataGm2UbOp, TilingDataGm2Ub)
REGISTER_TILING_DATA_CLASS(TilingDataLargeReduceOp, TilingDataLargeReduce)
REGISTER_TILING_DATA_CLASS(MsRmsNorm, TilingData4RmsNorm)
}  // namespace optiling

#endif  // RMSNORM_TILING_DATA_DEF_H
