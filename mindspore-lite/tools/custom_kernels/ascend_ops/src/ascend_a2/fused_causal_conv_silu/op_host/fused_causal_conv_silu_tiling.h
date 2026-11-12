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

/* fused_causal_conv_silu_tiling.h Host tiling for FusedCausalConvSilu (ascend910b, AIV vector kernel). */

#ifndef __OP_HOST_FUSED_CAUSAL_CONV_SILU_TILING_H__
#define __OP_HOST_FUSED_CAUSAL_CONV_SILU_TILING_H__

#include <tiling/tiling_api.h>
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/shape.h"
#include "../op_kernel/fused_causal_conv_silu_tiling_data.h"

namespace optiling {

ge::graphStatus FusedCausalConvSiluTilingFunc(gert::TilingContext *context);

}

#endif
