/**
 * modified from
 * https://gitcode.com/cann/ops-nn/blob/master/matmul/quant_batch_matmul_v4/op_host/quant_batch_matmul_v4_tiling.h
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

#ifndef __OP_HOST_QUANT_MATMUL_W4A8_TILING_H__
#define __OP_HOST_QUANT_MATMUL_W4A8_TILING_H__

#include <exe_graph/runtime/tiling_context.h>

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "../op_kernel/quant_matmul_w4a8_tiling_data.h"

namespace Ops {
namespace NN {
namespace QuantMatmulW4a8 {

struct V4MCompileInfo {
  uint64_t ubSize{0};
  uint64_t l1Size{0};
  uint64_t l0ASize{0};
  uint64_t l0BSize{0};
  uint64_t l0CSize{0};
  uint64_t aicNum{0};
};

struct V4MInfo {
  int64_t M = 0;
  int64_t N = 0;
  int64_t K = 0;
  bool initFlag = false;
  const char *opName = "QuantMatmulW4a8";
};

// Standalone tiling class (no ops-nn TilingBaseClass dependency).
// Uses CANN standard APIs directly.
class QuantMatmulW4a8Tiling {
 public:
  explicit QuantMatmulW4a8Tiling(gert::TilingContext *context) : context_(context) { InitCompileInfo(); }
  ~QuantMatmulW4a8Tiling() = default;

  ge::graphStatus GetShapeAttrsInfo();
  ge::graphStatus DoOpTiling();
  ge::graphStatus PostTiling();

 private:
  void InitCompileInfo();
  ge::graphStatus CheckContext();
  bool SetMatmulTiling();

  gert::TilingContext *context_;
  QuantMatmulW4a8TilingData tilingData_;
  V4MInfo inputParams_;
  V4MCompileInfo compileInfo_;
  size_t tilingDataSize_ = 0;
  size_t workspaceSize_ = 0;
};

}  // namespace QuantMatmulW4a8
}  // namespace NN
}  // namespace Ops

#endif  // __OP_HOST_QUANT_MATMUL_W4A8_TILING_H__
