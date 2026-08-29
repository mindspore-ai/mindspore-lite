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
#include "lib/normalization/rmsnorm.h"

using namespace AscendC;  // NOLINT(build/namespaces)

namespace {
using DataType = half;
constexpr uint32_t SHARED_TMP_BYTES = 32U;
constexpr uint32_t AICORE_UB_BYTES = 120U * 1024U;
constexpr uint32_t MAX_ROWS_PER_TILE = 32U;
constexpr uint32_t FP32_BLOCK_ELEMENTS = 8U;
}  // namespace

class KernelMsRmsNorm {
 public:
  __aicore__ inline KernelMsRmsNorm() {}

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR y, uint64_t originM, uint64_t originK, float epsilon,
                              float reciprocalOfHLength, uint32_t hasGamma, uint64_t blockM, uint32_t splitM,
                              uint32_t splitK, uint32_t loopK, uint32_t tailK, uint32_t reduceSplitK,
                              uint32_t reduceLoopK, uint32_t reduceTailK) {
    this->originM = originM;
    this->originK = originK;
    this->epsilon = epsilon;
    this->reciprocalOfHLength = reciprocalOfHLength;
    this->hasGamma = hasGamma;
    this->blockM = blockM;
    this->splitM = splitM;
    this->splitK = splitK;
    this->loopK = loopK;
    this->tailK = tailK;
    this->reduceSplitK = reduceSplitK;
    this->reduceLoopK = reduceLoopK;
    this->reduceTailK = reduceTailK;

    const uint64_t total = originM * originK;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(x), total);
    if (hasGamma != 0U) {
      gammaGm.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(w), originK);
    }
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(y), total);

    const uint32_t rowBytes = splitK * sizeof(DataType);
    const uint32_t availableForRows = AICORE_UB_BYTES - rowBytes - SHARED_TMP_BYTES;
    const uint32_t rowsByUb = availableForRows / (2U * rowBytes);
    tileRows = rowsByUb < MAX_ROWS_PER_TILE ? rowsByUb : MAX_ROWS_PER_TILE;
    if (tileRows == 0U) {
      tileRows = 1U;
    }
    if (static_cast<uint64_t>(tileRows) > blockM) {
      tileRows = static_cast<uint32_t>(blockM);
    }

    pipe.InitBuffer(xBuffer, rowBytes * tileRows);
    pipe.InitBuffer(gammaBuffer, rowBytes);
    pipe.InitBuffer(yBuffer, rowBytes * tileRows);
    pipe.InitBuffer(sharedTmpBuffer, SHARED_TMP_BYTES);
  }

  __aicore__ inline void Process() {
    const uint32_t block = GetBlockIdx();
    if (!TilingIsValid(block)) {
      return;
    }

    LocalTensor<DataType> gammaLocal = gammaBuffer.Get<DataType>();
    if (hasGamma != 0U) {
      DataCopy(gammaLocal, gammaGm, splitK);
      pipe_barrier(PIPE_ALL);
    }

    const uint64_t firstRow = static_cast<uint64_t>(block) * blockM;
    const uint64_t rowsForBlock = originM - firstRow < blockM ? originM - firstRow : blockM;
    for (uint64_t localRow = 0; localRow < rowsForBlock; localRow += tileRows) {
      const uint32_t rows =
        static_cast<uint32_t>(rowsForBlock - localRow < tileRows ? rowsForBlock - localRow : tileRows);
      ProcessRows(firstRow + localRow, rows, gammaLocal);
    }
  }

 private:
  __aicore__ inline bool TilingIsValid(uint32_t block) const {
    if (block >= splitM || originM == 0U || originK == 0U || blockM == 0U || splitK == 0U || tailK == 0U ||
        tailK > splitK || hasGamma > 1U || reduceSplitK == 0U || reduceTailK == 0U || reduceTailK > reduceSplitK) {
      return false;
    }
    const uint64_t gm2UbCovered = static_cast<uint64_t>(loopK) * splitK + tailK;
    const uint64_t reduceCovered = static_cast<uint64_t>(reduceLoopK) * reduceSplitK + reduceTailK;
    return gm2UbCovered == originK && reduceCovered == originK && splitK == originK && loopK == 0U &&
           tailK == originK && static_cast<uint64_t>(splitM) * blockM >= originM;
  }

  __aicore__ inline void ProcessRows(uint64_t firstRow, uint32_t rows, LocalTensor<DataType> gammaLocal) {
    const uint64_t offset = firstRow * originK;
    const uint32_t elements = rows * splitK;
    LocalTensor<DataType> xLocal = xBuffer.Get<DataType>();
    LocalTensor<DataType> yLocal = yBuffer.Get<DataType>();
    LocalTensor<uint8_t> sharedTmp = sharedTmpBuffer.Get<uint8_t>();
    DataCopy(xLocal, xGm[offset], elements);
    pipe_barrier(PIPE_ALL);

    RmsNormTiling rmsTiling;
    rmsTiling.bLength = 1U;
    rmsTiling.sLength = rows;
    rmsTiling.hLength = splitK;
    rmsTiling.originalHLength = static_cast<uint32_t>(originK);
    rmsTiling.reciprocalOfHLength = reciprocalOfHLength;
    rmsTiling.mainBshLength = elements;
    rmsTiling.mainBsLength = rows;
    rmsTiling.mainBsLengthAlign = (rows + FP32_BLOCK_ELEMENTS - 1U) / FP32_BLOCK_ELEMENTS * FP32_BLOCK_ELEMENTS;
    rmsTiling.loopRound = 1U;
    rmsTiling.tailBshLength = 0U;
    rmsTiling.inputTailPos = elements;
    rmsTiling.tailBsLength = 0U;

    if (hasGamma != 0U) {
      AscendC::RmsNorm<DataType, false, HAS_GAMMA_RMSNORM_CONFIG>(yLocal, xLocal, gammaLocal, sharedTmp,
                                                                  static_cast<DataType>(epsilon), rmsTiling);
    } else {
      AscendC::RmsNorm<DataType, false, NO_GAMMA_RMSNORM_CONFIG>(yLocal, xLocal, gammaLocal, sharedTmp,
                                                                 static_cast<DataType>(epsilon), rmsTiling);
    }
    pipe_barrier(PIPE_ALL);
    DataCopy(yGm[offset], yLocal, elements);
    pipe_barrier(PIPE_ALL);
  }

  TPipe pipe;
  TBuf<> xBuffer;
  TBuf<> gammaBuffer;
  TBuf<> yBuffer;
  TBuf<TPosition::VECCALC> sharedTmpBuffer;
  GlobalTensor<DataType> xGm;
  GlobalTensor<DataType> gammaGm;
  GlobalTensor<DataType> yGm;
  uint64_t originM = 0;
  uint64_t originK = 0;
  float epsilon = 0.0f;
  float reciprocalOfHLength = 0.0f;
  uint32_t hasGamma = 0U;
  uint64_t blockM = 0;
  uint32_t splitM = 0U;
  uint32_t splitK = 0U;
  uint32_t loopK = 0U;
  uint32_t tailK = 0U;
  uint32_t reduceSplitK = 0U;
  uint32_t reduceLoopK = 0U;
  uint32_t reduceTailK = 0U;
  uint32_t tileRows = 0U;
};

extern "C" __aicore__ void ms_rms_norm_impl(GM_ADDR x, GM_ADDR w, GM_ADDR y, GM_ADDR workspace, uint64_t originM,
                                            uint64_t originK, float epsilon, float reciprocalOfHLength,
                                            uint32_t hasGamma, uint64_t blockM, uint32_t splitM, uint32_t splitK,
                                            uint32_t loopK, uint32_t tailK, uint32_t reduceSplitK, uint32_t reduceLoopK,
                                            uint32_t reduceTailK) {
  (void)workspace;
  KernelMsRmsNorm op;
  op.Init(x, w, y, originM, originK, epsilon, reciprocalOfHLength, hasGamma, blockM, splitM, splitK, loopK, tailK,
          reduceSplitK, reduceLoopK, reduceTailK);
  op.Process();
}
