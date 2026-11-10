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

/* fused_causal_conv_silu.cpp FusedCausalConvSilu device entry (ascend910b, AIV vector only). */

#include "kernel_operator.h"  // NOLINT(build/include_subdir): AscendC op headers are included relative to the op directory
#include "fused_causal_conv_silu_tiling_data.h"  // NOLINT(build/include_subdir): AscendC op headers are included relative to the op directory

using namespace AscendC;  // NOLINT(build/namespaces): AscendC kernels are written inside the AscendC namespace

namespace {

constexpr int64_t TILE_C = FUSED_CONV_TILE_C;
constexpr int64_t TR = FUSED_CONV_TR;
constexpr int64_t WIN = TR + 3;
constexpr int64_t ALIGN = 32;
constexpr int64_t BYTES_BF16 = 2;    // sizeof(bfloat16) as int64_t to keep offset arithmetic
constexpr int64_t BYTES_F32 = 4;     // sizeof(float)    as int64_t to keep offset arithmetic
constexpr int64_t NUM_TAP_ROWS = 4;  // conv tap rows (wB/wF per tap)
constexpr int64_t VAL_ELEMS = 8;     // seq_len int32 values loaded from GM

__aicore__ inline void SyncMte2ToV(TPipe *p) {
  int32_t id = static_cast<int32_t>(p->FetchEventID(HardEvent::MTE2_V));
  SetFlag<HardEvent::MTE2_V>(id);
  WaitFlag<HardEvent::MTE2_V>(id);
}
__aicore__ inline void SyncMte2ToS(TPipe *p) {
  int32_t id = static_cast<int32_t>(p->FetchEventID(HardEvent::MTE2_S));
  SetFlag<HardEvent::MTE2_S>(id);
  WaitFlag<HardEvent::MTE2_S>(id);
}
__aicore__ inline void SyncMte2ToMte3(TPipe *p) {
  int32_t id = static_cast<int32_t>(p->FetchEventID(HardEvent::MTE2_MTE3));
  SetFlag<HardEvent::MTE2_MTE3>(id);
  WaitFlag<HardEvent::MTE2_MTE3>(id);
}
__aicore__ inline void SyncVToMte3(TPipe *p) {
  int32_t id = static_cast<int32_t>(p->FetchEventID(HardEvent::V_MTE3));
  SetFlag<HardEvent::V_MTE3>(id);
  WaitFlag<HardEvent::V_MTE3>(id);
}
__aicore__ inline void SyncMte3ToMte2(TPipe *p) {
  int32_t id = static_cast<int32_t>(p->FetchEventID(HardEvent::MTE3_MTE2));
  SetFlag<HardEvent::MTE3_MTE2>(id);
  WaitFlag<HardEvent::MTE3_MTE2>(id);
}
__aicore__ inline void SyncMte3ToV(TPipe *p) {
  int32_t id = static_cast<int32_t>(p->FetchEventID(HardEvent::MTE3_V));
  SetFlag<HardEvent::MTE3_V>(id);
  WaitFlag<HardEvent::MTE3_V>(id);
}

struct UbScratch {
  TBuf<TPosition::VECCALC> buf;
  LocalTensor<bfloat16_t> wB[NUM_TAP_ROWS];
  LocalTensor<float> wF[NUM_TAP_ROWS];
  LocalTensor<bfloat16_t> biasB;
  LocalTensor<float> biasF;
  LocalTensor<float> acc;
  LocalTensor<bfloat16_t> yRound;
  LocalTensor<float> xF2;
  LocalTensor<float> tmpF;
  LocalTensor<float> tExp;
  LocalTensor<float> tDen;
  LocalTensor<float> onesF;
  LocalTensor<float> zerosF;
  LocalTensor<float> tSig;
  LocalTensor<float> yF;
  LocalTensor<bfloat16_t> stateRow;
  LocalTensor<int32_t> val;
  uint32_t offXB;
  uint32_t offXF;
  uint32_t offYS;
};

__aicore__ inline LocalTensor<bfloat16_t> RowB(UbScratch &ub, uint32_t base, int64_t r) {
  return ub.buf.GetWithOffset<bfloat16_t>(static_cast<uint32_t>(TILE_C),
                                          base + static_cast<uint32_t>(r * TILE_C * BYTES_BF16));
}
__aicore__ inline LocalTensor<float> RowF(UbScratch &ub, uint32_t base, int64_t r) {
  return ub.buf.GetWithOffset<float>(static_cast<uint32_t>(TILE_C),
                                     base + static_cast<uint32_t>(r * TILE_C * BYTES_F32));
}

__aicore__ inline void InitScratch(TPipe *pipe, UbScratch &ub) {
  auto align = [](int64_t b) { return (b + ALIGN - 1) / ALIGN * ALIGN; };
  int64_t o = 0;
  int64_t offsW[NUM_TAP_ROWS];
  int64_t offsWF[NUM_TAP_ROWS];
  for (int j = 0; j < NUM_TAP_ROWS; j++) {
    offsW[j] = o;
    o = align(o + TILE_C * BYTES_BF16);
  }
  for (int j = 0; j < NUM_TAP_ROWS; j++) {
    offsWF[j] = o;
    o = align(o + TILE_C * BYTES_F32);
  }
  int64_t oBiasB = o;
  o = align(o + TILE_C * BYTES_BF16);
  int64_t oBiasF = o;
  o = align(o + TILE_C * BYTES_F32);
  int64_t oAcc = o;
  o = align(o + TILE_C * BYTES_F32);
  int64_t oYR = o;
  o = align(o + TILE_C * BYTES_BF16);
  int64_t oXF2 = o;
  o = align(o + TILE_C * BYTES_F32);
  int64_t oTmp = o;
  o = align(o + TILE_C * BYTES_F32);
  int64_t oExp = o;
  o = align(o + TILE_C * BYTES_F32);
  int64_t oDen = o;
  o = align(o + TILE_C * BYTES_F32);
  int64_t oOne = o;
  o = align(o + TILE_C * BYTES_F32);
  int64_t oZero = o;
  o = align(o + TILE_C * BYTES_F32);
  int64_t oSig = o;
  o = align(o + TILE_C * BYTES_F32);
  int64_t oYF = o;
  o = align(o + TILE_C * BYTES_F32);
  int64_t oStR = o;
  o = align(o + TILE_C * BYTES_BF16);
  int64_t oVal = o;
  o = align(o + VAL_ELEMS * BYTES_F32);
  int64_t oXB = o;
  o = align(o + WIN * TILE_C * BYTES_BF16);
  int64_t oXF = o;
  o = align(o + WIN * TILE_C * BYTES_F32);
  int64_t oYS = o;
  o = align(o + TR * TILE_C * BYTES_BF16);
  pipe->InitBuffer(ub.buf, o);
  for (int j = 0; j < NUM_TAP_ROWS; j++) {
    ub.wB[j] = ub.buf.GetWithOffset<bfloat16_t>(static_cast<uint32_t>(TILE_C), static_cast<uint32_t>(offsW[j]));
    ub.wF[j] = ub.buf.GetWithOffset<float>(static_cast<uint32_t>(TILE_C), static_cast<uint32_t>(offsWF[j]));
  }
  ub.biasB = ub.buf.GetWithOffset<bfloat16_t>(static_cast<uint32_t>(TILE_C), static_cast<uint32_t>(oBiasB));
  ub.biasF = ub.buf.GetWithOffset<float>(static_cast<uint32_t>(TILE_C), static_cast<uint32_t>(oBiasF));
  ub.acc = ub.buf.GetWithOffset<float>(static_cast<uint32_t>(TILE_C), static_cast<uint32_t>(oAcc));
  ub.yRound = ub.buf.GetWithOffset<bfloat16_t>(static_cast<uint32_t>(TILE_C), static_cast<uint32_t>(oYR));
  ub.xF2 = ub.buf.GetWithOffset<float>(static_cast<uint32_t>(TILE_C), static_cast<uint32_t>(oXF2));
  ub.tmpF = ub.buf.GetWithOffset<float>(static_cast<uint32_t>(TILE_C), static_cast<uint32_t>(oTmp));
  ub.tExp = ub.buf.GetWithOffset<float>(static_cast<uint32_t>(TILE_C), static_cast<uint32_t>(oExp));
  ub.tDen = ub.buf.GetWithOffset<float>(static_cast<uint32_t>(TILE_C), static_cast<uint32_t>(oDen));
  ub.onesF = ub.buf.GetWithOffset<float>(static_cast<uint32_t>(TILE_C), static_cast<uint32_t>(oOne));
  ub.zerosF = ub.buf.GetWithOffset<float>(static_cast<uint32_t>(TILE_C), static_cast<uint32_t>(oZero));
  ub.tSig = ub.buf.GetWithOffset<float>(static_cast<uint32_t>(TILE_C), static_cast<uint32_t>(oSig));
  ub.yF = ub.buf.GetWithOffset<float>(static_cast<uint32_t>(TILE_C), static_cast<uint32_t>(oYF));
  ub.stateRow = ub.buf.GetWithOffset<bfloat16_t>(static_cast<uint32_t>(TILE_C), static_cast<uint32_t>(oStR));
  ub.val = ub.buf.GetWithOffset<int32_t>(VAL_ELEMS, static_cast<uint32_t>(oVal));
  ub.offXB = static_cast<uint32_t>(oXB);
  ub.offXF = static_cast<uint32_t>(oXF);
  ub.offYS = static_cast<uint32_t>(oYS);
}

}  // namespace

extern "C" __global__ __aicore__ void fused_causal_conv_silu(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR seq_len,
                                                             GM_ADDR y, GM_ADDR conv_state, GM_ADDR workspaceGM,
                                                             GM_ADDR tilingGM) {
  REGISTER_TILING_DEFAULT(FusedCausalConvSiluTilingData);
  GET_TILING_DATA(tiling, tilingGM);
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if ASCEND_IS_AIV {
    (void)workspaceGM;
    const int64_t T = tiling.T;
    const int64_t C = tiling.C;
    const int64_t S = tiling.S;
    const int64_t K = tiling.K;
    const int64_t cores = tiling.usedCoreNum;
    const int64_t bid = static_cast<int64_t>(GetBlockIdx());
    const int64_t nTiles = C / TILE_C;
    const int64_t nChunks = (T + TR - 1) / TR;
    const int64_t nUnits = nTiles * nChunks;

    GlobalTensor<bfloat16_t> xGm, wGm, bGm, yGm, stGm;
    GlobalTensor<int32_t> valGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(x), T * S);
    wGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(weight), K * C);
    bGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(bias), C);
    valGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(seq_len), VAL_ELEMS);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(y), T * C);
    stGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(conv_state), K * C);

    TPipe pipe;
    UbScratch ub;
    InitScratch(&pipe, ub);

    DataCopy(ub.val, valGm, VAL_ELEMS);
    SyncMte2ToS(&pipe);

    const int64_t rlRaw = static_cast<int64_t>(ub.val.GetValue(0));
    const int64_t realLen = rlRaw < 1 ? 1 : (rlRaw > T ? T : rlRaw);
    Duplicate(ub.onesF, 1.0F, static_cast<int32_t>(TILE_C));
    PipeBarrier<PIPE_V>();
    Duplicate(ub.zerosF, 0.0F, static_cast<int32_t>(TILE_C));
    PipeBarrier<PIPE_V>();

    for (int64_t u = bid; u < nUnits; u += cores) {
      const int64_t tile = u / nChunks;
      const int64_t chunk = u % nChunks;
      const int64_t c0 = tile * TILE_C;
      const int64_t t0 = chunk * TR;
      const int64_t rows = (T - t0) < TR ? (T - t0) : TR;
      const int64_t win = rows + 3;

      for (int64_t j = 0; j < K; j++) {
        DataCopy(ub.wB[j], wGm[j * C + c0], static_cast<uint32_t>(TILE_C));
      }
      DataCopy(ub.biasB, bGm[c0], static_cast<uint32_t>(TILE_C));
      if (t0 == 0) {
        for (int64_t r = 0; r < 3; r++) {
          Duplicate(RowB(ub, ub.offXB, r), static_cast<bfloat16_t>(0), static_cast<int32_t>(TILE_C));
          PipeBarrier<PIPE_V>();
        }
      }
      const int64_t r0 = (t0 == 0) ? 3 : 0;
      for (int64_t r = r0; r < win; r++) {
        DataCopy(RowB(ub, ub.offXB, r), xGm[(t0 - 3 + r) * S + c0], static_cast<uint32_t>(TILE_C));
      }
      SyncMte2ToV(&pipe);

      for (int64_t j = 0; j < K; j++) {
        Cast(ub.wF[j], ub.wB[j], RoundMode::CAST_NONE, static_cast<int32_t>(TILE_C));
        PipeBarrier<PIPE_V>();
      }
      Cast(ub.biasF, ub.biasB, RoundMode::CAST_NONE, static_cast<int32_t>(TILE_C));
      PipeBarrier<PIPE_V>();
      for (int64_t r = 0; r < win; r++) {
        Cast(RowF(ub, ub.offXF, r), RowB(ub, ub.offXB, r), RoundMode::CAST_NONE, static_cast<int32_t>(TILE_C));
        PipeBarrier<PIPE_V>();
      }

      for (int64_t i = 0; i < rows; i++) {
        const LocalTensor<float> x0 = RowF(ub, ub.offXF, i);
        const LocalTensor<float> x1 = RowF(ub, ub.offXF, i + 1);
        const LocalTensor<float> x2 = RowF(ub, ub.offXF, i + 2);
        const LocalTensor<float> x3 = RowF(ub, ub.offXF, i + 3);
        Mul(ub.acc, x0, ub.wF[0], static_cast<int32_t>(TILE_C));
        PipeBarrier<PIPE_V>();
        Mul(ub.tmpF, x1, ub.wF[1], static_cast<int32_t>(TILE_C));
        PipeBarrier<PIPE_V>();
        Add(ub.acc, ub.acc, ub.tmpF, static_cast<int32_t>(TILE_C));
        PipeBarrier<PIPE_V>();
        Mul(ub.tmpF, x2, ub.wF[2], static_cast<int32_t>(TILE_C));
        PipeBarrier<PIPE_V>();
        Add(ub.acc, ub.acc, ub.tmpF, static_cast<int32_t>(TILE_C));
        PipeBarrier<PIPE_V>();
        Mul(ub.tmpF, x3, ub.wF[3], static_cast<int32_t>(TILE_C));
        PipeBarrier<PIPE_V>();
        Add(ub.acc, ub.acc, ub.tmpF, static_cast<int32_t>(TILE_C));
        PipeBarrier<PIPE_V>();
        Add(ub.acc, ub.acc, ub.biasF, static_cast<int32_t>(TILE_C));
        PipeBarrier<PIPE_V>();

        Cast(ub.yRound, ub.acc, RoundMode::CAST_RINT, static_cast<int32_t>(TILE_C));
        PipeBarrier<PIPE_V>();
        Cast(ub.xF2, ub.yRound, RoundMode::CAST_NONE, static_cast<int32_t>(TILE_C));
        PipeBarrier<PIPE_V>();

        Sub(ub.tmpF, ub.zerosF, ub.xF2, static_cast<int32_t>(TILE_C));
        PipeBarrier<PIPE_V>();
        Exp(ub.tExp, ub.tmpF, static_cast<int32_t>(TILE_C));
        PipeBarrier<PIPE_V>();
        Add(ub.tDen, ub.tExp, ub.onesF, static_cast<int32_t>(TILE_C));
        PipeBarrier<PIPE_V>();
        Div(ub.tSig, ub.onesF, ub.tDen, static_cast<int32_t>(TILE_C));
        PipeBarrier<PIPE_V>();
        Mul(ub.yF, ub.xF2, ub.tSig, static_cast<int32_t>(TILE_C));
        PipeBarrier<PIPE_V>();
        Cast(RowB(ub, ub.offYS, i), ub.yF, RoundMode::CAST_RINT, static_cast<int32_t>(TILE_C));
        PipeBarrier<PIPE_V>();
      }
      SyncVToMte3(&pipe);

      for (int64_t i = 0; i < rows; i++) {
        DataCopy(yGm[(t0 + i) * C + c0], RowB(ub, ub.offYS, i), static_cast<uint32_t>(TILE_C));
      }
      SyncMte3ToV(&pipe);
      SyncMte3ToMte2(&pipe);

      if (chunk == 0) {
        for (int64_t j = 0; j < K; j++) {
          const int64_t row = (realLen - K + j) < 0 ? 0 : (realLen - K + j);
          DataCopy(ub.stateRow, xGm[row * S + c0], static_cast<uint32_t>(TILE_C));
          SyncMte2ToMte3(&pipe);
          DataCopy(stGm[j * C + c0], ub.stateRow, static_cast<uint32_t>(TILE_C));
          SyncMte3ToMte2(&pipe);
        }
      }
    }
  }
}
