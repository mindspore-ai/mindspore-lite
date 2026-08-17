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

#ifndef CHUNK_GATED_DELTA_RULE_KERNEL_H_
#define CHUNK_GATED_DELTA_RULE_KERNEL_H_

#include "kernel_operator.h"                     // NOLINT(build/include_subdir)
#include "chunk_gated_delta_rule_tiling_data.h"  // NOLINT(build/include_subdir)

using namespace AscendC;  // NOLINT(build/namespaces)

// NOTE: CGDR + helpers intentionally live in the GLOBAL namespace (not a user
// namespace). On some CANN toolchains the op-kernel autogen expands the tiling
// macros (REGISTER_TILING_DEFAULT / GET_TILING_DATA) and, when a user namespace
// is open, emits the tiling-data type / tiling-key as <ns>::-qualified symbols
// that then fail to resolve ("unknown type 'ChunkGatedDeltaRuleTilingData'",
// "undeclared identifier '..._tilingkey'") and abort the device-kernel compile.
// Keeping everything global matches the CANN sample-operator convention and
// removes the namespace as an autogen variable. Each op is its own translation
// unit, so global symbols here do not collide with other ops.
constexpr uint64_t BUFFER_NUM = 1;
constexpr uint64_t FP16_NUM_PER_BLOCK = 16;
constexpr uint64_t FP32_NUM_PER_BLOCK = 8;
constexpr int64_t BLOCK_BYTES = 32;
constexpr uint64_t CUBE_STAGE_SLOT_BYTES = 64 * 1024;
constexpr uint32_t CUBE_STAGE_SLOT_COUNT = 2;
// Number of Taylor-series terms used by ScalarExp to approximate exp on a scalar.
constexpr int kExpTaylorTerms = 12;

template <typename T>
__aicore__ inline void CopyToGm(GlobalTensor<T> dstGm, LocalTensor<T> inLocal, DataCopyExtParams copyParamsIn) {
  int64_t elem = copyParamsIn.blockLen / sizeof(T);
  int64_t numPerBlock = BLOCK_BYTES / sizeof(T);
  int64_t alignElem = AlignUp(elem, numPerBlock);
  if (likely(alignElem == elem)) {
    DataCopyParams copyParams = {static_cast<uint16_t>(copyParamsIn.blockCount),
                                 static_cast<uint16_t>(alignElem / numPerBlock), 0, 0};
    DataCopy(dstGm, inLocal, copyParams);
  } else {
    DataCopyParams copyParams = {1, static_cast<uint16_t>(alignElem / numPerBlock), 0, 0};
    for (uint32_t i = 0; i < copyParamsIn.blockCount; i++) {
      DataCopy(dstGm[i * elem], inLocal[i * alignElem], copyParams);
      PipeBarrier<PIPE_MTE3>();
    }
  }
}

struct CGDRInitParams {
  GM_ADDR query;
  GM_ADDR key;
  GM_ADDR value;
  GM_ADDR beta;
  GM_ADDR initialState;
  GM_ADDR actualSeqLengths;
  GM_ADDR gOptional;
  GM_ADDR attnOut;
  GM_ADDR finalState;
  GM_ADDR workspace;
};

template <typename inType, typename outType, uint32_t kSpecializedDk>
class ChunkGatedDeltaRule {
 public:
  __aicore__ inline explicit ChunkGatedDeltaRule(const ChunkGatedDeltaRuleTilingData *tilingData) {
    B_ = tilingData->b;
    T_ = tilingData->t;
    NK_ = tilingData->hqk;
    realK_ = tilingData->dk;
    NV_ = tilingData->hv;
    realV_ = tilingData->dv;
    scale_ = tilingData->scaleValue;
    chunkSize_ = tilingData->chunkSize;
    numChunks_ = tilingData->numChunks;
    hasGamma_ = tilingData->hasGamma;
    vStep_ = tilingData->vStep;
    restUbSize_ = tilingData->ubRestBytes;
    uint32_t naturalAlignK = Ceil(tilingData->dk, FP16_NUM_PER_BLOCK) * FP16_NUM_PER_BLOCK;
    alignK_ = (kSpecializedDk != 0 && tilingData->dk <= kSpecializedDk) ? kSpecializedDk : naturalAlignK;
    stateStrideK_ = alignK_;
    stateWorkspaceStrideV_ = Ceil(tilingData->dv, FP32_NUM_PER_BLOCK) * FP32_NUM_PER_BLOCK;
    load_ = 0;
    usedblk_ = 0;
    avgload_ = 0;
    // Set in Init()/InitLocalBuffers(); zero-initialized here so every member is
    // defined before first use (the kernel constructor runs before those calls).
    pipe_ = nullptr;
    vStepAligned_ = 0;
    blockIdx_ = 0;
    workspaceAddr_ = nullptr;
  }

  __aicore__ inline void Init(const CGDRInitParams &initParams, TPipe *pipe) {
    uint64_t blockDim = GetBlockNum();
    blockIdx_ = GetBlockIdx();
    if (blockIdx_ >= blockDim) {
      return;
    }
    pipe_ = pipe;
    SetGlobalTensors(initParams);
    InitLocalBuffers();
  }

  __aicore__ inline void SetGlobalTensors(const CGDRInitParams &initParams) {
    queryGm_.SetGlobalBuffer((__gm__ inType *)initParams.query);
    keyGm_.SetGlobalBuffer((__gm__ inType *)initParams.key);
    valueGm_.SetGlobalBuffer((__gm__ inType *)initParams.value);
    betaGm_.SetGlobalBuffer((__gm__ inType *)initParams.beta);
    initStateGm_.SetGlobalBuffer((__gm__ inType *)initParams.initialState);
    actualSeqLengthsGm_.SetGlobalBuffer((__gm__ int32_t *)initParams.actualSeqLengths);
    if (hasGamma_ != 0) {
      gGm_.SetGlobalBuffer((__gm__ float *)initParams.gOptional);
    }
    finalStateGm_.SetGlobalBuffer((__gm__ outType *)initParams.finalState);
    attnOutGm_.SetGlobalBuffer((__gm__ outType *)initParams.attnOut);
    stateWorkspaceGm_.SetGlobalBuffer((__gm__ float *)initParams.workspace);
    workspaceAddr_ = initParams.workspace;
  }

  __aicore__ inline void InitLocalBuffers() {
    uint32_t cs = chunkSize_;
    uint32_t ak = alignK_;
    uint32_t avStepAligned = Ceil(vStep_, FP32_NUM_PER_BLOCK) * FP32_NUM_PER_BLOCK;
    vStepAligned_ = avStepAligned;

    // stateOutQueue: max of state tile [dk, vStep] and compact chunk output [cs, vStep].
    uint32_t stateTileBytes = stateStrideK_ * avStepAligned * sizeof(outType);
    uint32_t chunkOutBytes = cs * avStepAligned * sizeof(outType);
    uint32_t outQueueBytes = stateTileBytes;
    if (chunkOutBytes > outQueueBytes) outQueueBytes = chunkOutBytes;
    pipe_->InitBuffer(stateOutQueue_, BUFFER_NUM, outQueueBytes);

    // tmpBuff layout (all FP32):
    //   chunkKFp32:      cs * alignK
    //   kCumdecayFp32:   cs * alignK
    //   decayMaskFp32:   cs * cs
    //   chunkVFp32:      cs * avStepAligned
    //   chunkScoresFp32 / stateInFp32:
    //     single V tile: overlapped (state is loaded only after scores' last use)
    //     multiple tiles: separate (scores must remain live across every tile)
    //   chunkAttnOutFp32: cs * avStepAligned
    //   gCumsumFp32:     cs
    //   deltaFp32:       max(dkAlignedFp32, cs, vStepAligned)
    //   dotProductFp32:  max(dkAlignedFp32, cs)
    //   expGCumFp32:     cs  (precomputed exp(gCumsum))
    pipe_->InitBuffer(tmpBuff, restUbSize_);
    uint32_t off = 0;

    chunkKFp32 = tmpBuff.GetWithOffset<float>(cs * ak, off);
    off += cs * ak * sizeof(float);

    kCumdecayFp32 = tmpBuff.GetWithOffset<float>(cs * ak, off);
    off += cs * ak * sizeof(float);

    decayMaskFp32 = tmpBuff.GetWithOffset<float>(cs * cs, off);
    off += cs * cs * sizeof(float);

    chunkVFp32 = tmpBuff.GetWithOffset<float>(cs * avStepAligned, off);
    off += cs * avStepAligned * sizeof(float);

    if (vStep_ >= realV_ || ShouldSplitVTiles()) {
      uint32_t overlapSize = (cs * cs > stateStrideK_ * avStepAligned) ? cs * cs : stateStrideK_ * avStepAligned;
      chunkScoresFp32 = tmpBuff.GetWithOffset<float>(overlapSize, off);
      stateInFp32 = tmpBuff.GetWithOffset<float>(stateStrideK_ * avStepAligned, off);
      off += overlapSize * sizeof(float);
    } else {
      chunkScoresFp32 = tmpBuff.GetWithOffset<float>(cs * cs, off);
      off += cs * cs * sizeof(float);
      stateInFp32 = tmpBuff.GetWithOffset<float>(stateStrideK_ * avStepAligned, off);
      off += stateStrideK_ * avStepAligned * sizeof(float);
    }

    chunkAttnOutFp32 = tmpBuff.GetWithOffset<float>(cs * avStepAligned, off);
    off += cs * avStepAligned * sizeof(float);

    gCumsumFp32 = tmpBuff.GetWithOffset<float>(cs, off);
    off += cs * sizeof(float);

    uint32_t dotProductSize = (stateStrideK_ > cs) ? stateStrideK_ : cs;
    uint32_t deltaSize = (dotProductSize > avStepAligned) ? dotProductSize : avStepAligned;
    deltaFp32 = tmpBuff.GetWithOffset<float>(deltaSize, off);
    off += deltaSize * sizeof(float);

    dotProductFp32 = tmpBuff.GetWithOffset<float>(dotProductSize, off);
    off += dotProductSize * sizeof(float);

    expGCumFp32 = tmpBuff.GetWithOffset<float>(cs, off);
    off += cs * sizeof(float);

    betaFp32 = tmpBuff.GetWithOffset<float>(cs, off);
  }

  __aicore__ inline void ComputeAvgload() {
    uint64_t realT = 0;
    for (uint64_t batch_i = 0; batch_i < B_; batch_i++) {
      int32_t seqLen = actualSeqLengthsGm_.GetValue(batch_i);
      if (seqLen > 0) {
        realT += static_cast<uint64_t>(seqLen);
      }
    }
    uint64_t workItemsPerHead = ShouldSplitVTiles() ? Ceil(realV_, vStep_) : 1;
    avgload_ = Ceil(realT * NV_ * workItemsPerHead, GetBlockNum());
  }

  __aicore__ inline void Process() {
    ComputeAvgload();
    int32_t seq0 = 0;
    for (uint64_t batch_i = 0; batch_i < B_; batch_i++) {
      int32_t seqLen = actualSeqLengthsGm_.GetValue(batch_i);
      int32_t seq1 = seq0 + seqLen;
      if (seqLen <= 0) {
        seq0 = seq1;
        continue;
      }

      if (ShouldSplitVTiles()) {
        uint32_t vTileCount = Ceil(realV_, vStep_);
        for (uint64_t head_i = 0; head_i < NV_; head_i++) {
          for (uint32_t vTileIdx = 0; vTileIdx < vTileCount; vTileIdx++) {
            if (!IsCurrentBlock(seqLen)) continue;
            ProcessHeadVTile(seq0, seq1, head_i, batch_i, vTileIdx);
          }
        }
      } else {
        for (uint64_t head_i = 0; head_i < NV_; head_i++) {
          if (!IsCurrentBlock(seqLen)) continue;
          ProcessHead(seq0, seq1, head_i, batch_i);
        }
      }
      seq0 = seq1;
    }
  }

 private:
  __aicore__ inline bool ShouldSplitVTiles() { return realV_ > vStep_; }

  __aicore__ inline __gm__ uint8_t *GetCubeStageBase(uint32_t slot) {
    uint64_t stateWorkspaceElements = static_cast<uint64_t>(B_) * NV_ * realK_ * stateWorkspaceStrideV_;
    uint64_t stageByteOffset =
      stateWorkspaceElements * sizeof(float) +
      (static_cast<uint64_t>(blockIdx_) * CUBE_STAGE_SLOT_COUNT + slot) * CUBE_STAGE_SLOT_BYTES;
    return workspaceAddr_ + stageByteOffset;
  }

  __aicore__ inline bool IsCubeFastPath(uint32_t chunkLen, uint32_t avFp32) {
    if constexpr (kSpecializedDk != 0) {
      return realK_ <= kSpecializedDk && chunkLen == 64 && vStepAligned_ == kSpecializedDk && avFp32 == kSpecializedDk;
    }
    return false;
  }

  __aicore__ inline bool IsFusedValueOutputCubeFastPath(uint32_t chunkLen, uint32_t avFp32) {
    return IsCubeFastPath(chunkLen, avFp32);
  }

  // Copy a row-major GM slab to a zero-padded UB matrix using dav-2002
  // supported primitives. Aligned rows use one strided standard DataCopy;
  // non-aligned rows copy complete 32-byte blocks and fill at most 15 tail
  // elements through Scalar. The remaining 64/80/96/128 Cube tile stays zero.
  __aicore__ inline void LoadPaddedRows(LocalTensor<float> dst, LocalTensor<inType> staging, GlobalTensor<inType> src,
                                        uint64_t srcOffset, uint32_t rows, uint32_t gmRowElements) {
    Duplicate(staging, static_cast<inType>(0), rows * alignK_);
    PipeBarrier<PIPE_V>();
    event_t vectorToMte2 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(vectorToMte2);
    WaitFlag<HardEvent::V_MTE2>(vectorToMte2);

    uint32_t fullBlocks = realK_ / FP16_NUM_PER_BLOCK;
    uint32_t tailElements = realK_ % FP16_NUM_PER_BLOCK;
    uint32_t srcRowBlocks = gmRowElements / FP16_NUM_PER_BLOCK;
    uint32_t dstRowBlocks = alignK_ / FP16_NUM_PER_BLOCK;
    bool canUseStridedFullBlocks = fullBlocks > 0 && gmRowElements % FP16_NUM_PER_BLOCK == 0 &&
                                   alignK_ % FP16_NUM_PER_BLOCK == 0 && srcRowBlocks >= fullBlocks &&
                                   dstRowBlocks >= fullBlocks && srcRowBlocks - fullBlocks <= 65535U &&
                                   dstRowBlocks - fullBlocks <= 65535U;
    if (likely(canUseStridedFullBlocks)) {
      DataCopyParams copyParams{static_cast<uint16_t>(rows), static_cast<uint16_t>(fullBlocks),
                                static_cast<uint16_t>(srcRowBlocks - fullBlocks),
                                static_cast<uint16_t>(dstRowBlocks - fullBlocks)};
      DataCopy(staging, src[srcOffset], copyParams);
    } else if (fullBlocks > 0) {
      DataCopyParams rowParams{1, static_cast<uint16_t>(fullBlocks), 0, 0};
      for (uint32_t row = 0; row < rows; ++row) {
        DataCopy(staging[row * alignK_], src[srcOffset + static_cast<uint64_t>(row) * gmRowElements], rowParams);
      }
    }

    if (unlikely(tailElements != 0)) {
      uint32_t tailOffset = fullBlocks * FP16_NUM_PER_BLOCK;
      bool copyTailAsBlock = tailElements > FP16_NUM_PER_BLOCK / 2 && rows > 1 &&
                             gmRowElements % FP16_NUM_PER_BLOCK == 0 && alignK_ % FP16_NUM_PER_BLOCK == 0 &&
                             srcRowBlocks > 0 && dstRowBlocks > 0 && srcRowBlocks - 1 <= 65535U &&
                             dstRowBlocks - 1 <= 65535U;
      if (copyTailAsBlock) {
        // Every row except the last one may safely read through the short tail
        // into the following token/head. Clear those extra elements below.
        DataCopyParams tailParams{static_cast<uint16_t>(rows - 1), 1, static_cast<uint16_t>(srcRowBlocks - 1),
                                  static_cast<uint16_t>(dstRowBlocks - 1)};
        DataCopy(staging[tailOffset], src[srcOffset + tailOffset], tailParams);
      }
      event_t mte2ToScalar = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_S));
      SetFlag<HardEvent::MTE2_S>(mte2ToScalar);
      WaitFlag<HardEvent::MTE2_S>(mte2ToScalar);
      uint32_t scalarStartRow = copyTailAsBlock ? rows - 1 : 0;
      if (copyTailAsBlock) {
        for (uint32_t row = 0; row + 1 < rows; ++row) {
          for (uint32_t col = realK_; col < tailOffset + FP16_NUM_PER_BLOCK; ++col) {
            staging.SetValue(row * alignK_ + col, static_cast<inType>(0));
          }
        }
      }
      for (uint32_t row = scalarStartRow; row < rows; ++row) {
        uint64_t gmRowOffset = srcOffset + static_cast<uint64_t>(row) * gmRowElements;
        for (uint32_t col = tailOffset; col < realK_; ++col) {
          staging.SetValue(row * alignK_ + col, src.GetValue(gmRowOffset + col));
        }
      }
      TQueSync<PIPE_S, PIPE_V> scalarToVector;
      scalarToVector.SetFlag(0);
      scalarToVector.WaitFlag(0);
    } else {
      event_t mte2ToVector = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
      SetFlag<HardEvent::MTE2_V>(mte2ToVector);
      WaitFlag<HardEvent::MTE2_V>(mte2ToVector);
    }
    Cast(dst, staging, RoundMode::CAST_NONE, rows * alignK_);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline bool IsCurrentBlock(int32_t seqlen) {
    load_ += seqlen;
    bool ret = (blockIdx_ == usedblk_ && seqlen > 0);
    if (load_ >= avgload_) {
      load_ = 0;
      usedblk_++;
    }
    return ret;
  }

  __aicore__ inline float LoadG(int32_t t, uint64_t head_i) {
    if (hasGamma_ == 0) {
      return 0.0f;
    }
    return gGm_.GetValue(t * NV_ + head_i);
  }

  __aicore__ inline float LoadBeta(int32_t t, uint64_t head_i) {
    inType bVal = betaGm_.GetValue(t * NV_ + head_i);
    return static_cast<float>(bVal);
  }

  __aicore__ inline float ScalarExp(float val) {
    float absVal = (val >= 0.0f) ? val : -val;
    int k = 0;
    float reduced = absVal;
    while (reduced > 1.0f) {
      reduced *= 0.5f;
      k++;
    }
    float result = 1.0f;
    float term = 1.0f;
    for (int i = 1; i <= kExpTaylorTerms; i++) {
      term *= reduced / static_cast<float>(i);
      result += term;
    }
    for (int i = 0; i < k; i++) {
      result *= result;
    }
    if (val < 0.0f) {
      result = 1.0f / result;
    }
    return result;
  }

  // Multiply a contiguous FP32 row pair on the vector pipe, then preserve the
  // partial sums on the vector pipe. Each 64-element repeat writes one scalar
  // at an eight-element stride; only those partials are accumulated by scalar.
  __aicore__ inline float DotFp32(LocalTensor<float> lhs, LocalTensor<float> rhs, uint32_t count) {
    Mul(dotProductFp32, lhs, rhs, count);
    constexpr uint32_t kReduceRepeatElements = 64;
    constexpr uint32_t kReduceDstStride = 8;
    constexpr uint32_t kReduceSrcRepeatStride = 8;
    uint32_t fullRepeats = count / kReduceRepeatElements;
    uint32_t tail = count % kReduceRepeatElements;
    if (fullRepeats > 0) {
      WholeReduceSum(dotProductFp32, dotProductFp32, kReduceRepeatElements, fullRepeats, kReduceDstStride, 1,
                     kReduceSrcRepeatStride);
    }
    TQueSync<PIPE_V, PIPE_S> mulSync;
    mulSync.SetFlag(0);
    mulSync.WaitFlag(0);
    float sum = 0.0f;
    for (uint32_t i = 0; i < fullRepeats; i++) {
      sum += dotProductFp32.GetValue(i * kReduceDstStride);
    }
    uint32_t tailOffset = fullRepeats * kReduceRepeatElements;
    for (uint32_t i = 0; i < tail; i++) {
      sum += dotProductFp32.GetValue(tailOffset + i);
    }
    return sum;
  }

  template <uint32_t kBlock>
  __aicore__ inline void LoadMatmulBlockA(LocalTensor<half> a1Local, LocalTensor<half> a2Local,
                                          GlobalTensor<half> src) {
    DataCopy(a1Local, src, Nd2NzParams{1, kBlock, kBlock, 0, kBlock, kBlock, 1, 0});
    SetFlag<HardEvent::MTE2_MTE1>(0);
    WaitFlag<HardEvent::MTE2_MTE1>(0);
    for (uint32_t blockIdx = 0; blockIdx < kBlock / 16; ++blockIdx) {
      LoadData(a2Local[blockIdx * kBlock * 16], a1Local[blockIdx * 512 / sizeof(half)],
               LoadData2DParams{0, kBlock / 16, kBlock / 16, 0, 0, false, 0});
    }
  }

  template <uint32_t kBlock>
  __aicore__ inline void LoadMatmulBlockB(LocalTensor<half> b1Local, LocalTensor<half> b2Local,
                                          GlobalTensor<half> src) {
    DataCopy(b1Local, src, Nd2NzParams{1, kBlock, kBlock, 0, kBlock, kBlock, 1, 0});
    SetFlag<HardEvent::MTE2_MTE1>(0);
    WaitFlag<HardEvent::MTE2_MTE1>(0);
    for (uint32_t blockIdx = 0; blockIdx < kBlock / 16; ++blockIdx) {
      LoadData(b2Local[blockIdx * kBlock * 16], b1Local[blockIdx * 512 / sizeof(half)],
               LoadData2DParams{0, kBlock / 16, kBlock / 16, 0, 0, true, 0});
    }
  }

  template <uint32_t kBlock>
  __aicore__ inline void MmadMatmulBlock(LocalTensor<float> c1Local, LocalTensor<half> a2Local,
                                         LocalTensor<half> b2Local, bool init) {
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    Mmad(c1Local, a2Local, b2Local, MmadParams{kBlock, kBlock, kBlock, 0, false, init});
    SetFlag<HardEvent::M_MTE1>(0);
    WaitFlag<HardEvent::M_MTE1>(0);
  }

  // Execute one compensated 32x32 FP32 matrix product on Cube. Both operands
  // are split into FP16 high/low parts and accumulated in FP32 L0C. Retaining
  // every cross term keeps the block solve close to the original FP32
  // recurrence while moving the dense cross-block work off the scalar pipe.
  template <uint32_t kBlock>
  __aicore__ inline void MatmulBlockFp32Compensated(LocalTensor<float> srcA, LocalTensor<float> srcB,
                                                    LocalTensor<float> residualA, LocalTensor<float> residualB,
                                                    LocalTensor<float> dst) {
    constexpr uint32_t kMatrixElements = kBlock * kBlock;
    __gm__ uint8_t *stageBase = GetCubeStageBase(0);
    GlobalTensor<half> aHiGm;
    GlobalTensor<half> bHiGm;
    GlobalTensor<half> aLoGm;
    GlobalTensor<half> bLoGm;
    aHiGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(stageBase), kMatrixElements);
    bHiGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(stageBase + kMatrixElements * sizeof(half)), kMatrixElements);
    aLoGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(stageBase + 2 * kMatrixElements * sizeof(half)),
                          kMatrixElements);
    bLoGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(stageBase + 3 * kMatrixElements * sizeof(half)),
                          kMatrixElements);

    StageHalfWithResidual(srcA, residualA, aHiGm, aLoGm, kMatrixElements);
    StageHalfWithResidual(srcB, residualB, bHiGm, bLoGm, kMatrixElements);
    PipeBarrier<PIPE_ALL>();

    LocalMemAllocator<Hardware::L1> l1Allocator;
    LocalMemAllocator<Hardware::L0A> l0aAllocator;
    LocalMemAllocator<Hardware::L0B> l0bAllocator;
    LocalMemAllocator<Hardware::L0C> l0cAllocator;
    LocalTensor<half> a1Local = l1Allocator.Alloc<TPosition::A1, half>(kMatrixElements);
    LocalTensor<half> b1Local = l1Allocator.Alloc<TPosition::B1, half>(kMatrixElements);
    LocalTensor<half> a2Local = l0aAllocator.Alloc<TPosition::A2, half>(kMatrixElements);
    LocalTensor<half> b2Local = l0bAllocator.Alloc<TPosition::B2, half>(kMatrixElements);
    LocalTensor<float> c1Local = l0cAllocator.Alloc<TPosition::CO1, float>(kMatrixElements);

    LoadMatmulBlockA<kBlock>(a1Local, a2Local, aHiGm);
    LoadMatmulBlockB<kBlock>(b1Local, b2Local, bHiGm);
    MmadMatmulBlock<kBlock>(c1Local, a2Local, b2Local, true);
    LoadMatmulBlockA<kBlock>(a1Local, a2Local, aLoGm);
    MmadMatmulBlock<kBlock>(c1Local, a2Local, b2Local, false);
    LoadMatmulBlockA<kBlock>(a1Local, a2Local, aHiGm);
    LoadMatmulBlockB<kBlock>(b1Local, b2Local, bLoGm);
    MmadMatmulBlock<kBlock>(c1Local, a2Local, b2Local, false);
    LoadMatmulBlockA<kBlock>(a1Local, a2Local, aLoGm);
    MmadMatmulBlock<kBlock>(c1Local, a2Local, b2Local, false);

    LocalTensor<float> cNz = residualA;
    DataCopyParams cCopyParams{static_cast<uint16_t>(kBlock / 16), static_cast<uint16_t>(kBlock / 16), 0, 0};
    DataCopyEnhancedParams cCopyEnhanced;
    cCopyEnhanced.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    DataCopy(cNz, c1Local, cCopyParams, cCopyEnhanced);
    PipeBarrier<PIPE_ALL>();
    constexpr uint16_t kNdBlockLen = 16 * sizeof(float) / 32;
    constexpr uint16_t kNzSrcStride = (kBlock / 16 * 16 * 16 - 16) * sizeof(float) / 32;
    DataCopyParams nzToNdParams{static_cast<uint16_t>(kBlock / 16), kNdBlockLen, kNzSrcStride, 0};
    for (uint32_t row = 0; row < kBlock; ++row) {
      DataCopy(dst[row * kBlock], cNz[row * 16], nzToNdParams);
    }
    PipeBarrier<PIPE_ALL>();
  }

  // Solve one 16x16 diagonal block entirely on the Vector pipe. For every
  // source row k, Gather extracts A[k+1:, k], Brcb expands its coefficients,
  // and one strided MulAddDst applies the outer-product update
  //
  //   U[k+1:, :] += A[k+1:, k] * U[k, :].
  //
  // The only scalar work left is the fixed 15-step dependency schedule and
  // setup of the 16 diagonal/offset entries; the O(block^3) arithmetic and
  // all coefficient application stay on PIPE_V.
  __aicore__ inline void ComputeRecursiveAttnDiagonalBlock(LocalTensor<float> &buf, uint32_t rowBase, uint32_t ld) {
    constexpr uint32_t kBlock = 16;
    constexpr uint32_t kMatrixElements = kBlock * kBlock;
    LocalTensor<float> inverse = chunkVFp32;
    LocalTensor<float> coefficientBlocks = chunkVFp32[kMatrixElements];
    LocalTensor<float> column = deltaFp32;
    LocalTensor<uint32_t> gatherOffsets = dotProductFp32.template ReinterpretCast<uint32_t>();

    Duplicate(inverse, 0.0f, kMatrixElements);
    TQueSync<PIPE_V, PIPE_S> initVectorToScalar;
    initVectorToScalar.SetFlag(0);
    initVectorToScalar.WaitFlag(0);
    for (uint32_t i = 0; i < kBlock; ++i) {
      inverse.SetValue(i * kBlock + i, 1.0f);
      gatherOffsets.SetValue(i, i * ld * sizeof(float));
    }
    TQueSync<PIPE_S, PIPE_V> initScalarToVector;
    initScalarToVector.SetFlag(0);
    initScalarToVector.WaitFlag(0);

    for (uint32_t k = 0; k + 1 < kBlock; ++k) {
      uint32_t validRows = kBlock - k - 1;
      Gather(column, buf[(rowBase + k + 1) * ld + rowBase + k], gatherOffsets, static_cast<uint32_t>(0), validRows);
      PipeBarrier<PIPE_V>();
      uint8_t brcbRepeats = static_cast<uint8_t>(Ceil(validRows, FP32_NUM_PER_BLOCK));
      Brcb(coefficientBlocks, column, brcbRepeats, BrcbRepeatParams{1, 8});
      PipeBarrier<PIPE_V>();
      MulAddDst(inverse[(k + 1) * kBlock], coefficientBlocks, inverse[k * kBlock], k + 1,
                static_cast<uint8_t>(validRows), BinaryRepeatParams{1, 0, 1, kBlock / FP32_NUM_PER_BLOCK, 1, 0});
      PipeBarrier<PIPE_V>();
    }

    for (uint32_t row = 0; row < kBlock; ++row) {
      Muls(buf[(rowBase + row) * ld + rowBase], inverse[row * kBlock], 1.0f, kBlock);
    }
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void ComputeRecursiveAttnBlocked32(LocalTensor<float> &buf, uint32_t rowBase, uint32_t ld) {
    constexpr uint32_t kBlock = 16;
    constexpr uint32_t kMatrixElements = kBlock * kBlock;
    ComputeRecursiveAttnDiagonalBlock(buf, rowBase, ld);
    ComputeRecursiveAttnDiagonalBlock(buf, rowBase + kBlock, ld);

    LocalTensor<float> srcA = chunkVFp32;
    LocalTensor<float> srcB = chunkVFp32[kMatrixElements];
    LocalTensor<float> residualA = chunkVFp32[2 * kMatrixElements];
    LocalTensor<float> residualB = chunkVFp32[3 * kMatrixElements];
    LocalTensor<float> product = chunkVFp32[4 * kMatrixElements];

    for (uint32_t row = 0; row < kBlock; ++row) {
      Muls(srcA[row * kBlock], buf[(rowBase + kBlock + row) * ld + rowBase], 1.0f, kBlock);
      Muls(srcB[row * kBlock], buf[(rowBase + row) * ld + rowBase], 1.0f, kBlock);
    }
    PipeBarrier<PIPE_V>();
    MatmulBlockFp32Compensated<kBlock>(srcA, srcB, residualA, residualB, product);

    for (uint32_t row = 0; row < kBlock; ++row) {
      Muls(srcA[row * kBlock], buf[(rowBase + kBlock + row) * ld + rowBase + kBlock], 1.0f, kBlock);
    }
    PipeBarrier<PIPE_V>();
    MatmulBlockFp32Compensated<kBlock>(srcA, product, residualA, residualB, product);

    for (uint32_t row = 0; row < kBlock; ++row) {
      Muls(buf[(rowBase + kBlock + row) * ld + rowBase], product[row * kBlock], 1.0f, kBlock);
    }
    PipeBarrier<PIPE_V>();
  }

  // For a 64x64 strict-lower A, invert I-A with a 2x2 block factorization:
  //   U00=(I-A00)^-1, U11=(I-A11)^-1,
  //   U10=U11*A10*U00.
  // The diagonal solves remain FP32 Vector recurrences; compensated Cube
  // products handle the dense cross block without weakening FP32 stability.
  __aicore__ inline void ComputeRecursiveAttnBlocked64(LocalTensor<float> &buf, uint32_t ld) {
    constexpr uint32_t kBlock = 32;
    constexpr uint32_t kMatrixElements = kBlock * kBlock;
    ComputeRecursiveAttnBlocked32(buf, 0, ld);
    ComputeRecursiveAttnBlocked32(buf, kBlock, ld);

    LocalTensor<float> srcA = chunkVFp32;
    LocalTensor<float> srcB = chunkVFp32[kMatrixElements];
    LocalTensor<float> residualA = chunkVFp32[2 * kMatrixElements];
    LocalTensor<float> residualB = chunkVFp32[3 * kMatrixElements];
    LocalTensor<float> product = chunkVFp32[4 * kMatrixElements];

    for (uint32_t row = 0; row < kBlock; ++row) {
      Muls(srcA[row * kBlock], buf[(row + kBlock) * ld], 1.0f, kBlock);
      Muls(srcB[row * kBlock], buf[row * ld], 1.0f, kBlock);
    }
    PipeBarrier<PIPE_V>();
    MatmulBlockFp32Compensated<kBlock>(srcA, srcB, residualA, residualB, product);

    for (uint32_t row = 0; row < kBlock; ++row) {
      Muls(srcA[row * kBlock], buf[(row + kBlock) * ld + kBlock], 1.0f, kBlock);
    }
    PipeBarrier<PIPE_V>();
    MatmulBlockFp32Compensated<kBlock>(srcA, product, residualA, residualB, product);

    for (uint32_t row = 0; row < kBlock; ++row) {
      Muls(buf[(row + kBlock) * ld], product[row * kBlock], 1.0f, kBlock);
    }
    PipeBarrier<PIPE_V>();
  }

  // Compute recursive intra-chunk attn matrix in-place.
  // buf is stored with leading dimension ld (= chunkSize_), and only [chunkLen, chunkLen] is valid.
  // buf is lower-triangular (excluding diagonal), with upper tri (incl diagonal) = 0.
  __aicore__ inline void ComputeRecursiveAttn(LocalTensor<float> &buf, uint32_t chunkLen, uint32_t ld) {
    constexpr uint32_t kBlockedScratchMinV = (kSpecializedDk == 0) ? 128 : kSpecializedDk;
    if (likely(chunkLen == 64 && ld == 64 && vStepAligned_ >= kBlockedScratchMinV)) {
      ComputeRecursiveAttnBlocked64(buf, ld);
      return;
    }
    for (uint32_t i = 1; i < chunkLen; i++) {
      for (uint32_t k = 0; k < i; k++) {
        deltaFp32.SetValue(k, buf.GetValue(i * ld + k));
      }
      LocalTensor<float> outRow = buf[i * ld];
      // For each k, update the prefix j<k in one vector operation:
      // out[j] += original_row[k] * recursive_row_k[j].
      // For every j this visits k=j+1..i-1 in the same order as the scalar
      // recurrence, while element k itself remains untouched.
      for (uint32_t k = 1; k < i; k++) {
        float coeff = deltaFp32.GetValue(k);
        Axpy(outRow, buf[k * ld], coeff, k);
        PipeBarrier<PIPE_V>();
      }
    }
    TQueSync<PIPE_V, PIPE_S> recursiveSync;
    recursiveSync.SetFlag(0);
    recursiveSync.WaitFlag(0);
  }

  __aicore__ inline void WriteAttnTileToGm(int32_t t_start, uint32_t chunkLen, uint64_t head_i, uint32_t v_i,
                                           uint32_t curV, uint32_t avFp32) {
    uint32_t totalElem = chunkLen * avFp32;
    uint32_t alignedElem = Ceil(totalElem, FP32_NUM_PER_BLOCK) * FP32_NUM_PER_BLOCK;
    Muls(chunkAttnOutFp32, chunkAttnOutFp32, 1.0f, alignedElem);
    PipeBarrier<PIPE_V>();
    LocalTensor<outType> outLocal = stateOutQueue_.AllocTensor<outType>();
    Cast(outLocal, chunkAttnOutFp32, RoundMode::CAST_NONE, alignedElem);
    PipeBarrier<PIPE_V>();
    stateOutQueue_.EnQue<outType>(outLocal);
    outLocal = stateOutQueue_.DeQue<outType>();
    constexpr uint32_t outElemPerBlock = BLOCK_BYTES / sizeof(outType);
    uint64_t gmRowStride = static_cast<uint64_t>(NV_) * realV_;
    bool canUse2DStore = curV % outElemPerBlock == 0 && avFp32 % outElemPerBlock == 0 &&
                         gmRowStride % outElemPerBlock == 0 && (avFp32 - curV) / outElemPerBlock <= UINT16_MAX &&
                         (gmRowStride - curV) / outElemPerBlock <= UINT16_MAX;
    uint64_t outOff = (static_cast<uint64_t>(t_start) * NV_ + head_i) * realV_ + v_i;
    if (likely(canUse2DStore)) {
      DataCopyParams outParams{static_cast<uint16_t>(chunkLen), static_cast<uint16_t>(curV / outElemPerBlock),
                               static_cast<uint16_t>((avFp32 - curV) / outElemPerBlock),
                               static_cast<uint16_t>((gmRowStride - curV) / outElemPerBlock)};
      DataCopy(attnOutGm_[outOff], outLocal, outParams);
    } else {
      for (uint32_t i = 0; i < chunkLen; i++) {
        uint64_t rowOff = (static_cast<uint64_t>(t_start + i) * NV_ + head_i) * realV_ + v_i;
        DataCopyExtParams outParams{1, static_cast<uint32_t>(curV * sizeof(outType)), 0, 0, 0};
        CopyToGm(attnOutGm_[rowOff], outLocal[i * avFp32], outParams);
        PipeBarrier<PIPE_MTE3>();
      }
    }
    stateOutQueue_.FreeTensor(outLocal);
  }

  __aicore__ inline void ProcessHead(int32_t seq0, int32_t seq1, uint64_t head_i, uint64_t batch_i) {
    uint64_t nvPerNk = (NV_ >= NK_) ? (NV_ / NK_) : 1;
    uint64_t qkHead = head_i / nvPerNk;
    uint64_t stateBaseOffset = (batch_i * NV_ + head_i) * realK_ * realV_;
    uint64_t workspaceStateBaseOffset = (batch_i * NV_ + head_i) * realK_ * stateWorkspaceStrideV_;
    uint32_t cs = chunkSize_;
    int32_t totalLen = seq1 - seq0;
    uint32_t numC = Ceil(static_cast<uint32_t>(totalLen), cs);
    for (uint32_t c = 0; c < numC; c++) {
      ProcessChunk(seq0, seq1, head_i, c, qkHead, stateBaseOffset, workspaceStateBaseOffset, c + 1 == numC);
    }
  }

  __aicore__ inline void ProcessHeadVTile(int32_t seq0, int32_t seq1, uint64_t head_i, uint64_t batch_i,
                                          uint32_t vTileIdx) {
    uint64_t nvPerNk = (NV_ >= NK_) ? (NV_ / NK_) : 1;
    uint64_t qkHead = head_i / nvPerNk;
    uint64_t stateBaseOffset = (batch_i * NV_ + head_i) * realK_ * realV_;
    uint64_t workspaceStateBaseOffset = (batch_i * NV_ + head_i) * realK_ * stateWorkspaceStrideV_;
    uint32_t cs = chunkSize_;
    int32_t totalLen = seq1 - seq0;
    uint32_t numC = Ceil(static_cast<uint32_t>(totalLen), cs);
    for (uint32_t c = 0; c < numC; c++) {
      ProcessChunkVTile(seq0, seq1, head_i, c, qkHead, stateBaseOffset, workspaceStateBaseOffset, c + 1 == numC,
                        vTileIdx);
    }
  }

  // ---- per-chunk phases (ProcessHead delegates one chunk to ProcessChunk) ----

  __aicore__ inline void ProcessChunk(int32_t seq0, int32_t seq1, uint64_t head_i, uint32_t c, uint64_t qkHead,
                                      uint64_t stateBaseOffset, uint64_t workspaceStateBaseOffset, bool isLastChunk) {
    uint32_t cs = chunkSize_;
    uint32_t avFp32 = vStepAligned_;
    int32_t totalLen = seq1 - seq0;
    int32_t t_start = seq0 + c * cs;
    uint32_t chunkLen =
      (static_cast<uint32_t>(totalLen) - c * cs >= cs) ? cs : static_cast<uint32_t>(totalLen) - c * cs;

    LoadChunkKey(qkHead, t_start, chunkLen);
    LoadChunkCoefficients(t_start, head_i, chunkLen);
    ComputeKBeta(chunkLen);
    PrepareDecayAndExp(chunkLen);
    PrepareAttnQueryCache(qkHead, t_start, chunkLen);
    ComputeAttnMatrix(qkHead, t_start, chunkLen);
    ComputeKCumdecay(chunkLen);
    ProcessVTiles(t_start, chunkLen, head_i, qkHead, avFp32, stateBaseOffset, workspaceStateBaseOffset, c, isLastChunk);
  }

  __aicore__ inline void ProcessChunkVTile(int32_t seq0, int32_t seq1, uint64_t head_i, uint32_t c, uint64_t qkHead,
                                           uint64_t stateBaseOffset, uint64_t workspaceStateBaseOffset,
                                           bool isLastChunk, uint32_t vTileIdx) {
    uint32_t cs = chunkSize_;
    uint32_t avFp32 = vStepAligned_;
    int32_t totalLen = seq1 - seq0;
    int32_t t_start = seq0 + c * cs;
    uint32_t chunkLen =
      (static_cast<uint32_t>(totalLen) - c * cs >= cs) ? cs : static_cast<uint32_t>(totalLen) - c * cs;

    LoadChunkKey(qkHead, t_start, chunkLen);
    LoadChunkCoefficients(t_start, head_i, chunkLen);
    ComputeKBeta(chunkLen);
    PrepareDecayAndExp(chunkLen);
    PrepareAttnQueryCache(qkHead, t_start, chunkLen);
    ComputeAttnMatrix(qkHead, t_start, chunkLen);
    ComputeKCumdecay(chunkLen);
    uint32_t v_i = vTileIdx * vStep_;
    uint32_t curV = (v_i + vStep_ > realV_) ? realV_ - v_i : vStep_;
    ProcessOneVTile(t_start, chunkLen, head_i, qkHead, avFp32, stateBaseOffset, workspaceStateBaseOffset, c,
                    isLastChunk, v_i, curV);
  }

  // Phase 1: load K for this chunk into chunkKFp32.
  __aicore__ inline void LoadChunkKey(uint64_t qkHead, int32_t t_start, uint32_t chunkLen) {
    uint64_t stagingCapacity = static_cast<uint64_t>(2) * chunkSize_ * vStepAligned_;
    uint64_t keyElements = static_cast<uint64_t>(chunkLen) * alignK_;
    uint64_t gmRowStride = static_cast<uint64_t>(NK_) * realK_;
    bool canUseDma = chunkLen <= UINT16_MAX && keyElements <= stagingCapacity;
    if (likely(canUseDma)) {
      LocalTensor<inType> keyLocal = chunkVFp32.template ReinterpretCast<inType>();
      uint64_t qkOff = (static_cast<uint64_t>(t_start) * NK_ + qkHead) * realK_;
      LoadPaddedRows(chunkKFp32, keyLocal, keyGm_, qkOff, chunkLen, static_cast<uint32_t>(gmRowStride));
      return;
    }

    for (uint32_t i = 0; i < chunkLen; i++) {
      int32_t t = t_start + i;
      uint64_t qkOff = (static_cast<uint64_t>(t) * NK_ + qkHead) * realK_;
      for (uint32_t j = 0; j < realK_; j++) {
        chunkKFp32.SetValue(i * alignK_ + j, static_cast<float>(keyGm_.GetValue(qkOff + j)));
      }
    }
  }

  __aicore__ inline void BuildCoefficientOffsets(LocalTensor<uint32_t> offsets, uint32_t chunkLen) {
    LocalTensor<int32_t> signedOffsets = offsets.template ReinterpretCast<int32_t>();
    CreateVecIndex<int32_t>(signedOffsets, 0, chunkLen);
    PipeBarrier<PIPE_V>();
    Muls(signedOffsets, signedOffsets, static_cast<int32_t>(NV_ * sizeof(float)), chunkLen);
    PipeBarrier<PIPE_V>();
  }

  // Inclusive prefix sum for at most one 64-row chunk. Ping-ponging between
  // two UB tensors avoids the loop-carried scalar GetValue/SetValue chain.
  __aicore__ inline void PrefixSumChunkG(uint32_t chunkLen) {
    LocalTensor<float> src = gCumsumFp32;
    LocalTensor<float> dst = deltaFp32;
    bool resultInGCumsum = true;
    for (uint32_t offset = 1; offset < chunkLen; offset <<= 1) {
      Muls(dst, src, 1.0f, chunkLen);
      PipeBarrier<PIPE_V>();
      Add(dst[offset], src[offset], src, chunkLen - offset);
      PipeBarrier<PIPE_V>();
      LocalTensor<float> tmp = src;
      src = dst;
      dst = tmp;
      resultInGCumsum = !resultInGCumsum;
    }
    if (!resultInGCumsum) {
      Muls(gCumsumFp32, src, 1.0f, chunkLen);
      PipeBarrier<PIPE_V>();
    }
  }

  // Load the complete [chunk, Hv] beta/g slabs once, then gather this head
  // from UB. The same byte-offset vector is reused for beta and g.
  __aicore__ inline void LoadChunkCoefficients(int32_t t_start, uint64_t head_i, uint32_t chunkLen) {
    uint64_t slabElements = static_cast<uint64_t>(chunkLen) * NV_;
    uint64_t slabCapacity = static_cast<uint64_t>(chunkSize_) * vStepAligned_;
    LocalTensor<uint32_t> coefficientOffsets = dotProductFp32.template ReinterpretCast<uint32_t>();
    bool offsetsReady = false;
    bool canUseBetaDma = slabElements <= slabCapacity && slabElements * sizeof(inType) % BLOCK_BYTES == 0;
    if (likely(canUseBetaDma)) {
      LocalTensor<inType> betaLocal = chunkVFp32.template ReinterpretCast<inType>();
      event_t vectorToMte2 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE2));
      SetFlag<HardEvent::V_MTE2>(vectorToMte2);
      WaitFlag<HardEvent::V_MTE2>(vectorToMte2);
      DataCopy(betaLocal, betaGm_[static_cast<uint64_t>(t_start) * NV_], slabElements);
      event_t mte2ToVector = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
      SetFlag<HardEvent::MTE2_V>(mte2ToVector);
      WaitFlag<HardEvent::MTE2_V>(mte2ToVector);
      Cast(chunkVFp32, betaLocal, RoundMode::CAST_NONE, slabElements);
      PipeBarrier<PIPE_V>();
      BuildCoefficientOffsets(coefficientOffsets, chunkLen);
      offsetsReady = true;
      Gather(betaFp32, chunkVFp32, coefficientOffsets, static_cast<uint32_t>(head_i * sizeof(float)), chunkLen);
      PipeBarrier<PIPE_V>();
    } else {
      for (uint32_t i = 0; i < chunkLen; ++i) {
        betaFp32.SetValue(i, LoadBeta(t_start + i, head_i));
      }
    }

    float gSum = 0.0f;
    bool canUseGDma = hasGamma_ != 0 && slabElements <= slabCapacity && slabElements * sizeof(float) % BLOCK_BYTES == 0;
    if (likely(canUseGDma)) {
      event_t vectorToMte2 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE2));
      SetFlag<HardEvent::V_MTE2>(vectorToMte2);
      WaitFlag<HardEvent::V_MTE2>(vectorToMte2);
      DataCopy(chunkVFp32, gGm_[static_cast<uint64_t>(t_start) * NV_], slabElements);
      event_t mte2ToVector = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
      SetFlag<HardEvent::MTE2_V>(mte2ToVector);
      WaitFlag<HardEvent::MTE2_V>(mte2ToVector);
      if (!offsetsReady) {
        BuildCoefficientOffsets(coefficientOffsets, chunkLen);
      }
      Gather(gCumsumFp32, chunkVFp32, coefficientOffsets, static_cast<uint32_t>(head_i * sizeof(float)), chunkLen);
      PipeBarrier<PIPE_V>();
      PrefixSumChunkG(chunkLen);
    } else if (hasGamma_ == 0) {
      Duplicate(gCumsumFp32, 0.0f, chunkLen);
      PipeBarrier<PIPE_V>();
    } else {
      for (uint32_t i = 0; i < chunkLen; ++i) {
        gSum += LoadG(t_start + i, head_i);
        gCumsumFp32.SetValue(i, gSum);
      }
    }
  }

  // Phase 2: k_beta = K * beta -> kCumdecayFp32. Brcb expands each
  // per-row beta into one 32-byte block; high-dimensional Mul then broadcasts
  // that block across every K block while advancing one beta per row.
  __aicore__ inline void ComputeKBeta(uint32_t chunkLen) {
    TQueSync<PIPE_S, PIPE_V> inputSync;
    inputSync.SetFlag(0);
    inputSync.WaitFlag(0);

    LocalTensor<float> betaBlocks = chunkVFp32;
    uint8_t brcbRepeats = static_cast<uint8_t>(Ceil(chunkLen, FP32_NUM_PER_BLOCK));
    Brcb(betaBlocks, betaFp32, brcbRepeats, BrcbRepeatParams{1, 8});
    PipeBarrier<PIPE_V>();

    // The padded K columns must remain exact zeros because all Cube products
    // consume the complete bucket width.
    Duplicate(kCumdecayFp32, 0.0f, chunkLen * alignK_);
    PipeBarrier<PIPE_V>();

    uint8_t rowStride = static_cast<uint8_t>(alignK_ / FP32_NUM_PER_BLOCK);
    BinaryRepeatParams mulParams{1, 1, 0, rowStride, rowStride, 1};
    uint32_t kOffset = 0;
    for (; kOffset + 64 <= realK_; kOffset += 64) {
      Mul(kCumdecayFp32[kOffset], chunkKFp32[kOffset], betaBlocks, static_cast<uint64_t>(64),
          static_cast<uint8_t>(chunkLen), mulParams);
    }
    if (kOffset < realK_) {
      Mul(kCumdecayFp32[kOffset], chunkKFp32[kOffset], betaBlocks, static_cast<uint64_t>(realK_ - kOffset),
          static_cast<uint8_t>(chunkLen), mulParams);
    }
    PipeBarrier<PIPE_V>();
    TQueSync<PIPE_V, PIPE_S> betaSync;
    betaSync.SetFlag(0);
    betaSync.WaitFlag(0);
  }

  // Phase 3: precompute exp(gCumsum) and decay_mask[i][j] = exp(g_cumsum[i] -
  // g_cumsum[j]).
  __aicore__ inline void PrepareDecayAndExp(uint32_t chunkLen) {
    uint32_t cs = chunkSize_;
    Muls(gCumsumFp32, gCumsumFp32, 1.0f, chunkLen);
    PipeBarrier<PIPE_V>();
    Exp(expGCumFp32, gCumsumFp32, chunkLen);
    PipeBarrier<PIPE_V>();

    // g is a log-decay and its prefix sum can easily be below -100 for real
    // Qwen activations. Factoring exp(g_i - g_j) into exp(g_i)*exp(-g_j)
    // overflows the second factor. Build all rows with high-dimensional Vector
    // instructions, clear the non-causal upper triangle before Exp, then issue
    // one strided Exp for the complete causal matrix.
    uint8_t rowStride = static_cast<uint8_t>(cs / FP32_NUM_PER_BLOCK);
    UnaryRepeatParams rowBroadcastParams{1, 1, rowStride, 0};
    Muls(decayMaskFp32, gCumsumFp32, -1.0f, static_cast<uint64_t>(chunkLen), static_cast<uint8_t>(chunkLen),
         rowBroadcastParams);
    PipeBarrier<PIPE_V>();

    LocalTensor<float> gBlocks = chunkVFp32;
    uint8_t brcbRepeats = static_cast<uint8_t>(Ceil(chunkLen, FP32_NUM_PER_BLOCK));
    Brcb(gBlocks, gCumsumFp32, brcbRepeats, BrcbRepeatParams{1, 8});
    PipeBarrier<PIPE_V>();
    BinaryRepeatParams rowAddParams{1, 1, 0, rowStride, rowStride, 1};
    Add(decayMaskFp32, decayMaskFp32, gBlocks, static_cast<uint64_t>(chunkLen), static_cast<uint8_t>(chunkLen),
        rowAddParams);
    PipeBarrier<PIPE_V>();

    for (uint32_t row = 0; row + 1 < chunkLen; ++row) {
      Duplicate(decayMaskFp32[row * cs + row + 1], 0.0f, chunkLen - row - 1);
    }
    PipeBarrier<PIPE_V>();
    UnaryRepeatParams rowExpParams{1, 1, rowStride, rowStride};
    Exp(decayMaskFp32, decayMaskFp32, static_cast<uint64_t>(chunkLen), static_cast<uint8_t>(chunkLen), rowExpParams);
    PipeBarrier<PIPE_V>();
    TQueSync<PIPE_V, PIPE_S> decaySync;
    decaySync.SetFlag(0);
    decaySync.WaitFlag(0);
  }

  __aicore__ inline bool CanCacheAttnQuery(uint32_t chunkLen) {
    uint64_t queryElements = static_cast<uint64_t>(chunkLen) * alignK_;
    uint64_t stagingCapacity = static_cast<uint64_t>(2) * chunkSize_ * vStepAligned_;
    uint64_t cacheCapacity = static_cast<uint64_t>(chunkSize_) * vStepAligned_;
    return chunkLen <= UINT16_MAX && queryElements <= stagingCapacity && queryElements <= cacheCapacity;
  }

  // chunkAttnOutFp32 is unused until V-tile processing. When it can hold a
  // full Q chunk, reuse chunkVFp32 as FP16 staging and cache Q*scale once for
  // the attention QK dot products.
  __aicore__ inline void PrepareAttnQueryCache(uint64_t qkHead, int32_t t_start, uint32_t chunkLen) {
    if (!CanCacheAttnQuery(chunkLen)) {
      return;
    }
    LocalTensor<inType> queryLocal = chunkVFp32.template ReinterpretCast<inType>();
    uint64_t qkOff = (static_cast<uint64_t>(t_start) * NK_ + qkHead) * realK_;
    LoadPaddedRows(chunkAttnOutFp32, queryLocal, queryGm_, qkOff, chunkLen, NK_ * realK_);
    Muls(chunkAttnOutFp32, chunkAttnOutFp32, scale_, chunkLen * alignK_);
    PipeBarrier<PIPE_V>();
  }

  // Phase 4: attn = -((k_beta @ K^T) * decay_mask) (lower tri), recursive accumulation, identity
  // diagonal; then attn_i = (Q @ K^T) * decay_mask overwrites decayMaskFp32 (lower tri + diag).
  __aicore__ inline void ComputeAttnMatrix(uint64_t qkHead, int32_t t_start, uint32_t chunkLen) {
    uint32_t cs = chunkSize_;
    if constexpr (kSpecializedDk != 0) {
      if (likely(realK_ <= kSpecializedDk && chunkLen == 64 && vStepAligned_ >= kSpecializedDk &&
                 CanCacheAttnQuery(chunkLen))) {
        ComputeAttnProductsCube<kSpecializedDk>();
        ComputeRecursiveAttn(chunkScoresFp32, chunkLen, cs);
        for (uint32_t i = 0; i < chunkLen; i++) {
          chunkScoresFp32.SetValue(i * cs + i, 1.0f);
        }
        return;
      }
    }
    for (uint32_t i = 1; i < chunkLen; i++) {
      for (uint32_t jj = 0; jj < i; jj++) {
        float dot = DotFp32(kCumdecayFp32[i * alignK_], chunkKFp32[jj * alignK_], realK_);
        chunkScoresFp32.SetValue(i * cs + jj, -dot * decayMaskFp32.GetValue(i * cs + jj));
      }
    }
    ComputeRecursiveAttn(chunkScoresFp32, chunkLen, cs);
    for (uint32_t i = 0; i < chunkLen; i++) {
      chunkScoresFp32.SetValue(i * cs + i, 1.0f);
    }
    if (CanCacheAttnQuery(chunkLen)) {
      for (uint32_t i = 0; i < chunkLen; i++) {
        LocalTensor<float> queryRow = chunkAttnOutFp32[i * alignK_];
        for (uint32_t jj = 0; jj <= i; jj++) {
          float dot = DotFp32(queryRow, chunkKFp32[jj * alignK_], realK_);
          float decay = decayMaskFp32.GetValue(i * cs + jj);
          decayMaskFp32.SetValue(i * cs + jj, dot * decay);
        }
      }
    } else {
      for (uint32_t i = 0; i < chunkLen; i++) {
        int32_t t_i = t_start + i;
        uint64_t qkOff_i = (static_cast<uint64_t>(t_i) * NK_ + qkHead) * realK_;
        for (uint32_t d = 0; d < realK_; d++) {
          deltaFp32.SetValue(d, static_cast<float>(queryGm_.GetValue(qkOff_i + d)) * scale_);
        }
        TQueSync<PIPE_S, PIPE_V> querySync;
        querySync.SetFlag(0);
        querySync.WaitFlag(0);
        for (uint32_t jj = 0; jj <= i; jj++) {
          float dot = DotFp32(deltaFp32, chunkKFp32[jj * alignK_], realK_);
          float decay = decayMaskFp32.GetValue(i * cs + jj);
          decayMaskFp32.SetValue(i * cs + jj, dot * decay);
        }
      }
    }
  }

  // Batch the two dense attention products on Cube while K^T remains resident
  // in L0B:
  //   scores = -(K*beta) @ K^T * decay
  //   attn_i = (Q*scale) @ K^T * decay.
  // This replaces 4096 row-pair DotFp32 reductions on the fixed 64x128 path.
  template <uint32_t kMatmulK>
  __aicore__ inline void ComputeAttnProductsCube() {
    constexpr uint32_t kMatmulM = 64;
    constexpr uint32_t kMatmulN = 64;
    constexpr uint32_t kAElements = kMatmulM * kMatmulK;
    constexpr uint32_t kBElements = kMatmulK * kMatmulN;
    constexpr uint32_t kCElements = kMatmulM * kMatmulN;
    __gm__ uint8_t *stageBase = GetCubeStageBase(0);
    GlobalTensor<half> aStageGm;
    GlobalTensor<half> bStageGm;
    GlobalTensor<half> aResidualStageGm;
    __gm__ uint8_t *nextStageBase = GetCubeStageBase(1);
    GlobalTensor<half> nextAStageGm;
    GlobalTensor<half> nextAResidualStageGm;
    aStageGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(stageBase), kAElements);
    bStageGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(stageBase + kAElements * sizeof(half)), kBElements);
    aResidualStageGm.SetGlobalBuffer(
      reinterpret_cast<__gm__ half *>(stageBase + (kAElements + kBElements) * sizeof(half)), kAElements);
    nextAStageGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(nextStageBase), kAElements);
    nextAResidualStageGm.SetGlobalBuffer(
      reinterpret_cast<__gm__ half *>(nextStageBase + (kAElements + kBElements) * sizeof(half)), kAElements);

    // Transpose the FP32 K cache through FP16 in UB; K originated as FP16, so
    // this conversion is exact and avoids another GM read of the source tensor.
    LocalTensor<half> keyLocal = stateOutQueue_.AllocTensor<half>();
    LocalTensor<half> keyTransposed = chunkVFp32.template ReinterpretCast<half>();
    Cast(keyLocal, chunkKFp32, RoundMode::CAST_NONE, kAElements);
    PipeBarrier<PIPE_V>();
    TransDataTo5HDParams transposeParams;
    transposeParams.repeatTimes = static_cast<uint8_t>(kMatmulK / 16);
    transposeParams.srcRepStride = 1;
    transposeParams.dstRepStride = kMatmulM;
    for (uint32_t block = 0; block < kMatmulM / 16; ++block) {
      LocalTensor<half> srcRows[16];
      LocalTensor<half> dstRows[16];
      for (uint32_t row = 0; row < 16; ++row) {
        srcRows[row] = keyLocal[block * 16 * kMatmulK + row * kMatmulK];
        dstRows[row] = keyTransposed[block * 16 + row * kMatmulM];
      }
      TransDataTo5HD<half>(dstRows, srcRows, transposeParams);
    }
    PipeBarrier<PIPE_V>();
    event_t vectorToMte3 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(vectorToMte3);
    WaitFlag<HardEvent::V_MTE3>(vectorToMte3);
    DataCopy(bStageGm, keyTransposed, kBElements);
    stateOutQueue_.FreeTensor(keyLocal);
    StageHalfWithResidual(kCumdecayFp32, chunkVFp32, aStageGm, aResidualStageGm, kAElements);
    PipeBarrier<PIPE_ALL>();

    LocalMemAllocator<Hardware::L1> l1Allocator;
    LocalMemAllocator<Hardware::L0A> l0aAllocator;
    LocalMemAllocator<Hardware::L0B> l0bAllocator;
    LocalMemAllocator<Hardware::L0C> l0cAllocator;
    LocalTensor<half> a1Local = l1Allocator.Alloc<TPosition::A1, half>(kAElements);
    LocalTensor<half> b1Local = l1Allocator.Alloc<TPosition::B1, half>(kBElements);
    LocalTensor<half> a2Local = l0aAllocator.Alloc<TPosition::A2, half>(kAElements);
    LocalTensor<half> b2Local = l0bAllocator.Alloc<TPosition::B2, half>(kBElements);
    LocalTensor<float> c1Local = l0cAllocator.Alloc<TPosition::CO1, float>(kCElements);

    DataCopy(b1Local, bStageGm, Nd2NzParams{1, kMatmulK, kMatmulN, 0, kMatmulN, kMatmulK, 1, 0});
    DataCopy(a1Local, aStageGm, Nd2NzParams{1, kMatmulM, kMatmulK, 0, kMatmulK, kMatmulM, 1, 0});
    SetFlag<HardEvent::MTE2_MTE1>(0);
    WaitFlag<HardEvent::MTE2_MTE1>(0);
    for (uint32_t mBlock = 0; mBlock < kMatmulM / 16; ++mBlock) {
      LoadData(a2Local[mBlock * kMatmulK * 16], a1Local[mBlock * 512 / sizeof(half)],
               LoadData2DParams{0, kMatmulK / 16, kMatmulM / 16, 0, 0, false, 0});
    }
    for (uint32_t kBlock = 0; kBlock < kMatmulK / 16; ++kBlock) {
      LoadData(b2Local[kBlock * kMatmulN * 16], b1Local[kBlock * 512 / sizeof(half)],
               LoadData2DParams{0, kMatmulN / 16, kMatmulK / 16, 0, 0, true, 0});
    }
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    Mmad(c1Local, a2Local, b2Local, MmadParams{kMatmulM, kMatmulN, kMatmulK, 0, false, true});

    SetFlag<HardEvent::M_MTE1>(0);
    WaitFlag<HardEvent::M_MTE1>(0);
    DataCopy(a1Local, aResidualStageGm, Nd2NzParams{1, kMatmulM, kMatmulK, 0, kMatmulK, kMatmulM, 1, 0});
    SetFlag<HardEvent::MTE2_MTE1>(0);
    WaitFlag<HardEvent::MTE2_MTE1>(0);
    for (uint32_t mBlock = 0; mBlock < kMatmulM / 16; ++mBlock) {
      LoadData(a2Local[mBlock * kMatmulK * 16], a1Local[mBlock * 512 / sizeof(half)],
               LoadData2DParams{0, kMatmulK / 16, kMatmulM / 16, 0, 0, false, 0});
    }
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    Mmad(c1Local, a2Local, b2Local, MmadParams{kMatmulM, kMatmulN, kMatmulK, 0, false, false});

    // Stage Q into a disjoint GM slot while the first compensated product is
    // executing. The MTE3->MTE2 dependency below prevents the next consumer
    // from observing a partially written slot.
    StageHalfWithResidual(chunkAttnOutFp32, chunkVFp32, nextAStageGm, nextAResidualStageGm, kAElements);
    CopyAttnCubeResult(c1Local, chunkScoresFp32);

    // Reuse resident K^T for QK^T. Q*scale was cached in
    // chunkAttnOutFp32 by PrepareAttnQueryCache.
    event_t stageReady = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(stageReady);
    WaitFlag<HardEvent::MTE3_MTE2>(stageReady);
    DataCopy(a1Local, nextAStageGm, Nd2NzParams{1, kMatmulM, kMatmulK, 0, kMatmulK, kMatmulM, 1, 0});
    SetFlag<HardEvent::MTE2_MTE1>(0);
    WaitFlag<HardEvent::MTE2_MTE1>(0);
    for (uint32_t mBlock = 0; mBlock < kMatmulM / 16; ++mBlock) {
      LoadData(a2Local[mBlock * kMatmulK * 16], a1Local[mBlock * 512 / sizeof(half)],
               LoadData2DParams{0, kMatmulK / 16, kMatmulM / 16, 0, 0, false, 0});
    }
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    Mmad(c1Local, a2Local, b2Local, MmadParams{kMatmulM, kMatmulN, kMatmulK, 0, false, true});
    SetFlag<HardEvent::M_MTE1>(0);
    WaitFlag<HardEvent::M_MTE1>(0);
    DataCopy(a1Local, nextAResidualStageGm, Nd2NzParams{1, kMatmulM, kMatmulK, 0, kMatmulK, kMatmulM, 1, 0});
    SetFlag<HardEvent::MTE2_MTE1>(0);
    WaitFlag<HardEvent::MTE2_MTE1>(0);
    for (uint32_t mBlock = 0; mBlock < kMatmulM / 16; ++mBlock) {
      LoadData(a2Local[mBlock * kMatmulK * 16], a1Local[mBlock * 512 / sizeof(half)],
               LoadData2DParams{0, kMatmulK / 16, kMatmulM / 16, 0, 0, false, 0});
    }
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    Mmad(c1Local, a2Local, b2Local, MmadParams{kMatmulM, kMatmulN, kMatmulK, 0, false, false});
    CopyAttnCubeResult(c1Local, chunkAttnOutFp32);

    Mul(chunkScoresFp32, chunkScoresFp32, decayMaskFp32, kCElements);
    PipeBarrier<PIPE_V>();
    Muls(chunkScoresFp32, chunkScoresFp32, -1.0f, kCElements);
    Mul(decayMaskFp32, chunkAttnOutFp32, decayMaskFp32, kCElements);
    PipeBarrier<PIPE_V>();
    for (uint32_t row = 0; row < kMatmulM; ++row) {
      Duplicate(chunkScoresFp32[row * kMatmulN + row], 0.0f, kMatmulN - row);
      if (row + 1 < kMatmulN) {
        Duplicate(decayMaskFp32[row * kMatmulN + row + 1], 0.0f, kMatmulN - row - 1);
      }
    }
    TQueSync<PIPE_V, PIPE_S> attentionSync;
    attentionSync.SetFlag(0);
    attentionSync.WaitFlag(0);
  }

  __aicore__ inline void CopyAttnCubeResult(LocalTensor<float> &c1Local, LocalTensor<float> dst) {
    constexpr uint32_t kMatmulM = 64;
    constexpr uint32_t kMatmulN = 64;
    SetFlag<HardEvent::M_MTE1>(0);
    WaitFlag<HardEvent::M_MTE1>(0);
    LocalTensor<float> cNz = chunkVFp32;
    DataCopyParams cCopyParams{static_cast<uint16_t>(kMatmulN / 16), static_cast<uint16_t>(kMatmulM / 16), 0, 0};
    DataCopyEnhancedParams cCopyEnhanced;
    cCopyEnhanced.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    DataCopy(cNz, c1Local, cCopyParams, cCopyEnhanced);
    PipeBarrier<PIPE_ALL>();
    constexpr uint16_t kNdBlockLen = 16 * sizeof(float) / 32;
    constexpr uint16_t kNzSrcStride = (kMatmulM / 16 * 16 * 16 - 16) * sizeof(float) / 32;
    DataCopyParams nzToNdParams{static_cast<uint16_t>(kMatmulN / 16), kNdBlockLen, kNzSrcStride, 0};
    for (uint32_t row = 0; row < kMatmulM; ++row) {
      DataCopy(dst[row * kMatmulN], cNz[row * 16], nzToNdParams);
    }
    PipeBarrier<PIPE_ALL>();
  }

  // Phase 5: k_cumdecay = attn @ (k_beta * exp(g_cumsum)); result stored back into chunkKFp32.
  __aicore__ inline void ComputeKCumdecay(uint32_t chunkLen) {
    uint32_t cs = chunkSize_;
    for (uint32_t i = 0; i < chunkLen; i++) {
      float gExp = expGCumFp32.GetValue(i);
      Muls(kCumdecayFp32[i * alignK_], kCumdecayFp32[i * alignK_], gExp, realK_);
    }
    TQueSync<PIPE_V, PIPE_S> decaySync;
    decaySync.SetFlag(0);
    decaySync.WaitFlag(0);
    if constexpr (kSpecializedDk != 0) {
      if (likely(realK_ <= kSpecializedDk && chunkLen == 64 && vStepAligned_ >= kSpecializedDk)) {
        ComputeKCumdecayCube<kSpecializedDk>();
        return;
      }
    }
    for (uint32_t i = 0; i < chunkLen; i++) {
      LocalTensor<float> outRow = chunkKFp32[i * alignK_];
      Duplicate(outRow, 0.0f, realK_);
      PipeBarrier<PIPE_V>();
      for (uint32_t k = 0; k <= i; k++) {
        float score = chunkScoresFp32.GetValue(i * cs + k);
        Axpy(outRow, kCumdecayFp32[k * alignK_], score, realK_);
        PipeBarrier<PIPE_V>();
      }
    }
    TQueSync<PIPE_V, PIPE_S> outputSync;
    outputSync.SetFlag(0);
    outputSync.WaitFlag(0);
  }

  // Fixed-shape k_cumdecay = attn @ (k_beta * exp(g)) as one
  // 64x64x128 Cube product, replacing the lower-triangle Axpy nest.
  template <uint32_t kMatmulN>
  __aicore__ inline void ComputeKCumdecayCube() {
    constexpr uint32_t kMatmulM = 64;
    constexpr uint32_t kMatmulK = 64;
    constexpr uint32_t kAElements = kMatmulM * kMatmulK;
    constexpr uint32_t kBElements = kMatmulK * kMatmulN;
    constexpr uint32_t kCElements = kMatmulM * kMatmulN;
    __gm__ uint8_t *stageBase = GetCubeStageBase(0);
    GlobalTensor<half> aStageGm;
    GlobalTensor<half> bStageGm;
    GlobalTensor<half> aResidualStageGm;
    aStageGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(stageBase), kAElements);
    bStageGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(stageBase + kAElements * sizeof(half)), kBElements);
    aResidualStageGm.SetGlobalBuffer(
      reinterpret_cast<__gm__ half *>(stageBase + (kAElements + kBElements) * sizeof(half)), kAElements);

    TQueSync<PIPE_S, PIPE_V> recursiveToVector;
    recursiveToVector.SetFlag(0);
    recursiveToVector.WaitFlag(0);
    Muls(chunkAttnOutFp32, chunkScoresFp32, 1.0f, kAElements);
    PipeBarrier<PIPE_V>();
    for (uint32_t row = 0; row + 1 < kMatmulM; ++row) {
      Duplicate(chunkAttnOutFp32[row * kMatmulK + row + 1], 0.0f, kMatmulK - row - 1);
    }
    PipeBarrier<PIPE_V>();
    StageHalfWithResidual(chunkAttnOutFp32, chunkVFp32, aStageGm, aResidualStageGm, kAElements);
    StageHalf(kCumdecayFp32, bStageGm, kBElements);
    PipeBarrier<PIPE_ALL>();

    LocalMemAllocator<Hardware::L1> l1Allocator;
    LocalMemAllocator<Hardware::L0A> l0aAllocator;
    LocalMemAllocator<Hardware::L0B> l0bAllocator;
    LocalMemAllocator<Hardware::L0C> l0cAllocator;
    LocalTensor<half> a1Local = l1Allocator.Alloc<TPosition::A1, half>(kAElements);
    LocalTensor<half> b1Local = l1Allocator.Alloc<TPosition::B1, half>(kBElements);
    LocalTensor<half> a2Local = l0aAllocator.Alloc<TPosition::A2, half>(kAElements);
    LocalTensor<half> b2Local = l0bAllocator.Alloc<TPosition::B2, half>(kBElements);
    LocalTensor<float> c1Local = l0cAllocator.Alloc<TPosition::CO1, float>(kCElements);
    DataCopy(a1Local, aStageGm, Nd2NzParams{1, kMatmulM, kMatmulK, 0, kMatmulK, kMatmulM, 1, 0});
    DataCopy(b1Local, bStageGm, Nd2NzParams{1, kMatmulK, kMatmulN, 0, kMatmulN, kMatmulK, 1, 0});
    SetFlag<HardEvent::MTE2_MTE1>(0);
    WaitFlag<HardEvent::MTE2_MTE1>(0);
    for (uint32_t mBlock = 0; mBlock < kMatmulM / 16; ++mBlock) {
      LoadData(a2Local[mBlock * kMatmulK * 16], a1Local[mBlock * 512 / sizeof(half)],
               LoadData2DParams{0, kMatmulK / 16, kMatmulM / 16, 0, 0, false, 0});
    }
    for (uint32_t kBlock = 0; kBlock < kMatmulK / 16; ++kBlock) {
      LoadData(b2Local[kBlock * kMatmulN * 16], b1Local[kBlock * 512 / sizeof(half)],
               LoadData2DParams{0, kMatmulN / 16, kMatmulK / 16, 0, 0, true, 0});
    }
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    Mmad(c1Local, a2Local, b2Local, MmadParams{kMatmulM, kMatmulN, kMatmulK, 0, false, true});
    SetFlag<HardEvent::M_MTE1>(0);
    WaitFlag<HardEvent::M_MTE1>(0);
    DataCopy(a1Local, aResidualStageGm, Nd2NzParams{1, kMatmulM, kMatmulK, 0, kMatmulK, kMatmulM, 1, 0});
    SetFlag<HardEvent::MTE2_MTE1>(0);
    WaitFlag<HardEvent::MTE2_MTE1>(0);
    for (uint32_t mBlock = 0; mBlock < kMatmulM / 16; ++mBlock) {
      LoadData(a2Local[mBlock * kMatmulK * 16], a1Local[mBlock * 512 / sizeof(half)],
               LoadData2DParams{0, kMatmulK / 16, kMatmulM / 16, 0, 0, false, 0});
    }
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    Mmad(c1Local, a2Local, b2Local, MmadParams{kMatmulM, kMatmulN, kMatmulK, 0, false, false});
    SetFlag<HardEvent::M_MTE1>(0);
    WaitFlag<HardEvent::M_MTE1>(0);
    LocalTensor<float> cNz = chunkVFp32;
    DataCopyParams cCopyParams{static_cast<uint16_t>(kMatmulN / 16), static_cast<uint16_t>(kMatmulM / 16), 0, 0};
    DataCopyEnhancedParams cCopyEnhanced;
    cCopyEnhanced.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    DataCopy(cNz, c1Local, cCopyParams, cCopyEnhanced);
    PipeBarrier<PIPE_ALL>();
    constexpr uint16_t kNdBlockLen = 16 * sizeof(float) / 32;
    constexpr uint16_t kNzSrcStride = (kMatmulM / 16 * 16 * 16 - 16) * sizeof(float) / 32;
    DataCopyParams nzToNdParams{static_cast<uint16_t>(kMatmulN / 16), kNdBlockLen, kNzSrcStride, 0};
    for (uint32_t row = 0; row < kMatmulM; ++row) {
      DataCopy(chunkKFp32[row * kMatmulN], cNz[row * 16], nzToNdParams);
    }
    PipeBarrier<PIPE_ALL>();
  }

  // ---- Phase 6: per v-tile processing (attn matrix and state tile overlap) ----

  __aicore__ inline void ProcessVTiles(int32_t t_start, uint32_t chunkLen, uint64_t head_i, uint64_t qkHead,
                                       uint32_t avFp32, uint64_t stateBaseOffset, uint64_t workspaceStateBaseOffset,
                                       uint32_t c, bool isLastChunk) {
    for (uint32_t v_i = 0; v_i < realV_; v_i += vStep_) {
      uint32_t curV = (v_i + vStep_ > realV_) ? realV_ - v_i : vStep_;
      ProcessOneVTile(t_start, chunkLen, head_i, qkHead, avFp32, stateBaseOffset, workspaceStateBaseOffset, c,
                      isLastChunk, v_i, curV);
    }
  }

  __aicore__ inline void ProcessOneVTile(int32_t t_start, uint32_t chunkLen, uint64_t head_i, uint64_t qkHead,
                                         uint32_t avFp32, uint64_t stateBaseOffset, uint64_t workspaceStateBaseOffset,
                                         uint32_t c, bool isLastChunk, uint32_t v_i, uint32_t curV) {
    LoadVBeta(t_start, head_i, chunkLen, v_i, curV, avFp32);
    bool useFusedCubeFastPath = IsFusedValueOutputCubeFastPath(chunkLen, avFp32);
    if (likely(useFusedCubeFastPath)) {
      ComputeValueAndVNewCubeDispatch(stateBaseOffset, workspaceStateBaseOffset, v_i, curV, c);
    } else {
      ComputeValueNew(chunkLen, curV, avFp32);
      LoadStateTile(stateBaseOffset, workspaceStateBaseOffset, v_i, curV, c);
      ComputeVNew(chunkLen, avFp32);
    }
    if (likely(useFusedCubeFastPath)) {
      ComputeOutputCubeDispatch(t_start, qkHead);
    } else {
      ComputeAttnInter(t_start, qkHead, chunkLen, avFp32);
      AccumOutput(chunkLen, avFp32);
    }
    WriteAttnTileToGm(t_start, chunkLen, head_i, v_i, curV, avFp32);
    UpdateAndWriteState(t_start, chunkLen, v_i, curV, qkHead, stateBaseOffset, workspaceStateBaseOffset, avFp32,
                        isLastChunk);
  }

  // Step 1: v_beta = V * beta -> chunkVFp32.
  __aicore__ inline void LoadVBeta(int32_t t_start, uint64_t head_i, uint32_t chunkLen, uint32_t v_i, uint32_t curV,
                                   uint32_t avFp32) {
    // The compact output tile is dead until ComputeValueNew, so reuse it as
    // FP16 staging and replace chunkLen*curV scalar GM loads with one strided
    // standard DataCopy. Only use the fast path when both row width and source
    // gap are exactly representable in 32-byte blocks.
    uint64_t srcRowGapElems = static_cast<uint64_t>(NV_) * realV_ - curV;
    uint64_t srcRowGapBytes = srcRowGapElems * sizeof(inType);
    uint64_t dstRowGapBytes = static_cast<uint64_t>(avFp32 - curV) * sizeof(inType);
    if (likely((curV * sizeof(inType)) % BLOCK_BYTES == 0 && srcRowGapBytes % BLOCK_BYTES == 0 &&
               dstRowGapBytes % BLOCK_BYTES == 0 && srcRowGapBytes / BLOCK_BYTES <= 65535 &&
               dstRowGapBytes / BLOCK_BYTES <= 65535)) {
      LocalTensor<inType> valueLocal = chunkAttnOutFp32.template ReinterpretCast<inType>();
      uint64_t vOff = (static_cast<uint64_t>(t_start) * NV_ + head_i) * realV_ + v_i;
      uint16_t rowBlocks = static_cast<uint16_t>(curV * sizeof(inType) / BLOCK_BYTES);
      uint16_t srcRowGapBlocks = static_cast<uint16_t>(srcRowGapBytes / BLOCK_BYTES);
      uint16_t dstRowGapBlocks = static_cast<uint16_t>(dstRowGapBytes / BLOCK_BYTES);
      DataCopyParams copyParams{static_cast<uint16_t>(chunkLen), rowBlocks, srcRowGapBlocks, dstRowGapBlocks};

      // Zero the aligned destination rows first, then let MTE2 write valid V
      // elements directly at the final row stride. This makes the subsequent
      // FP16->FP32 Cast contiguous and removes one Cast launch per row.
      Duplicate(valueLocal, static_cast<inType>(0), chunkLen * avFp32);
      PipeBarrier<PIPE_V>();

      event_t scalarToMte2 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::S_MTE2));
      SetFlag<HardEvent::S_MTE2>(scalarToMte2);
      WaitFlag<HardEvent::S_MTE2>(scalarToMte2);
      event_t vectorToMte2 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE2));
      SetFlag<HardEvent::V_MTE2>(vectorToMte2);
      WaitFlag<HardEvent::V_MTE2>(vectorToMte2);
      DataCopy(valueLocal, valueGm_[vOff], copyParams);
      event_t mte2ToVector = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
      SetFlag<HardEvent::MTE2_V>(mte2ToVector);
      WaitFlag<HardEvent::MTE2_V>(mte2ToVector);

      Cast(chunkVFp32, valueLocal, RoundMode::CAST_NONE, chunkLen * avFp32);
      PipeBarrier<PIPE_V>();

      // Broadcast one beta per row into 32-byte blocks and scale the complete
      // aligned V tile with high-dimensional Mul instructions. Padding stays
      // zero, while scalar beta reads and row-wise Muls launches disappear.
      LocalTensor<float> betaBlocks = chunkAttnOutFp32;
      uint8_t brcbRepeats = static_cast<uint8_t>(Ceil(chunkLen, FP32_NUM_PER_BLOCK));
      Brcb(betaBlocks, betaFp32, brcbRepeats, BrcbRepeatParams{1, 8});
      PipeBarrier<PIPE_V>();
      uint8_t rowStride = static_cast<uint8_t>(avFp32 / FP32_NUM_PER_BLOCK);
      BinaryRepeatParams mulParams{1, 1, 0, rowStride, rowStride, 1};
      uint32_t vOffset = 0;
      for (; vOffset + 64 <= avFp32; vOffset += 64) {
        Mul(chunkVFp32[vOffset], chunkVFp32[vOffset], betaBlocks, static_cast<uint64_t>(64),
            static_cast<uint8_t>(chunkLen), mulParams);
      }
      if (vOffset < avFp32) {
        Mul(chunkVFp32[vOffset], chunkVFp32[vOffset], betaBlocks, static_cast<uint64_t>(avFp32 - vOffset),
            static_cast<uint8_t>(chunkLen), mulParams);
      }
      PipeBarrier<PIPE_V>();
      TQueSync<PIPE_V, PIPE_S> valueSync;
      valueSync.SetFlag(0);
      valueSync.WaitFlag(0);
      return;
    }

    Duplicate(chunkVFp32, 0.0f, chunkLen * avFp32);
    PipeBarrier<PIPE_V>();
    for (uint32_t i = 0; i < chunkLen; i++) {
      int32_t t = t_start + i;
      uint64_t vOff = (static_cast<uint64_t>(t) * NV_ + head_i) * realV_;
      float beta_val = betaFp32.GetValue(i);
      for (uint32_t v = 0; v < curV; v++) {
        float v_val = static_cast<float>(valueGm_.GetValue(vOff + v_i + v));
        chunkVFp32.SetValue(i * avFp32 + v, v_val * beta_val);
      }
    }
  }

  // Step 2: value_new_tile = attn @ v_beta_tile (lower-tri attn, sum k <= i) ->
  // chunkAttnOutFp32.
  __aicore__ inline void ComputeValueNew(uint32_t chunkLen, uint32_t curV, uint32_t avFp32) {
    if (likely(IsCubeFastPath(chunkLen, avFp32))) {
      ComputeValueNewCube();
      return;
    }
    (void)curV;
    Duplicate(chunkAttnOutFp32, 0.0f, chunkLen * avFp32);
    PipeBarrier<PIPE_V>();
    for (uint32_t i = 0; i < chunkLen; i++) {
      LocalTensor<float> outRow = chunkAttnOutFp32[i * avFp32];
      for (uint32_t k = 0; k <= i; k++) {
        float score = chunkScoresFp32.GetValue(i * chunkSize_ + k);
        Axpy(outRow, chunkVFp32[k * avFp32], score, avFp32);
        PipeBarrier<PIPE_V>();
      }
    }
  }

  // Step 3: load state tile into stateInFp32 (overlaps chunkScoresFp32 — must run after Step 2).
  __aicore__ inline void LoadStateTile(uint64_t stateBaseOffset, uint64_t workspaceStateBaseOffset, uint32_t v_i,
                                       uint32_t curV, uint32_t c) {
    uint32_t stateTileElem = stateStrideK_ * vStepAligned_;
    Duplicate(stateInFp32, 0.0f, stateTileElem);
    PipeBarrier<PIPE_V>();
    if (c > 0) {
      PipeBarrier<PIPE_ALL>();
      event_t vectorToMte2 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE2));
      SetFlag<HardEvent::V_MTE2>(vectorToMte2);
      WaitFlag<HardEvent::V_MTE2>(vectorToMte2);
      uint32_t alignedV = Ceil(curV, FP32_NUM_PER_BLOCK) * FP32_NUM_PER_BLOCK;
      if constexpr (kSpecializedDk != 0) {
        uint16_t rowBlocks = static_cast<uint16_t>(alignedV / FP32_NUM_PER_BLOCK);
        uint16_t srcGapBlocks = static_cast<uint16_t>((stateWorkspaceStrideV_ - alignedV) / FP32_NUM_PER_BLOCK);
        uint16_t dstGapBlocks = static_cast<uint16_t>((vStepAligned_ - alignedV) / FP32_NUM_PER_BLOCK);
        DataCopyParams stateParams{static_cast<uint16_t>(realK_), rowBlocks, srcGapBlocks, dstGapBlocks};
        uint64_t rowOff = workspaceStateBaseOffset + v_i;
        DataCopy(stateInFp32, stateWorkspaceGm_[rowOff], stateParams);
      } else {
        DataCopyParams stateParams{1, static_cast<uint16_t>(alignedV / FP32_NUM_PER_BLOCK), 0, 0};
        for (uint32_t d = 0; d < realK_; d++) {
          uint64_t rowOff = workspaceStateBaseOffset + static_cast<uint64_t>(d) * stateWorkspaceStrideV_ + v_i;
          DataCopy(stateInFp32[d * vStepAligned_], stateWorkspaceGm_[rowOff], stateParams);
        }
      }
      PipeBarrier<PIPE_ALL>();
      event_t mte2ToVector = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
      SetFlag<HardEvent::MTE2_V>(mte2ToVector);
      WaitFlag<HardEvent::MTE2_V>(mte2ToVector);
      return;
    }

    // The public interface follows 910B and stores state as [B, Nv, Dv, Dk].
    // The internal FP32 tile is [Dk, vStep], so transpose while loading.
    // For 16-aligned tiles, replace the scalar element-wise transpose with one
    // contiguous GM->UB transfer followed by vnchwconv and an FP16->FP32 cast.
    // kCumdecayFp32 and chunkVFp32 are dead at this point and provide the two
    // temporary FP16 matrices when both borrowed buffers are large enough.
    constexpr uint32_t kTransposeBlock = 16;
    uint32_t kBlockCount = realK_ / kTransposeBlock;
    uint32_t vBlockCount = vStepAligned_ / kTransposeBlock;
    uint32_t stateElementCount = realK_ * vStepAligned_;
    uint32_t stateNdCapacity = 2 * chunkSize_ * alignK_;
    uint32_t stateTransposedCapacity = 2 * chunkSize_ * vStepAligned_;
    bool supportedVTile = curV == vStepAligned_;
    if constexpr (kSpecializedDk != 0) {
      supportedVTile = true;
    }
    if (likely(realK_ % kTransposeBlock == 0 && vStepAligned_ % kTransposeBlock == 0 && supportedVTile &&
               kBlockCount <= UINT8_MAX && stateElementCount <= stateNdCapacity &&
               stateElementCount <= stateTransposedCapacity)) {
      LocalTensor<inType> stateNd = kCumdecayFp32.template ReinterpretCast<inType>();
      LocalTensor<inType> stateTransposed = chunkVFp32.template ReinterpretCast<inType>();

      if constexpr (kSpecializedDk != 0) {
        if (curV < vStepAligned_) {
          Duplicate(stateNd, static_cast<inType>(0), stateElementCount);
        }
      }
      event_t vectorToMte2 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE2));
      SetFlag<HardEvent::V_MTE2>(vectorToMte2);
      WaitFlag<HardEvent::V_MTE2>(vectorToMte2);
      uint64_t stateOffset = stateBaseOffset + static_cast<uint64_t>(v_i) * realK_;
      DataCopy(stateNd, initStateGm_[stateOffset], curV * realK_);
      event_t mte2ToVector = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
      SetFlag<HardEvent::MTE2_V>(mte2ToVector);
      WaitFlag<HardEvent::MTE2_V>(mte2ToVector);

      TransDataTo5HDParams transposeParams;
      transposeParams.repeatTimes = static_cast<uint8_t>(kBlockCount);
      transposeParams.srcRepStride = (kBlockCount == 1) ? 0 : 1;
      transposeParams.dstRepStride = (kBlockCount == 1) ? 0 : vStepAligned_;
      for (uint32_t block = 0; block < vBlockCount; block++) {
        LocalTensor<inType> srcRows[kTransposeBlock];
        LocalTensor<inType> dstRows[kTransposeBlock];
        for (uint32_t row = 0; row < kTransposeBlock; row++) {
          srcRows[row] = stateNd[block * kTransposeBlock * realK_ + row * realK_];
          dstRows[row] = stateTransposed[block * kTransposeBlock + row * vStepAligned_];
        }
        TransDataTo5HD<inType>(dstRows, srcRows, transposeParams);
      }
      PipeBarrier<PIPE_V>();
      Cast(stateInFp32, stateTransposed, RoundMode::CAST_NONE, stateElementCount);
      PipeBarrier<PIPE_V>();
      return;
    }

    TQueSync<PIPE_V, PIPE_S> clearStateSync;
    clearStateSync.SetFlag(0);
    clearStateSync.WaitFlag(0);
    for (uint32_t d = 0; d < realK_; d++) {
      for (uint32_t v = 0; v < curV; v++) {
        uint64_t stateOffset = stateBaseOffset + static_cast<uint64_t>(v_i + v) * realK_ + d;
        stateInFp32.SetValue(d * vStepAligned_ + v, static_cast<float>(initStateGm_.GetValue(stateOffset)));
      }
    }
    // Initial state is loaded by scalar GetValue/SetValue and is consumed by
    // vector Muls in ComputeVNew/ComputeAttnInter.
    TQueSync<PIPE_S, PIPE_V> initialStateSync;
    initialStateSync.SetFlag(0);
    initialStateSync.WaitFlag(0);
  }

  // Step 4: v_new = value_new - (k_cumdecay @ state) -> chunkVFp32 (reused).
  __aicore__ inline void ComputeVNew(uint32_t chunkLen, uint32_t avFp32) {
    if (likely(IsCubeFastPath(chunkLen, avFp32))) {
      ComputeVNewCube(chunkLen, avFp32);
      return;
    }
    for (uint32_t i = 0; i < chunkLen; i++) {
      LocalTensor<float> vpRow = chunkVFp32[i * avFp32];
      Duplicate(vpRow, 0.0f, vStepAligned_);
      PipeBarrier<PIPE_V>();
      for (uint32_t d = 0; d < realK_; d++) {
        float kcd = chunkKFp32.GetValue(i * alignK_ + d);
        LocalTensor<float> stRow = stateInFp32[d * vStepAligned_];
        Axpy(vpRow, stRow, kcd, vStepAligned_);
        PipeBarrier<PIPE_V>();
      }
      Muls(vpRow, vpRow, -1.0f, vStepAligned_);
      PipeBarrier<PIPE_V>();
      Add(vpRow, vpRow, chunkAttnOutFp32[i * avFp32], vStepAligned_);
      PipeBarrier<PIPE_V>();
    }
  }

  __aicore__ inline void StageHalfWithResidual(LocalTensor<float> src, LocalTensor<float> scratch,
                                               GlobalTensor<half> hiStageGm, GlobalTensor<half> loStageGm,
                                               uint32_t elementCount) {
    LocalTensor<half> stageLocal = stateOutQueue_.AllocTensor<half>();
    Cast(stageLocal, src, RoundMode::CAST_NONE, elementCount);
    PipeBarrier<PIPE_V>();
    stateOutQueue_.EnQue<half>(stageLocal);
    stageLocal = stateOutQueue_.DeQue<half>();
    DataCopy(hiStageGm, stageLocal, elementCount);

    event_t mte3ToVector = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(mte3ToVector);
    WaitFlag<HardEvent::MTE3_V>(mte3ToVector);
    Cast(scratch, stageLocal, RoundMode::CAST_NONE, elementCount);
    PipeBarrier<PIPE_V>();
    Sub(scratch, src, scratch, elementCount);
    PipeBarrier<PIPE_V>();
    Cast(stageLocal, scratch, RoundMode::CAST_NONE, elementCount);
    PipeBarrier<PIPE_V>();
    stateOutQueue_.EnQue<half>(stageLocal);
    stageLocal = stateOutQueue_.DeQue<half>();
    DataCopy(loStageGm, stageLocal, elementCount);
    stateOutQueue_.FreeTensor(stageLocal);
  }

  __aicore__ inline void StageHalf(LocalTensor<float> src, GlobalTensor<half> stageGm, uint32_t elementCount) {
    LocalTensor<half> stageLocal = stateOutQueue_.AllocTensor<half>();
    Cast(stageLocal, src, RoundMode::CAST_NONE, elementCount);
    PipeBarrier<PIPE_V>();
    stateOutQueue_.EnQue<half>(stageLocal);
    stageLocal = stateOutQueue_.DeQue<half>();
    DataCopy(stageGm, stageLocal, elementCount);
    stateOutQueue_.FreeTensor(stageLocal);
  }

  // Fixed-shape fused Step 2 + Step 4:
  //   v_new = lower(attn) @ v_beta - k_cumdecay @ state.
  // Both products accumulate in the same FP32 L0C matrix. Compared with the
  // former pair of Cube functions this removes one L0C->UB conversion, the
  // intermediate value_new tile, and the final Vector negate/add pass.
  __aicore__ inline void ComputeValueAndVNewCubeDispatch(uint64_t stateBaseOffset, uint64_t workspaceStateBaseOffset,
                                                         uint32_t v_i, uint32_t curV, uint32_t c) {
    if constexpr (kSpecializedDk != 0) {
      ComputeValueAndVNewCube<kSpecializedDk, kSpecializedDk>(stateBaseOffset, workspaceStateBaseOffset, v_i, curV, c);
    }
  }

  template <uint32_t kStateK, uint32_t kMatmulN>
  __aicore__ inline void ComputeValueAndVNewCube(uint64_t stateBaseOffset, uint64_t workspaceStateBaseOffset,
                                                 uint32_t v_i, uint32_t curV, uint32_t c) {
    constexpr uint32_t kMatmulM = 64;
    constexpr uint32_t kAttnK = 64;
    constexpr uint32_t kAAttnElements = kMatmulM * kAttnK;
    constexpr uint32_t kBAttnElements = kAttnK * kMatmulN;
    constexpr uint32_t kAStateElements = kMatmulM * kStateK;
    constexpr uint32_t kBStateElements = kStateK * kMatmulN;
    constexpr uint32_t kCElements = kMatmulM * kMatmulN;
    __gm__ uint8_t *stageBase = GetCubeStageBase(0);

    GlobalTensor<half> aStageGm;
    GlobalTensor<half> bStageGm;
    GlobalTensor<half> aResidualStageGm;
    aStageGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(stageBase), kAStateElements);
    bStageGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(stageBase + kAStateElements * sizeof(half)),
                             kBStateElements);
    aResidualStageGm.SetGlobalBuffer(
      reinterpret_cast<__gm__ half *>(stageBase + (kAStateElements + kBStateElements) * sizeof(half)), kAStateElements);

    // Materialize lower(attn); upper-triangle values are not part of the
    // original recurrence and must not contribute to the dense Cube product.
    Muls(kCumdecayFp32, chunkScoresFp32, 1.0f, kAAttnElements);
    PipeBarrier<PIPE_V>();
    for (uint32_t row = 0; row + 1 < kMatmulM; ++row) {
      Duplicate(kCumdecayFp32[row * kAttnK + row + 1], 0.0f, kAttnK - row - 1);
    }
    PipeBarrier<PIPE_V>();
    StageHalfWithResidual(kCumdecayFp32, chunkAttnOutFp32, aStageGm, aResidualStageGm, kAAttnElements);
    StageHalf(chunkVFp32, bStageGm, kBAttnElements);
    PipeBarrier<PIPE_ALL>();

    LocalMemAllocator<Hardware::L1> l1Allocator;
    LocalMemAllocator<Hardware::L0A> l0aAllocator;
    LocalMemAllocator<Hardware::L0B> l0bAllocator;
    LocalMemAllocator<Hardware::L0C> l0cAllocator;
    LocalTensor<half> a1Local = l1Allocator.Alloc<TPosition::A1, half>(kAStateElements);
    LocalTensor<half> b1Local = l1Allocator.Alloc<TPosition::B1, half>(kBStateElements);
    LocalTensor<half> a2Local = l0aAllocator.Alloc<TPosition::A2, half>(kAStateElements);
    LocalTensor<half> b2Local = l0bAllocator.Alloc<TPosition::B2, half>(kBStateElements);
    LocalTensor<float> c1Local = l0cAllocator.Alloc<TPosition::CO1, float>(kCElements);

    DataCopy(a1Local, aStageGm, Nd2NzParams{1, kMatmulM, kAttnK, 0, kAttnK, kMatmulM, 1, 0});
    DataCopy(b1Local, bStageGm, Nd2NzParams{1, kAttnK, kMatmulN, 0, kMatmulN, kAttnK, 1, 0});
    SetFlag<HardEvent::MTE2_MTE1>(0);
    WaitFlag<HardEvent::MTE2_MTE1>(0);
    for (uint32_t mBlock = 0; mBlock < kMatmulM / 16; ++mBlock) {
      LoadData(a2Local[mBlock * kAttnK * 16], a1Local[mBlock * 512 / sizeof(half)],
               LoadData2DParams{0, kAttnK / 16, kMatmulM / 16, 0, 0, false, 0});
    }
    for (uint32_t kBlock = 0; kBlock < kAttnK / 16; ++kBlock) {
      LoadData(b2Local[kBlock * kMatmulN * 16], b1Local[kBlock * 512 / sizeof(half)],
               LoadData2DParams{0, kMatmulN / 16, kAttnK / 16, 0, 0, true, 0});
    }
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    Mmad(c1Local, a2Local, b2Local, MmadParams{kMatmulM, kMatmulN, kAttnK, 0, false, true});

    SetFlag<HardEvent::M_MTE1>(0);
    WaitFlag<HardEvent::M_MTE1>(0);
    DataCopy(a1Local, aResidualStageGm, Nd2NzParams{1, kMatmulM, kAttnK, 0, kAttnK, kMatmulM, 1, 0});
    SetFlag<HardEvent::MTE2_MTE1>(0);
    WaitFlag<HardEvent::MTE2_MTE1>(0);
    for (uint32_t mBlock = 0; mBlock < kMatmulM / 16; ++mBlock) {
      LoadData(a2Local[mBlock * kAttnK * 16], a1Local[mBlock * 512 / sizeof(half)],
               LoadData2DParams{0, kAttnK / 16, kMatmulM / 16, 0, 0, false, 0});
    }
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    Mmad(c1Local, a2Local, b2Local, MmadParams{kMatmulM, kMatmulN, kAttnK, 0, false, false});

    // The first product no longer needs chunkScores/v_beta. Wait for Cube,
    // then reuse the same staging slot for the second product.
    SetFlag<HardEvent::M_MTE1>(0);
    WaitFlag<HardEvent::M_MTE1>(0);
    LoadStateTile(stateBaseOffset, workspaceStateBaseOffset, v_i, curV, c);
    Muls(chunkKFp32, chunkKFp32, -1.0f, kAStateElements);
    PipeBarrier<PIPE_V>();
    StageHalfWithResidual(chunkKFp32, kCumdecayFp32, aStageGm, aResidualStageGm, kAStateElements);
    StageHalf(stateInFp32, bStageGm, kBStateElements);
    PipeBarrier<PIPE_ALL>();

    DataCopy(a1Local, aStageGm, Nd2NzParams{1, kMatmulM, kStateK, 0, kStateK, kMatmulM, 1, 0});
    DataCopy(b1Local, bStageGm, Nd2NzParams{1, kStateK, kMatmulN, 0, kMatmulN, kStateK, 1, 0});
    SetFlag<HardEvent::MTE2_MTE1>(0);
    WaitFlag<HardEvent::MTE2_MTE1>(0);
    for (uint32_t mBlock = 0; mBlock < kMatmulM / 16; ++mBlock) {
      LoadData(a2Local[mBlock * kStateK * 16], a1Local[mBlock * 512 / sizeof(half)],
               LoadData2DParams{0, kStateK / 16, kMatmulM / 16, 0, 0, false, 0});
    }
    for (uint32_t kBlock = 0; kBlock < kStateK / 16; ++kBlock) {
      LoadData(b2Local[kBlock * kMatmulN * 16], b1Local[kBlock * 512 / sizeof(half)],
               LoadData2DParams{0, kMatmulN / 16, kStateK / 16, 0, 0, true, 0});
    }
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    Mmad(c1Local, a2Local, b2Local, MmadParams{kMatmulM, kMatmulN, kStateK, 0, false, false});

    SetFlag<HardEvent::M_MTE1>(0);
    WaitFlag<HardEvent::M_MTE1>(0);
    DataCopy(a1Local, aResidualStageGm, Nd2NzParams{1, kMatmulM, kStateK, 0, kStateK, kMatmulM, 1, 0});
    SetFlag<HardEvent::MTE2_MTE1>(0);
    WaitFlag<HardEvent::MTE2_MTE1>(0);
    for (uint32_t mBlock = 0; mBlock < kMatmulM / 16; ++mBlock) {
      LoadData(a2Local[mBlock * kStateK * 16], a1Local[mBlock * 512 / sizeof(half)],
               LoadData2DParams{0, kStateK / 16, kMatmulM / 16, 0, 0, false, 0});
    }
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    Mmad(c1Local, a2Local, b2Local, MmadParams{kMatmulM, kMatmulN, kStateK, 0, false, false});

    SetFlag<HardEvent::M_MTE1>(0);
    WaitFlag<HardEvent::M_MTE1>(0);
    LocalTensor<float> cNz = chunkKFp32;
    DataCopyParams cCopyParams{static_cast<uint16_t>(kMatmulN / 16), static_cast<uint16_t>(kMatmulM / 16), 0, 0};
    DataCopyEnhancedParams cCopyEnhanced;
    cCopyEnhanced.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    DataCopy(cNz, c1Local, cCopyParams, cCopyEnhanced);
    PipeBarrier<PIPE_ALL>();

    constexpr uint16_t kNdBlockLen = 16 * sizeof(float) / 32;
    constexpr uint16_t kNzSrcStride = (kMatmulM / 16 * 16 * 16 - 16) * sizeof(float) / 32;
    DataCopyParams nzToNdParams{static_cast<uint16_t>(kMatmulN / 16), kNdBlockLen, kNzSrcStride, 0};
    for (uint32_t row = 0; row < kMatmulM; ++row) {
      DataCopy(chunkVFp32[row * kMatmulN], cNz[row * 16], nzToNdParams);
    }
    PipeBarrier<PIPE_ALL>();
  }

  // Fixed-shape Step 2: value_new = lower(attn) @ v_beta. The old path
  // launched one Axpy for every lower-triangle coefficient. Build the two
  // regular matrices once and execute the complete 64x64x128 product on Cube.
  __aicore__ inline void ComputeValueNewCube() {
    constexpr uint32_t kMatmulM = 64;
    constexpr uint32_t kMatmulK = 64;
    constexpr uint32_t kMatmulN = 128;
    constexpr uint32_t kAElements = kMatmulM * kMatmulK;
    constexpr uint32_t kBElements = kMatmulK * kMatmulN;
    constexpr uint32_t kCElements = kMatmulM * kMatmulN;
    __gm__ uint8_t *stageBase = GetCubeStageBase(0);

    GlobalTensor<half> aStageGm;
    GlobalTensor<half> bStageGm;
    GlobalTensor<half> aResidualStageGm;
    aStageGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(stageBase), kAElements);
    bStageGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(stageBase + kAElements * sizeof(half)), kBElements);
    aResidualStageGm.SetGlobalBuffer(
      reinterpret_cast<__gm__ half *>(stageBase + (kAElements + kBElements) * sizeof(half)), kAElements);

    // The source buffer may contain values above the diagonal. Preserve the
    // original k<=i semantics by clearing only that upper triangle.
    Muls(kCumdecayFp32, chunkScoresFp32, 1.0f, kAElements);
    PipeBarrier<PIPE_V>();
    for (uint32_t row = 0; row + 1 < kMatmulM; ++row) {
      Duplicate(kCumdecayFp32[row * kMatmulK + row + 1], 0.0f, kMatmulK - row - 1);
    }
    PipeBarrier<PIPE_V>();
    StageHalfWithResidual(kCumdecayFp32, chunkAttnOutFp32, aStageGm, aResidualStageGm, kAElements);
    StageHalf(chunkVFp32, bStageGm, kBElements);
    PipeBarrier<PIPE_ALL>();

    LocalMemAllocator<Hardware::L1> l1Allocator;
    LocalMemAllocator<Hardware::L0A> l0aAllocator;
    LocalMemAllocator<Hardware::L0B> l0bAllocator;
    LocalMemAllocator<Hardware::L0C> l0cAllocator;
    LocalTensor<half> a1Local = l1Allocator.Alloc<TPosition::A1, half>(kAElements);
    LocalTensor<half> b1Local = l1Allocator.Alloc<TPosition::B1, half>(kBElements);
    LocalTensor<half> a2Local = l0aAllocator.Alloc<TPosition::A2, half>(kAElements);
    LocalTensor<half> b2Local = l0bAllocator.Alloc<TPosition::B2, half>(kBElements);
    LocalTensor<float> c1Local = l0cAllocator.Alloc<TPosition::CO1, float>(kCElements);

    DataCopy(a1Local, aStageGm, Nd2NzParams{1, kMatmulM, kMatmulK, 0, kMatmulK, kMatmulM, 1, 0});
    DataCopy(b1Local, bStageGm, Nd2NzParams{1, kMatmulK, kMatmulN, 0, kMatmulN, kMatmulK, 1, 0});
    SetFlag<HardEvent::MTE2_MTE1>(0);
    WaitFlag<HardEvent::MTE2_MTE1>(0);
    for (uint32_t mBlock = 0; mBlock < kMatmulM / 16; ++mBlock) {
      LoadData(a2Local[mBlock * kMatmulK * 16], a1Local[mBlock * 512 / sizeof(half)],
               LoadData2DParams{0, kMatmulK / 16, kMatmulM / 16, 0, 0, false, 0});
    }
    for (uint32_t kBlock = 0; kBlock < kMatmulK / 16; ++kBlock) {
      LoadData(b2Local[kBlock * kMatmulN * 16], b1Local[kBlock * 512 / sizeof(half)],
               LoadData2DParams{0, kMatmulN / 16, kMatmulK / 16, 0, 0, true, 0});
    }
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    Mmad(c1Local, a2Local, b2Local, MmadParams{kMatmulM, kMatmulN, kMatmulK, 0, false, true});

    SetFlag<HardEvent::M_MTE1>(0);
    WaitFlag<HardEvent::M_MTE1>(0);
    DataCopy(a1Local, aResidualStageGm, Nd2NzParams{1, kMatmulM, kMatmulK, 0, kMatmulK, kMatmulM, 1, 0});
    SetFlag<HardEvent::MTE2_MTE1>(0);
    WaitFlag<HardEvent::MTE2_MTE1>(0);
    for (uint32_t mBlock = 0; mBlock < kMatmulM / 16; ++mBlock) {
      LoadData(a2Local[mBlock * kMatmulK * 16], a1Local[mBlock * 512 / sizeof(half)],
               LoadData2DParams{0, kMatmulK / 16, kMatmulM / 16, 0, 0, false, 0});
    }
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    Mmad(c1Local, a2Local, b2Local, MmadParams{kMatmulM, kMatmulN, kMatmulK, 0, false, false});

    SetFlag<HardEvent::M_MTE1>(0);
    WaitFlag<HardEvent::M_MTE1>(0);
    LocalTensor<float> cNz = kCumdecayFp32;
    DataCopyParams cCopyParams{static_cast<uint16_t>(kMatmulN / 16), static_cast<uint16_t>(kMatmulM / 16), 0, 0};
    DataCopyEnhancedParams cCopyEnhanced;
    cCopyEnhanced.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    DataCopy(cNz, c1Local, cCopyParams, cCopyEnhanced);
    PipeBarrier<PIPE_ALL>();

    constexpr uint16_t kNdBlockLen = 16 * sizeof(float) / 32;
    constexpr uint16_t kNzSrcStride = (kMatmulM / 16 * 16 * 16 - 16) * sizeof(float) / 32;
    DataCopyParams nzToNdParams{static_cast<uint16_t>(kMatmulN / 16), kNdBlockLen, kNzSrcStride, 0};
    for (uint32_t row = 0; row < kMatmulM; ++row) {
      DataCopy(chunkAttnOutFp32[row * kMatmulN], cNz[row * 16], nzToNdParams);
    }
    PipeBarrier<PIPE_ALL>();
  }

  __aicore__ inline void ComputeVNewCube(uint32_t chunkLen, uint32_t avFp32) {
    constexpr uint32_t kMatmulM = 64;
    constexpr uint32_t kMatmulK = 128;
    constexpr uint32_t kMatmulN = 128;
    constexpr uint32_t kAElements = kMatmulM * kMatmulK;
    constexpr uint32_t kBElements = kMatmulK * kMatmulN;
    constexpr uint32_t kCElements = kMatmulM * kMatmulN;
    // The workspace stages A_hi, A_lo and B_hi. kAElements equals kCElements
    // for this fixed fast path, but spelling out the actual contents avoids
    // confusing the A residual buffer with an output-C buffer.
    __gm__ uint8_t *stageBase = GetCubeStageBase(0);

    GlobalTensor<half> aStageGm;
    GlobalTensor<half> bStageGm;
    GlobalTensor<half> aResidualStageGm;
    aStageGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(stageBase), kAElements);
    bStageGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(stageBase + kAElements * sizeof(half)), kBElements);
    aResidualStageGm.SetGlobalBuffer(
      reinterpret_cast<__gm__ half *>(stageBase + (kAElements + kBElements) * sizeof(half)), kAElements);

    StageHalfWithResidual(chunkKFp32, kCumdecayFp32, aStageGm, aResidualStageGm, kAElements);
    StageHalf(stateInFp32, bStageGm, kBElements);
    PipeBarrier<PIPE_ALL>();

    LocalMemAllocator<Hardware::L1> l1Allocator;
    LocalMemAllocator<Hardware::L0A> l0aAllocator;
    LocalMemAllocator<Hardware::L0B> l0bAllocator;
    LocalMemAllocator<Hardware::L0C> l0cAllocator;
    LocalTensor<half> a1Local = l1Allocator.Alloc<TPosition::A1, half>(kAElements);
    LocalTensor<half> b1Local = l1Allocator.Alloc<TPosition::B1, half>(kBElements);
    LocalTensor<half> a2Local = l0aAllocator.Alloc<TPosition::A2, half>(kAElements);
    LocalTensor<half> b2Local = l0bAllocator.Alloc<TPosition::B2, half>(kBElements);
    LocalTensor<float> c1Local = l0cAllocator.Alloc<TPosition::CO1, float>(kCElements);

    DataCopy(a1Local, aStageGm, Nd2NzParams{1, kMatmulM, kMatmulK, 0, kMatmulK, kMatmulM, 1, 0});
    DataCopy(b1Local, bStageGm, Nd2NzParams{1, kMatmulK, kMatmulN, 0, kMatmulN, kMatmulK, 1, 0});
    SetFlag<HardEvent::MTE2_MTE1>(0);
    WaitFlag<HardEvent::MTE2_MTE1>(0);
    for (uint32_t mBlock = 0; mBlock < kMatmulM / 16; ++mBlock) {
      LoadData(a2Local[mBlock * kMatmulK * 16], a1Local[mBlock * 512 / sizeof(half)],
               LoadData2DParams{0, kMatmulK / 16, kMatmulM / 16, 0, 0, false, 0});
    }
    for (uint32_t kBlock = 0; kBlock < kMatmulK / 16; ++kBlock) {
      LoadData(b2Local[kBlock * kMatmulN * 16], b1Local[kBlock * 512 / sizeof(half)],
               LoadData2DParams{0, kMatmulN / 16, kMatmulK / 16, 0, 0, true, 0});
    }
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    Mmad(c1Local, a2Local, b2Local, MmadParams{kMatmulM, kMatmulN, kMatmulK, 0, false, true});

    SetFlag<HardEvent::M_MTE1>(0);
    WaitFlag<HardEvent::M_MTE1>(0);
    // Accumulate A_lo * B_hi into the same L0C matrix. This compensates most
    // of the FP16 quantization error while keeping the main product on Cube.
    DataCopy(a1Local, aResidualStageGm, Nd2NzParams{1, kMatmulM, kMatmulK, 0, kMatmulK, kMatmulM, 1, 0});
    SetFlag<HardEvent::MTE2_MTE1>(0);
    WaitFlag<HardEvent::MTE2_MTE1>(0);
    for (uint32_t mBlock = 0; mBlock < kMatmulM / 16; ++mBlock) {
      LoadData(a2Local[mBlock * kMatmulK * 16], a1Local[mBlock * 512 / sizeof(half)],
               LoadData2DParams{0, kMatmulK / 16, kMatmulM / 16, 0, 0, false, 0});
    }
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    Mmad(c1Local, a2Local, b2Local, MmadParams{kMatmulM, kMatmulN, kMatmulK, 0, false, false});

    SetFlag<HardEvent::M_MTE1>(0);
    WaitFlag<HardEvent::M_MTE1>(0);

    LocalTensor<float> cNz = chunkKFp32;
    // On dav_m200, raw Mmad writes L0C in NZ order and Fixpipe is unavailable.
    // Copy N/16 fractals, each containing M/16 cube blocks, into UB in NZ order.
    DataCopyParams cCopyParams{static_cast<uint16_t>(kMatmulN / 16), static_cast<uint16_t>(kMatmulM / 16), 0, 0};
    DataCopyEnhancedParams cCopyEnhanced;
    cCopyEnhanced.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    DataCopy(cNz, c1Local, cCopyParams, cCopyEnhanced);
    PipeBarrier<PIPE_ALL>();

    // Convert the UB NZ matrix to row-major ND one output row at a time.
    // Each row contributes 16 FP32 values from every N fractal.
    constexpr uint16_t kNdBlockLen = 16 * sizeof(float) / 32;
    constexpr uint16_t kNzSrcStride = (kMatmulM / 16 * 16 * 16 - 16) * sizeof(float) / 32;
    DataCopyParams nzToNdParams{static_cast<uint16_t>(kMatmulN / 16), kNdBlockLen, kNzSrcStride, 0};
    for (uint32_t row = 0; row < kMatmulM; ++row) {
      DataCopy(chunkVFp32[row * kMatmulN], cNz[row * 16], nzToNdParams);
    }
    PipeBarrier<PIPE_ALL>();
    Muls(chunkVFp32, chunkVFp32, -1.0f, kCElements);
    PipeBarrier<PIPE_V>();
    Add(chunkVFp32, chunkVFp32, chunkAttnOutFp32, kCElements);
    PipeBarrier<PIPE_V>();
  }

  // Fuse Step 5 and Step 6 for the fixed 64x128 fast path:
  //   output = (Q * exp(g)) @ state + lower(attn_i) @ v_new.
  // Both products accumulate in one L0C matrix. A high/low FP16 split keeps
  // the same FP32-accumulation accuracy strategy used by ComputeVNewCube.
  __aicore__ inline void ComputeOutputCubeDispatch(int32_t t_start, uint64_t qkHead) {
    if constexpr (kSpecializedDk != 0) {
      ComputeOutputCube<kSpecializedDk, kSpecializedDk>(t_start, qkHead);
    }
  }

  template <uint32_t kStateK, uint32_t kMatmulN>
  __aicore__ inline void ComputeOutputCube(int32_t t_start, uint64_t qkHead) {
    constexpr uint32_t kMatmulM = 64;
    constexpr uint32_t kAttnK = 64;
    constexpr uint32_t kAStateElements = kMatmulM * kStateK;
    constexpr uint32_t kBStateElements = kStateK * kMatmulN;
    constexpr uint32_t kAAttnElements = kMatmulM * kAttnK;
    constexpr uint32_t kBAttnElements = kAttnK * kMatmulN;
    constexpr uint32_t kCElements = kMatmulM * kMatmulN;
    __gm__ uint8_t *stageBase = GetCubeStageBase(0);

    GlobalTensor<half> aStageGm;
    GlobalTensor<half> bStageGm;
    GlobalTensor<half> aResidualStageGm;
    aStageGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(stageBase), kAStateElements);
    bStageGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(stageBase + kAStateElements * sizeof(half)),
                             kBStateElements);
    aResidualStageGm.SetGlobalBuffer(
      reinterpret_cast<__gm__ half *>(stageBase + (kAStateElements + kBStateElements) * sizeof(half)), kAStateElements);

    // Cache the complete Q chunk with one strided DMA, then apply scale and
    // exp(g) one row at a time. This replaces 8192 scalar GM GetValue calls.
    LocalTensor<inType> queryLocal = chunkKFp32.template ReinterpretCast<inType>();
    uint64_t queryOffset = (static_cast<uint64_t>(t_start) * NK_ + qkHead) * realK_;
    LoadPaddedRows(kCumdecayFp32, queryLocal, queryGm_, queryOffset, kMatmulM, NK_ * realK_);
    for (uint32_t row = 0; row < kMatmulM; ++row) {
      float rowScale = scale_ * expGCumFp32.GetValue(row);
      Muls(kCumdecayFp32[row * kStateK], kCumdecayFp32[row * kStateK], rowScale, kStateK);
    }
    PipeBarrier<PIPE_V>();
    StageHalfWithResidual(kCumdecayFp32, chunkAttnOutFp32, aStageGm, aResidualStageGm, kAStateElements);
    PipeBarrier<PIPE_ALL>();

    LocalMemAllocator<Hardware::L1> l1Allocator;
    LocalMemAllocator<Hardware::L0A> l0aAllocator;
    LocalMemAllocator<Hardware::L0B> l0bAllocator;
    LocalMemAllocator<Hardware::L0C> l0cAllocator;
    LocalTensor<half> a1Local = l1Allocator.Alloc<TPosition::A1, half>(kAStateElements);
    LocalTensor<half> b1Local = l1Allocator.Alloc<TPosition::B1, half>(kBStateElements);
    LocalTensor<half> a2Local = l0aAllocator.Alloc<TPosition::A2, half>(kAStateElements);
    LocalTensor<half> b2Local = l0bAllocator.Alloc<TPosition::B2, half>(kBStateElements);
    LocalTensor<float> c1Local = l0cAllocator.Alloc<TPosition::CO1, float>(kCElements);

    DataCopy(a1Local, aStageGm, Nd2NzParams{1, kMatmulM, kStateK, 0, kStateK, kMatmulM, 1, 0});
    DataCopy(b1Local, bStageGm, Nd2NzParams{1, kStateK, kMatmulN, 0, kMatmulN, kStateK, 1, 0});
    SetFlag<HardEvent::MTE2_MTE1>(0);
    WaitFlag<HardEvent::MTE2_MTE1>(0);
    for (uint32_t mBlock = 0; mBlock < kMatmulM / 16; ++mBlock) {
      LoadData(a2Local[mBlock * kStateK * 16], a1Local[mBlock * 512 / sizeof(half)],
               LoadData2DParams{0, kStateK / 16, kMatmulM / 16, 0, 0, false, 0});
    }
    for (uint32_t kBlock = 0; kBlock < kStateK / 16; ++kBlock) {
      LoadData(b2Local[kBlock * kMatmulN * 16], b1Local[kBlock * 512 / sizeof(half)],
               LoadData2DParams{0, kMatmulN / 16, kStateK / 16, 0, 0, true, 0});
    }
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    Mmad(c1Local, a2Local, b2Local, MmadParams{kMatmulM, kMatmulN, kStateK, 0, false, true});

    SetFlag<HardEvent::M_MTE1>(0);
    WaitFlag<HardEvent::M_MTE1>(0);
    DataCopy(a1Local, aResidualStageGm, Nd2NzParams{1, kMatmulM, kStateK, 0, kStateK, kMatmulM, 1, 0});
    SetFlag<HardEvent::MTE2_MTE1>(0);
    WaitFlag<HardEvent::MTE2_MTE1>(0);
    for (uint32_t mBlock = 0; mBlock < kMatmulM / 16; ++mBlock) {
      LoadData(a2Local[mBlock * kStateK * 16], a1Local[mBlock * 512 / sizeof(half)],
               LoadData2DParams{0, kStateK / 16, kMatmulM / 16, 0, 0, false, 0});
    }
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    Mmad(c1Local, a2Local, b2Local, MmadParams{kMatmulM, kMatmulN, kStateK, 0, false, false});

    // Build lower(attn_i) in one dead K buffer. Only the upper triangle needs
    // clearing; the lower triangle and diagonal are already contiguous.
    SetFlag<HardEvent::M_MTE1>(0);
    WaitFlag<HardEvent::M_MTE1>(0);
    Muls(kCumdecayFp32, decayMaskFp32, 1.0f, kAAttnElements);
    PipeBarrier<PIPE_V>();
    for (uint32_t row = 0; row + 1 < kMatmulM; ++row) {
      Duplicate(kCumdecayFp32[row * kAttnK + row + 1], 0.0f, kAttnK - row - 1);
    }
    PipeBarrier<PIPE_V>();
    StageHalfWithResidual(kCumdecayFp32, chunkAttnOutFp32, aStageGm, aResidualStageGm, kAAttnElements);
    StageHalf(chunkVFp32, bStageGm, kBAttnElements);
    PipeBarrier<PIPE_ALL>();

    DataCopy(a1Local, aStageGm, Nd2NzParams{1, kMatmulM, kAttnK, 0, kAttnK, kMatmulM, 1, 0});
    DataCopy(b1Local, bStageGm, Nd2NzParams{1, kAttnK, kMatmulN, 0, kMatmulN, kAttnK, 1, 0});
    SetFlag<HardEvent::MTE2_MTE1>(0);
    WaitFlag<HardEvent::MTE2_MTE1>(0);
    for (uint32_t mBlock = 0; mBlock < kMatmulM / 16; ++mBlock) {
      LoadData(a2Local[mBlock * kAttnK * 16], a1Local[mBlock * 512 / sizeof(half)],
               LoadData2DParams{0, kAttnK / 16, kMatmulM / 16, 0, 0, false, 0});
    }
    for (uint32_t kBlock = 0; kBlock < kAttnK / 16; ++kBlock) {
      LoadData(b2Local[kBlock * kMatmulN * 16], b1Local[kBlock * 512 / sizeof(half)],
               LoadData2DParams{0, kMatmulN / 16, kAttnK / 16, 0, 0, true, 0});
    }
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    Mmad(c1Local, a2Local, b2Local, MmadParams{kMatmulM, kMatmulN, kAttnK, 0, false, false});

    SetFlag<HardEvent::M_MTE1>(0);
    WaitFlag<HardEvent::M_MTE1>(0);
    DataCopy(a1Local, aResidualStageGm, Nd2NzParams{1, kMatmulM, kAttnK, 0, kAttnK, kMatmulM, 1, 0});
    SetFlag<HardEvent::MTE2_MTE1>(0);
    WaitFlag<HardEvent::MTE2_MTE1>(0);
    for (uint32_t mBlock = 0; mBlock < kMatmulM / 16; ++mBlock) {
      LoadData(a2Local[mBlock * kAttnK * 16], a1Local[mBlock * 512 / sizeof(half)],
               LoadData2DParams{0, kAttnK / 16, kMatmulM / 16, 0, 0, false, 0});
    }
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    Mmad(c1Local, a2Local, b2Local, MmadParams{kMatmulM, kMatmulN, kAttnK, 0, false, false});

    SetFlag<HardEvent::M_MTE1>(0);
    WaitFlag<HardEvent::M_MTE1>(0);
    LocalTensor<float> cNz = kCumdecayFp32;
    DataCopyParams cCopyParams{static_cast<uint16_t>(kMatmulN / 16), static_cast<uint16_t>(kMatmulM / 16), 0, 0};
    DataCopyEnhancedParams cCopyEnhanced;
    cCopyEnhanced.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    DataCopy(cNz, c1Local, cCopyParams, cCopyEnhanced);
    PipeBarrier<PIPE_ALL>();

    constexpr uint16_t kNdBlockLen = 16 * sizeof(float) / 32;
    constexpr uint16_t kNzSrcStride = (kMatmulM / 16 * 16 * 16 - 16) * sizeof(float) / 32;
    DataCopyParams nzToNdParams{static_cast<uint16_t>(kMatmulN / 16), kNdBlockLen, kNzSrcStride, 0};
    for (uint32_t row = 0; row < kMatmulM; ++row) {
      DataCopy(chunkAttnOutFp32[row * kMatmulN], cNz[row * 16], nzToNdParams);
    }
    PipeBarrier<PIPE_ALL>();
  }

  // Step 5: attn_inter = (q * exp(g_cumsum)) @ state -> chunkAttnOutFp32.
  __aicore__ inline void ComputeAttnInter(int32_t t_start, uint64_t qkHead, uint32_t chunkLen, uint32_t avFp32) {
    for (uint32_t i = 0; i < chunkLen; i++) {
      LocalTensor<float> outRow = chunkAttnOutFp32[i * avFp32];
      Duplicate(outRow, 0.0f, vStepAligned_);
      PipeBarrier<PIPE_V>();
      int32_t t = t_start + i;
      uint64_t qkOff = (static_cast<uint64_t>(t) * NK_ + qkHead) * realK_;
      float gExp = expGCumFp32.GetValue(i);
      for (uint32_t d = 0; d < realK_; d++) {
        float qd = static_cast<float>(queryGm_.GetValue(qkOff + d)) * scale_ * gExp;
        LocalTensor<float> stRow = stateInFp32[d * vStepAligned_];
        Axpy(outRow, stRow, qd, vStepAligned_);
        PipeBarrier<PIPE_V>();
      }
    }
  }

  // Step 6: output = attn_inter + attn_i @ v_new (accumulate into chunkAttnOutFp32).
  __aicore__ inline void AccumOutput(uint32_t chunkLen, uint32_t avFp32) {
    uint32_t cs = chunkSize_;
    for (uint32_t i = 0; i < chunkLen; i++) {
      LocalTensor<float> outRow = chunkAttnOutFp32[i * avFp32];
      for (uint32_t jj = 0; jj <= i; jj++) {
        float a = decayMaskFp32.GetValue(i * cs + jj);
        LocalTensor<float> vRow = chunkVFp32[jj * avFp32];
        Axpy(outRow, vRow, a, vStepAligned_);
        PipeBarrier<PIPE_V>();
      }
    }
  }

  // Fixed-shape state update:
  //   state = exp(g_last) * state + (K * exp(g_last - g))^T @ v_new.
  // The weighted K^T @ v_new product uses Cube. The 64/80/96 buckets convert the
  // complete NZ result to contiguous ND and use one wide Vector Add; the 128
  // bucket retains the segmented fallback.
  __aicore__ inline void ComputeStateUpdateCubeDispatch(int32_t t_start, uint64_t qkHead) {
    if constexpr (kSpecializedDk != 0) {
      ComputeStateUpdateCube<kSpecializedDk, kSpecializedDk>(t_start, qkHead);
    }
  }

  template <uint32_t kStateRows, uint32_t kMatmulN>
  __aicore__ inline void ComputeStateUpdateCube(int32_t t_start, uint64_t qkHead) {
    // The 64/80/96 buckets fit one complete M tile in L0C; the 128 bucket keeps
    // the conservative 64-row split.
    constexpr uint32_t kTileM = (kSpecializedDk <= 96) ? kSpecializedDk : 64;
    constexpr uint32_t kMatmulK = 64;
    constexpr uint32_t kMTileCount = (kStateRows + kTileM - 1) / kTileM;
    constexpr uint32_t kAElements = kStateRows * kMatmulK;
    constexpr uint32_t kATileElements = kTileM * kMatmulK;
    constexpr uint32_t kBElements = kMatmulK * kMatmulN;
    constexpr uint32_t kStateElements = kStateRows * kMatmulN;
    constexpr uint32_t kCTileElements = kTileM * kMatmulN;
    __gm__ uint8_t *stageBase = GetCubeStageBase(0);
    GlobalTensor<half> aStageGm;
    GlobalTensor<half> bStageGm;
    GlobalTensor<half> aResidualStageGm;
    aStageGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(stageBase), kAElements);
    bStageGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(stageBase + kAElements * sizeof(half)), kBElements);
    aResidualStageGm.SetGlobalBuffer(
      reinterpret_cast<__gm__ half *>(stageBase + (kAElements + kBElements) * sizeof(half)), kAElements);

    float gLast = gCumsumFp32.GetValue(kMatmulK - 1);
    float gLastExp = expGCumFp32.GetValue(kMatmulK - 1);
    Muls(stateInFp32, stateInFp32, gLastExp, kStateElements);
    Duplicate(deltaFp32, gLast, kMatmulK);
    PipeBarrier<PIPE_V>();
    Sub(deltaFp32, deltaFp32, gCumsumFp32, kMatmulK);
    PipeBarrier<PIPE_V>();
    Exp(deltaFp32, deltaFp32, kMatmulK);
    PipeBarrier<PIPE_V>();

    // Load K in [64,128] ND order, apply one decay coefficient per row, then
    // transpose the FP16 high/low parts to the [128,64] A matrix.
    LocalTensor<inType> keyLocal = chunkKFp32.template ReinterpretCast<inType>();
    uint64_t keyOffset = (static_cast<uint64_t>(t_start) * NK_ + qkHead) * realK_;
    LoadPaddedRows(chunkAttnOutFp32, keyLocal, keyGm_, keyOffset, kMatmulK, NK_ * realK_);
    TQueSync<PIPE_V, PIPE_S> decaySync;
    decaySync.SetFlag(0);
    decaySync.WaitFlag(0);
    for (uint32_t row = 0; row < kMatmulK; ++row) {
      float rowScale = deltaFp32.GetValue(row);
      Muls(chunkAttnOutFp32[row * kStateRows], chunkAttnOutFp32[row * kStateRows], rowScale, kStateRows);
    }
    PipeBarrier<PIPE_V>();

    LocalTensor<half> stageLocal = stateOutQueue_.AllocTensor<half>();
    LocalTensor<half> transposedLocal = kCumdecayFp32.template ReinterpretCast<half>();
    Cast(stageLocal, chunkAttnOutFp32, RoundMode::CAST_NONE, kAElements);
    PipeBarrier<PIPE_V>();
    TransDataTo5HDParams transposeParams;
    transposeParams.repeatTimes = static_cast<uint8_t>(kStateRows / 16);
    transposeParams.srcRepStride = 1;
    transposeParams.dstRepStride = kMatmulK;
    for (uint32_t block = 0; block < kMatmulK / 16; ++block) {
      LocalTensor<half> srcRows[16];
      LocalTensor<half> dstRows[16];
      for (uint32_t row = 0; row < 16; ++row) {
        srcRows[row] = stageLocal[block * 16 * kStateRows + row * kStateRows];
        dstRows[row] = transposedLocal[block * 16 + row * kMatmulK];
      }
      TransDataTo5HD<half>(dstRows, srcRows, transposeParams);
    }
    PipeBarrier<PIPE_V>();
    event_t vectorToMte3 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(vectorToMte3);
    WaitFlag<HardEvent::V_MTE3>(vectorToMte3);
    DataCopy(aStageGm, transposedLocal, kAElements);

    event_t mte3ToVector = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(mte3ToVector);
    WaitFlag<HardEvent::MTE3_V>(mte3ToVector);
    Cast(chunkKFp32, stageLocal, RoundMode::CAST_NONE, kAElements);
    PipeBarrier<PIPE_V>();
    Sub(chunkKFp32, chunkAttnOutFp32, chunkKFp32, kAElements);
    PipeBarrier<PIPE_V>();
    Cast(stageLocal, chunkKFp32, RoundMode::CAST_NONE, kAElements);
    PipeBarrier<PIPE_V>();
    for (uint32_t block = 0; block < kMatmulK / 16; ++block) {
      LocalTensor<half> srcRows[16];
      LocalTensor<half> dstRows[16];
      for (uint32_t row = 0; row < 16; ++row) {
        srcRows[row] = stageLocal[block * 16 * kStateRows + row * kStateRows];
        dstRows[row] = transposedLocal[block * 16 + row * kMatmulK];
      }
      TransDataTo5HD<half>(dstRows, srcRows, transposeParams);
    }
    PipeBarrier<PIPE_V>();
    vectorToMte3 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(vectorToMte3);
    WaitFlag<HardEvent::V_MTE3>(vectorToMte3);
    DataCopy(aResidualStageGm, transposedLocal, kAElements);
    stateOutQueue_.FreeTensor(stageLocal);
    StageHalf(chunkVFp32, bStageGm, kBElements);
    PipeBarrier<PIPE_ALL>();

    LocalMemAllocator<Hardware::L1> l1Allocator;
    LocalMemAllocator<Hardware::L0A> l0aAllocator;
    LocalMemAllocator<Hardware::L0B> l0bAllocator;
    LocalMemAllocator<Hardware::L0C> l0cAllocator;
    LocalTensor<half> a1Local = l1Allocator.Alloc<TPosition::A1, half>(kATileElements);
    LocalTensor<half> b1Local = l1Allocator.Alloc<TPosition::B1, half>(kBElements);
    LocalTensor<half> a2Local = l0aAllocator.Alloc<TPosition::A2, half>(kATileElements);
    LocalTensor<half> b2Local = l0bAllocator.Alloc<TPosition::B2, half>(kBElements);
    LocalTensor<float> c1Local = l0cAllocator.Alloc<TPosition::CO1, float>(kCTileElements);

    // B is shared by both 64-row M tiles and remains resident in L0B.
    DataCopy(b1Local, bStageGm, Nd2NzParams{1, kMatmulK, kMatmulN, 0, kMatmulN, kMatmulK, 1, 0});
    SetFlag<HardEvent::MTE2_MTE1>(0);
    WaitFlag<HardEvent::MTE2_MTE1>(0);
    for (uint32_t kBlock = 0; kBlock < kMatmulK / 16; ++kBlock) {
      LoadData(b2Local[kBlock * kMatmulN * 16], b1Local[kBlock * 512 / sizeof(half)],
               LoadData2DParams{0, kMatmulN / 16, kMatmulK / 16, 0, 0, true, 0});
    }
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);

    for (uint32_t mTile = 0; mTile < kMTileCount; ++mTile) {
      uint32_t curTileM = (mTile + 1) * kTileM <= kStateRows ? kTileM : kStateRows - mTile * kTileM;
      uint32_t aOffset = mTile * kATileElements;
      DataCopy(
        a1Local, aStageGm[aOffset],
        Nd2NzParams{1, static_cast<uint16_t>(curTileM), kMatmulK, 0, kMatmulK, static_cast<uint16_t>(curTileM), 1, 0});
      SetFlag<HardEvent::MTE2_MTE1>(0);
      WaitFlag<HardEvent::MTE2_MTE1>(0);
      for (uint32_t mBlock = 0; mBlock < curTileM / 16; ++mBlock) {
        LoadData(a2Local[mBlock * kMatmulK * 16], a1Local[mBlock * 512 / sizeof(half)],
                 LoadData2DParams{0, kMatmulK / 16, static_cast<uint16_t>(curTileM / 16), 0, 0, false, 0});
      }
      SetFlag<HardEvent::MTE1_M>(0);
      WaitFlag<HardEvent::MTE1_M>(0);
      Mmad(c1Local, a2Local, b2Local, MmadParams{static_cast<uint16_t>(curTileM), kMatmulN, kMatmulK, 0, false, true});

      SetFlag<HardEvent::M_MTE1>(0);
      WaitFlag<HardEvent::M_MTE1>(0);
      DataCopy(
        a1Local, aResidualStageGm[aOffset],
        Nd2NzParams{1, static_cast<uint16_t>(curTileM), kMatmulK, 0, kMatmulK, static_cast<uint16_t>(curTileM), 1, 0});
      SetFlag<HardEvent::MTE2_MTE1>(0);
      WaitFlag<HardEvent::MTE2_MTE1>(0);
      for (uint32_t mBlock = 0; mBlock < curTileM / 16; ++mBlock) {
        LoadData(a2Local[mBlock * kMatmulK * 16], a1Local[mBlock * 512 / sizeof(half)],
                 LoadData2DParams{0, kMatmulK / 16, static_cast<uint16_t>(curTileM / 16), 0, 0, false, 0});
      }
      SetFlag<HardEvent::MTE1_M>(0);
      WaitFlag<HardEvent::MTE1_M>(0);
      Mmad(c1Local, a2Local, b2Local, MmadParams{static_cast<uint16_t>(curTileM), kMatmulN, kMatmulK, 0, false, false});

      SetFlag<HardEvent::M_MTE1>(0);
      WaitFlag<HardEvent::M_MTE1>(0);
      LocalTensor<float> cNz = chunkKFp32;
      if constexpr (kSpecializedDk <= 96) {
        // chunkK/kCumdecay/decayMask are dead after staging A. Their combined
        // contiguous UB region holds the complete 64/80/96-bucket NZ result.
        cNz = tmpBuff.GetWithOffset<float>(kStateElements, 0);
      }
      DataCopyParams cCopyParams{static_cast<uint16_t>(kMatmulN / 16), static_cast<uint16_t>(curTileM / 16), 0, 0};
      DataCopyEnhancedParams cCopyEnhanced;
      cCopyEnhanced.blockMode = BlockMode::BLOCK_MODE_MATRIX;
      DataCopy(cNz, c1Local, cCopyParams, cCopyEnhanced);
      PipeBarrier<PIPE_ALL>();

      constexpr uint16_t kNdBlockLen = 16 * sizeof(float) / BLOCK_BYTES;
      if constexpr (kSpecializedDk <= 96) {
        // Fuse NZ->ND addressing with state accumulation. One repeat-stride
        // Add handles every row of a 16-column NZ fractal.
        constexpr uint32_t kNzFractalElements = kStateRows * 16;
        constexpr uint8_t kNdDstRepStride = kMatmulN * sizeof(float) / BLOCK_BYTES;
        constexpr uint8_t kNzSrcRepStride = 16 * sizeof(float) / BLOCK_BYTES;
        for (uint32_t nBlock = 0; nBlock < kMatmulN / 16; ++nBlock) {
          Add(stateInFp32[nBlock * 16], stateInFp32[nBlock * 16], cNz[nBlock * kNzFractalElements], 16,
              static_cast<uint8_t>(kStateRows),
              BinaryRepeatParams{1, 1, 1, kNdDstRepStride, kNdDstRepStride, kNzSrcRepStride});
        }
        PipeBarrier<PIPE_V>();
      } else {
        uint16_t nzSrcStride = static_cast<uint16_t>((curTileM / 16 * 16 * 16 - 16) * sizeof(float) / BLOCK_BYTES);
        DataCopyParams nzToNdParams{static_cast<uint16_t>(kMatmulN / 16), kNdBlockLen, nzSrcStride, 0};
        uint32_t nzFractalStride = (curTileM / 16) * 16 * 16;
        for (uint32_t row = 0; row < curTileM; ++row) {
          uint32_t stateRow = mTile * kTileM + row;
          for (uint32_t nBlock = 0; nBlock < kMatmulN / 16; ++nBlock) {
            uint32_t ndOffset = stateRow * kMatmulN + nBlock * 16;
            uint32_t nzOffset = row * 16 + nBlock * nzFractalStride;
            Add(stateInFp32[ndOffset], stateInFp32[ndOffset], cNz[nzOffset], 16);
          }
        }
      }
      PipeBarrier<PIPE_V>();
    }
  }

  // Step 7: decay state, add (K * exp(gLast - g))^T @ v_new, then write the state tile back to GM.
  __aicore__ inline void UpdateAndWriteState(int32_t t_start, uint32_t chunkLen, uint32_t v_i, uint32_t curV,
                                             uint64_t qkHead, uint64_t stateBaseOffset,
                                             uint64_t workspaceStateBaseOffset, uint32_t avFp32, bool isLastChunk) {
    if (likely(IsCubeFastPath(chunkLen, avFp32))) {
      ComputeStateUpdateCubeDispatch(t_start, qkHead);
    } else {
      float gLastExp = expGCumFp32.GetValue(chunkLen - 1);
      for (uint32_t d = 0; d < realK_; d++) {
        LocalTensor<float> stRow = stateInFp32[d * vStepAligned_];
        Muls(stRow, stRow, gLastExp, vStepAligned_);
        PipeBarrier<PIPE_V>();
      }
      for (uint32_t i = 0; i < chunkLen; i++) {
        int32_t t = t_start + i;
        uint64_t qkOff = (static_cast<uint64_t>(t) * NK_ + qkHead) * realK_;
        float diffExp = ScalarExp(gCumsumFp32.GetValue(chunkLen - 1) - gCumsumFp32.GetValue(i));
        LocalTensor<float> vRow = chunkVFp32[i * avFp32];
        for (uint32_t d = 0; d < realK_; d++) {
          float coeff = static_cast<float>(keyGm_.GetValue(qkOff + d)) * diffExp;
          LocalTensor<float> stRow = stateInFp32[d * vStepAligned_];
          Axpy(stRow, vRow, coeff, vStepAligned_);
          PipeBarrier<PIPE_V>();
        }
      }
    }
    // Preserve the state in FP32 between chunks. Casting to final_state (FP16)
    // after every chunk accumulates quantization error and makes the result
    // depend on chunk_size.
    if (!isLastChunk) {
      PipeBarrier<PIPE_ALL>();
      event_t vectorToMte3 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
      SetFlag<HardEvent::V_MTE3>(vectorToMte3);
      WaitFlag<HardEvent::V_MTE3>(vectorToMte3);
      uint32_t alignedV = Ceil(curV, FP32_NUM_PER_BLOCK) * FP32_NUM_PER_BLOCK;
      if constexpr (kSpecializedDk != 0) {
        uint16_t rowBlocks = static_cast<uint16_t>(alignedV / FP32_NUM_PER_BLOCK);
        uint16_t srcGapBlocks = static_cast<uint16_t>((vStepAligned_ - alignedV) / FP32_NUM_PER_BLOCK);
        uint16_t dstGapBlocks = static_cast<uint16_t>((stateWorkspaceStrideV_ - alignedV) / FP32_NUM_PER_BLOCK);
        DataCopyParams stateParams{static_cast<uint16_t>(realK_), rowBlocks, srcGapBlocks, dstGapBlocks};
        uint64_t rowOff = workspaceStateBaseOffset + v_i;
        DataCopy(stateWorkspaceGm_[rowOff], stateInFp32, stateParams);
      } else {
        DataCopyParams stateParams{1, static_cast<uint16_t>(alignedV / FP32_NUM_PER_BLOCK), 0, 0};
        for (uint32_t d = 0; d < realK_; d++) {
          uint64_t rowOff = workspaceStateBaseOffset + static_cast<uint64_t>(d) * stateWorkspaceStrideV_ + v_i;
          DataCopy(stateWorkspaceGm_[rowOff], stateInFp32[d * vStepAligned_], stateParams);
        }
      }
      PipeBarrier<PIPE_ALL>();
      event_t mte3ToMte2 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE3_MTE2));
      SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2);
      WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2);
      return;
    }

    // Only the final chunk is cast to the public FP16 final_state output.
    uint32_t alignedElem = stateStrideK_ * vStepAligned_;
    Muls(stateInFp32, stateInFp32, 1.0f, alignedElem);
    PipeBarrier<PIPE_V>();
    LocalTensor<outType> stateLocal = stateOutQueue_.AllocTensor<outType>();
    Cast(stateLocal, stateInFp32, RoundMode::CAST_NONE, alignedElem);
    PipeBarrier<PIPE_V>();
    stateOutQueue_.EnQue<outType>(stateLocal);
    stateLocal = stateOutQueue_.DeQue<outType>();

    // The public final_state is [B, Nv, Dv, Dk], while stateLocal is the
    // internal [Dk, vStep] tile. For aligned full tiles, transpose in Vector
    // and write one contiguous block instead of performing scalar GM stores.
    constexpr uint32_t kTransposeBlock = 16;
    uint32_t kBlockCount = realK_ / kTransposeBlock;
    uint32_t vBlockCount = curV / kTransposeBlock;
    uint32_t stateElementCount = realK_ * curV;
    bool supportedVTile = curV == vStepAligned_;
    if constexpr (kSpecializedDk != 0) {
      vBlockCount = vStepAligned_ / kTransposeBlock;
      stateElementCount = realK_ * vStepAligned_;
      supportedVTile = true;
    }
    uint32_t outputStateElementCount = realK_ * curV;
    uint32_t transposeCapacity = 2 * chunkSize_ * alignK_;
    if (likely(realK_ % kTransposeBlock == 0 && vStepAligned_ % kTransposeBlock == 0 && supportedVTile &&
               vBlockCount <= UINT8_MAX && stateElementCount <= transposeCapacity)) {
      LocalTensor<outType> stateTransposed = kCumdecayFp32.template ReinterpretCast<outType>();
      TransDataTo5HDParams transposeParams;
      transposeParams.repeatTimes = static_cast<uint8_t>(vBlockCount);
      transposeParams.srcRepStride = (vBlockCount == 1) ? 0 : 1;
      transposeParams.dstRepStride = (vBlockCount == 1) ? 0 : realK_;
      for (uint32_t block = 0; block < kBlockCount; block++) {
        LocalTensor<outType> srcRows[kTransposeBlock];
        LocalTensor<outType> dstRows[kTransposeBlock];
        for (uint32_t row = 0; row < kTransposeBlock; row++) {
          srcRows[row] = stateLocal[block * kTransposeBlock * vStepAligned_ + row * vStepAligned_];
          dstRows[row] = stateTransposed[block * kTransposeBlock + row * realK_];
        }
        TransDataTo5HD<outType>(dstRows, srcRows, transposeParams);
      }
      PipeBarrier<PIPE_V>();
      event_t vectorToMte3 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
      SetFlag<HardEvent::V_MTE3>(vectorToMte3);
      WaitFlag<HardEvent::V_MTE3>(vectorToMte3);
      uint64_t stateOffset = stateBaseOffset + static_cast<uint64_t>(v_i) * realK_;
      DataCopy(finalStateGm_[stateOffset], stateTransposed, outputStateElementCount);
      stateOutQueue_.FreeTensor(stateLocal);
      return;
    }

    TQueSync<PIPE_V, PIPE_S> finalStateSync;
    finalStateSync.SetFlag(0);
    finalStateSync.WaitFlag(0);
    for (uint32_t v = 0; v < curV; v++) {
      uint64_t stateRowOffset = stateBaseOffset + static_cast<uint64_t>(v_i + v) * realK_;
      for (uint32_t d = 0; d < realK_; d++) {
        finalStateGm_.SetValue(stateRowOffset + d, stateLocal.GetValue(d * vStepAligned_ + v));
      }
    }
    stateOutQueue_.FreeTensor(stateLocal);
  }

 private:
  GlobalTensor<inType> queryGm_;
  GlobalTensor<inType> keyGm_;
  GlobalTensor<inType> valueGm_;
  GlobalTensor<float> gGm_;
  GlobalTensor<inType> betaGm_;
  GlobalTensor<inType> initStateGm_;
  GlobalTensor<int32_t> actualSeqLengthsGm_;
  GlobalTensor<outType> finalStateGm_;
  GlobalTensor<outType> attnOutGm_;
  GlobalTensor<float> stateWorkspaceGm_;
  GM_ADDR workspaceAddr_;
  TPipe *pipe_;
  TQue<QuePosition::VECOUT, 1> stateOutQueue_;
  TBuf<TPosition::VECCALC> tmpBuff;

  LocalTensor<float> chunkKFp32;        // cs * alignK (original K)
  LocalTensor<float> kCumdecayFp32;     // cs * alignK (k_beta then k_cumdecay)
  LocalTensor<float> decayMaskFp32;     // cs * cs
  LocalTensor<float> chunkVFp32;        // cs * avFp32 (v_beta, then v_new)
  LocalTensor<float> chunkScoresFp32;   // cs * cs (attn matrix)
  LocalTensor<float> stateInFp32;       // dk * avStepAligned (state [DK, vStep])
  LocalTensor<float> chunkAttnOutFp32;  // cs * avFp32 (output)
  LocalTensor<float> gCumsumFp32;       // cs
  LocalTensor<float> deltaFp32;         // temp: max(stateStrideK_, chunkSize_, vStepAligned_)
  LocalTensor<float> dotProductFp32;    // dot-product multiply/reduction scratch
  LocalTensor<float> expGCumFp32;       // precomputed exp(gCumsum) [cs]
  LocalTensor<float> betaFp32;          // beta for the current chunk/head [cs]

  uint32_t B_;
  uint32_t T_;
  uint32_t NK_;
  uint32_t alignK_;
  uint32_t realK_;
  uint32_t NV_;
  uint32_t realV_;
  uint32_t vStep_;
  uint32_t chunkSize_;
  uint32_t numChunks_;
  uint32_t hasGamma_;
  uint32_t restUbSize_;
  uint32_t vStepAligned_;
  uint32_t stateStrideK_;
  uint32_t stateWorkspaceStrideV_;
  uint32_t load_;
  uint32_t usedblk_;
  uint32_t avgload_;
  float scale_;
  uint64_t blockIdx_;
};

#endif  // CHUNK_GATED_DELTA_RULE_KERNEL_H_
