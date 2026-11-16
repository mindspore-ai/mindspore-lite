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

// NOTE: only the kernel ENTRY (in chunk_gated_delta_rule.cpp) must stay in the
// global namespace: on some CANN toolchains the op-kernel autogen expands the
// tiling macros (REGISTER_TILING_DEFAULT / GET_TILING_DATA) and, when a user
// namespace is open around them, emits the tiling-data type / tiling-key as
// <ns>::-qualified symbols that then fail to resolve ("unknown type
// 'ChunkGatedDeltaRuleTilingData'", "undeclared identifier '..._tilingkey'")
// and aborts the device-kernel compile. The kernel class, its helpers and the
// constants live in the ChunkGatedDeltaRule namespace (mirroring the ascend_a2
// arch22 port); the entry refers to them with explicit qualification, which the
// autogen tolerates (it is only an OPEN user namespace around the macros that
// breaks symbol resolution). Each op is its own translation unit.
namespace ChunkGatedDeltaRule {

constexpr uint64_t BUFFER_NUM = 1;
constexpr uint64_t FP16_NUM_PER_BLOCK = 16;
constexpr uint64_t FP32_NUM_PER_BLOCK = 8;
constexpr int64_t BLOCK_BYTES = 32;
constexpr uint64_t CUBE_STAGE_SLOT_BYTES = 64 * 1024;
constexpr uint32_t CUBE_STAGE_SLOT_COUNT = 4;
// Per-core GM cache for the exact FP16 K^T tile (max Dk bucket 128 x 64 half).
constexpr uint64_t KEY_CACHE_BYTES_PER_CORE = 128 * 64 * sizeof(uint16_t);
// NZ fractal geometry on dav_m200: one fractal is 16x16 elements; one L0A/L0B
// fractal row occupies 16 * 32 = 512 bytes.
constexpr uint32_t kNzFractalDim = 16;
constexpr uint32_t kNzFractalRowBytes = 512;
// Maximum value of the 16-bit DataCopy stride fields (in 32-byte blocks).
constexpr uint64_t kDataCopyStrideBlockLimit = 65535;
// Chunk length that selects the fast (specialized) Cube path.
constexpr uint32_t kCubeFastPathChunkLen = 64;
// Number of columns processed per high-dimensional Mul in the beta-scale loops.
constexpr uint32_t kVectorMulBlockSize = 64;
// Number of Taylor-series terms used by ScalarExp to approximate exp on a scalar.
constexpr int kExpTaylorTerms = 12;
// Number of value heads that share one Q/K head in the paired fast path
// (a value-head pair runs the shared attention prework once for both heads).
constexpr uint32_t kPairValueHeads = 2;
// Per-core cube-stage slot that caches the raw attention products of a pair.
constexpr uint32_t kAttnRawSlot = 2;
// Width of the 2x2 block factorization used by the blocked recursive inverse,
// and the width of each of its diagonal sub-blocks.
constexpr uint32_t kBlockDiagWidth = 32;
constexpr uint32_t kBlockDiagHalf = 16;
// 16x16 granularity of the Vector transpose (TransDataTo5HD) and of the 16x16
// diagonal solves inside the blocked recursive inverse.
constexpr uint32_t kTransposeBlockDim = 16;
// Implemented fast-path decisions (kept as compile-time documentation of the
// numerics; they are unconditional in this kernel):
//   * recursive-inverse block products use two-term compensation
//     (A_hi + A_lo) * B_hi;
//   * the per-row exp(g) scaling chains are Brcb-broadcast vector Muls;
//   * the output stage is fused into the value stage (state B reuse);
//   * the decay mask upper triangle is cleared after Exp;
//   * adjacent value heads that share a qkHead are processed as a pair.

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
    keyCacheValid_ = 0;
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

  // Convert an NZ (fractal) FP32 tile produced by L0C->UB into ND rows.
  // The L0C->UB blob is column-group major: one 16-column group holds all
  // kRows rows as kRows*16 contiguous floats (row r at offset r*16). A whole
  // group therefore moves with a single strided DataCopy whose repeat stride
  // skips (kCols - 16) floats, replacing the former one-descriptor-per-row
  // loop (64 descriptors -> kCols/16) with a byte-identical move.
  template <uint32_t kRows, uint32_t kCols>
  __aicore__ inline void NzTileToNd(LocalTensor<float> dst, LocalTensor<float> cNz) {
    constexpr uint32_t kLaneFloats = kNzFractalDim;
    constexpr uint32_t kGroupFloats = kRows * kLaneFloats;
    constexpr uint16_t kLaneBlocks = kLaneFloats * sizeof(float) / BLOCK_BYTES;
    constexpr uint16_t kDstGapBlocks = (kCols - kLaneFloats) * sizeof(float) / BLOCK_BYTES;
    DataCopyParams params{static_cast<uint16_t>(kRows), kLaneBlocks, 0, kDstGapBlocks};
    for (uint32_t group = 0; group < kCols / kLaneFloats; ++group) {
      DataCopy(dst[group * kLaneFloats], cNz[group * kGroupFloats], params);
    }
    PipeBarrier<PIPE_ALL>();
  }

  __aicore__ inline void CopyL0CToUb(LocalTensor<float> dst, LocalTensor<float> src, const DataCopyParams &copyParams,
                                     const DataCopyEnhancedParams &copyEnhanced) {
    event_t mToV = static_cast<event_t>(pipe_->FetchEventID(HardEvent::M_V));
    event_t vToM = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_M));
    SetFlag<HardEvent::M_V>(mToV);
    WaitFlag<HardEvent::M_V>(mToV);
    DataCopy(dst, src, copyParams, copyEnhanced);
    SetFlag<HardEvent::V_M>(vToM);
    WaitFlag<HardEvent::V_M>(vToM);
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
    if (chunkOutBytes > outQueueBytes) {
      outQueueBytes = chunkOutBytes;
    }
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

    // A work item is one (batch, value head, V tile), so a core holds a single V
    // tile at a time: chunkScoresFp32 is dead before that tile loads the state,
    // and the two may therefore share one region sized by the larger of them.
    uint32_t overlapSize = (cs * cs > stateStrideK_ * avStepAligned) ? cs * cs : stateStrideK_ * avStepAligned;
    chunkScoresFp32 = tmpBuff.GetWithOffset<float>(overlapSize, off);
    stateInFp32 = tmpBuff.GetWithOffset<float>(stateStrideK_ * avStepAligned, off);
    off += overlapSize * sizeof(float);

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
      ProcessBatch(seq0, seq1, seqLen, batch_i);
      seq0 = seq1;
    }
  }

 private:
  // One work item is (batch, value head, V tile) when a head spans several V
  // tiles, otherwise (batch, value head) or (batch, value-head pair).
  __aicore__ inline void ProcessBatch(int32_t seq0, int32_t seq1, int32_t seqLen, uint64_t batch_i) {
    if (ShouldSplitVTiles()) {
      ProcessBatchVTiles(seq0, seq1, seqLen, batch_i);
    } else {
      ProcessBatchHeads(seq0, seq1, seqLen, batch_i);
    }
  }

  __aicore__ inline void ProcessBatchVTiles(int32_t seq0, int32_t seq1, int32_t seqLen, uint64_t batch_i) {
    uint32_t vTileCount = Ceil(realV_, vStep_);
    for (uint64_t head_i = 0; head_i < NV_; head_i++) {
      for (uint32_t vTileIdx = 0; vTileIdx < vTileCount; vTileIdx++) {
        if (!IsCurrentBlock(seqLen)) {
          continue;
        }
        ProcessHeadVTile(seq0, seq1, head_i, batch_i, vTileIdx);
      }
    }
  }

  __aicore__ inline void ProcessBatchHeads(int32_t seq0, int32_t seq1, int32_t seqLen, uint64_t batch_i) {
    const bool canPair = (NV_ % kPairValueHeads == 0) && (NK_ > 0) && (NV_ >= NK_) && (NV_ / NK_ == kPairValueHeads);
    if (canPair) {
      for (uint64_t head_i = 0; head_i < NV_; head_i += kPairValueHeads) {
        if (!IsCurrentBlockN(seqLen, kPairValueHeads)) {
          continue;
        }
        ProcessHeadPair(seq0, seq1, head_i, batch_i);
      }
      return;
    }
    for (uint64_t head_i = 0; head_i < NV_; head_i++) {
      if (!IsCurrentBlock(seqLen)) {
        continue;
      }
      ProcessHead(seq0, seq1, head_i, batch_i);
    }
  }

  __aicore__ inline bool ShouldSplitVTiles() { return realV_ > vStep_; }

  __aicore__ inline __gm__ uint8_t *GetKeyCacheBase() {
    uint64_t stateWorkspaceElements = static_cast<uint64_t>(B_) * NV_ * realK_ * stateWorkspaceStrideV_;
    return workspaceAddr_ + stateWorkspaceElements * sizeof(float) +
           static_cast<uint64_t>(GetBlockNum()) * CUBE_STAGE_SLOT_COUNT * CUBE_STAGE_SLOT_BYTES +
           static_cast<uint64_t>(blockIdx_) * KEY_CACHE_BYTES_PER_CORE;
  }

  __aicore__ inline __gm__ uint8_t *GetCubeStageBase(uint32_t slot) {
    uint64_t stateWorkspaceElements = static_cast<uint64_t>(B_) * NV_ * realK_ * stateWorkspaceStrideV_;
    uint64_t stageByteOffset =
      stateWorkspaceElements * sizeof(float) +
      (static_cast<uint64_t>(blockIdx_) * CUBE_STAGE_SLOT_COUNT + slot) * CUBE_STAGE_SLOT_BYTES;
    return workspaceAddr_ + stageByteOffset;
  }

  // ComputeRecursiveAttnBlocked64 solves the four diagonal 16x16 blocks with an
  // implicit unit diagonal and writes them back whole, so on that path the
  // explicit identity fill below would only rewrite 1.0 over 1.0.
  __aicore__ inline bool InverseDiagonalIsIdentity(uint32_t chunkLen) {
    if constexpr (kSpecializedDk == 0) {
      return false;
    }
    constexpr uint32_t kScratchMinV = (kSpecializedDk <= 64) ? 80 : kSpecializedDk;
    return chunkLen == kCubeFastPathChunkLen && chunkSize_ == kCubeFastPathChunkLen && vStepAligned_ >= kScratchMinV;
  }

  __aicore__ inline bool IsCubeFastPath(uint32_t chunkLen, uint32_t avFp32) {
    if constexpr (kSpecializedDk != 0) {
      return realK_ <= kSpecializedDk && chunkLen == kCubeFastPathChunkLen && vStepAligned_ == kSpecializedDk &&
             avFp32 == kSpecializedDk;
    }
    return false;
  }

  // Single source of truth for the staged attention fast path. StageAttnKeyOperands
  // publishes K^T into the per-core key cache only under this predicate, so every
  // reader of that cache (ComputeAttnMatrix / ProcessHeadPair) must gate on exactly
  // the same condition. A weaker condition (e.g. vStepAligned_ >= kSpecializedDk)
  // would let a consumer read a cache slot this chunk never wrote.
  __aicore__ inline bool CanUseCubeAttnPath(uint32_t chunkLen) {
    return IsCubeFastPath(chunkLen, vStepAligned_) && CanCacheAttnQuery(chunkLen);
  }

  // Copy a row-major GM slab to a zero-padded UB matrix using dav-2002
  // supported primitives. Aligned rows use one strided standard DataCopy;
  // non-aligned rows copy complete 32-byte blocks and fill at most 15 tail
  // elements through Scalar. The remaining 64/80/96/128 Cube tile stays zero.
  __aicore__ inline void LoadPaddedRows(LocalTensor<float> dst, LocalTensor<inType> staging, GlobalTensor<inType> src,
                                        uint64_t srcOffset, uint32_t rows, uint32_t gmRowElements) {
    uint32_t fullBlocks = realK_ / FP16_NUM_PER_BLOCK;
    uint32_t tailElements = realK_ % FP16_NUM_PER_BLOCK;
    uint32_t srcRowBlocks = gmRowElements / FP16_NUM_PER_BLOCK;
    uint32_t dstRowBlocks = alignK_ / FP16_NUM_PER_BLOCK;
    bool canUseStridedFullBlocks =
      fullBlocks > 0 && gmRowElements % FP16_NUM_PER_BLOCK == 0 && alignK_ % FP16_NUM_PER_BLOCK == 0 &&
      srcRowBlocks >= fullBlocks && dstRowBlocks >= fullBlocks &&
      srcRowBlocks - fullBlocks <= kDataCopyStrideBlockLimit && dstRowBlocks - fullBlocks <= kDataCopyStrideBlockLimit;
    // The zero fill only matters for padded columns/rows. When the copy covers
    // the whole tile (full chunk, realK == alignK, block aligned), every staged
    // element is overwritten and the fill is pure overhead.
    bool fullTileCopy = canUseStridedFullBlocks && tailElements == 0 && realK_ == alignK_;
    if (!fullTileCopy) {
      Duplicate(staging, static_cast<inType>(0), rows * alignK_);
      PipeBarrier<PIPE_V>();
    }
    event_t vectorToMte2 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(vectorToMte2);
    WaitFlag<HardEvent::V_MTE2>(vectorToMte2);

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
                             srcRowBlocks > 0 && dstRowBlocks > 0 && srcRowBlocks - 1 <= kDataCopyStrideBlockLimit &&
                             dstRowBlocks - 1 <= kDataCopyStrideBlockLimit;
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

  __aicore__ inline bool IsCurrentBlock(int32_t seqlen) { return IsCurrentBlockN(seqlen, 1); }

  // Same work distribution, but a work item may cover several value heads.
  __aicore__ inline bool IsCurrentBlockN(int32_t seqlen, uint32_t weight) {
    load_ += static_cast<uint32_t>(seqlen) * weight;
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
  __aicore__ inline void LoadMatmulBlockA(LocalTensor<half> a1Local, LocalTensor<half> a2Local) {
    for (uint32_t blockIdx = 0; blockIdx < kBlock / 16; ++blockIdx) {
      LoadData(a2Local[blockIdx * kBlock * 16], a1Local[blockIdx * 512 / sizeof(half)],
               LoadData2DParams{0, kBlock / 16, kBlock / 16, 0, 0, false, 0});
    }
  }

  template <uint32_t kBlock>
  __aicore__ inline void LoadMatmulBlockB(LocalTensor<half> b1Local, LocalTensor<half> b2Local) {
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

  // Execute one compensated kBlock x kBlock FP32 product on Cube. A is split
  // into FP16 high/low parts while B is used at FP16 precision, so
  // (A_hi + A_lo) * B_hi is accumulated in FP32 L0C. The dropped A_hi * B_lo
  // term is O(2^-11) of B's magnitude, far below the FP16 output resolution,
  // and dropping it saves one Cube pass per block. This keeps the block solve
  // close to the original FP32 recurrence while moving the dense cross-block
  // work off the scalar pipe.
  template <uint32_t kBlock>
  __aicore__ inline void MatmulBlockFp32Compensated(LocalTensor<float> srcA, LocalTensor<float> srcB,
                                                    LocalTensor<float> residualA, LocalTensor<float> dst) {
    constexpr uint32_t kMatrixElements = kBlock * kBlock;

    LocalMemAllocator<AscendC::Hardware::L1> l1Allocator;
    LocalMemAllocator<AscendC::Hardware::L0A> l0aAllocator;
    LocalMemAllocator<AscendC::Hardware::L0B> l0bAllocator;
    LocalMemAllocator<AscendC::Hardware::L0C> l0cAllocator;
    LocalTensor<half> aHiLocal = l1Allocator.Alloc<TPosition::A1, half>(kMatrixElements);
    LocalTensor<half> aLoLocal = l1Allocator.Alloc<TPosition::A1, half>(kMatrixElements);
    LocalTensor<half> b1Local = l1Allocator.Alloc<TPosition::B1, half>(kMatrixElements);
    LocalTensor<half> a2Local = l0aAllocator.Alloc<TPosition::A2, half>(kMatrixElements);
    LocalTensor<half> b2Local = l0bAllocator.Alloc<TPosition::B2, half>(kMatrixElements);
    LocalTensor<float> c1Local = l0cAllocator.Alloc<TPosition::CO1, float>(kMatrixElements);

    // Two-term compensation: (A_hi + A_lo) * B_hi. B's FP16 quantization is
    // O(2^-11) of its magnitude, far below the FP16 output resolution, so the
    // A_hi * B_lo cross term is dropped (one Cube pass saved per block).
    StageHalfWithResidualToL1<kBlock, kBlock>(srcA, residualA, aHiLocal, aLoLocal);
    StageHalfToL1<kBlock, kBlock>(srcB, b1Local);
    SetFlag<HardEvent::MTE3_MTE1>(0);
    WaitFlag<HardEvent::MTE3_MTE1>(0);

    LoadMatmulBlockA<kBlock>(aHiLocal, a2Local);
    LoadMatmulBlockB<kBlock>(b1Local, b2Local);
    MmadMatmulBlock<kBlock>(c1Local, a2Local, b2Local, true);
    LoadMatmulBlockA<kBlock>(aLoLocal, a2Local);
    MmadMatmulBlock<kBlock>(c1Local, a2Local, b2Local, false);

    LocalTensor<float> cNz = residualA;
    DataCopyParams cCopyParams{static_cast<uint16_t>(kBlock / 16), static_cast<uint16_t>(kBlock / 16), 0, 0};
    DataCopyEnhancedParams cCopyEnhanced;
    cCopyEnhanced.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    CopyL0CToUb(cNz, c1Local, cCopyParams, cCopyEnhanced);
    PipeBarrier<PIPE_ALL>();
    NzTileToNd<kBlock, kBlock>(dst, cNz);
  }

  // Solve one 16x16 diagonal block entirely on the Vector pipe. For every
  // source row k, Gather extracts A[k+1:, k], Brcb expands its coefficients,
  // and one strided MulAddDst applies the outer-product update
  //   U[k+1:, :] += A[k+1:, k] * U[k, :].
  // The only scalar work left is the fixed 15-step dependency schedule and
  // setup of the 16 diagonal/offset entries; the O(block^3) arithmetic and
  // all coefficient application stay on PIPE_V.
  // Solve all four 16x16 diagonal blocks of the 64x64 strict-lower A in one
  // batched Vector recurrence. The blocks are independent, so a single Gather
  // plus one Brcb feed four independent MulAddDst per step, collapsing four
  // serial 15-step chains into 15 steps whose only dependency is the k chain.
  // Every element still receives exactly the same FP32 multiply-add sequence
  // as the former per-block version, so the result is bit-identical.
  // The Gather offset table is independent of k: block b row j sits at
  // b*16*(ld+1) + j*ld relative to (k+1, k), so only srcBase moves with k.
  __aicore__ inline void ComputeRecursiveAttnDiagonals64(LocalTensor<float> &buf, uint32_t ld) {
    constexpr uint32_t kBlock = 16;
    constexpr uint32_t kBlocks = 4;
    constexpr uint32_t kBlockElements = kBlock * kBlock;
    constexpr uint32_t kColumnCount = kBlocks * kBlock;
    LocalTensor<float> inverse = chunkVFp32;
    LocalTensor<float> coefficientBlocks = chunkVFp32[kBlocks * kBlockElements];
    LocalTensor<float> column = deltaFp32;
    LocalTensor<uint32_t> gatherOffsets = dotProductFp32.template ReinterpretCast<uint32_t>();

    Duplicate(inverse, 0.0f, kBlocks * kBlockElements);
    TQueSync<PIPE_V, PIPE_S> initVectorToScalar;
    initVectorToScalar.SetFlag(0);
    initVectorToScalar.WaitFlag(0);
    for (uint32_t b = 0; b < kBlocks; ++b) {
      for (uint32_t d = 0; d < kBlock; ++d) {
        inverse.SetValue(b * kBlockElements + d * kBlock + d, 1.0f);
      }
    }
    for (uint32_t b = 0; b < kBlocks; ++b) {
      for (uint32_t j = 0; j < kBlock; ++j) {
        gatherOffsets.SetValue(b * kBlock + j, (b * kBlock * (ld + 1) + j * ld) * sizeof(float));
      }
    }
    TQueSync<PIPE_S, PIPE_V> initScalarToVector;
    initScalarToVector.SetFlag(0);
    initScalarToVector.WaitFlag(0);

    for (uint32_t k = 0; k + 1 < kBlock; ++k) {
      uint32_t validRows = kBlock - k - 1;
      // All three operations are PIPE_V and execute in issue order, so same-pipe
      // barriers between them are redundant; only the RAW chain must be intact.
      Gather(column, buf, gatherOffsets, static_cast<uint32_t>(((k + 1) * ld + k) * sizeof(float)), kColumnCount);
      Brcb(coefficientBlocks, column, static_cast<uint8_t>(Ceil(kColumnCount, FP32_NUM_PER_BLOCK)),
           BrcbRepeatParams{1, 8});
      for (uint32_t b = 0; b < kBlocks; ++b) {
        MulAddDst(inverse[b * kBlockElements + (k + 1) * kBlock], coefficientBlocks[b * kBlock * FP32_NUM_PER_BLOCK],
                  inverse[b * kBlockElements + k * kBlock], k + 1, static_cast<uint8_t>(validRows),
                  BinaryRepeatParams{1, 0, 1, kBlock / FP32_NUM_PER_BLOCK, 1, 0});
      }
    }

    for (uint32_t b = 0; b < kBlocks; ++b) {
      CopyRowsFp32(buf[b * kBlock * ld + b * kBlock], ld, inverse[b * kBlockElements], kBlock, kBlock, kBlock);
    }
  }

  // One strided FP32 row-block copy: `rows` x `cols` floats with independent
  // source/destination leading dimensions. Replaces the former per-row
  // Muls(x, 1.0f) loops (16-32 Vector instructions each) by a single
  // DataCopy; the copy is bit-identical because it moves the same bytes.
  __aicore__ inline void CopyRowsFp32(LocalTensor<float> dst, uint32_t dstLd, LocalTensor<float> src, uint32_t srcLd,
                                      uint32_t rows, uint32_t cols) {
    DataCopyParams params{static_cast<uint16_t>(rows), static_cast<uint16_t>(cols * sizeof(float) / BLOCK_BYTES),
                          static_cast<uint16_t>((srcLd - cols) * sizeof(float) / BLOCK_BYTES),
                          static_cast<uint16_t>((dstLd - cols) * sizeof(float) / BLOCK_BYTES)};
    DataCopy(dst, src, params);
    PipeBarrier<PIPE_ALL>();
  }

  // Build diag(A0, A1) into a contiguous 32x32 tile from two 16x16 sub-blocks
  // of a row-strided (ld) buffer; the off-diagonal 16x16 blocks are exact zeros.
  __aicore__ inline void BuildBlockDiag32(LocalTensor<float> dst, LocalTensor<float> &buf, uint32_t ld, uint32_t r0,
                                          uint32_t c0, uint32_t r1, uint32_t c1) {
    Duplicate(dst, 0.0f, kBlockDiagWidth * kBlockDiagWidth);
    PipeBarrier<PIPE_V>();
    CopyRowsFp32(dst, kBlockDiagWidth, buf[r0 * ld + c0], ld, kBlockDiagHalf, kBlockDiagHalf);
    CopyRowsFp32(dst[kBlockDiagHalf * kBlockDiagWidth + kBlockDiagHalf], kBlockDiagWidth, buf[r1 * ld + c1], ld,
                 kBlockDiagHalf, kBlockDiagHalf);
  }

  __aicore__ inline void WriteBlockDiag32(LocalTensor<float> &buf, uint32_t ld, LocalTensor<float> src, uint32_t r0,
                                          uint32_t c0, uint32_t r1, uint32_t c1) {
    CopyRowsFp32(buf[r0 * ld + c0], ld, src, kBlockDiagWidth, kBlockDiagHalf, kBlockDiagHalf);
    CopyRowsFp32(buf[r1 * ld + c1], ld, src[kBlockDiagHalf * kBlockDiagWidth + kBlockDiagHalf], kBlockDiagWidth,
                 kBlockDiagHalf, kBlockDiagHalf);
  }

  // For a 64x64 strict-lower A, invert I-A with a 2x2 block factorization:
  //   U00=(I-A00)^-1, U11=(I-A11)^-1,
  //   U10=U11*A10*U00.
  // The diagonal solves remain FP32 Vector recurrences; compensated Cube
  // products handle the dense cross block without weakening FP32 stability.
  __aicore__ inline void ComputeRecursiveAttnBlocked64(LocalTensor<float> &buf, uint32_t ld) {
    constexpr uint32_t kBlock = kBlockDiagWidth;
    constexpr uint32_t kMatrixElements = kBlock * kBlock;
    // Four independent 16x16 diagonal inverses, solved as one batched recurrence.
    ComputeRecursiveAttnDiagonals64(buf, ld);

    LocalTensor<float> srcA = chunkVFp32;
    LocalTensor<float> srcB = chunkVFp32[kMatrixElements];
    LocalTensor<float> residualA = chunkVFp32[2 * kMatrixElements];
    LocalTensor<float> product = chunkVFp32[3 * kMatrixElements];

    // The two 32x32 sub-blocks are independent, so batch their dependent
    // product pairs into block-diagonal 32x32 products: exact zeros contribute
    // +0.0f only, keeping the FP32 accumulation bit-identical.
    // A10 of both sub-blocks.
    BuildBlockDiag32(srcA, buf, ld, kBlockDiagHalf, 0, kBlock + kBlockDiagHalf, kBlock);
    // U00 of both sub-blocks.
    BuildBlockDiag32(srcB, buf, ld, 0, 0, kBlock, kBlock);
    MatmulBlockFp32Compensated<kBlock>(srcA, srcB, residualA, product);
    // U11 of both sub-blocks.
    BuildBlockDiag32(srcB, buf, ld, kBlockDiagHalf, kBlockDiagHalf, kBlock + kBlockDiagHalf, kBlock + kBlockDiagHalf);
    MatmulBlockFp32Compensated<kBlock>(srcB, product, residualA, product);
    WriteBlockDiag32(buf, ld, product, kBlockDiagHalf, 0, kBlock + kBlockDiagHalf, kBlock);

    // Top-level 2x2 factorization for the full 64x64 block (unchanged).
    CopyRowsFp32(srcA, kBlock, buf[kBlock * ld], ld, kBlock, kBlock);
    CopyRowsFp32(srcB, kBlock, buf, ld, kBlock, kBlock);
    MatmulBlockFp32Compensated<kBlock>(srcA, srcB, residualA, product);

    CopyRowsFp32(srcA, kBlock, buf[kBlock * ld + kBlock], ld, kBlock, kBlock);
    MatmulBlockFp32Compensated<kBlock>(srcA, product, residualA, product);

    CopyRowsFp32(buf[kBlock * ld], ld, product, kBlock, kBlock, kBlock);
  }

  // Compute recursive intra-chunk attn matrix in-place.
  // buf is stored with leading dimension ld (= chunkSize_), and only [chunkLen, chunkLen] is valid.
  // buf is lower-triangular (excluding diagonal), with upper tri (incl diagonal) = 0.
  __aicore__ inline void ComputeRecursiveAttn(LocalTensor<float> &buf, uint32_t chunkLen, uint32_t ld) {
    // ComputeRecursiveAttnBlocked64 carves 4 disjoint kBlock*kBlock blocks out of
    // chunkVFp32 (A, B, A residual and the product tile), so the scratch must
    // hold cs * vStepAligned_ >= 4 * kBlock * kBlock elements. The guard below
    // is kept at the historical vStep>=80 threshold: lowering it would make
    // small-Dk shapes switch from the Vector recurrence to this blocked path,
    // which changes their rounding, so the switch is deliberately not made.
    constexpr uint32_t kBlockedScratchMinV = (kSpecializedDk == 0) ? 128 : (kSpecializedDk <= 64 ? 80 : kSpecializedDk);
    // The blocked inverse is only used together with the complete Cube fast path
    // (realK_ within the compiled bucket). Scalar-built attention data takes the
    // plain recurrence below: mixing it with the blocked path is not supported.
    if (likely(realK_ <= static_cast<uint32_t>(kSpecializedDk) && chunkLen == kCubeFastPathChunkLen &&
               ld == kCubeFastPathChunkLen && vStepAligned_ >= kBlockedScratchMinV)) {
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
    // Touch the finished tile on the Vector pipe before the cast. Removing this
    // pass measured ~0.2% slower end to end (it lets the NZ->ND store retire
    // before the cast starts), so it is kept deliberately.
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

  // Per-core GM slot 2 caches the raw (pre-beta/decay) attention products of
  // the chunk currently shared by a value-head pair.
  __aicore__ inline void SaveAttnRawToGm() {
    constexpr uint32_t kCElems = 64 * 64;
    __gm__ uint8_t *slot2 = GetCubeStageBase(kAttnRawSlot);
    GlobalTensor<float> gmScores;
    GlobalTensor<float> gmQk;
    gmScores.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(slot2), kCElems);
    gmQk.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(slot2) + kCElems, kCElems);
    event_t vToMte3 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(vToMte3);
    WaitFlag<HardEvent::V_MTE3>(vToMte3);
    DataCopy(gmScores, chunkScoresFp32, kCElems);
    DataCopy(gmQk, chunkAttnOutFp32, kCElems);
    event_t mte3ToMte2 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2);
    PipeBarrier<PIPE_ALL>();
  }

  __aicore__ inline void RestoreAttnRawFromGm() {
    constexpr uint32_t kCElems = 64 * 64;
    __gm__ uint8_t *slot2 = GetCubeStageBase(kAttnRawSlot);
    GlobalTensor<float> gmScores;
    GlobalTensor<float> gmQk;
    gmScores.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(slot2), kCElems);
    gmQk.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(slot2) + kCElems, kCElems);
    DataCopy(chunkScoresFp32, gmScores, kCElems);
    DataCopy(chunkAttnOutFp32, gmQk, kCElems);
    event_t mte2ToVector = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(mte2ToVector);
    WaitFlag<HardEvent::MTE2_V>(mte2ToVector);
    PipeBarrier<PIPE_ALL>();
  }

  // Two value heads that share a qkHead are processed chunk by chunk: the
  // shared prework (K/K^T staging, Q cache, raw K@K^T and Q@K^T) runs once,
  // then each head applies its own beta/decay and runs the per-head pipeline.
  __aicore__ inline void ProcessHeadPair(int32_t seq0, int32_t seq1, uint64_t head_i, uint64_t batch_i) {
    uint64_t nvPerNk = (NV_ >= NK_) ? (NV_ / NK_) : 1;
    uint64_t qkHead = head_i / nvPerNk;
    uint32_t cs = chunkSize_;
    uint32_t avFp32 = vStepAligned_;
    int32_t totalLen = seq1 - seq0;
    uint32_t numC = Ceil(static_cast<uint32_t>(totalLen), cs);
    for (uint32_t c = 0; c < numC; c++) {
      int32_t t_start = seq0 + c * cs;
      uint32_t chunkLen =
        (static_cast<uint32_t>(totalLen) - c * cs >= cs) ? cs : static_cast<uint32_t>(totalLen) - c * cs;
      bool isLastChunk = (c + 1 == numC);
      bool pairFast = false;
      if constexpr (kSpecializedDk != 0) {
        // Must match the StageAttnKeyOperands predicate below, which is what
        // fills the key cache consumed by this chunk's attention products.
        pairFast = CanUseCubeAttnPath(chunkLen);
      }
      if (!pairFast) {
        for (uint32_t hv = 0; hv < kPairValueHeads; ++hv) {
          uint64_t hi = head_i + hv;
          uint64_t stateBaseOffset = (batch_i * NV_ + hi) * realK_ * realV_;
          uint64_t workspaceStateBaseOffset = (batch_i * NV_ + hi) * realK_ * stateWorkspaceStrideV_;
          ProcessChunk(seq0, seq1, hi, c, qkHead, stateBaseOffset, workspaceStateBaseOffset, isLastChunk);
        }
        continue;
      }
      keyCacheValid_ = 0;
      LoadChunkKey(qkHead, t_start, chunkLen);
      StageAttnKeyOperands(qkHead, t_start, chunkLen);
      // Head 0's coefficients must be loaded before the raw products: the
      // coefficient DMA stages through chunkAttnOutFp32, which holds the raw
      // Q @ K^T tile afterwards.
      LoadChunkCoefficients(t_start, head_i, chunkLen);
      ComputeKBeta(chunkLen);
      PrepareDecayAndExp(chunkLen);
      PrepareAttnQueryCache(qkHead, t_start, chunkLen);
      // Stop after the raw products: each head of the pair post-processes them
      // from the per-core GM slot below.
      ComputeAttnProductsCube<kSpecializedDk>(true);
      SaveAttnRawToGm();
      for (uint32_t hv = 0; hv < kPairValueHeads; ++hv) {
        uint64_t hi = head_i + hv;
        uint64_t stateBaseOffset = (batch_i * NV_ + hi) * realK_ * realV_;
        uint64_t workspaceStateBaseOffset = (batch_i * NV_ + hi) * realK_ * stateWorkspaceStrideV_;
        if (hv > 0) {
          // The first head's k_cumdecay overwrote the K cache and its
          // coefficients overwrote beta/g; reload both, then restore the raw
          // attention products from the per-core GM slot.
          LoadChunkKey(qkHead, t_start, chunkLen);
          LoadChunkCoefficients(t_start, hi, chunkLen);
          ComputeKBeta(chunkLen);
          PrepareDecayAndExp(chunkLen);
          RestoreAttnRawFromGm();
        }
        PostProcessAttnProducts();
        ComputeRecursiveAttn(chunkScoresFp32, chunkLen, cs);
        if (!InverseDiagonalIsIdentity(chunkLen)) {
          for (uint32_t i = 0; i < chunkLen; i++) {
            chunkScoresFp32.SetValue(i * cs + i, 1.0f);
          }
        }
        ComputeKCumdecay(chunkLen);
        ProcessVTiles(t_start, chunkLen, hi, qkHead, avFp32, stateBaseOffset, workspaceStateBaseOffset, c, isLastChunk);
      }
    }
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

    keyCacheValid_ = 0;
    LoadChunkKey(qkHead, t_start, chunkLen);
    StageAttnKeyOperands(qkHead, t_start, chunkLen);
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

    keyCacheValid_ = 0;
    LoadChunkKey(qkHead, t_start, chunkLen);
    StageAttnKeyOperands(qkHead, t_start, chunkLen);
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

  // Pre-stage the exact-FP16 K and K^T Cube operands used by the two
  // attention products. Runs before ComputeKBeta so the temporary FP16
  // transpose buffer (kCumdecayFp32, still free) can be reused safely.
  __aicore__ inline void StageAttnKeyOperands(uint64_t qkHead, int32_t t_start, uint32_t chunkLen) {
    if (likely(!CanUseCubeAttnPath(chunkLen))) {
      return;
    }
    constexpr uint32_t kMatmulM = 64;
    constexpr uint32_t kAElements = kMatmulM * kSpecializedDk;
    constexpr uint32_t kBElements = kSpecializedDk * kMatmulM;

    LocalTensor<half> keyTrans = chunkVFp32.template ReinterpretCast<half>();
    LocalTensor<half> keyT = kCumdecayFp32.template ReinterpretCast<half>();
    Cast(keyTrans, chunkKFp32, RoundMode::CAST_NONE, kAElements);
    PipeBarrier<PIPE_V>();
    TransDataTo5HDParams transposeParams;
    transposeParams.repeatTimes = static_cast<uint8_t>(kSpecializedDk / kTransposeBlockDim);
    transposeParams.srcRepStride = 1;
    transposeParams.dstRepStride = kMatmulM;
    for (uint32_t block = 0; block < kMatmulM / kTransposeBlockDim; ++block) {
      LocalTensor<half> srcRows[kTransposeBlockDim];
      LocalTensor<half> dstRows[kTransposeBlockDim];
      for (uint32_t row = 0; row < kTransposeBlockDim; ++row) {
        srcRows[row] = keyTrans[block * kTransposeBlockDim * kSpecializedDk + row * kSpecializedDk];
        dstRows[row] = keyT[block * kTransposeBlockDim + row * kMatmulM];
      }
      TransDataTo5HD<half>(dstRows, srcRows, transposeParams);
    }
    PipeBarrier<PIPE_V>();
    event_t vectorToMte3 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(vectorToMte3);
    WaitFlag<HardEvent::V_MTE3>(vectorToMte3);
    // K^T is stored once, in the per-core key cache: the attention product and
    // the later state update both read it from there, so the former duplicate
    // store into the shared cube stage slot is gone. The A operand (K, exact
    // FP16) is no longer published through GM either: its consumer re-casts
    // chunkKFp32 straight into the L1 NZ tile (see ComputeAttnProductsCube).
    GlobalTensor<half> keyCacheGm;
    keyCacheGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(GetKeyCacheBase()), kBElements);
    DataCopy(keyCacheGm, keyT, kBElements);
    keyCacheValid_ = 1;
    PipeBarrier<PIPE_ALL>();
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
    // CreateVecIndex and Muls are both PIPE_V (issue order is the dependency).
    CreateVecIndex<int32_t>(signedOffsets, 0, chunkLen);
    Muls(signedOffsets, signedOffsets, static_cast<int32_t>(NV_ * sizeof(float)), chunkLen);
  }

  // Inclusive prefix sum for at most one 64-row chunk. Ping-ponging between
  // two UB tensors avoids the loop-carried scalar GetValue/SetValue chain.
  __aicore__ inline void PrefixSumChunkG(uint32_t chunkLen) {
    LocalTensor<float> src = gCumsumFp32;
    LocalTensor<float> dst = deltaFp32;
    bool resultInGCumsum = true;
    // Muls/Add are both PIPE_V and execute in issue order, so the barriers
    // between them are redundant (the caller still publishes V->S afterwards).
    for (uint32_t offset = 1; offset < chunkLen; offset <<= 1) {
      Muls(dst, src, 1.0f, chunkLen);
      Add(dst[offset], src[offset], src, chunkLen - offset);
      LocalTensor<float> tmp = src;
      src = dst;
      dst = tmp;
      resultInGCumsum = !resultInGCumsum;
    }
    if (!resultInGCumsum) {
      Muls(gCumsumFp32, src, 1.0f, chunkLen);
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
      // Keep the FP16 source separate because an in-place widening Cast overwrites unread beta values.
      LocalTensor<inType> betaLocal = chunkAttnOutFp32.template ReinterpretCast<inType>();
      event_t vectorToMte2 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE2));
      SetFlag<HardEvent::V_MTE2>(vectorToMte2);
      WaitFlag<HardEvent::V_MTE2>(vectorToMte2);
      DataCopy(betaLocal, betaGm_[static_cast<uint64_t>(t_start) * NV_], slabElements);
      event_t mte2ToVector = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
      SetFlag<HardEvent::MTE2_V>(mte2ToVector);
      WaitFlag<HardEvent::MTE2_V>(mte2ToVector);
      // Cast / CreateVecIndex / Muls / Gather are all ordered by the S->V
      // handshake inside CreateVecIndex (it is a scalar SetValue loop on
      // dav_m200) plus same-pipe issue order, so no PIPE_V barrier is needed.
      Cast(chunkVFp32, betaLocal, RoundMode::CAST_NONE, slabElements);
      BuildCoefficientOffsets(coefficientOffsets, chunkLen);
      offsetsReady = true;
      Gather(betaFp32, chunkVFp32, coefficientOffsets, static_cast<uint32_t>(head_i * sizeof(float)), chunkLen);
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
      PrefixSumChunkG(chunkLen);
    } else if (hasGamma_ == 0) {
      Duplicate(gCumsumFp32, 0.0f, chunkLen);
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

    // Brcb / Duplicate / Mul are all PIPE_V and execute in issue order, so no
    // same-pipe barrier is needed between them.
    LocalTensor<float> betaBlocks = chunkVFp32;
    uint8_t brcbRepeats = static_cast<uint8_t>(Ceil(chunkLen, FP32_NUM_PER_BLOCK));
    Brcb(betaBlocks, betaFp32, brcbRepeats, BrcbRepeatParams{1, 8});

    // The padded K columns must remain exact zeros because all Cube products
    // consume the complete bucket width. A full-width, full-chunk tile is
    // entirely overwritten by the Mul below, so the fill is skipped there.
    if (!(realK_ == alignK_ && chunkLen == chunkSize_)) {
      Duplicate(kCumdecayFp32, 0.0f, chunkLen * alignK_);
    }

    uint8_t rowStride = static_cast<uint8_t>(alignK_ / FP32_NUM_PER_BLOCK);
    BinaryRepeatParams mulParams{1, 1, 0, rowStride, rowStride, 1};
    uint32_t kOffset = 0;
    for (; kOffset + kVectorMulBlockSize <= realK_; kOffset += kVectorMulBlockSize) {
      Mul(kCumdecayFp32[kOffset], chunkKFp32[kOffset], betaBlocks, static_cast<uint64_t>(kVectorMulBlockSize),
          static_cast<uint8_t>(chunkLen), mulParams);
    }
    if (kOffset < realK_) {
      Mul(kCumdecayFp32[kOffset], chunkKFp32[kOffset], betaBlocks, static_cast<uint64_t>(realK_ - kOffset),
          static_cast<uint8_t>(chunkLen), mulParams);
    }
    // The trailing V->S publication already orders the Mul chain for the scalar
    // readers, so the former PipeBarrier<PIPE_V>() here was redundant.
    TQueSync<PIPE_V, PIPE_S> betaSync;
    betaSync.SetFlag(0);
    betaSync.WaitFlag(0);
  }

  // Phase 3: precompute exp(gCumsum) and decay_mask[i][j] = exp(g_cumsum[i] -
  // g_cumsum[j]).
  __aicore__ inline void PrepareDecayAndExp(uint32_t chunkLen) {
    uint32_t cs = chunkSize_;
    // All operations below are PIPE_V; in-order issue makes same-pipe barriers
    // redundant, so they are omitted (the trailing V->S publication is kept).
    Muls(gCumsumFp32, gCumsumFp32, 1.0f, chunkLen);
    Exp(expGCumFp32, gCumsumFp32, chunkLen);

    // g is a log-decay and its prefix sum can easily be below -100 for real
    // Qwen activations. Factoring exp(g_i - g_j) into exp(g_i)*exp(-g_j)
    // overflows the second factor. Build all rows with high-dimensional Vector
    // instructions, clear the non-causal upper triangle before Exp, then issue
    // one strided Exp for the complete causal matrix.
    uint8_t rowStride = static_cast<uint8_t>(cs / FP32_NUM_PER_BLOCK);
    UnaryRepeatParams rowBroadcastParams{1, 1, rowStride, 0};
    Muls(decayMaskFp32, gCumsumFp32, -1.0f, static_cast<uint64_t>(chunkLen), static_cast<uint8_t>(chunkLen),
         rowBroadcastParams);

    LocalTensor<float> gBlocks = chunkVFp32;
    uint8_t brcbRepeats = static_cast<uint8_t>(Ceil(chunkLen, FP32_NUM_PER_BLOCK));
    Brcb(gBlocks, gCumsumFp32, brcbRepeats, BrcbRepeatParams{1, 8});
    BinaryRepeatParams rowAddParams{1, 1, 0, rowStride, rowStride, 1};
    Add(decayMaskFp32, decayMaskFp32, gBlocks, static_cast<uint64_t>(chunkLen), static_cast<uint8_t>(chunkLen),
        rowAddParams);

    UnaryRepeatParams rowExpParams{1, 1, rowStride, rowStride};
    Exp(decayMaskFp32, decayMaskFp32, static_cast<uint64_t>(chunkLen), static_cast<uint8_t>(chunkLen), rowExpParams);
    // Clear the upper triangle after Exp so it stays exactly zero (clearing
    // before Exp would make it exp(0)=1 and every consumer would have to mask
    // again). The raw attention products are then already lower-triangular.
    for (uint32_t row = 0; row + 1 < chunkLen; ++row) {
      Duplicate(decayMaskFp32[row * cs + row + 1], 0.0f, chunkLen - row - 1);
    }
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
    // Q is exact FP16; keep it unscaled and apply scale after the exact Q @ K^T.
    PipeBarrier<PIPE_V>();
  }

  // Phase 4: attn = -((k_beta @ K^T) * decay_mask) (lower tri), recursive accumulation, identity
  // diagonal; then attn_i = (Q @ K^T) * decay_mask overwrites decayMaskFp32 (lower tri + diag).
  __aicore__ inline void ComputeAttnMatrix(uint64_t qkHead, int32_t t_start, uint32_t chunkLen) {
    uint32_t cs = chunkSize_;
    if constexpr (kSpecializedDk != 0) {
      // keyCacheValid_ was just published by the StageAttnKeyOperands call that
      // precedes this function; require it so the Cube path can never read a
      // cache slot this chunk did not write.
      if (likely(CanUseCubeAttnPath(chunkLen) && keyCacheValid_ != 0)) {
        ComputeAttnProductsCube<kSpecializedDk>();
        ComputeRecursiveAttn(chunkScoresFp32, chunkLen, cs);
        if (!InverseDiagonalIsIdentity(chunkLen)) {
          for (uint32_t i = 0; i < chunkLen; i++) {
            chunkScoresFp32.SetValue(i * cs + i, 1.0f);
          }
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
    // The scores above are built by Scalar stores while the recursive solve reads
    // them on the Vector pipe; Scalar->Vector must be published explicitly.
    TQueSync<PIPE_S, PIPE_V> scoreBuildToVector;
    scoreBuildToVector.SetFlag(0);
    scoreBuildToVector.WaitFlag(0);
    ComputeRecursiveAttn(chunkScoresFp32, chunkLen, cs);
    for (uint32_t i = 0; i < chunkLen; i++) {
      chunkScoresFp32.SetValue(i * cs + i, 1.0f);
    }
    TQueSync<PIPE_S, PIPE_V> identityToVector;
    identityToVector.SetFlag(0);
    identityToVector.WaitFlag(0);
    if (CanCacheAttnQuery(chunkLen)) {
      for (uint32_t i = 0; i < chunkLen; i++) {
        LocalTensor<float> queryRow = chunkAttnOutFp32[i * alignK_];
        for (uint32_t jj = 0; jj <= i; jj++) {
          float dot = DotFp32(queryRow, chunkKFp32[jj * alignK_], realK_);
          float decay = decayMaskFp32.GetValue(i * cs + jj);
          decayMaskFp32.SetValue(i * cs + jj, dot * scale_ * decay);
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
          // Q was already scaled when loaded above.
          decayMaskFp32.SetValue(i * cs + jj, dot * decay);
        }
      }
    }
    // attn_i was stored into decayMaskFp32 by Scalar; publish it to Vector.
    TQueSync<PIPE_S, PIPE_V> attnInterToVector;
    attnInterToVector.SetFlag(0);
    attnInterToVector.WaitFlag(0);
  }

  // Batch the two dense attention products on Cube while K^T remains resident
  // in L0B. K and Q are exact FP16 inputs, so each product is a single exact
  // Cube pass; beta / decay / scale are applied afterwards in FP32:
  //   scores = -(K @ K^T * beta_row) * decay
  //   attn_i = (Q @ K^T) * decay * scale.
  // This replaces 4096 row-pair DotFp32 reductions on the fixed 64x128 path.
  template <uint32_t kMatmulK>
  __aicore__ inline void ComputeAttnProductsCube(bool rawOnly = false) {
    constexpr uint32_t kMatmulM = 64;
    constexpr uint32_t kMatmulN = 64;
    constexpr uint32_t kAElements = kMatmulM * kMatmulK;
    constexpr uint32_t kBElements = kMatmulK * kMatmulN;
    constexpr uint32_t kCElements = kMatmulM * kMatmulN;
    // Both A operands (K and Q, exact FP16) are staged from UB straight into
    // the L1 NZ tile: on dav_m200 Nd2Nz is a software GM->UB->L1 emulation, so
    // the GM round trip is pure overhead. Q is exact FP16, so a plain cast copy
    // is exact (no residual needed) and it stays in chunkAttnOutFp32 until the
    // second product needs it.
    PipeBarrier<PIPE_ALL>();

    LocalMemAllocator<AscendC::Hardware::L1> l1Allocator;
    LocalMemAllocator<AscendC::Hardware::L0A> l0aAllocator;
    LocalMemAllocator<AscendC::Hardware::L0B> l0bAllocator;
    LocalMemAllocator<AscendC::Hardware::L0C> l0cAllocator;
    LocalTensor<half> a1Local = l1Allocator.Alloc<TPosition::A1, half>(kAElements);
    LocalTensor<half> b1Local = l1Allocator.Alloc<TPosition::B1, half>(kBElements);
    LocalTensor<half> a2Local = l0aAllocator.Alloc<TPosition::A2, half>(kAElements);
    LocalTensor<half> b2Local = l0bAllocator.Alloc<TPosition::B2, half>(kBElements);
    LocalTensor<float> c1Local = l0cAllocator.Alloc<TPosition::CO1, float>(kCElements);

    // B = K^T: one contiguous GM->UB load of the per-core cached ND tile, then
    // the same strided UB->L1 move used by every other staging site. The cache
    // is the only producer of this tile (the former copy into the shared cube
    // stage slot is gone), and ComputeAttnMatrix only reaches here after
    // StageAttnKeyOperands published it for this chunk. chunkVFp32 is free here
    // (its last use was PrepareDecayAndExp's g broadcast).
    GlobalTensor<half> keyTSrc;
    keyTSrc.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(GetKeyCacheBase()), kBElements);
    LocalTensor<half> nzScratch = chunkVFp32.template ReinterpretCast<half>();
    DataCopy(nzScratch, keyTSrc, kBElements);
    event_t mte2ToMte3 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_MTE3));
    SetFlag<HardEvent::MTE2_MTE3>(mte2ToMte3);
    WaitFlag<HardEvent::MTE2_MTE3>(mte2ToMte3);
    CopyUbNdToL1Nz<kMatmulK, kMatmulN>(b1Local, nzScratch);
    StageHalfToL1<kMatmulM, kMatmulK>(chunkKFp32, a1Local);
    SetFlag<HardEvent::MTE3_MTE1>(0);
    WaitFlag<HardEvent::MTE3_MTE1>(0);
    for (uint32_t mBlock = 0; mBlock < kMatmulM / kNzFractalDim; ++mBlock) {
      LoadData(a2Local[mBlock * kMatmulK * kNzFractalDim], a1Local[mBlock * kNzFractalRowBytes / sizeof(half)],
               LoadData2DParams{0, kMatmulK / kNzFractalDim, kMatmulM / kNzFractalDim, 0, 0, false, 0});
    }
    for (uint32_t kBlock = 0; kBlock < kMatmulK / kNzFractalDim; ++kBlock) {
      LoadData(b2Local[kBlock * kMatmulN * kNzFractalDim], b1Local[kBlock * kNzFractalRowBytes / sizeof(half)],
               LoadData2DParams{0, kMatmulN / kNzFractalDim, kMatmulK / kNzFractalDim, 0, 0, true, 0});
    }
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    Mmad(c1Local, a2Local, b2Local, MmadParams{kMatmulM, kMatmulN, kMatmulK, 0, false, true});
    CopyAttnCubeResult(c1Local, chunkScoresFp32);

    // Reuse the resident K^T for the exact Q @ K^T product. a1Local was read
    // by the first product's LoadData, so the MTE3 rewrite must wait for MTE1.
    event_t mte1ToMte3 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE1_MTE3));
    SetFlag<HardEvent::MTE1_MTE3>(mte1ToMte3);
    WaitFlag<HardEvent::MTE1_MTE3>(mte1ToMte3);
    StageHalfToL1<kMatmulM, kMatmulK>(chunkAttnOutFp32, a1Local);
    SetFlag<HardEvent::MTE3_MTE1>(0);
    WaitFlag<HardEvent::MTE3_MTE1>(0);
    for (uint32_t mBlock = 0; mBlock < kMatmulM / kNzFractalDim; ++mBlock) {
      LoadData(a2Local[mBlock * kMatmulK * kNzFractalDim], a1Local[mBlock * kNzFractalRowBytes / sizeof(half)],
               LoadData2DParams{0, kMatmulK / kNzFractalDim, kMatmulM / kNzFractalDim, 0, 0, false, 0});
    }
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    Mmad(c1Local, a2Local, b2Local, MmadParams{kMatmulM, kMatmulN, kMatmulK, 0, false, true});
    CopyAttnCubeResult(c1Local, chunkAttnOutFp32);

    if (rawOnly) {
      TQueSync<PIPE_V, PIPE_S> rawSync;
      rawSync.SetFlag(0);
      rawSync.WaitFlag(0);
      return;
    }
    PostProcessAttnProducts();
  }

  // Convert the raw K@K^T / Q@K^T tiles into scores and attn_i for one value
  // head (beta row scaling, decay mask, scale).
  __aicore__ inline void PostProcessAttnProducts() {
    constexpr uint32_t kMatmulM = 64;
    constexpr uint32_t kMatmulN = 64;
    constexpr uint32_t kCElements = kMatmulM * kMatmulN;
    LocalTensor<float> betaBlocks = chunkVFp32;
    uint8_t brcbRepeats = static_cast<uint8_t>(Ceil(kMatmulM, FP32_NUM_PER_BLOCK));
    // Every operation below is PIPE_V and executes in issue order, so the
    // same-pipe barriers between them are redundant: only the final V->S
    // publication is required.
    Brcb(betaBlocks, betaFp32, brcbRepeats, BrcbRepeatParams{1, 8});
    // scores = -(raw * beta_row * decay): the negation is folded into the 64-entry
    // beta broadcast instead of a 4096-element Muls (sign flip is exact).
    Muls(betaBlocks, betaBlocks, -1.0f, static_cast<uint64_t>(kMatmulM) * FP32_NUM_PER_BLOCK);
    uint8_t rowStride = static_cast<uint8_t>(kMatmulN / FP32_NUM_PER_BLOCK);
    BinaryRepeatParams rowMulParams{1, 1, 0, rowStride, rowStride, 1};
    Mul(chunkScoresFp32, chunkScoresFp32, betaBlocks, static_cast<uint64_t>(kMatmulN), static_cast<uint8_t>(kMatmulM),
        rowMulParams);
    Mul(chunkScoresFp32, chunkScoresFp32, decayMaskFp32, kCElements);
    // attn_i is consumed in place from decayMaskFp32 downstream.
    Mul(decayMaskFp32, chunkAttnOutFp32, decayMaskFp32, kCElements);
    Muls(decayMaskFp32, decayMaskFp32, scale_, kCElements);
    TQueSync<PIPE_V, PIPE_S> attentionSync;
    attentionSync.SetFlag(0);
    attentionSync.WaitFlag(0);
  }

  __aicore__ inline void CopyAttnCubeResult(const LocalTensor<float> &c1Local, LocalTensor<float> dst) {
    constexpr uint32_t kMatmulM = 64;
    constexpr uint32_t kMatmulN = 64;
    SetFlag<HardEvent::M_MTE1>(0);
    WaitFlag<HardEvent::M_MTE1>(0);
    LocalTensor<float> cNz = chunkVFp32;
    DataCopyParams cCopyParams{static_cast<uint16_t>(kMatmulN / kNzFractalDim),
                               static_cast<uint16_t>(kMatmulM / kNzFractalDim), 0, 0};
    DataCopyEnhancedParams cCopyEnhanced;
    cCopyEnhanced.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    CopyL0CToUb(cNz, c1Local, cCopyParams, cCopyEnhanced);
    PipeBarrier<PIPE_ALL>();
    NzTileToNd<kMatmulM, kMatmulN>(dst, cNz);
  }

  // Phase 5: k_cumdecay = attn @ (k_beta * exp(g_cumsum)); result stored back into chunkKFp32.
  __aicore__ inline void ComputeKCumdecay(uint32_t chunkLen) {
    uint32_t cs = chunkSize_;
    // Broadcast exp(g) per row with Brcb + one high-dimensional Mul instead of
    // 64 scalar UB reads and 64 row-wise Muls. Each element still computes
    // exactly x * exp(g_i) in FP32, so the result is bit-identical.
    LocalTensor<float> gBlocks = chunkVFp32;
    uint8_t gBrcbRepeats = static_cast<uint8_t>(Ceil(chunkLen, FP32_NUM_PER_BLOCK));
    Brcb(gBlocks, expGCumFp32, gBrcbRepeats, BrcbRepeatParams{1, 8});
    PipeBarrier<PIPE_V>();
    uint8_t gRowStride = static_cast<uint8_t>(alignK_ / FP32_NUM_PER_BLOCK);
    BinaryRepeatParams gMulParams{1, 1, 0, gRowStride, gRowStride, 1};
    uint32_t gOffset = 0;
    for (; gOffset + kVectorMulBlockSize <= realK_; gOffset += kVectorMulBlockSize) {
      Mul(kCumdecayFp32[gOffset], kCumdecayFp32[gOffset], gBlocks, kVectorMulBlockSize, chunkLen, gMulParams);
    }
    if (gOffset < realK_) {
      Mul(kCumdecayFp32[gOffset], kCumdecayFp32[gOffset], gBlocks, realK_ - gOffset, chunkLen, gMulParams);
    }
    TQueSync<PIPE_V, PIPE_S> decaySync;
    decaySync.SetFlag(0);
    decaySync.WaitFlag(0);
    if constexpr (kSpecializedDk != 0) {
      if (likely(realK_ <= kSpecializedDk && chunkLen == kCubeFastPathChunkLen && vStepAligned_ >= kSpecializedDk)) {
        ComputeKCumdecayCube<kSpecializedDk>();
        return;
      }
    }
    // attn (chunkScoresFp32) was published by Scalar stores; the Cube variant
    // already carries this guard, the Vector path needs it too.
    TQueSync<PIPE_S, PIPE_V> scoresToVector;
    scoresToVector.SetFlag(0);
    scoresToVector.WaitFlag(0);
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

    TQueSync<PIPE_S, PIPE_V> recursiveToVector;
    recursiveToVector.SetFlag(0);
    recursiveToVector.WaitFlag(0);
    // (I-A)^-1 is lower triangular by construction: the upper triangle was
    // zeroed before the inverse and the diagonal is set to 1 afterwards, so no
    // extra mask/copy is needed before staging the attention operand.
    LocalMemAllocator<AscendC::Hardware::L1> l1Allocator;
    LocalMemAllocator<AscendC::Hardware::L0A> l0aAllocator;
    LocalMemAllocator<AscendC::Hardware::L0B> l0bAllocator;
    LocalMemAllocator<AscendC::Hardware::L0C> l0cAllocator;
    LocalTensor<half> a1Local = l1Allocator.Alloc<TPosition::A1, half>(kAElements);
    LocalTensor<half> aLoLocal = l1Allocator.Alloc<TPosition::A1, half>(kAElements);
    LocalTensor<half> b1Local = l1Allocator.Alloc<TPosition::B1, half>(kBElements);
    LocalTensor<half> a2Local = l0aAllocator.Alloc<TPosition::A2, half>(kAElements);
    LocalTensor<half> b2Local = l0bAllocator.Alloc<TPosition::B2, half>(kBElements);
    LocalTensor<float> c1Local = l0cAllocator.Alloc<TPosition::CO1, float>(kCElements);
    StageHalfWithResidualToL1<kMatmulM, kMatmulK>(chunkScoresFp32, chunkVFp32, a1Local, aLoLocal);
    StageHalfToL1<kMatmulK, kMatmulN>(kCumdecayFp32, b1Local);
    // The attention operand is no longer published through GM, so the V-tile
    // product re-stages it from chunkScoresFp32 instead of reusing a GM copy.
    SetFlag<HardEvent::MTE3_MTE1>(0);
    WaitFlag<HardEvent::MTE3_MTE1>(0);
    for (uint32_t mBlock = 0; mBlock < kMatmulM / kNzFractalDim; ++mBlock) {
      LoadData(a2Local[mBlock * kMatmulK * kNzFractalDim], a1Local[mBlock * kNzFractalRowBytes / sizeof(half)],
               LoadData2DParams{0, kMatmulK / kNzFractalDim, kMatmulM / kNzFractalDim, 0, 0, false, 0});
    }
    for (uint32_t kBlock = 0; kBlock < kMatmulK / kNzFractalDim; ++kBlock) {
      LoadData(b2Local[kBlock * kMatmulN * kNzFractalDim], b1Local[kBlock * kNzFractalRowBytes / sizeof(half)],
               LoadData2DParams{0, kMatmulN / kNzFractalDim, kMatmulK / kNzFractalDim, 0, 0, true, 0});
    }
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    Mmad(c1Local, a2Local, b2Local, MmadParams{kMatmulM, kMatmulN, kMatmulK, 0, false, true});
    SetFlag<HardEvent::M_MTE1>(0);
    WaitFlag<HardEvent::M_MTE1>(0);
    for (uint32_t mBlock = 0; mBlock < kMatmulM / kNzFractalDim; ++mBlock) {
      LoadData(a2Local[mBlock * kMatmulK * kNzFractalDim], aLoLocal[mBlock * kNzFractalRowBytes / sizeof(half)],
               LoadData2DParams{0, kMatmulK / kNzFractalDim, kMatmulM / kNzFractalDim, 0, 0, false, 0});
    }
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    Mmad(c1Local, a2Local, b2Local, MmadParams{kMatmulM, kMatmulN, kMatmulK, 0, false, false});
    SetFlag<HardEvent::M_MTE1>(0);
    WaitFlag<HardEvent::M_MTE1>(0);
    LocalTensor<float> cNz = chunkVFp32;
    DataCopyParams cCopyParams{static_cast<uint16_t>(kMatmulN / kNzFractalDim),
                               static_cast<uint16_t>(kMatmulM / kNzFractalDim), 0, 0};
    DataCopyEnhancedParams cCopyEnhanced;
    cCopyEnhanced.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    CopyL0CToUb(cNz, c1Local, cCopyParams, cCopyEnhanced);
    PipeBarrier<PIPE_ALL>();
    NzTileToNd<kMatmulM, kMatmulN>(chunkKFp32, cNz);
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
    bool useFusedCubeFastPath = IsCubeFastPath(chunkLen, avFp32);
    if (likely(useFusedCubeFastPath)) {
      ComputeValueAndVNewCubeDispatch(stateBaseOffset, workspaceStateBaseOffset, v_i, curV, c, t_start, qkHead);
    } else {
      ComputeValueNew(chunkLen, curV, avFp32);
      LoadStateTile(stateBaseOffset, workspaceStateBaseOffset, v_i, curV, c);
      ComputeVNew(chunkLen, avFp32);
    }
    if (!useFusedCubeFastPath) {
      ComputeAttnInter(t_start, qkHead, chunkLen, avFp32);
      AccumOutput(chunkLen, avFp32);
    }
    // On the fast path the output was already produced by
    // ComputeValueAndVNewCube (state B reuse).
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
               dstRowGapBytes % BLOCK_BYTES == 0 && srcRowGapBytes / BLOCK_BYTES <= kDataCopyStrideBlockLimit &&
               dstRowGapBytes / BLOCK_BYTES <= kDataCopyStrideBlockLimit)) {
      LocalTensor<inType> valueLocal = chunkAttnOutFp32.template ReinterpretCast<inType>();
      uint64_t vOff = (static_cast<uint64_t>(t_start) * NV_ + head_i) * realV_ + v_i;
      uint16_t rowBlocks = static_cast<uint16_t>(curV * sizeof(inType) / BLOCK_BYTES);
      uint16_t srcRowGapBlocks = static_cast<uint16_t>(srcRowGapBytes / BLOCK_BYTES);
      uint16_t dstRowGapBlocks = static_cast<uint16_t>(dstRowGapBytes / BLOCK_BYTES);
      DataCopyParams copyParams{static_cast<uint16_t>(chunkLen), rowBlocks, srcRowGapBlocks, dstRowGapBlocks};

      // Zero the aligned destination rows first, then let MTE2 write valid V
      // elements directly at the final row stride. This makes the subsequent
      // FP16->FP32 Cast contiguous and removes one Cast launch per row. A full
      // tile is written completely by the DMA, so the fill is skipped there.
      if (curV != avFp32 || chunkLen != chunkSize_) {
        Duplicate(valueLocal, static_cast<inType>(0), chunkLen * avFp32);
        PipeBarrier<PIPE_V>();
      }

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

      // Cast / Brcb / Mul are all PIPE_V on distinct buffers: issue order is
      // the only dependency, so no same-pipe barrier is needed.
      Cast(chunkVFp32, valueLocal, RoundMode::CAST_NONE, chunkLen * avFp32);

      // Broadcast one beta per row into 32-byte blocks and scale the complete
      // aligned V tile with high-dimensional Mul instructions. Padding stays
      // zero, while scalar beta reads and row-wise Muls launches disappear.
      LocalTensor<float> betaBlocks = chunkAttnOutFp32;
      uint8_t brcbRepeats = static_cast<uint8_t>(Ceil(chunkLen, FP32_NUM_PER_BLOCK));
      Brcb(betaBlocks, betaFp32, brcbRepeats, BrcbRepeatParams{1, 8});
      uint8_t rowStride = static_cast<uint8_t>(avFp32 / FP32_NUM_PER_BLOCK);
      BinaryRepeatParams mulParams{1, 1, 0, rowStride, rowStride, 1};
      uint32_t vOffset = 0;
      for (; vOffset + kVectorMulBlockSize <= avFp32; vOffset += kVectorMulBlockSize) {
        Mul(chunkVFp32[vOffset], chunkVFp32[vOffset], betaBlocks, static_cast<uint64_t>(kVectorMulBlockSize),
            static_cast<uint8_t>(chunkLen), mulParams);
      }
      if (vOffset < avFp32) {
        Mul(chunkVFp32[vOffset], chunkVFp32[vOffset], betaBlocks, static_cast<uint64_t>(avFp32 - vOffset),
            static_cast<uint8_t>(chunkLen), mulParams);
      }
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
  // chunkAttnOutFp32. Every IsCubeFastPath shape fuses steps 2 and 4 into one
  // Cube product (ComputeValueAndVNewCube), so only this Vector form is reachable.
  __aicore__ inline void ComputeValueNew(uint32_t chunkLen, uint32_t curV, uint32_t avFp32) {
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
    // The zero fill only exists to pad a partial V tile / padded K rows. When the
    // load below covers the complete [stateStrideK_, vStepAligned_] tile it is
    // dead work, so it is skipped (bit-identical result).
    if (curV != vStepAligned_ || realK_ != stateStrideK_) {
      Duplicate(stateInFp32, 0.0f, stateStrideK_ * vStepAligned_);
      PipeBarrier<PIPE_V>();
    }
    if (c > 0) {
      LoadStateFromWorkspace(workspaceStateBaseOffset, v_i, curV);
      return;
    }
    if (LoadInitialStateTransposed(stateBaseOffset, v_i, curV)) {
      return;
    }
    LoadInitialStateScalar(stateBaseOffset, v_i, curV);
  }

  // Reload a mid-sequence state tile from the FP32 workspace buffer.
  __aicore__ inline void LoadStateFromWorkspace(uint64_t workspaceStateBaseOffset, uint32_t v_i, uint32_t curV) {
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
  }

  // Load the initial state and transpose it into the internal [Dk, vStep] tile.
  // The public interface follows 910B and stores state as [B, Nv, Dv, Dk], so a
  // 16-aligned tile can use one contiguous GM->UB transfer followed by
  // vnchwconv and an FP16->FP32 cast instead of an element-wise transpose.
  // kCumdecayFp32 and chunkVFp32 are dead at this point and provide the two
  // temporary FP16 matrices when both borrowed buffers are large enough.
  // Returns false when the shape cannot use this path.
  __aicore__ inline bool LoadInitialStateTransposed(uint64_t stateBaseOffset, uint32_t v_i, uint32_t curV) {
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
    if (unlikely(realK_ % kTransposeBlock != 0 || vStepAligned_ % kTransposeBlock != 0 || !supportedVTile ||
                 kBlockCount > UINT8_MAX || stateElementCount > stateNdCapacity ||
                 stateElementCount > stateTransposedCapacity)) {
      return false;
    }
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
    return true;
  }

  // Scalar fallback for initial-state shapes the transposed path cannot handle.
  __aicore__ inline void LoadInitialStateScalar(uint64_t stateBaseOffset, uint32_t v_i, uint32_t curV) {
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
  // See ComputeValueNew: the fused Cube product owns every IsCubeFastPath shape.
  __aicore__ inline void ComputeVNew(uint32_t chunkLen, uint32_t avFp32) {
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

  // Copy a row-major FP16 [kRows, kCols] UB tile to L1 in the exact NZ (fractal)
  // byte order that Nd2Nz produced, without the GM round trip.
  //
  // NZ places element (r, c) at (c/16 * kRows + r) * 16 + c%16 halves, i.e. one
  // 16x16 fractal is 16 contiguous 32-byte rows. For a fixed 16-column group the
  // destination is therefore kRows contiguous 32-byte blocks while the source
  // reads one 32-byte block per row with a (kCols/16 - 1) block gap. On dav_m200
  // Nd2Nz is a software loop (GM->UB, UB transpose, nValue 32-byte UB->L1
  // bursts), so replacing it with this direct strided UB->L1 copy is
  // byte-identical by construction and removes the GM + transpose stages.
  template <uint32_t kRows, uint32_t kCols>
  __aicore__ inline void CopyUbNdToL1Nz(LocalTensor<half> dstL1, LocalTensor<half> srcUb) {
    static_assert(kRows % kNzFractalDim == 0 && kCols % kNzFractalDim == 0,
                  "NZ staging requires 16-aligned tile geometry");
    DataCopyParams nzParams{static_cast<uint16_t>(kRows), 1, static_cast<uint16_t>(kCols / kNzFractalDim - 1), 0};
    for (uint32_t colBlock = 0; colBlock < kCols / kNzFractalDim; ++colBlock) {
      DataCopy(dstL1[colBlock * kRows * kNzFractalDim], srcUb[colBlock * kNzFractalDim], nzParams);
    }
  }

  // Cast a row-major FP32 [kRows, kCols] tile to FP16 in UB and publish it
  // straight into the L1 NZ tile (byte-identical to a GM round trip through
  // Nd2Nz, without the GM staging copy).
  template <uint32_t kRows, uint32_t kCols>
  __aicore__ inline void StageHalfToL1(LocalTensor<float> src, LocalTensor<half> dstL1) {
    LocalTensor<half> stageLocal = stateOutQueue_.AllocTensor<half>();
    Cast(stageLocal, src, RoundMode::CAST_NONE, kRows * kCols);
    PipeBarrier<PIPE_V>();
    stateOutQueue_.EnQue<half>(stageLocal);
    stageLocal = stateOutQueue_.DeQue<half>();
    CopyUbNdToL1Nz<kRows, kCols>(dstL1, stageLocal);
    stateOutQueue_.FreeTensor(stageLocal);
  }

  // Two-term compensated variant of StageHalfToL1: the same hi/lo split as the
  // GM-staging form it replaced, but both halves are published straight into two
  // distinct L1 NZ tiles instead of the GM staging slots.
  template <uint32_t kRows, uint32_t kCols>
  __aicore__ inline void StageHalfWithResidualToL1(LocalTensor<float> src, LocalTensor<float> scratch,
                                                   LocalTensor<half> hiL1, LocalTensor<half> loL1) {
    constexpr uint32_t kCount = kRows * kCols;
    LocalTensor<half> stageLocal = stateOutQueue_.AllocTensor<half>();
    Cast(stageLocal, src, RoundMode::CAST_NONE, kCount);
    PipeBarrier<PIPE_V>();
    stateOutQueue_.EnQue<half>(stageLocal);
    stageLocal = stateOutQueue_.DeQue<half>();
    CopyUbNdToL1Nz<kRows, kCols>(hiL1, stageLocal);

    event_t mte3ToVector = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(mte3ToVector);
    WaitFlag<HardEvent::MTE3_V>(mte3ToVector);
    Cast(scratch, stageLocal, RoundMode::CAST_NONE, kCount);
    PipeBarrier<PIPE_V>();
    Sub(scratch, src, scratch, kCount);
    PipeBarrier<PIPE_V>();
    Cast(stageLocal, scratch, RoundMode::CAST_NONE, kCount);
    PipeBarrier<PIPE_V>();
    stateOutQueue_.EnQue<half>(stageLocal);
    stageLocal = stateOutQueue_.DeQue<half>();
    CopyUbNdToL1Nz<kRows, kCols>(loL1, stageLocal);
    stateOutQueue_.FreeTensor(stageLocal);
  }

  // Fixed-shape fused Step 2 + Step 4:
  //   v_new = lower(attn) @ v_beta - k_cumdecay @ state.
  // Both products accumulate in the same FP32 L0C matrix. Compared with the
  // former pair of Cube functions this removes one L0C->UB conversion, the
  // intermediate value_new tile, and the final Vector negate/add pass.
  __aicore__ inline void ComputeValueAndVNewCubeDispatch(uint64_t stateBaseOffset, uint64_t workspaceStateBaseOffset,
                                                         uint32_t v_i, uint32_t curV, uint32_t c, int32_t t_start,
                                                         uint64_t qkHead) {
    if constexpr (kSpecializedDk != 0) {
      ComputeValueAndVNewCube<kSpecializedDk, kSpecializedDk>(stateBaseOffset, workspaceStateBaseOffset, v_i, curV, c,
                                                              t_start, qkHead);
    }
  }

  template <uint32_t kStateK, uint32_t kMatmulN>
  __aicore__ inline void ComputeValueAndVNewCube(uint64_t stateBaseOffset, uint64_t workspaceStateBaseOffset,
                                                 uint32_t v_i, uint32_t curV, uint32_t c, int32_t t_start,
                                                 uint64_t qkHead) {
    constexpr uint32_t kMatmulM = 64;
    constexpr uint32_t kAttnK = 64;
    constexpr uint32_t kAAttnElements = kMatmulM * kAttnK;
    constexpr uint32_t kBAttnElements = kAttnK * kMatmulN;
    constexpr uint32_t kAStateElements = kMatmulM * kStateK;
    constexpr uint32_t kBStateElements = kStateK * kMatmulN;
    constexpr uint32_t kCElements = kMatmulM * kMatmulN;

    // lower(attn) is still live in chunkScoresFp32: LoadStateTile only clobbers
    // that buffer later, for the state product. Stage it straight from UB.
    // Muls and the strict-lower Duplicate fill are both PIPE_V, so the barrier
    // between them only duplicated issue-order semantics.
    Muls(kCumdecayFp32, chunkScoresFp32, 1.0f, kAAttnElements);
    for (uint32_t row = 0; row + 1 < kMatmulM; ++row) {
      Duplicate(kCumdecayFp32[row * kAttnK + row + 1], 0.0f, kAttnK - row - 1);
    }

    LocalMemAllocator<AscendC::Hardware::L1> l1Allocator;
    LocalMemAllocator<AscendC::Hardware::L0A> l0aAllocator;
    LocalMemAllocator<AscendC::Hardware::L0B> l0bAllocator;
    LocalMemAllocator<AscendC::Hardware::L0C> l0cAllocator;
    LocalTensor<half> a1Local = l1Allocator.Alloc<TPosition::A1, half>(kAStateElements);
    LocalTensor<half> aLoLocal = l1Allocator.Alloc<TPosition::A1, half>(kAStateElements);
    LocalTensor<half> b1Local = l1Allocator.Alloc<TPosition::B1, half>(kBStateElements);
    LocalTensor<half> a2Local = l0aAllocator.Alloc<TPosition::A2, half>(kAStateElements);
    LocalTensor<half> b2Local = l0bAllocator.Alloc<TPosition::B2, half>(kBStateElements);
    LocalTensor<float> c1Local = l0cAllocator.Alloc<TPosition::CO1, float>(kCElements);

    StageHalfWithResidualToL1<kMatmulM, kAttnK>(kCumdecayFp32, chunkAttnOutFp32, a1Local, aLoLocal);
    StageHalfToL1<kAttnK, kMatmulN>(chunkVFp32, b1Local);
    SetFlag<HardEvent::MTE3_MTE1>(0);
    WaitFlag<HardEvent::MTE3_MTE1>(0);
    for (uint32_t mBlock = 0; mBlock < kMatmulM / kNzFractalDim; ++mBlock) {
      LoadData(a2Local[mBlock * kAttnK * 16], a1Local[mBlock * 512 / sizeof(half)],
               LoadData2DParams{0, kAttnK / 16, kMatmulM / 16, 0, 0, false, 0});
    }
    for (uint32_t kBlock = 0; kBlock < kAttnK / 16; ++kBlock) {
      LoadData(b2Local[kBlock * kMatmulN * kNzFractalDim], b1Local[kBlock * kNzFractalRowBytes / sizeof(half)],
               LoadData2DParams{0, kMatmulN / 16, kAttnK / 16, 0, 0, true, 0});
    }
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    Mmad(c1Local, a2Local, b2Local, MmadParams{kMatmulM, kMatmulN, kAttnK, 0, false, true});

    SetFlag<HardEvent::M_MTE1>(0);
    WaitFlag<HardEvent::M_MTE1>(0);
    for (uint32_t mBlock = 0; mBlock < kMatmulM / kNzFractalDim; ++mBlock) {
      LoadData(a2Local[mBlock * kAttnK * 16], aLoLocal[mBlock * 512 / sizeof(half)],
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
    StageHalfWithResidualToL1<kMatmulM, kStateK>(chunkKFp32, kCumdecayFp32, a1Local, aLoLocal);
    StageHalfToL1<kStateK, kMatmulN>(stateInFp32, b1Local);
    SetFlag<HardEvent::MTE3_MTE1>(0);
    WaitFlag<HardEvent::MTE3_MTE1>(0);

    for (uint32_t mBlock = 0; mBlock < kMatmulM / kNzFractalDim; ++mBlock) {
      LoadData(a2Local[mBlock * kStateK * 16], a1Local[mBlock * 512 / sizeof(half)],
               LoadData2DParams{0, kStateK / 16, kMatmulM / 16, 0, 0, false, 0});
    }
    for (uint32_t kBlock = 0; kBlock < kStateK / 16; ++kBlock) {
      LoadData(b2Local[kBlock * kMatmulN * kNzFractalDim], b1Local[kBlock * kNzFractalRowBytes / sizeof(half)],
               LoadData2DParams{0, kMatmulN / 16, kStateK / 16, 0, 0, true, 0});
    }
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    Mmad(c1Local, a2Local, b2Local, MmadParams{kMatmulM, kMatmulN, kStateK, 0, false, false});

    SetFlag<HardEvent::M_MTE1>(0);
    WaitFlag<HardEvent::M_MTE1>(0);
    for (uint32_t mBlock = 0; mBlock < kMatmulM / kNzFractalDim; ++mBlock) {
      LoadData(a2Local[mBlock * kStateK * 16], aLoLocal[mBlock * 512 / sizeof(half)],
               LoadData2DParams{0, kStateK / 16, kMatmulM / 16, 0, 0, false, 0});
    }
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    Mmad(c1Local, a2Local, b2Local, MmadParams{kMatmulM, kMatmulN, kStateK, 0, false, false});

    SetFlag<HardEvent::M_MTE1>(0);
    WaitFlag<HardEvent::M_MTE1>(0);
    LocalTensor<float> cNz = chunkKFp32;
    DataCopyParams cCopyParams{static_cast<uint16_t>(kMatmulN / kNzFractalDim),
                               static_cast<uint16_t>(kMatmulM / kNzFractalDim), 0, 0};
    DataCopyEnhancedParams cCopyEnhanced;
    cCopyEnhanced.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    CopyL0CToUb(cNz, c1Local, cCopyParams, cCopyEnhanced);
    PipeBarrier<PIPE_ALL>();

    NzTileToNd<kMatmulM, kMatmulN>(chunkVFp32, cNz);

    // ---- Output stage fused here: out = (Q*scale*exp(g)) @ state +
    // lower(attn_i) @ v_new. The state B staged above is still resident in
    // b1Local/b2Local, so only its A operand is reloaded. chunkAttnOutFp32 is
    // free until the final copy, so it serves as Brcb target and staging
    // scratch; chunkKFp32 is free after the v_new conversion.
    LocalTensor<inType> queryLocal = chunkKFp32.template ReinterpretCast<inType>();
    uint64_t queryOffset = (static_cast<uint64_t>(t_start) * NK_ + qkHead) * realK_;
    LoadPaddedRows(kCumdecayFp32, queryLocal, queryGm_, queryOffset, kMatmulM, NK_ * realK_);
    // Same per-row scale * exp(g_i), broadcast via Brcb (bit-identical).
    // Muls / Brcb / Mul are all PIPE_V, so no same-pipe barrier is needed.
    Muls(deltaFp32, expGCumFp32, scale_, kMatmulM);
    {
      LocalTensor<float> qScaleBlocks = chunkAttnOutFp32;
      uint8_t qBrcbRepeats = static_cast<uint8_t>(Ceil(kMatmulM, FP32_NUM_PER_BLOCK));
      Brcb(qScaleBlocks, deltaFp32, qBrcbRepeats, BrcbRepeatParams{1, 8});
      uint8_t qRowStride = static_cast<uint8_t>(kStateK / FP32_NUM_PER_BLOCK);
      BinaryRepeatParams qMulParams{1, 1, 0, qRowStride, qRowStride, 1};
      uint32_t qOffset = 0;
      for (; qOffset + kVectorMulBlockSize <= kStateK; qOffset += kVectorMulBlockSize) {
        Mul(kCumdecayFp32[qOffset], kCumdecayFp32[qOffset], qScaleBlocks, kVectorMulBlockSize, kMatmulM, qMulParams);
      }
      if (qOffset < kStateK) {
        Mul(kCumdecayFp32[qOffset], kCumdecayFp32[qOffset], qScaleBlocks, kStateK - qOffset, kMatmulM, qMulParams);
      }
    }
    StageHalfWithResidualToL1<kMatmulM, kStateK>(kCumdecayFp32, chunkAttnOutFp32, a1Local, aLoLocal);
    SetFlag<HardEvent::MTE3_MTE1>(0);
    WaitFlag<HardEvent::MTE3_MTE1>(0);

    for (uint32_t mBlock = 0; mBlock < kMatmulM / kNzFractalDim; ++mBlock) {
      LoadData(a2Local[mBlock * kStateK * 16], a1Local[mBlock * 512 / sizeof(half)],
               LoadData2DParams{0, kStateK / 16, kMatmulM / 16, 0, 0, false, 0});
    }
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    Mmad(c1Local, a2Local, b2Local, MmadParams{kMatmulM, kMatmulN, kStateK, 0, false, true});

    SetFlag<HardEvent::M_MTE1>(0);
    WaitFlag<HardEvent::M_MTE1>(0);
    for (uint32_t mBlock = 0; mBlock < kMatmulM / kNzFractalDim; ++mBlock) {
      LoadData(a2Local[mBlock * kStateK * 16], aLoLocal[mBlock * 512 / sizeof(half)],
               LoadData2DParams{0, kStateK / 16, kMatmulM / 16, 0, 0, false, 0});
    }
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    Mmad(c1Local, a2Local, b2Local, MmadParams{kMatmulM, kMatmulN, kStateK, 0, false, false});

    SetFlag<HardEvent::M_MTE1>(0);
    WaitFlag<HardEvent::M_MTE1>(0);
    // (I-A)^-1 is lower triangular by construction and attn_i's upper
    // triangle is exactly zero, so no extra masking is needed here. attn_i is
    // staged straight out of decayMaskFp32; chunkAttnOutFp32 holds the dead
    // Q-scale residual and serves as the residual scratch, which drops the
    // former 16 KiB attn_i copy on the Vector pipe.
    StageHalfWithResidualToL1<kMatmulM, kAttnK>(decayMaskFp32, chunkAttnOutFp32, a1Local, aLoLocal);
    StageHalfToL1<kAttnK, kMatmulN>(chunkVFp32, b1Local);
    SetFlag<HardEvent::MTE3_MTE1>(0);
    WaitFlag<HardEvent::MTE3_MTE1>(0);

    for (uint32_t mBlock = 0; mBlock < kMatmulM / kNzFractalDim; ++mBlock) {
      LoadData(a2Local[mBlock * kAttnK * 16], a1Local[mBlock * 512 / sizeof(half)],
               LoadData2DParams{0, kAttnK / 16, kMatmulM / 16, 0, 0, false, 0});
    }
    for (uint32_t kBlock = 0; kBlock < kAttnK / 16; ++kBlock) {
      LoadData(b2Local[kBlock * kMatmulN * kNzFractalDim], b1Local[kBlock * kNzFractalRowBytes / sizeof(half)],
               LoadData2DParams{0, kMatmulN / 16, kAttnK / 16, 0, 0, true, 0});
    }
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    Mmad(c1Local, a2Local, b2Local, MmadParams{kMatmulM, kMatmulN, kAttnK, 0, false, false});

    SetFlag<HardEvent::M_MTE1>(0);
    WaitFlag<HardEvent::M_MTE1>(0);
    for (uint32_t mBlock = 0; mBlock < kMatmulM / kNzFractalDim; ++mBlock) {
      LoadData(a2Local[mBlock * kAttnK * 16], aLoLocal[mBlock * 512 / sizeof(half)],
               LoadData2DParams{0, kAttnK / 16, kMatmulM / 16, 0, 0, false, 0});
    }
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    Mmad(c1Local, a2Local, b2Local, MmadParams{kMatmulM, kMatmulN, kAttnK, 0, false, false});

    SetFlag<HardEvent::M_MTE1>(0);
    WaitFlag<HardEvent::M_MTE1>(0);
    LocalTensor<float> cNzOut = kCumdecayFp32;
    CopyL0CToUb(cNzOut, c1Local, cCopyParams, cCopyEnhanced);
    NzTileToNd<kMatmulM, kMatmulN>(chunkAttnOutFp32, cNzOut);
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

  /**
   * @brief Fixed-shape state update: state = exp(g_last) * state +
   *        (K * exp(g_last - g))^T @ v_new. The weighted K^T @ v_new product
   *        uses Cube for every bucket; the complete NZ result is converted in
   *        place and accumulated with one wide Vector Add per 16-column group.
   * @param t_start Start position on the T dimension.
   * @param qkHead  Q/K head index for this dispatch.
   */
  __aicore__ inline void ComputeStateUpdateCubeDispatch(int32_t t_start, uint64_t qkHead) {
    if constexpr (kSpecializedDk != 0) {
      ComputeStateUpdateCube<kSpecializedDk, kSpecializedDk>(t_start, qkHead);
    }
  }

  template <uint32_t kStateRows, uint32_t kMatmulN>
  __aicore__ inline void ComputeStateUpdateCube(int32_t t_start, uint64_t qkHead) {
    // state = exp(g_last) * state + K^T @ (exp(g_last - g) * v_new).
    // K originates as an FP16 input, so it is exact in FP16. Folding the decay
    // into v_new keeps the Cube A operand exact (no hi/lo split), so only the
    // B operand is compensated: one A transpose/stage instead of two.
    // The whole K row-block is accumulated in one L0C tile (a 128x128 FP32
    // result is 64 KiB, within the 128 KiB L0C), so the 128 bucket no longer
    // pays a second L0C->UB conversion plus NZ->ND Add pass per chunk.
    constexpr uint32_t kTileM = kSpecializedDk;
    constexpr uint32_t kMatmulK = 64;
    constexpr uint32_t kMTileCount = (kStateRows + kTileM - 1) / kTileM;
    constexpr uint32_t kAElements = kStateRows * kMatmulK;
    constexpr uint32_t kATileElements = kTileM * kMatmulK;
    constexpr uint32_t kBElements = kMatmulK * kMatmulN;
    constexpr uint32_t kStateElements = kStateRows * kMatmulN;
    constexpr uint32_t kCTileElements = kTileM * kMatmulN;
    __gm__ uint8_t *stageBase = GetCubeStageBase(0);
    GlobalTensor<half> aStageGm;
    aStageGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(stageBase), kAElements);

    float gLast = gCumsumFp32.GetValue(kMatmulK - 1);
    float gLastExp = expGCumFp32.GetValue(kMatmulK - 1);
    // The Muls/Duplicate/Sub/Exp/Brcb chain is entirely PIPE_V, so same-pipe
    // barriers between its steps are redundant (issue order is the dependency).
    Muls(stateInFp32, stateInFp32, gLastExp, kStateElements);
    Duplicate(deltaFp32, gLast, kMatmulK);
    Sub(deltaFp32, deltaFp32, gCumsumFp32, kMatmulK);
    Exp(deltaFp32, deltaFp32, kMatmulK);

    // v_new *= exp(g_last - g) with a row-broadcast high-dimensional Mul.
    LocalTensor<float> deltaBlocks = chunkAttnOutFp32;
    uint8_t brcbRepeats = static_cast<uint8_t>(Ceil(kMatmulK, FP32_NUM_PER_BLOCK));
    Brcb(deltaBlocks, deltaFp32, brcbRepeats, BrcbRepeatParams{1, 8});
    uint8_t vRowStride = static_cast<uint8_t>(kMatmulN / FP32_NUM_PER_BLOCK);
    BinaryRepeatParams vMulParams{1, 1, 0, vRowStride, vRowStride, 1};
    uint32_t vOffset = 0;
    for (; vOffset + kVectorMulBlockSize <= kMatmulN; vOffset += kVectorMulBlockSize) {
      Mul(chunkVFp32[vOffset], chunkVFp32[vOffset], deltaBlocks, static_cast<uint64_t>(kVectorMulBlockSize),
          static_cast<uint8_t>(kMatmulK), vMulParams);
    }
    if (vOffset < kMatmulN) {
      Mul(chunkVFp32[vOffset], chunkVFp32[vOffset], deltaBlocks, static_cast<uint64_t>(kMatmulN - vOffset),
          static_cast<uint8_t>(kMatmulK), vMulParams);
    }

    LocalMemAllocator<AscendC::Hardware::L1> l1Allocator;
    LocalMemAllocator<AscendC::Hardware::L0A> l0aAllocator;
    LocalMemAllocator<AscendC::Hardware::L0B> l0bAllocator;
    LocalMemAllocator<AscendC::Hardware::L0C> l0cAllocator;
    LocalTensor<half> a1Local = l1Allocator.Alloc<TPosition::A1, half>(kATileElements);
    LocalTensor<half> b1Local = l1Allocator.Alloc<TPosition::B1, half>(kBElements);
    LocalTensor<half> bLoLocal = l1Allocator.Alloc<TPosition::B1, half>(kBElements);
    LocalTensor<half> a2Local = l0aAllocator.Alloc<TPosition::A2, half>(kATileElements);
    LocalTensor<half> b2Local = l0bAllocator.Alloc<TPosition::B2, half>(2 * kBElements);
    LocalTensor<float> c1Local = l0cAllocator.Alloc<TPosition::CO1, float>(kCTileElements);

    // Both B halves stay resident in L0B across every M tile.
    StageHalfWithResidualToL1<kMatmulK, kMatmulN>(chunkVFp32, chunkAttnOutFp32, b1Local, bLoLocal);

    // A = K^T in exact FP16. StageAttnKeyOperands already cached this tile in
    // GM for the whole chunk; only fall back to a fresh transpose if it did not.
    if (keyCacheValid_ != 0) {
      aStageGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(GetKeyCacheBase()), kAElements);
    } else {
      LocalTensor<inType> keyLocal = chunkKFp32.template ReinterpretCast<inType>();
      uint64_t keyOffset = (static_cast<uint64_t>(t_start) * NK_ + qkHead) * realK_;
      LoadPaddedRows(chunkAttnOutFp32, keyLocal, keyGm_, keyOffset, kMatmulK, NK_ * realK_);
      TQueSync<PIPE_V, PIPE_S> keySync;
      keySync.SetFlag(0);
      keySync.WaitFlag(0);
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
      stateOutQueue_.FreeTensor(stageLocal);
      PipeBarrier<PIPE_ALL>();
    }

    SetFlag<HardEvent::MTE3_MTE1>(0);
    WaitFlag<HardEvent::MTE3_MTE1>(0);
    for (uint32_t kBlock = 0; kBlock < kMatmulK / kNzFractalDim; ++kBlock) {
      LoadData(b2Local[kBlock * kMatmulN * kNzFractalDim], b1Local[kBlock * kNzFractalRowBytes / sizeof(half)],
               LoadData2DParams{0, kMatmulN / kNzFractalDim, kMatmulK / kNzFractalDim, 0, 0, true, 0});
      LoadData(b2Local[kBElements + kBlock * kMatmulN * kNzFractalDim],
               bLoLocal[kBlock * kNzFractalRowBytes / sizeof(half)],
               LoadData2DParams{0, kMatmulN / kNzFractalDim, kMatmulK / kNzFractalDim, 0, 0, true, 0});
    }
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);

    constexpr uint16_t kNdDstRepStride = static_cast<uint16_t>(kMatmulN * sizeof(float) / BLOCK_BYTES);
    constexpr uint8_t kNzSrcRepStride = static_cast<uint8_t>(16 * sizeof(float) / BLOCK_BYTES);
    for (uint32_t mTile = 0; mTile < kMTileCount; ++mTile) {
      uint32_t curTileM = (mTile + 1) * kTileM <= kStateRows ? kTileM : kStateRows - mTile * kTileM;
      uint32_t aOffset = mTile * kATileElements;
      DataCopy(
        a1Local, aStageGm[aOffset],
        Nd2NzParams{1, static_cast<uint16_t>(curTileM), kMatmulK, 0, kMatmulK, static_cast<uint16_t>(curTileM), 1, 0});
      SetFlag<HardEvent::MTE2_MTE1>(0);
      WaitFlag<HardEvent::MTE2_MTE1>(0);
      for (uint32_t mBlock = 0; mBlock < curTileM / 16; ++mBlock) {
        LoadData(a2Local[mBlock * kMatmulK * kNzFractalDim], a1Local[mBlock * kNzFractalRowBytes / sizeof(half)],
                 LoadData2DParams{0, kMatmulK / 16, static_cast<uint16_t>(curTileM / 16), 0, 0, false, 0});
      }
      SetFlag<HardEvent::MTE1_M>(0);
      WaitFlag<HardEvent::MTE1_M>(0);
      Mmad(c1Local, a2Local, b2Local, MmadParams{static_cast<uint16_t>(curTileM), kMatmulN, kMatmulK, 0, false, true});
      Mmad(c1Local, a2Local, b2Local[kBElements],
           MmadParams{static_cast<uint16_t>(curTileM), kMatmulN, kMatmulK, 0, false, false});
      SetFlag<HardEvent::M_MTE1>(0);
      WaitFlag<HardEvent::M_MTE1>(0);

      // The L0C->UB copy below writes curTileM x kMatmulN floats, i.e. the whole
      // kStateElements slab. That can exceed the cs * alignK footprint of
      // chunkKFp32, which aliases this region (exactly 2x for the 128 bucket), so
      // the scratch must be declared with its real extent instead of that view.
      LocalTensor<float> cNz = tmpBuff.GetWithOffset<float>(kStateElements, 0);
      DataCopyParams cCopyParams{static_cast<uint16_t>(kMatmulN / 16), static_cast<uint16_t>(curTileM / 16), 0, 0};
      DataCopyEnhancedParams cCopyEnhanced;
      cCopyEnhanced.blockMode = BlockMode::BLOCK_MODE_MATRIX;
      CopyL0CToUb(cNz, c1Local, cCopyParams, cCopyEnhanced);
      PipeBarrier<PIPE_ALL>();

      // Wide NZ -> ND accumulation: one strided Add per 16-column fractal
      // instead of one per row.
      uint32_t stateBase = mTile * kTileM * kMatmulN;
      for (uint32_t nBlock = 0; nBlock < kMatmulN / 16; ++nBlock) {
        Add(stateInFp32[stateBase + nBlock * 16], stateInFp32[stateBase + nBlock * 16], cNz[nBlock * curTileM * 16], 16,
            static_cast<uint8_t>(curTileM),
            BinaryRepeatParams{1, 1, 1, kNdDstRepStride, kNdDstRepStride, kNzSrcRepStride});
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
  uint32_t keyCacheValid_;
  float scale_;
  uint64_t blockIdx_;
};

}  // namespace ChunkGatedDeltaRule

#endif  // CHUNK_GATED_DELTA_RULE_KERNEL_H_
