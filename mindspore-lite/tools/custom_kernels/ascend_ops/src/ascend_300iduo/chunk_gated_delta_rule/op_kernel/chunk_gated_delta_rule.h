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

template <typename inType, typename outType>
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
    alignK_ = Ceil(tilingData->dk, FP16_NUM_PER_BLOCK) * FP16_NUM_PER_BLOCK;
    stateStrideK_ = Ceil(tilingData->dk, FP32_NUM_PER_BLOCK) * FP32_NUM_PER_BLOCK;
    stateWorkspaceStrideV_ = Ceil(tilingData->dv, FP32_NUM_PER_BLOCK) * FP32_NUM_PER_BLOCK;
    load_ = 0;
    usedblk_ = 0;
    avgload_ = 0;
    // Set in Init()/InitLocalBuffers(); zero-initialized here so every member is
    // defined before first use (the kernel constructor runs before those calls).
    pipe_ = nullptr;
    vStepAligned_ = 0;
    blockIdx_ = 0;
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
  __aicore__ inline bool ShouldSplitVTiles() { return realV_ > vStep_ && GetBlockNum() > NV_; }

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

  // Compute recursive intra-chunk attn matrix in-place.
  // buf is stored with leading dimension ld (= chunkSize_), and only [chunkLen, chunkLen] is valid.
  // buf is lower-triangular (excluding diagonal), with upper tri (incl diagonal) = 0.
  __aicore__ inline void ComputeRecursiveAttn(LocalTensor<float> &buf, uint32_t chunkLen, uint32_t ld) {
    for (uint32_t i = 1; i < chunkLen; i++) {
      for (uint32_t k = 0; k < i; k++) {
        deltaFp32.SetValue(k, buf.GetValue(i * ld + k));
      }
      TQueSync<PIPE_S, PIPE_V> rowSync;
      rowSync.SetFlag(0);
      rowSync.WaitFlag(0);
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
    ComputeGCumsum(t_start, head_i, chunkLen);
    ComputeKBeta(t_start, head_i, chunkLen);
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
    ComputeGCumsum(t_start, head_i, chunkLen);
    ComputeKBeta(t_start, head_i, chunkLen);
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
    uint64_t keyElements = static_cast<uint64_t>(chunkLen) * realK_;
    uint64_t gmRowStride = static_cast<uint64_t>(NK_) * realK_;
    uint64_t srcGapBytes = (gmRowStride - realK_) * sizeof(inType);
    bool canUseDma = chunkLen <= UINT16_MAX && keyElements <= stagingCapacity && alignK_ == realK_ &&
                     (realK_ * sizeof(inType)) % BLOCK_BYTES == 0 && srcGapBytes % BLOCK_BYTES == 0 &&
                     srcGapBytes / BLOCK_BYTES <= UINT16_MAX;
    if (likely(canUseDma)) {
      LocalTensor<inType> keyLocal = chunkVFp32.template ReinterpretCast<inType>();
      uint64_t qkOff = (static_cast<uint64_t>(t_start) * NK_ + qkHead) * realK_;
      DataCopyParams keyParams{static_cast<uint16_t>(chunkLen),
                               static_cast<uint16_t>(realK_ * sizeof(inType) / BLOCK_BYTES),
                               static_cast<uint16_t>(srcGapBytes / BLOCK_BYTES), 0};
      event_t scalarToMte2 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::S_MTE2));
      SetFlag<HardEvent::S_MTE2>(scalarToMte2);
      WaitFlag<HardEvent::S_MTE2>(scalarToMte2);
      event_t vectorToMte2 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE2));
      SetFlag<HardEvent::V_MTE2>(vectorToMte2);
      WaitFlag<HardEvent::V_MTE2>(vectorToMte2);
      DataCopy(keyLocal, keyGm_[qkOff], keyParams);
      event_t mte2ToVector = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
      SetFlag<HardEvent::MTE2_V>(mte2ToVector);
      WaitFlag<HardEvent::MTE2_V>(mte2ToVector);
      Cast(chunkKFp32, keyLocal, RoundMode::CAST_NONE, chunkLen * realK_);
      PipeBarrier<PIPE_V>();
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

  // Phase 1b: cumulative sum of g into gCumsumFp32.
  __aicore__ inline void ComputeGCumsum(int32_t t_start, uint64_t head_i, uint32_t chunkLen) {
    float gSum = 0.0f;
    for (uint32_t i = 0; i < chunkLen; i++) {
      gSum += LoadG(t_start + i, head_i);
      gCumsumFp32.SetValue(i, gSum);
    }
  }

  // Phase 2: k_beta = K * beta -> kCumdecayFp32.
  __aicore__ inline void ComputeKBeta(int32_t t_start, uint64_t head_i, uint32_t chunkLen) {
    // LoadChunkKey and ComputeGCumsum populate UB through the scalar pipe.
    // Synchronize before the first vector read of chunkKFp32.
    TQueSync<PIPE_S, PIPE_V> inputSync;
    inputSync.SetFlag(0);
    inputSync.WaitFlag(0);
    TQueSync<PIPE_S, PIPE_V> betaValueSync;
    for (uint32_t i = 0; i < chunkLen; i++) {
      float beta_val = LoadBeta(t_start + i, head_i);
      betaValueSync.SetFlag(0);
      betaValueSync.WaitFlag(0);
      Muls(kCumdecayFp32[i * alignK_], chunkKFp32[i * alignK_], beta_val, realK_);
    }
    TQueSync<PIPE_V, PIPE_S> betaSync;
    betaSync.SetFlag(0);
    betaSync.WaitFlag(0);
  }

  // Phase 3: precompute exp(gCumsum) and decay_mask[i][j] = exp(g_cumsum[i] - g_cumsum[j]).
  __aicore__ inline void PrepareDecayAndExp(uint32_t chunkLen) {
    uint32_t cs = chunkSize_;
    Muls(gCumsumFp32, gCumsumFp32, 1.0f, cs);
    PipeBarrier<PIPE_V>();
    Exp(expGCumFp32, gCumsumFp32, cs);
    PipeBarrier<PIPE_V>();
    TQueSync<PIPE_V, PIPE_S> expSync;
    expSync.SetFlag(0);
    expSync.WaitFlag(0);
    for (uint32_t i = 0; i < chunkLen; i++) {
      float gi = gCumsumFp32.GetValue(i);
      Duplicate(deltaFp32, gi, cs);
      PipeBarrier<PIPE_V>();
      Sub(deltaFp32, deltaFp32, gCumsumFp32, cs);
      PipeBarrier<PIPE_V>();
      Exp(decayMaskFp32[i * cs], deltaFp32, cs);
      PipeBarrier<PIPE_V>();
    }
  }

  __aicore__ inline bool CanCacheAttnQuery(uint32_t chunkLen) {
    uint64_t queryElements = static_cast<uint64_t>(chunkLen) * realK_;
    uint64_t stagingCapacity = static_cast<uint64_t>(2) * chunkSize_ * vStepAligned_;
    uint64_t cacheCapacity = static_cast<uint64_t>(chunkSize_) * vStepAligned_;
    uint64_t gmRowStride = static_cast<uint64_t>(NK_) * realK_;
    uint64_t srcGapBytes = (gmRowStride - realK_) * sizeof(inType);
    return chunkLen <= UINT16_MAX && alignK_ == realK_ && queryElements <= stagingCapacity &&
           queryElements <= cacheCapacity && (realK_ * sizeof(inType)) % BLOCK_BYTES == 0 &&
           srcGapBytes % BLOCK_BYTES == 0 && srcGapBytes / BLOCK_BYTES <= UINT16_MAX;
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
    uint64_t srcGapBytes = static_cast<uint64_t>(NK_ - 1) * realK_ * sizeof(inType);
    DataCopyParams queryParams{static_cast<uint16_t>(chunkLen),
                               static_cast<uint16_t>(realK_ * sizeof(inType) / BLOCK_BYTES),
                               static_cast<uint16_t>(srcGapBytes / BLOCK_BYTES), 0};
    event_t scalarToMte2 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::S_MTE2));
    SetFlag<HardEvent::S_MTE2>(scalarToMte2);
    WaitFlag<HardEvent::S_MTE2>(scalarToMte2);
    event_t vectorToMte2 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(vectorToMte2);
    WaitFlag<HardEvent::V_MTE2>(vectorToMte2);
    DataCopy(queryLocal, queryGm_[qkOff], queryParams);
    event_t mte2ToVector = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(mte2ToVector);
    WaitFlag<HardEvent::MTE2_V>(mte2ToVector);
    Cast(chunkAttnOutFp32, queryLocal, RoundMode::CAST_NONE, chunkLen * realK_);
    PipeBarrier<PIPE_V>();
    Muls(chunkAttnOutFp32, chunkAttnOutFp32, scale_, chunkLen * realK_);
    PipeBarrier<PIPE_V>();
  }

  // Phase 4: attn = -((k_beta @ K^T) * decay_mask) (lower tri), recursive accumulation, identity
  // diagonal; then attn_i = (Q @ K^T) * decay_mask overwrites decayMaskFp32 (lower tri + diag).
  __aicore__ inline void ComputeAttnMatrix(uint64_t qkHead, int32_t t_start, uint32_t chunkLen) {
    uint32_t cs = chunkSize_;
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

  // Phase 5: k_cumdecay = attn @ (k_beta * exp(g_cumsum)); result stored back into chunkKFp32.
  __aicore__ inline void ComputeKCumdecay(uint32_t chunkLen) {
    uint32_t cs = chunkSize_;
    TQueSync<PIPE_S, PIPE_V> expValueSync;
    for (uint32_t i = 0; i < chunkLen; i++) {
      float gExp = expGCumFp32.GetValue(i);
      expValueSync.SetFlag(0);
      expValueSync.WaitFlag(0);
      Muls(kCumdecayFp32[i * alignK_], kCumdecayFp32[i * alignK_], gExp, realK_);
    }
    TQueSync<PIPE_V, PIPE_S> decaySync;
    decaySync.SetFlag(0);
    decaySync.WaitFlag(0);
    TQueSync<PIPE_S, PIPE_V> coefficientSync;
    for (uint32_t i = 0; i < chunkLen; i++) {
      LocalTensor<float> outRow = chunkKFp32[i * alignK_];
      Duplicate(outRow, 0.0f, realK_);
      PipeBarrier<PIPE_V>();
      for (uint32_t k = 0; k <= i; k++) {
        float score = chunkScoresFp32.GetValue(i * cs + k);
        coefficientSync.SetFlag(0);
        coefficientSync.WaitFlag(0);
        Axpy(outRow, kCumdecayFp32[k * alignK_], score, realK_);
        PipeBarrier<PIPE_V>();
      }
    }
    TQueSync<PIPE_V, PIPE_S> outputSync;
    outputSync.SetFlag(0);
    outputSync.WaitFlag(0);
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
    ComputeValueNew(chunkLen, curV, avFp32);
    LoadStateTile(stateBaseOffset, workspaceStateBaseOffset, v_i, curV, c);
    ComputeVNew(chunkLen, avFp32);
    ComputeAttnInter(t_start, qkHead, chunkLen, avFp32);
    AccumOutput(chunkLen, avFp32);
    WriteAttnTileToGm(t_start, chunkLen, head_i, v_i, curV, avFp32);
    UpdateAndWriteState(t_start, chunkLen, v_i, curV, qkHead, stateBaseOffset, workspaceStateBaseOffset, avFp32,
                        isLastChunk);
  }

  // Step 1: v_beta = V * beta -> chunkVFp32.
  __aicore__ inline void LoadVBeta(int32_t t_start, uint64_t head_i, uint32_t chunkLen, uint32_t v_i, uint32_t curV,
                                   uint32_t avFp32) {
    Duplicate(chunkVFp32, 0.0f, chunkLen * avFp32);
    PipeBarrier<PIPE_V>();

    // The compact output tile is dead until ComputeValueNew, so reuse it as
    // FP16 staging and replace chunkLen*curV scalar GM loads with one strided
    // standard DataCopy. Only use the fast path when both row width and source
    // gap are exactly representable in 32-byte blocks.
    uint64_t srcRowGapElems = static_cast<uint64_t>(NV_) * realV_ - curV;
    uint64_t srcRowGapBytes = srcRowGapElems * sizeof(inType);
    if (likely((curV * sizeof(inType)) % BLOCK_BYTES == 0 && srcRowGapBytes % BLOCK_BYTES == 0 &&
               srcRowGapBytes / BLOCK_BYTES <= 65535)) {
      LocalTensor<inType> valueLocal = chunkAttnOutFp32.template ReinterpretCast<inType>();
      uint64_t vOff = (static_cast<uint64_t>(t_start) * NV_ + head_i) * realV_ + v_i;
      uint16_t rowBlocks = static_cast<uint16_t>(curV * sizeof(inType) / BLOCK_BYTES);
      uint16_t srcRowGapBlocks = static_cast<uint16_t>(srcRowGapBytes / BLOCK_BYTES);
      DataCopyParams copyParams{static_cast<uint16_t>(chunkLen), rowBlocks, srcRowGapBlocks, 0};

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

      TQueSync<PIPE_S, PIPE_V> betaValueSync;
      for (uint32_t i = 0; i < chunkLen; i++) {
        float betaVal = LoadBeta(t_start + i, head_i);
        betaValueSync.SetFlag(0);
        betaValueSync.WaitFlag(0);
        Cast(chunkVFp32[i * avFp32], valueLocal[i * curV], RoundMode::CAST_NONE, curV);
        PipeBarrier<PIPE_V>();
        Muls(chunkVFp32[i * avFp32], chunkVFp32[i * avFp32], betaVal, curV);
        PipeBarrier<PIPE_V>();
      }
      TQueSync<PIPE_V, PIPE_S> valueSync;
      valueSync.SetFlag(0);
      valueSync.WaitFlag(0);
      return;
    }

    for (uint32_t i = 0; i < chunkLen; i++) {
      int32_t t = t_start + i;
      uint64_t vOff = (static_cast<uint64_t>(t) * NV_ + head_i) * realV_;
      float beta_val = LoadBeta(t, head_i);
      for (uint32_t v = 0; v < curV; v++) {
        float v_val = static_cast<float>(valueGm_.GetValue(vOff + v_i + v));
        chunkVFp32.SetValue(i * avFp32 + v, v_val * beta_val);
      }
    }
  }

  // Step 2: value_new_tile = attn @ v_beta_tile (lower-tri attn, sum k <= i) -> chunkAttnOutFp32.
  __aicore__ inline void ComputeValueNew(uint32_t chunkLen, uint32_t curV, uint32_t avFp32) {
    (void)curV;
    Duplicate(chunkAttnOutFp32, 0.0f, chunkLen * avFp32);
    PipeBarrier<PIPE_V>();
    TQueSync<PIPE_S, PIPE_V> coefficientSync;
    for (uint32_t i = 0; i < chunkLen; i++) {
      LocalTensor<float> outRow = chunkAttnOutFp32[i * avFp32];
      for (uint32_t k = 0; k <= i; k++) {
        float score = chunkScoresFp32.GetValue(i * chunkSize_ + k);
        coefficientSync.SetFlag(0);
        coefficientSync.WaitFlag(0);
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
      DataCopyParams stateParams{1, static_cast<uint16_t>(alignedV / FP32_NUM_PER_BLOCK), 0, 0};
      for (uint32_t d = 0; d < realK_; d++) {
        uint64_t rowOff = workspaceStateBaseOffset + static_cast<uint64_t>(d) * stateWorkspaceStrideV_ + v_i;
        DataCopy(stateInFp32[d * vStepAligned_], stateWorkspaceGm_[rowOff], stateParams);
      }
      PipeBarrier<PIPE_ALL>();
      event_t mte2ToVector = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
      SetFlag<HardEvent::MTE2_V>(mte2ToVector);
      WaitFlag<HardEvent::MTE2_V>(mte2ToVector);
      return;
    }

    // The public interface follows 910B and stores state as [B, Nv, Dv, Dk].
    // The internal FP32 tile is [Dk, vStep], so transpose while loading.
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
    TQueSync<PIPE_S, PIPE_V> coefficientSync;
    for (uint32_t i = 0; i < chunkLen; i++) {
      LocalTensor<float> vpRow = chunkVFp32[i * avFp32];
      Duplicate(vpRow, 0.0f, vStepAligned_);
      PipeBarrier<PIPE_V>();
      for (uint32_t d = 0; d < realK_; d++) {
        float kcd = chunkKFp32.GetValue(i * alignK_ + d);
        LocalTensor<float> stRow = stateInFp32[d * vStepAligned_];
        coefficientSync.SetFlag(0);
        coefficientSync.WaitFlag(0);
        Axpy(vpRow, stRow, kcd, vStepAligned_);
        PipeBarrier<PIPE_V>();
      }
      Muls(vpRow, vpRow, -1.0f, vStepAligned_);
      PipeBarrier<PIPE_V>();
      Add(vpRow, vpRow, chunkAttnOutFp32[i * avFp32], vStepAligned_);
      PipeBarrier<PIPE_V>();
    }
  }

  // Step 5: attn_inter = (q * exp(g_cumsum)) @ state -> chunkAttnOutFp32.
  __aicore__ inline void ComputeAttnInter(int32_t t_start, uint64_t qkHead, uint32_t chunkLen, uint32_t avFp32) {
    TQueSync<PIPE_S, PIPE_V> coefficientSync;
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
        coefficientSync.SetFlag(0);
        coefficientSync.WaitFlag(0);
        Axpy(outRow, stRow, qd, vStepAligned_);
        PipeBarrier<PIPE_V>();
      }
    }
  }

  // Step 6: output = attn_inter + attn_i @ v_new (accumulate into chunkAttnOutFp32).
  __aicore__ inline void AccumOutput(uint32_t chunkLen, uint32_t avFp32) {
    uint32_t cs = chunkSize_;
    TQueSync<PIPE_S, PIPE_V> coefficientSync;
    for (uint32_t i = 0; i < chunkLen; i++) {
      LocalTensor<float> outRow = chunkAttnOutFp32[i * avFp32];
      for (uint32_t jj = 0; jj <= i; jj++) {
        float a = decayMaskFp32.GetValue(i * cs + jj);
        LocalTensor<float> vRow = chunkVFp32[jj * avFp32];
        coefficientSync.SetFlag(0);
        coefficientSync.WaitFlag(0);
        Axpy(outRow, vRow, a, vStepAligned_);
        PipeBarrier<PIPE_V>();
      }
    }
  }

  // Step 7: decay state, add (K * exp(gLast - g))^T @ v_new, then write the state tile back to GM.
  __aicore__ inline void UpdateAndWriteState(int32_t t_start, uint32_t chunkLen, uint32_t v_i, uint32_t curV,
                                             uint64_t qkHead, uint64_t stateBaseOffset,
                                             uint64_t workspaceStateBaseOffset, uint32_t avFp32, bool isLastChunk) {
    TQueSync<PIPE_S, PIPE_V> coefficientSync;
    float gLastExp = expGCumFp32.GetValue(chunkLen - 1);
    coefficientSync.SetFlag(0);
    coefficientSync.WaitFlag(0);
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
        coefficientSync.SetFlag(0);
        coefficientSync.WaitFlag(0);
        Axpy(stRow, vRow, coeff, vStepAligned_);
        PipeBarrier<PIPE_V>();
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
      DataCopyParams stateParams{1, static_cast<uint16_t>(alignedV / FP32_NUM_PER_BLOCK), 0, 0};
      for (uint32_t d = 0; d < realK_; d++) {
        uint64_t rowOff = workspaceStateBaseOffset + static_cast<uint64_t>(d) * stateWorkspaceStrideV_ + v_i;
        DataCopy(stateWorkspaceGm_[rowOff], stateInFp32[d * vStepAligned_], stateParams);
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
