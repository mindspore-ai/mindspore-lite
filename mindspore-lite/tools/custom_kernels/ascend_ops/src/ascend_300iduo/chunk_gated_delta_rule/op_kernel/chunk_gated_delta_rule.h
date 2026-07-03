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
  GM_ADDR g;
  GM_ADDR beta;
  GM_ADDR initialState;
  GM_ADDR cuSeqlens;
  GM_ADDR ssmStateIndices;
  GM_ADDR attnOut;
  GM_ADDR finalState;
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
    vStep_ = tilingData->vStep;
    restUbSize_ = tilingData->ubRestBytes;
    alignK_ = Ceil(tilingData->dk, FP16_NUM_PER_BLOCK) * FP16_NUM_PER_BLOCK;
    stateStrideK_ = Ceil(tilingData->dk, FP32_NUM_PER_BLOCK) * FP32_NUM_PER_BLOCK;
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
    gGm_.SetGlobalBuffer((__gm__ inType *)initParams.g);
    betaGm_.SetGlobalBuffer((__gm__ inType *)initParams.beta);
    initStateGm_.SetGlobalBuffer((__gm__ inType *)initParams.initialState);
    cuSeqlensGm_.SetGlobalBuffer((__gm__ int32_t *)initParams.cuSeqlens);
    ssmStateIndicesGm_.SetGlobalBuffer((__gm__ int32_t *)initParams.ssmStateIndices);
    finalStateGm_.SetGlobalBuffer((__gm__ outType *)initParams.finalState);
    attnOutGm_.SetGlobalBuffer((__gm__ outType *)initParams.attnOut);
  }

  __aicore__ inline void InitLocalBuffers() {
    uint32_t cs = chunkSize_;
    uint32_t ak = alignK_;
    uint32_t avFp32 = Ceil(realV_, FP32_NUM_PER_BLOCK) * FP32_NUM_PER_BLOCK;
    uint32_t avStepAligned = Ceil(vStep_, FP32_NUM_PER_BLOCK) * FP32_NUM_PER_BLOCK;
    vStepAligned_ = avStepAligned;

    // stateOutQueue: max of state tile [dk * avStepAligned] and chunk attn output [cs*avFp32]
    uint32_t stateTileBytes = stateStrideK_ * avStepAligned * sizeof(outType);
    uint32_t chunkOutBytes = cs * avFp32 * sizeof(outType);
    uint32_t outQueueBytes = stateTileBytes;
    if (chunkOutBytes > outQueueBytes) outQueueBytes = chunkOutBytes;
    pipe_->InitBuffer(stateOutQueue_, BUFFER_NUM, outQueueBytes);

    // tmpBuff layout (all FP32):
    //   chunkKFp32:      cs * alignK
    //   kCumdecayFp32:   cs * alignK
    //   decayMaskFp32:   cs * cs
    //   chunkVFp32:      cs * avFp32
    //   chunkScoresFp32 / stateInFp32: max(cs*cs, dk*avStepAligned)  [overlapped]
    //   chunkAttnOutFp32: cs * avFp32
    //   gCumsumFp32:     cs
    //   deltaFp32:       max(dkAlignedFp32, cs)
    //   expGCumFp32:     cs  (precomputed exp(gCumsum))
    pipe_->InitBuffer(tmpBuff, restUbSize_);
    uint32_t off = 0;

    chunkKFp32 = tmpBuff.GetWithOffset<float>(cs * ak, off);
    off += cs * ak * sizeof(float);

    kCumdecayFp32 = tmpBuff.GetWithOffset<float>(cs * ak, off);
    off += cs * ak * sizeof(float);

    decayMaskFp32 = tmpBuff.GetWithOffset<float>(cs * cs, off);
    off += cs * cs * sizeof(float);

    chunkVFp32 = tmpBuff.GetWithOffset<float>(cs * avFp32, off);
    off += cs * avFp32 * sizeof(float);

    // Overlapped: max of chunkScoresFp32 (cs*cs) and stateInFp32 (dk*avStepAligned)
    uint32_t overlapSize = (cs * cs > stateStrideK_ * avStepAligned) ? cs * cs : stateStrideK_ * avStepAligned;
    chunkScoresFp32 = tmpBuff.GetWithOffset<float>(overlapSize, off);
    stateInFp32 = tmpBuff.GetWithOffset<float>(stateStrideK_ * avStepAligned, off);
    off += overlapSize * sizeof(float);

    chunkAttnOutFp32 = tmpBuff.GetWithOffset<float>(cs * avFp32, off);
    off += cs * avFp32 * sizeof(float);

    gCumsumFp32 = tmpBuff.GetWithOffset<float>(cs, off);
    off += cs * sizeof(float);

    uint32_t deltaSize = (stateStrideK_ > cs) ? stateStrideK_ : cs;
    deltaFp32 = tmpBuff.GetWithOffset<float>(deltaSize, off);
    off += deltaSize * sizeof(float);

    expGCumFp32 = tmpBuff.GetWithOffset<float>(cs, off);
  }

  __aicore__ inline void ComputeAvgload() {
    uint64_t realT = 0;
    for (uint64_t batch_i = 0; batch_i < B_; batch_i++) {
      int32_t s0 = cuSeqlensGm_.GetValue(batch_i);
      int32_t s1 = cuSeqlensGm_.GetValue(batch_i + 1);
      realT += static_cast<uint64_t>(s1 - s0);
    }
    avgload_ = Ceil(realT * NV_, GetBlockNum());
  }

  __aicore__ inline void Process() {
    ComputeAvgload();
    for (uint64_t batch_i = 0; batch_i < B_; batch_i++) {
      int32_t seq0 = cuSeqlensGm_.GetValue(batch_i);
      int32_t seq1 = cuSeqlensGm_.GetValue(batch_i + 1);
      int32_t seqLen = seq1 - seq0;
      if (seqLen <= 0) {
        continue;
      }

      for (uint64_t head_i = 0; head_i < NV_; head_i++) {
        if (!IsCurrentBlock(seqLen)) continue;
        ProcessHead(seq0, seq1, head_i, batch_i);
      }
    }
  }

 private:
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
    inType gVal = gGm_.GetValue(t * NV_ + head_i);
    return static_cast<float>(gVal);
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

  // Compute recursive intra-chunk attn matrix in-place.
  // buf is stored with leading dimension ld (= chunkSize_), and only [chunkLen, chunkLen] is valid.
  // buf is lower-triangular (excluding diagonal), with upper tri (incl diagonal) = 0.
  __aicore__ inline void ComputeRecursiveAttn(LocalTensor<float> &buf, uint32_t chunkLen, uint32_t ld) {
    for (uint32_t i = 1; i < chunkLen; i++) {
      for (uint32_t k = 0; k < i; k++) {
        deltaFp32.SetValue(k, buf.GetValue(i * ld + k));
      }
      for (uint32_t j = 0; j < i; j++) {
        float sum = 0.0f;
        for (uint32_t k = j + 1; k < i; k++) {
          sum += deltaFp32.GetValue(k) * buf.GetValue(k * ld + j);
        }
        buf.SetValue(i * ld + j, deltaFp32.GetValue(j) + sum);
      }
    }
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
    for (uint32_t i = 0; i < chunkLen; i++) {
      uint64_t outOff = (static_cast<uint64_t>(t_start + i) * NV_ + head_i) * realV_ + v_i;
      DataCopyExtParams outParams{1, static_cast<uint32_t>(curV * sizeof(outType)), 0, 0, 0};
      CopyToGm(attnOutGm_[outOff], outLocal[i * avFp32], outParams);
      PipeBarrier<PIPE_MTE3>();
    }
    stateOutQueue_.FreeTensor(outLocal);
  }

  __aicore__ inline void ProcessHead(int32_t seq0, int32_t seq1, uint64_t head_i, uint64_t batch_i) {
    (void)batch_i;  // state is indexed via ssm_state_indices, not batch_i
    uint64_t nvPerNk = (NV_ >= NK_) ? (NV_ / NK_) : 1;
    uint64_t qkHead = head_i / nvPerNk;
    int32_t stateIdx = ssmStateIndicesGm_.GetValue(seq0);
    uint64_t stateBaseOffset = (static_cast<uint64_t>(stateIdx) * NV_ + head_i) * realK_ * realV_;
    uint32_t cs = chunkSize_;
    int32_t totalLen = seq1 - seq0;
    uint32_t numC = Ceil(static_cast<uint32_t>(totalLen), cs);
    for (uint32_t c = 0; c < numC; c++) {
      ProcessChunk(seq0, seq1, head_i, c, qkHead, stateBaseOffset);
    }
  }

  // ---- per-chunk phases (ProcessHead delegates one chunk to ProcessChunk) ----

  __aicore__ inline void ProcessChunk(int32_t seq0, int32_t seq1, uint64_t head_i, uint32_t c, uint64_t qkHead,
                                      uint64_t stateBaseOffset) {
    uint32_t cs = chunkSize_;
    uint32_t avFp32 = Ceil(realV_, FP32_NUM_PER_BLOCK) * FP32_NUM_PER_BLOCK;
    int32_t totalLen = seq1 - seq0;
    int32_t t_start = seq0 + c * cs;
    uint32_t chunkLen =
      (static_cast<uint32_t>(totalLen) - c * cs >= cs) ? cs : static_cast<uint32_t>(totalLen) - c * cs;

    LoadChunkKey(qkHead, t_start, chunkLen);
    ComputeGCumsum(t_start, head_i, chunkLen);
    ComputeKBeta(t_start, head_i, chunkLen);
    PrepareDecayAndExp(chunkLen);
    ComputeAttnMatrix(qkHead, t_start, chunkLen);
    ComputeKCumdecay(chunkLen);
    ProcessVTiles(t_start, chunkLen, head_i, qkHead, avFp32, stateBaseOffset, c);
  }

  // Phase 1: load K for this chunk into chunkKFp32.
  __aicore__ inline void LoadChunkKey(uint64_t qkHead, int32_t t_start, uint32_t chunkLen) {
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
    for (uint32_t i = 0; i < chunkLen; i++) {
      float beta_val = LoadBeta(t_start + i, head_i);
      for (uint32_t j = 0; j < realK_; j++) {
        float k_val = chunkKFp32.GetValue(i * alignK_ + j);
        kCumdecayFp32.SetValue(i * alignK_ + j, k_val * beta_val);
      }
    }
  }

  // Phase 3: precompute exp(gCumsum) and decay_mask[i][j] = exp(g_cumsum[i] - g_cumsum[j]).
  __aicore__ inline void PrepareDecayAndExp(uint32_t chunkLen) {
    uint32_t cs = chunkSize_;
    Muls(gCumsumFp32, gCumsumFp32, 1.0f, cs);
    PipeBarrier<PIPE_V>();
    Exp(expGCumFp32, gCumsumFp32, cs);
    PipeBarrier<PIPE_V>();
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

  // Phase 4: attn = -((k_beta @ K^T) * decay_mask) (lower tri), recursive accumulation, identity
  // diagonal; then attn_i = (Q @ K^T) * decay_mask overwrites decayMaskFp32 (lower tri + diag).
  __aicore__ inline void ComputeAttnMatrix(uint64_t qkHead, int32_t t_start, uint32_t chunkLen) {
    uint32_t cs = chunkSize_;
    for (uint32_t i = 1; i < chunkLen; i++) {
      for (uint32_t jj = 0; jj < i; jj++) {
        float dot = 0.0f;
        for (uint32_t d = 0; d < realK_; d++) {
          dot += kCumdecayFp32.GetValue(i * alignK_ + d) * chunkKFp32.GetValue(jj * alignK_ + d);
        }
        chunkScoresFp32.SetValue(i * cs + jj, -dot * decayMaskFp32.GetValue(i * cs + jj));
      }
    }
    ComputeRecursiveAttn(chunkScoresFp32, chunkLen, cs);
    for (uint32_t i = 0; i < chunkLen; i++) {
      chunkScoresFp32.SetValue(i * cs + i, 1.0f);
    }
    for (uint32_t i = 0; i < chunkLen; i++) {
      int32_t t_i = t_start + i;
      uint64_t qkOff_i = (static_cast<uint64_t>(t_i) * NK_ + qkHead) * realK_;
      for (uint32_t d = 0; d < realK_; d++) {
        deltaFp32.SetValue(d, static_cast<float>(queryGm_.GetValue(qkOff_i + d)) * scale_);
      }
      for (uint32_t jj = 0; jj <= i; jj++) {
        float dot = 0.0f;
        for (uint32_t d = 0; d < realK_; d++) {
          dot += deltaFp32.GetValue(d) * chunkKFp32.GetValue(jj * alignK_ + d);
        }
        float decay = decayMaskFp32.GetValue(i * cs + jj);
        decayMaskFp32.SetValue(i * cs + jj, dot * decay);
      }
    }
  }

  // Phase 5: k_cumdecay = attn @ (k_beta * exp(g_cumsum)); result stored back into chunkKFp32.
  __aicore__ inline void ComputeKCumdecay(uint32_t chunkLen) {
    uint32_t cs = chunkSize_;
    for (uint32_t i = 0; i < chunkLen; i++) {
      float gExp = expGCumFp32.GetValue(i);
      for (uint32_t j = 0; j < realK_; j++) {
        float val = kCumdecayFp32.GetValue(i * alignK_ + j);
        kCumdecayFp32.SetValue(i * alignK_ + j, val * gExp);
      }
    }
    for (uint32_t i = 0; i < chunkLen; i++) {
      for (uint32_t d = 0; d < realK_; d++) {
        float sum = 0.0f;
        for (uint32_t k = 0; k <= i; k++) {
          sum += chunkScoresFp32.GetValue(i * cs + k) * kCumdecayFp32.GetValue(k * alignK_ + d);
        }
        deltaFp32.SetValue(d, sum);
      }
      for (uint32_t d = 0; d < realK_; d++) {
        chunkKFp32.SetValue(i * alignK_ + d, deltaFp32.GetValue(d));
      }
    }
  }

  // ---- Phase 6: per v-tile processing (attn matrix and state tile overlap) ----

  __aicore__ inline void ProcessVTiles(int32_t t_start, uint32_t chunkLen, uint64_t head_i, uint64_t qkHead,
                                       uint32_t avFp32, uint64_t stateBaseOffset, uint32_t c) {
    for (uint32_t v_i = 0; v_i < realV_; v_i += vStep_) {
      uint32_t curV = (v_i + vStep_ > realV_) ? realV_ - v_i : vStep_;
      LoadVBeta(t_start, head_i, chunkLen, v_i, curV, avFp32);
      ComputeValueNew(chunkLen, curV, avFp32);
      LoadStateTile(stateBaseOffset, v_i, curV, c);
      ComputeVNew(chunkLen, avFp32);
      ComputeAttnInter(t_start, qkHead, chunkLen, avFp32);
      AccumOutput(chunkLen, avFp32);
      WriteAttnTileToGm(t_start, chunkLen, head_i, v_i, curV, avFp32);
      UpdateAndWriteState(t_start, chunkLen, v_i, curV, qkHead, stateBaseOffset, avFp32);
    }
  }

  // Step 1: v_beta = V * beta -> chunkVFp32.
  __aicore__ inline void LoadVBeta(int32_t t_start, uint64_t head_i, uint32_t chunkLen, uint32_t v_i, uint32_t curV,
                                   uint32_t avFp32) {
    Duplicate(chunkVFp32, 0.0f, chunkLen * avFp32);
    PipeBarrier<PIPE_V>();
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
    uint32_t cs = chunkSize_;
    Duplicate(chunkAttnOutFp32, 0.0f, chunkLen * avFp32);
    PipeBarrier<PIPE_V>();
    for (uint32_t i = 0; i < chunkLen; i++) {
      for (uint32_t v = 0; v < curV; v++) {
        float sum = 0.0f;
        for (uint32_t k = 0; k <= i; k++) {
          sum += chunkScoresFp32.GetValue(i * cs + k) * chunkVFp32.GetValue(k * avFp32 + v);
        }
        chunkAttnOutFp32.SetValue(i * avFp32 + v, sum);
      }
    }
  }

  // Step 3: load state tile into stateInFp32 (overlaps chunkScoresFp32 — must run after Step 2).
  __aicore__ inline void LoadStateTile(uint64_t stateBaseOffset, uint32_t v_i, uint32_t curV, uint32_t c) {
    uint32_t stateTileElem = stateStrideK_ * vStepAligned_;
    Duplicate(stateInFp32, 0.0f, stateTileElem);
    PipeBarrier<PIPE_V>();
    for (uint32_t d = 0; d < realK_; d++) {
      uint64_t rowOff = stateBaseOffset + static_cast<uint64_t>(d) * realV_ + v_i;
      for (uint32_t v = 0; v < curV; v++) {
        float sval = 0.0f;
        if (c == 0) {
          sval = static_cast<float>(initStateGm_.GetValue(rowOff + v));
        } else {
          sval = static_cast<float>(finalStateGm_.GetValue(rowOff + v));
        }
        stateInFp32.SetValue(d * vStepAligned_ + v, sval);
      }
    }
  }

  // Step 4: v_new = value_new - (k_cumdecay @ state) -> chunkVFp32 (reused).
  __aicore__ inline void ComputeVNew(uint32_t chunkLen, uint32_t avFp32) {
    for (uint32_t i = 0; i < chunkLen; i++) {
      LocalTensor<float> vpRow = chunkVFp32[i * avFp32];
      Duplicate(vpRow, 0.0f, vStepAligned_);
      PipeBarrier<PIPE_V>();
      for (uint32_t d = 0; d < realK_; d++) {
        float kcd = chunkKFp32.GetValue(i * alignK_ + d);
        LocalTensor<float> stRow = stateInFp32[d * vStepAligned_];
        Muls(deltaFp32, stRow, kcd, vStepAligned_);
        PipeBarrier<PIPE_V>();
        Add(vpRow, vpRow, deltaFp32, vStepAligned_);
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
        Muls(deltaFp32, stRow, qd, vStepAligned_);
        PipeBarrier<PIPE_V>();
        Add(outRow, outRow, deltaFp32, vStepAligned_);
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
        Muls(deltaFp32, vRow, a, vStepAligned_);
        PipeBarrier<PIPE_V>();
        Add(outRow, outRow, deltaFp32, vStepAligned_);
        PipeBarrier<PIPE_V>();
      }
    }
  }

  // Step 7: decay state, add (K * exp(gLast - g))^T @ v_new, then write the state tile back to GM.
  __aicore__ inline void UpdateAndWriteState(int32_t t_start, uint32_t chunkLen, uint32_t v_i, uint32_t curV,
                                             uint64_t qkHead, uint64_t stateBaseOffset, uint32_t avFp32) {
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
        Muls(deltaFp32, vRow, coeff, vStepAligned_);
        PipeBarrier<PIPE_V>();
        Add(stRow, stRow, deltaFp32, vStepAligned_);
        PipeBarrier<PIPE_V>();
      }
    }
    // Write updated state back to GM
    uint32_t alignedElem = stateStrideK_ * vStepAligned_;
    Muls(stateInFp32, stateInFp32, 1.0f, alignedElem);
    PipeBarrier<PIPE_V>();
    LocalTensor<outType> stateLocal = stateOutQueue_.AllocTensor<outType>();
    Cast(stateLocal, stateInFp32, RoundMode::CAST_NONE, alignedElem);
    PipeBarrier<PIPE_V>();
    stateOutQueue_.EnQue<outType>(stateLocal);
    stateLocal = stateOutQueue_.DeQue<outType>();
    for (uint32_t d = 0; d < realK_; d++) {
      uint64_t rowOff = stateBaseOffset + static_cast<uint64_t>(d) * realV_ + v_i;
      DataCopyExtParams stateParams{1, static_cast<uint32_t>(curV * sizeof(outType)), 0, 0, 0};
      CopyToGm(finalStateGm_[rowOff], stateLocal[d * vStepAligned_], stateParams);
      PipeBarrier<PIPE_MTE3>();
    }
    stateOutQueue_.FreeTensor(stateLocal);
  }

 private:
  GlobalTensor<inType> queryGm_;
  GlobalTensor<inType> keyGm_;
  GlobalTensor<inType> valueGm_;
  GlobalTensor<inType> gGm_;
  GlobalTensor<inType> betaGm_;
  GlobalTensor<inType> initStateGm_;
  GlobalTensor<int32_t> cuSeqlensGm_;
  GlobalTensor<int32_t> ssmStateIndicesGm_;
  GlobalTensor<outType> finalStateGm_;
  GlobalTensor<outType> attnOutGm_;
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
  LocalTensor<float> deltaFp32;         // temp: max(stateStrideK_, chunkSize_)
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
  uint32_t restUbSize_;
  uint32_t vStepAligned_;
  uint32_t stateStrideK_;
  uint32_t load_;
  uint32_t usedblk_;
  uint32_t avgload_;
  float scale_;
  uint64_t blockIdx_;
};

#endif  // CHUNK_GATED_DELTA_RULE_KERNEL_H_
