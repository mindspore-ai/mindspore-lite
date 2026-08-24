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

#ifndef RECURRENT_GATED_DELTA_RULE_KERNEL_H_
#define RECURRENT_GATED_DELTA_RULE_KERNEL_H_

#include "kernel_operator.h"                         // NOLINT(build/include_subdir)
#include "recurrent_gated_delta_rule_tiling_data.h"  // NOLINT(build/include_subdir)

using namespace AscendC;  // NOLINT(build/namespaces)
constexpr uint64_t BUFFER_NUM = 1;
constexpr uint32_t MAX_OUT_BUFFER_NUM = 2;
constexpr uint64_t MAX_MTP = 8;
constexpr uint64_t FP16_NUM_PER_BLOCK = 16;
constexpr uint64_t FP32_NUM_PER_BLOCK = 8;
constexpr uint32_t REPEAT_LENGTH = 64;
constexpr uint32_t MAX_REPEAT_TIME = 255;
constexpr uint32_t MAX_CAST_ELEMENTS = REPEAT_LENGTH * MAX_REPEAT_TIME;
constexpr int64_t BLOCK_BYTES = 32;

template <HardEvent event>
__aicore__ inline void SetWaitFlag(HardEvent evt) {
  event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
  SetFlag<event>(eventId);
  WaitFlag<event>(eventId);
}

template <typename T>
__aicore__ inline void DataCopyPadCustom(LocalTensor<T> inLocal, GlobalTensor<T> srcGm,
                                         DataCopyExtParams tokenCopyParams, DataCopyPadExtParams<T> padParams) {
  int64_t elem = tokenCopyParams.blockLen / sizeof(T);
  int64_t numPerBlock = BLOCK_BYTES / sizeof(T);
  int64_t alignElem = AlignUp(elem, numPerBlock);
  int64_t srcStrideElem = tokenCopyParams.srcStride / sizeof(T);
  int64_t gmStepPerRow = elem + srcStrideElem;

  if (likely(alignElem == elem && srcStrideElem == 0)) {
    DataCopyParams copyParams = {tokenCopyParams.blockCount, static_cast<uint16_t>(alignElem / numPerBlock), 0, 0};
    DataCopy(inLocal, srcGm, copyParams);
  } else {
    DataCopyParams copyParams = {1, static_cast<uint16_t>(alignElem / numPerBlock), 0, 0};
    for (uint32_t i = 0; i < tokenCopyParams.blockCount; i++) {
      DataCopy(inLocal[i * alignElem], srcGm[i * gmStepPerRow], copyParams);
    }
  }
}

template <typename DST, typename SRC>
__aicore__ inline void DataCopyCustom(DST dst, SRC src, DataCopyParams copyParams) {
  int64_t alignBytes = AlignUp(static_cast<int64_t>(copyParams.blockLen), BLOCK_BYTES);
  int64_t blocks = alignBytes / BLOCK_BYTES;
  DataCopyParams aligned = {copyParams.blockCount, static_cast<uint16_t>(blocks), 0, 0};
  DataCopy(dst, src, aligned);
}

template <typename T, bool needBack = false, bool isAtomic = false>
__aicore__ inline void DataCopyCustom(GlobalTensor<T> dstGm, LocalTensor<T> inLocal, DataCopyExtParams copyParamsIn) {
  int64_t elem = copyParamsIn.blockLen / sizeof(T);
  int64_t numPerBlock = sizeof(T) == 0 ? 1 : BLOCK_BYTES / sizeof(T);
  int64_t alignElem = AlignUp(elem, numPerBlock);

  if (likely(alignElem == elem)) {
    DataCopyParams copyParams = {static_cast<uint16_t>(copyParamsIn.blockCount),
                                 static_cast<uint16_t>(alignElem / numPerBlock), 0, 0};
    DataCopy(dstGm, inLocal, copyParams);
  } else {
    if (copyParamsIn.blockCount == 1) {
      if constexpr (needBack) {
        int64_t elemAlignDown = numPerBlock == 0 ? 0 : elem / numPerBlock * numPerBlock;
        if (elemAlignDown != 0) {
          DataCopyParams copyParams = {static_cast<uint16_t>(copyParamsIn.blockCount),
                                       static_cast<uint16_t>(elemAlignDown / numPerBlock), 0, 0};
          DataCopy(dstGm, inLocal, copyParams);
          SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
          SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
          for (uint32_t i = 0; i < numPerBlock; i++) {
            inLocal.SetValue(alignElem - 1 - i, inLocal.GetValue(elem - 1 - i));
          }
          SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
          DataCopyParams copyParamslast = {1, 1, 0, 0};
          DataCopy(dstGm[elem - numPerBlock], inLocal[elemAlignDown], copyParamslast);
        } else {
          T tmp[BLOCK_BYTES / sizeof(T)];
          SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
          SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
          for (uint32_t i = 0; i < elem; i++) {
            tmp[i] = inLocal.GetValue(elem - 1 - i);
          }
          DataCopyParams copyParamslast = {1, 1, 0, 0};
          SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::S_MTE2);
          SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
          DataCopy(inLocal, dstGm[elem - numPerBlock], copyParamslast);
          SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
          for (uint32_t i = 0; i < elem; i++) {
            inLocal.SetValue(numPerBlock - 1 - i, tmp[i]);
          }
          SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
          DataCopy(dstGm[elem - numPerBlock], inLocal, copyParamslast);
        }
      } else if constexpr (isAtomic) {
        SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        for (uint32_t i = 0; i < alignElem - elem; i++) {
          inLocal.SetValue(alignElem - 1 - i, T(0));
        }
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyParams copyParams = {static_cast<uint16_t>(copyParamsIn.blockCount),
                                     static_cast<uint16_t>(alignElem / numPerBlock), 0, 0};
        DataCopy(dstGm, inLocal, copyParams);
      } else {
        DataCopyParams copyParams = {static_cast<uint16_t>(copyParamsIn.blockCount),
                                     static_cast<uint16_t>(alignElem / numPerBlock), 0, 0};
        DataCopy(dstGm, inLocal, copyParams);
      }
    } else {
      DataCopyParams copyParams = {1, static_cast<uint16_t>(alignElem / numPerBlock), 0, 0};
      for (uint32_t i = 0; i < copyParamsIn.blockCount; i++) {
        DataCopy(dstGm[i * elem], inLocal[i * alignElem], copyParams);
        PipeBarrier<PIPE_MTE3>();
      }
    }
  }
}

struct RGDRInitParams {
  GM_ADDR query;
  GM_ADDR key;
  GM_ADDR value;
  GM_ADDR gama;
  GM_ADDR gamaK;
  GM_ADDR beta;
  GM_ADDR initState;
  GM_ADDR cuSeqlens;
  GM_ADDR ssmStateIndices;
  GM_ADDR numAcceptedTokens;
  GM_ADDR attnOut;
  GM_ADDR finalState;
};

template <typename inType, typename outType>
class RGDR {
 public:
  __aicore__ inline explicit RGDR(const RecurrentGatedDeltaRuleTilingData *tilingData)
      : pipe_(nullptr), gama_(1.0f), gamaK_(1.0f), beta_(0.0f), blockIdx(0) {
    B_ = tilingData->b;
    T_ = tilingData->t;
    NK_ = tilingData->nk;
    realK_ = tilingData->dk;
    NV_ = tilingData->nv;
    realV_ = tilingData->dv;
    scale_ = tilingData->scale;
    hasAcceptedTokens_ = (tilingData->hasAcceptedTokens == 1);
    hasGama_ = (tilingData->hasGama == 1);
    hasGamaK_ = (tilingData->hasGamaK == 1);
    gamaKScalar_ = (tilingData->gamaKScalar == 1);
    cuSeqlensIsPrefix_ = (tilingData->cuSeqlensIsPrefix == 1);
    cuSeqlensIsInt64_ = (tilingData->cuSeqlensIsInt64 == 1);
    ssmStateIndicesIsInt64_ = (tilingData->ssmStateIndicesIsInt64 == 1);
    vStep_ = tilingData->vStep;
    stateOutBufferNum_ = (tilingData->stateOutBufferNum == MAX_OUT_BUFFER_NUM) ? MAX_OUT_BUFFER_NUM : BUFFER_NUM;
    attnOutBufferNum_ = (tilingData->attnOutBufferNum == MAX_OUT_BUFFER_NUM) ? MAX_OUT_BUFFER_NUM : BUFFER_NUM;
    restUbSize_ = tilingData->ubRestBytes;
    uint64_t taskUnits = static_cast<uint64_t>(B_) * static_cast<uint64_t>(NV_);
    workBlockDim_ = taskUnits < tilingData->vectorCoreNum ? taskUnits : tilingData->vectorCoreNum;
    if (workBlockDim_ == 0) {
      workBlockDim_ = 1;
    }
    alignK_ = Ceil(tilingData->dk, FP16_NUM_PER_BLOCK) * FP16_NUM_PER_BLOCK;
    alignV_ = Ceil(tilingData->dv, FP16_NUM_PER_BLOCK) * FP16_NUM_PER_BLOCK;
  }

  __aicore__ inline void Init(const RGDRInitParams &initParams, TPipe *pipe) {
    blockIdx = GetBlockIdx();
    if (blockIdx >= workBlockDim_) {
      return;
    }
    pipe_ = pipe;
    SetGlobalTensors(initParams);
    InitLocalBuffers();
  }

  __aicore__ inline void SetGlobalTensors(const RGDRInitParams &initParams) {
    queryGm_.SetGlobalBuffer((__gm__ inType *)initParams.query);
    keyGm_.SetGlobalBuffer((__gm__ inType *)initParams.key);
    valueGm_.SetGlobalBuffer((__gm__ inType *)initParams.value);
    gamaGm_.SetGlobalBuffer((__gm__ float *)initParams.gama);
    gamaKGm_.SetGlobalBuffer((__gm__ float *)initParams.gamaK);
    betaGm_.SetGlobalBuffer((__gm__ inType *)initParams.beta);
    initStateGm_.SetGlobalBuffer((__gm__ inType *)initParams.initState);
    cuSeqlens32Gm_.SetGlobalBuffer((__gm__ int32_t *)initParams.cuSeqlens);
    cuSeqlens64Gm_.SetGlobalBuffer((__gm__ int64_t *)initParams.cuSeqlens);
    ssmStateIndices32Gm_.SetGlobalBuffer((__gm__ int32_t *)initParams.ssmStateIndices);
    ssmStateIndices64Gm_.SetGlobalBuffer((__gm__ int64_t *)initParams.ssmStateIndices);
    numAcceptedTokensGm_.SetGlobalBuffer((__gm__ int32_t *)initParams.numAcceptedTokens);
    finalStateGm_.SetGlobalBuffer((__gm__ outType *)initParams.finalState);
    attnOutGm_.SetGlobalBuffer((__gm__ outType *)initParams.attnOut);
  }

  __aicore__ inline void InitLocalBuffers() {
    uint32_t cubeSize = alignK_ * vStep_ * sizeof(float);
    uint32_t vSize = MAX_MTP * alignV_ * sizeof(float);
    uint32_t kSize = MAX_MTP * alignK_ * sizeof(float);
    uint32_t betaUbSize = Ceil(MAX_MTP * NV_, FP16_NUM_PER_BLOCK) * FP16_NUM_PER_BLOCK * sizeof(float);
    pipe_->InitBuffer(qInQueue_, BUFFER_NUM, MAX_MTP * alignK_ * sizeof(inType));
    pipe_->InitBuffer(kInQueue_, BUFFER_NUM, MAX_MTP * alignK_ * sizeof(inType));
    pipe_->InitBuffer(vInQueue_, BUFFER_NUM, MAX_MTP * alignV_ * sizeof(inType));
    pipe_->InitBuffer(stateInQueue_, BUFFER_NUM, alignK_ * vStep_ * sizeof(inType));
    if (hasGamaK_ && !gamaKScalar_) {
      pipe_->InitBuffer(gamaKInQueue_, BUFFER_NUM, MAX_MTP * alignK_ * sizeof(float));
    }
    pipe_->InitBuffer(betaInQueue_, BUFFER_NUM, MAX_MTP * NV_ * sizeof(inType));
    pipe_->InitBuffer(stateOutQueue_, stateOutBufferNum_, alignK_ * vStep_ * sizeof(outType));
    pipe_->InitBuffer(attnOutQueue_, attnOutBufferNum_, vStep_ * sizeof(outType));
    pipe_->InitBuffer(tmpBuff, restUbSize_);
    uint32_t buffOffset = 0;
    attnInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(vStep_), buffOffset);
    buffOffset += vStep_ * sizeof(float);
    vInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(MAX_MTP * alignV_), buffOffset);
    buffOffset += vSize;
    qInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(MAX_MTP * alignK_), buffOffset);
    buffOffset += kSize;
    kInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(MAX_MTP * alignK_), buffOffset);
    buffOffset += kSize;
    stateInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(alignK_ * vStep_), buffOffset);
    buffOffset += cubeSize;
    broadTmpInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(alignK_ * vStep_), buffOffset);
    buffOffset += cubeSize;
    betaInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(betaUbSize), buffOffset);
    buffOffset += betaUbSize;
    if (hasGama_) {
      uint32_t gamaSize = MAX_MTP * NV_ * sizeof(float);
      gamaInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(MAX_MTP * NV_), buffOffset);
      buffOffset += gamaSize;
    }
    if (hasGamaK_ && gamaKScalar_) {
      uint32_t gamaKScalarSize = Ceil(MAX_MTP * NV_, FP32_NUM_PER_BLOCK) * FP32_NUM_PER_BLOCK * sizeof(float);
      gamaKScalarInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(gamaKScalarSize), buffOffset);
    }
  }

  __aicore__ inline void WaitVectorToScalar() {
    TEventID eventId = pipe_->FetchEventID(HardEvent::V_S);
    SetFlag<HardEvent::V_S>(eventId);
    WaitFlag<HardEvent::V_S>(eventId);
  }

  __aicore__ inline void Process() {
    uint64_t blockDim = workBlockDim_;
    for (uint64_t batch_i = 0; batch_i < B_; batch_i++) {
      int32_t seq0 = 0;
      int32_t seq1 = 0;
      int32_t seqLen = 0;
      if (cuSeqlensIsPrefix_) {
        seq0 = GetCuSeqlen(batch_i);
        seq1 = GetCuSeqlen(batch_i + 1);
        seqLen = seq1 - seq0;
      } else {
        seqLen = GetCuSeqlen(batch_i);
        seq0 = 0;
        for (uint64_t i = 0; i < batch_i; i++) {
          seq0 += GetCuSeqlen(i);
        }
        seq1 = seq0 + seqLen;
      }
      if (seqLen <= 0) {
        continue;
      }
      if (seqLen > static_cast<int32_t>(MAX_MTP)) {
        return;
      }
      if (seq0 < 0 || seq1 < 0 || seq0 > static_cast<int32_t>(T_) || seq1 > static_cast<int32_t>(T_) || seq0 > seq1) {
        return;
      }
      uint32_t copyFlag = 0;
      uint64_t stateOffset;
      for (uint64_t head_i = 0; head_i < NV_; head_i++) {
        if (blockDim > 0 &&
            (static_cast<uint64_t>(batch_i) * static_cast<uint64_t>(NV_) + head_i) % blockDim != blockIdx) {
          continue;
        }
        copyFlag++;
        if (copyFlag == 1) {
          int32_t stateTokenIdx = seq0;
          if (hasAcceptedTokens_) {
            int32_t acceptedTokenNum = numAcceptedTokensGm_.GetValue(batch_i);
            if (acceptedTokenNum > 0 && acceptedTokenNum <= seqLen) {
              stateTokenIdx = seq0 + acceptedTokenNum - 1;
            }
          }
          stateOffset = GetSsmStateIndex(stateTokenIdx);
          CopyInGamaBeta(seq0, seq1);
        }
        ProcessHead(seq0, seq1, head_i, stateOffset);
      }
    }
  }

 private:
  __aicore__ inline int32_t GetCuSeqlen(uint64_t idx) const {
    if (cuSeqlensIsInt64_) {
      return static_cast<int32_t>(cuSeqlens64Gm_.GetValue(idx));
    }
    return cuSeqlens32Gm_.GetValue(idx);
  }

  __aicore__ inline int32_t GetSsmStateIndex(uint64_t idx) const {
    if (ssmStateIndicesIsInt64_) {
      return static_cast<int32_t>(ssmStateIndices64Gm_.GetValue(idx));
    }
    return ssmStateIndices32Gm_.GetValue(idx);
  }

  __aicore__ inline void CastInputToFp32(LocalTensor<float> dstTensor, LocalTensor<inType> srcTensor,
                                         uint32_t elementCount) {
    for (uint32_t offset = 0; offset < elementCount; offset += MAX_CAST_ELEMENTS) {
      uint32_t currentCount = Std::min(MAX_CAST_ELEMENTS, elementCount - offset);
      Cast(dstTensor[offset], srcTensor[offset], RoundMode::CAST_NONE, currentCount);
      PipeBarrier<PIPE_V>();
    }
  }

  __aicore__ inline void CopyInQKV(uint64_t vOffset, uint64_t qkOffset, int32_t seqLen) {
    LocalTensor<inType> qLocal = qInQueue_.AllocTensor<inType>();
    LocalTensor<inType> kLocal = kInQueue_.AllocTensor<inType>();
    LocalTensor<inType> vLocal = vInQueue_.AllocTensor<inType>();
    DataCopyExtParams qkInParams{static_cast<uint16_t>(seqLen), static_cast<uint32_t>(realK_ * sizeof(inType)),
                                 static_cast<uint32_t>((NK_ - 1) * realK_ * sizeof(inType)), 0, 0};
    DataCopyExtParams vInParams{static_cast<uint16_t>(seqLen), static_cast<uint32_t>(realV_ * sizeof(inType)),
                                static_cast<uint32_t>((NV_ - 1) * realV_ * sizeof(inType)), 0, 0};
    DataCopyPadExtParams<inType> qkPadParams{true, 0, static_cast<uint8_t>(alignK_ - realK_), 0};
    DataCopyPadExtParams<inType> vPadParams{true, 0, static_cast<uint8_t>(alignV_ - realV_), 0};
    if (hasGamaK_ && !gamaKScalar_) {
      uint32_t alignKGamma = Ceil(realK_, FP32_NUM_PER_BLOCK) * FP32_NUM_PER_BLOCK;
      uint32_t stride = alignKGamma < alignK_ ? 1 : 0;
      DataCopyExtParams gkInParams{static_cast<uint16_t>(seqLen), static_cast<uint32_t>(realK_ * sizeof(float)),
                                   static_cast<uint32_t>((NV_ - 1) * realK_ * sizeof(float)), stride, 0};
      DataCopyPadExtParams<float> gkPadParams{true, 0, static_cast<uint8_t>(alignKGamma - realK_), 0};
      LocalTensor<float> gamaKLocal = gamaKInQueue_.AllocTensor<float>();
      Duplicate<float>(gamaKLocal, 0, alignK_ * seqLen);
      TEventID evevtIdVtoMte2 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
      SetFlag<HardEvent::V_MTE2>(evevtIdVtoMte2);
      WaitFlag<HardEvent::V_MTE2>(evevtIdVtoMte2);
      DataCopyPadCustom(gamaKLocal, gamaKGm_[vOffset / realV_ * realK_], gkInParams, gkPadParams);
      gamaKInQueue_.EnQue<float>(gamaKLocal);
      gamaKInUb = gamaKInQueue_.DeQue<float>();
      Exp(gamaKInUb, gamaKInUb, alignK_ * seqLen);
      PipeBarrier<PIPE_V>();
    }
    DataCopyPadCustom(qLocal, queryGm_[qkOffset], qkInParams, qkPadParams);
    DataCopyPadCustom(kLocal, keyGm_[qkOffset], qkInParams, qkPadParams);
    DataCopyPadCustom(vLocal, valueGm_[vOffset], vInParams, vPadParams);
    qInQueue_.EnQue<inType>(qLocal);
    kInQueue_.EnQue<inType>(kLocal);
    vInQueue_.EnQue<inType>(vLocal);
    qLocal = qInQueue_.DeQue<inType>();
    kLocal = kInQueue_.DeQue<inType>();
    vLocal = vInQueue_.DeQue<inType>();
    Cast(qInUb, qLocal, RoundMode::CAST_NONE, alignK_ * seqLen);
    Cast(kInUb, kLocal, RoundMode::CAST_NONE, alignK_ * seqLen);
    Cast(vInUb, vLocal, RoundMode::CAST_NONE, alignV_ * seqLen);
    PipeBarrier<PIPE_V>();
    Muls(qInUb, qInUb, scale_, seqLen * alignK_);
    SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
    qInQueue_.FreeTensor(qLocal);
    kInQueue_.FreeTensor(kLocal);
    vInQueue_.FreeTensor(vLocal);
  }

  __aicore__ inline void PrefetchState(uint64_t stateOffest, uint32_t curSingleV) {
    LocalTensor<inType> stateLocal = stateInQueue_.AllocTensor<inType>();
    if (likely(realK_ == alignK_)) {
      uint32_t blockLen = curSingleV * alignK_ / FP16_NUM_PER_BLOCK;
      DataCopyParams stateInParams{1, static_cast<uint16_t>(blockLen), 0, 0};
      DataCopy(stateLocal, initStateGm_[stateOffest], stateInParams);
    } else {
      DataCopyExtParams stateInParams{static_cast<uint16_t>(curSingleV), static_cast<uint32_t>(realK_ * sizeof(inType)),
                                      0, 0, 0};
      DataCopyPadExtParams<inType> padParams{true, 0, static_cast<uint8_t>(alignK_ - realK_), 0};
      DataCopyPadCustom(stateLocal, initStateGm_[stateOffest], stateInParams, padParams);
    }
    stateInQueue_.EnQue<inType>(stateLocal);
  }

  __aicore__ inline void LoadPrefetchedState(uint32_t curSingleV) {
    uint32_t alignedV = Ceil(curSingleV, FP16_NUM_PER_BLOCK) * FP16_NUM_PER_BLOCK;
    LocalTensor<inType> stateLocal = stateInQueue_.DeQue<inType>();
    CastInputToFp32(stateInUb, stateLocal, alignK_ * alignedV);
    SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
    stateInQueue_.FreeTensor(stateLocal);
  }

  __aicore__ inline float DotFp32(LocalTensor<float> lhsTensor, LocalTensor<float> rhsTensor) {
    Mul(broadTmpInUb, lhsTensor, rhsTensor, alignK_);
    constexpr uint32_t kReduceDstStride = FP32_NUM_PER_BLOCK;
    uint32_t fullRepeats = alignK_ / REPEAT_LENGTH;
    uint32_t tail = alignK_ % REPEAT_LENGTH;
    if (fullRepeats > 0) {
      WholeReduceSum(broadTmpInUb, broadTmpInUb, REPEAT_LENGTH, fullRepeats, kReduceDstStride, 1,
                     REPEAT_LENGTH / FP32_NUM_PER_BLOCK);
    }
    WaitVectorToScalar();
    float result = 0.0f;
    for (uint32_t i = 0; i < fullRepeats; ++i) {
      result += broadTmpInUb.GetValue(i * kReduceDstStride);
    }
    uint32_t tailOffset = fullRepeats * REPEAT_LENGTH;
    for (uint32_t i = 0; i < tail; ++i) {
      result += broadTmpInUb.GetValue(tailOffset + i);
    }
    return result;
  }

  __aicore__ inline void CastFp32ToOutput(LocalTensor<outType> dstTensor, LocalTensor<float> srcTensor,
                                          uint32_t elementCount) {
    for (uint32_t offset = 0; offset < elementCount; offset += MAX_CAST_ELEMENTS) {
      uint32_t currentCount = Std::min(MAX_CAST_ELEMENTS, elementCount - offset);
      Cast(dstTensor[offset], srcTensor[offset], RoundMode::CAST_NONE, currentCount);
      PipeBarrier<PIPE_V>();
    }
  }

  __aicore__ inline void Compute(uint32_t curSingleV, uint64_t curQKOffset, uint64_t curVOffset) {
    uint32_t alignedV = Ceil(curSingleV, FP16_NUM_PER_BLOCK) * FP16_NUM_PER_BLOCK;
    if (hasGama_) {
      Muls(stateInUb, stateInUb, gama_, alignK_ * alignedV);
    }
    if (hasGamaK_) {
      if (gamaKScalar_) {
        Muls(stateInUb, stateInUb, gamaK_, alignK_ * alignedV);
      } else {
        for (uint32_t v = 0; v < alignedV; ++v) {
          Mul(stateInUb[v * alignK_], stateInUb[v * alignK_], gamaKInUb[curQKOffset], alignK_);
        }
      }
    }
    if (hasGama_ || hasGamaK_) {
      PipeBarrier<PIPE_V>();
    }
    for (uint32_t v = 0; v < curSingleV; ++v) {
      float memory = DotFp32(stateInUb[v * alignK_], kInUb[curQKOffset]);
      float delta = (vInUb.GetValue(curVOffset + v) - memory) * beta_;
      Muls(broadTmpInUb, kInUb[curQKOffset], delta, alignK_);
      PipeBarrier<PIPE_V>();
      Add(stateInUb[v * alignK_], stateInUb[v * alignK_], broadTmpInUb, alignK_);
      PipeBarrier<PIPE_V>();
    }
    for (uint32_t v = 0; v < curSingleV; ++v) {
      attnInUb.SetValue(v, DotFp32(stateInUb[v * alignK_], qInUb[curQKOffset]));
    }
    TQueSync<PIPE_S, PIPE_V> scalarToVectorSync;
    scalarToVectorSync.SetFlag(0);
    scalarToVectorSync.WaitFlag(0);
    LocalTensor<outType> attnOutLocal = attnOutQueue_.AllocTensor<outType>();
    CastFp32ToOutput(attnOutLocal, attnInUb, alignedV);
    attnOutQueue_.EnQue<outType>(attnOutLocal);
    LocalTensor<outType> stateOutLocal = stateOutQueue_.AllocTensor<outType>();
    CastFp32ToOutput(stateOutLocal, stateInUb, alignK_ * alignedV);
    stateOutQueue_.EnQue<outType>(stateOutLocal);
  }

  __aicore__ inline void CopyOutAttn(uint64_t attnOffset, uint32_t curSingleV) {
    LocalTensor<outType> attnLocal = attnOutQueue_.DeQue<outType>();
    DataCopyParams attnOutParams{1, static_cast<uint16_t>(curSingleV * sizeof(outType)), 0, 0};
    DataCopyCustom(attnOutGm_[attnOffset], attnLocal, attnOutParams);
    // The queue has a single output buffer on 310P.  Do not return it to
    // the Vector producer until the asynchronous MTE3 copy has finished.
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    attnOutQueue_.FreeTensor(attnLocal);
  }

  __aicore__ inline void CopyOutState(uint64_t stateOffset, uint32_t curSingleV) {
    LocalTensor<outType> stateOutLocal = stateOutQueue_.DeQue<outType>();
    for (uint32_t v = 0; v < curSingleV; ++v) {
      DataCopyParams stateOutParams{1, static_cast<uint16_t>(realK_ * sizeof(outType)), 0, 0};
      DataCopyCustom(finalStateGm_[stateOffset + static_cast<uint64_t>(v) * realK_],
                     stateOutLocal[static_cast<uint64_t>(v) * alignK_], stateOutParams);
    }
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    stateOutQueue_.FreeTensor(stateOutLocal);
  }

  __aicore__ inline void CopyInGamaBeta(int32_t seq0, int32_t seq1) {
    int32_t seqLen = seq1 - seq0;
    uint64_t bBatchSize = Ceil(seqLen * NV_, FP16_NUM_PER_BLOCK) * FP16_NUM_PER_BLOCK;
    LocalTensor<inType> betaLocal = betaInQueue_.AllocTensor<inType>();
    DataCopyParams betaInParams{1, static_cast<uint16_t>(seqLen * NV_ * sizeof(inType)), 0, 0};
    DataCopyCustom(betaLocal, betaGm_[seq0 * NV_], betaInParams);
    betaInQueue_.EnQue<inType>(betaLocal);
    betaLocal = betaInQueue_.DeQue<inType>();
    Cast(betaInUb, betaLocal, RoundMode::CAST_NONE, bBatchSize);
    SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
    betaInQueue_.FreeTensor(betaLocal);
    if (hasGama_) {
      DataCopyParams gamaInParams{1, static_cast<uint16_t>(seqLen * NV_ * sizeof(float)), 0, 0};
      DataCopyCustom(gamaInUb, gamaGm_[seq0 * NV_], gamaInParams);
      SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
      Exp(gamaInUb, gamaInUb, seqLen * NV_);
      PipeBarrier<PIPE_V>();
    }
    if (hasGamaK_ && gamaKScalar_) {
      DataCopyParams gamaKInParams{1, static_cast<uint16_t>(seqLen * NV_ * sizeof(float)), 0, 0};
      DataCopyCustom(gamaKScalarInUb, gamaKGm_[seq0 * NV_], gamaKInParams);
      PipeBarrier<PIPE_MTE2>();
      uint32_t gamaKAlignSize = Ceil(seqLen * NV_, FP32_NUM_PER_BLOCK) * FP32_NUM_PER_BLOCK;
      Exp(gamaKScalarInUb, gamaKScalarInUb, gamaKAlignSize);
      PipeBarrier<PIPE_V>();
    }
    // beta/gama/gamaK are read with LocalTensor::GetValue in
    // ProcessHead.  Explicitly synchronize Vector writes to Scalar reads.
    WaitVectorToScalar();
  }

  __aicore__ inline void ReleaseGamaKInput() {
    if (hasGamaK_ && !gamaKScalar_) {
      gamaKInQueue_.FreeTensor(gamaKInUb);
    }
  }

  __aicore__ inline void QueueAttnOutput(uint64_t attnOffset, uint32_t curSingleV, uint64_t &pendingOffset,
                                         bool &hasPending) {
    if (attnOutBufferNum_ == BUFFER_NUM) {
      CopyOutAttn(attnOffset, curSingleV);
      return;
    }
    if (hasPending) {
      CopyOutAttn(pendingOffset, curSingleV);
    }
    pendingOffset = attnOffset;
    hasPending = true;
  }

  __aicore__ inline void QueueStateOutput(uint64_t stateOffset, uint32_t curSingleV, uint64_t &pendingOffset,
                                          bool &hasPending) {
    if (stateOutBufferNum_ == BUFFER_NUM) {
      CopyOutState(stateOffset, curSingleV);
      return;
    }
    if (hasPending) {
      CopyOutState(pendingOffset, curSingleV);
    }
    pendingOffset = stateOffset;
    hasPending = true;
  }

  __aicore__ inline void ProcessSequenceChunk(int32_t seq0, int32_t seq1, uint64_t head_i, uint64_t v_i,
                                              uint32_t curSingleV) {
    uint64_t pendingAttnOffset = 0;
    uint64_t pendingStateOffset = 0;
    bool hasPendingAttn = false;
    bool hasPendingState = false;
    for (uint64_t seq_i = seq0; seq_i < seq1; seq_i++) {
      uint64_t gbOffset = head_i + (seq_i - seq0) * NV_;
      uint64_t curQKOffset = (seq_i - seq0) * alignK_;
      uint64_t curVOffset = (seq_i - seq0) * alignV_ + v_i;
      uint64_t attnOffset = (seq_i * NV_ + head_i) * realV_ + v_i;
      uint64_t curStateOutOffset =
        (static_cast<uint64_t>(GetSsmStateIndex(seq_i)) * NV_ + head_i) * static_cast<uint64_t>(realK_) * realV_ +
        v_i * static_cast<uint64_t>(realK_);
      gama_ = hasGama_ ? gamaInUb.GetValue(gbOffset) : 1;
      beta_ = betaInUb.GetValue(gbOffset);
      gamaK_ = (hasGamaK_ && gamaKScalar_) ? gamaKScalarInUb.GetValue(gbOffset) : 1;
      Compute(curSingleV, curQKOffset, curVOffset);
      QueueAttnOutput(attnOffset, curSingleV, pendingAttnOffset, hasPendingAttn);
      QueueStateOutput(curStateOutOffset, curSingleV, pendingStateOffset, hasPendingState);
    }
    if (hasPendingAttn) {
      CopyOutAttn(pendingAttnOffset, curSingleV);
    }
    if (hasPendingState) {
      CopyOutState(pendingStateOffset, curSingleV);
    }
  }

  __aicore__ inline void ProcessHead(int32_t seq0, int32_t seq1, uint64_t head_i, uint64_t stateOffset) {
    uint64_t vOffset = (seq0 * NV_ + head_i) * realV_;
    uint64_t qkOffset = (seq0 * NK_ + head_i / (NV_ / NK_)) * realK_;
    CopyInQKV(vOffset, qkOffset, seq1 - seq0);
    if (realV_ == 0) {
      ReleaseGamaKInput();
      return;
    }
    uint32_t nextSingleV = realV_ > vStep_ ? vStep_ : realV_;
    uint64_t nextStateOffset = (stateOffset * NV_ + head_i) * static_cast<uint64_t>(realV_) * realK_;
    PrefetchState(nextStateOffset, nextSingleV);
    for (uint64_t v_i = 0; v_i < realV_; v_i += vStep_) {
      uint32_t curSingleV = v_i + vStep_ > realV_ ? realV_ - v_i : vStep_;
      LoadPrefetchedState(curSingleV);
      uint64_t nextVOffset = v_i + vStep_;
      if (nextVOffset < realV_) {
        nextSingleV = nextVOffset + vStep_ > realV_ ? realV_ - nextVOffset : vStep_;
        nextStateOffset = (stateOffset * NV_ + head_i) * static_cast<uint64_t>(realV_) * realK_ +
                          nextVOffset * static_cast<uint64_t>(realK_);
        PrefetchState(nextStateOffset, nextSingleV);
      }
      ProcessSequenceChunk(seq0, seq1, head_i, v_i, curSingleV);
    }
    ReleaseGamaKInput();
  }

 private:
  GlobalTensor<inType> queryGm_;
  GlobalTensor<inType> keyGm_;
  GlobalTensor<inType> valueGm_;
  GlobalTensor<inType> betaGm_;
  GlobalTensor<float> gamaGm_;
  GlobalTensor<float> gamaKGm_;
  GlobalTensor<inType> initStateGm_;
  GlobalTensor<int32_t> cuSeqlens32Gm_;
  GlobalTensor<int64_t> cuSeqlens64Gm_;
  GlobalTensor<int32_t> ssmStateIndices32Gm_;
  GlobalTensor<int64_t> ssmStateIndices64Gm_;
  GlobalTensor<int32_t> numAcceptedTokensGm_;
  GlobalTensor<outType> finalStateGm_;
  GlobalTensor<outType> attnOutGm_;
  TPipe *pipe_;
  TQue<QuePosition::VECIN, 1> qInQueue_;
  TQue<QuePosition::VECIN, 1> kInQueue_;
  TQue<QuePosition::VECIN, 1> vInQueue_;
  TQue<QuePosition::VECIN, 1> gamaKInQueue_;
  TQue<QuePosition::VECIN, 1> betaInQueue_;
  TQue<QuePosition::VECIN, 1> stateInQueue_;
  TQue<QuePosition::VECOUT, MAX_OUT_BUFFER_NUM> attnOutQueue_;
  TQue<QuePosition::VECOUT, MAX_OUT_BUFFER_NUM> stateOutQueue_;
  TBuf<TPosition::VECCALC> tmpBuff;
  LocalTensor<float> qInUb;
  LocalTensor<float> kInUb;
  LocalTensor<float> vInUb;
  LocalTensor<float> gamaInUb;
  LocalTensor<float> gamaKInUb;
  LocalTensor<float> betaInUb;
  LocalTensor<float> broadTmpInUb;
  LocalTensor<float> attnInUb;
  LocalTensor<float> stateInUb;
  LocalTensor<float> gamaKScalarInUb;
  uint32_t B_;
  uint32_t T_;
  uint32_t NK_;
  uint32_t alignK_;
  uint32_t realK_;
  uint32_t NV_;
  uint32_t alignV_;
  uint32_t realV_;
  uint32_t vStep_;
  uint32_t stateOutBufferNum_;
  uint32_t attnOutBufferNum_;
  uint32_t restUbSize_;
  bool hasAcceptedTokens_;
  bool hasGama_;
  bool hasGamaK_;
  bool gamaKScalar_;
  bool cuSeqlensIsPrefix_;
  bool cuSeqlensIsInt64_;
  bool ssmStateIndicesIsInt64_;
  float gama_;
  float gamaK_;
  float beta_;
  float scale_;
  uint64_t blockIdx;
  uint64_t workBlockDim_;
};

#endif  // RECURRENT_GATED_DELTA_RULE_KERNEL_H_
