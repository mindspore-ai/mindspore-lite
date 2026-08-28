/** * copy from https://gitcode.com/cann/ops-transformer/tree/master/attention/chunk_gated_delta_rule
 *
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

/*!
 * \file chunk_gated_delta_rule_stage2.h
 * \brief
 */
#ifndef CHUNK_GATED_DELTA_RULE_STAGE2_H
#define CHUNK_GATED_DELTA_RULE_STAGE2_H

#include "kernel_tiling/kernel_tiling.h"
#include "chunk_gated_delta_rule_utils.h"
#include "../chunk_gated_delta_rule_tiling_data.h"

namespace ChunkGatedDeltaRule {
using namespace AscendC;

template <typename lowType>
using aT2 = MatmulType<TPosition::GM, CubeFormat::ND, lowType, true>;
template <typename lowType>
using bT2 = MatmulType<TPosition::GM, CubeFormat::ND, lowType, true>;
template <typename lowType>
using cT2 = MatmulType<TPosition::GM, CubeFormat::ND, lowType>;
template <typename lowType>
using StageTwoMT = matmul::MatmulImpl<aT2<lowType>, bT2<lowType>, cT2<lowType>>;

template <typename lowType>
struct StageTwoParams {
  GlobalTensor<lowType> qPrime;      // (Nv, Sp, Dk)
  GlobalTensor<lowType> vInner;      // (Nv, Sp, Dv)
  GlobalTensor<float> gCum;          // (Nv, Sp)
  GlobalTensor<lowType> kCumdecay;   // (Nv, Sp, Dk)
  GlobalTensor<lowType> curState;    // (Nv, Dv, Dk)
  GlobalTensor<lowType> finalState;  // (Nv, Dv, Dk)
  GlobalTensor<lowType> kg;
  GlobalTensor<lowType> out;
  GM_ADDR ws;
  StageTwoMT<lowType> *mm1;
  TPipe *pipe;
  ChunkGroup *cg;
  int64_t Nv;
  int64_t Nk;
  int64_t Dv;
  int64_t Dk;
  bool gOptional;
};

template <typename lowType>
class Stage2 {
 public:
  __aicore__ inline void Init(StageTwoParams<lowType> *initParams, int32_t coreNum) {
    sTP_ = initParams;
    pipe_ = sTP_->pipe;
    chunkSize_ = sTP_->cg->chunkSize;
    seqLength_ = sTP_->cg->length;
    Sp_ = (seqLength_ + chunkSize_ - 1) / chunkSize_ * chunkSize_;
    chunkNum_ = Sp_ / chunkSize_;
    coreNum_ = coreNum;
    Nv_ = sTP_->Nv;
    Nk_ = sTP_->Nk;
    Dv_ = sTP_->Dv;
    Dk_ = sTP_->Dk;
    curDk_ = Ceil(Dk_, BLOCK_SIZE / sizeof(lowType)) * (BLOCK_SIZE / sizeof(lowType));
    curChunkSize_ = chunkSize_;
    gOptional_ = sTP_->gOptional;
    InitLocalBuffers();
  }

  __aicore__ inline void InitLocalBuffers() {
    if ASCEND_IS_AIC {
      return;
    }
    pipe_->InitBuffer(inQueue_, BUFFER_NUM_ONE,
                      Std::max((uint64_t)chunkSize_, (uint64_t)Dv_) * curDk_ * sizeof(lowType));
    uint64_t outQueueSize =
      Std::max((uint64_t)chunkSize_ * chunkSize_ * sizeof(lowType), (uint64_t)Dv_ * curDk_ * sizeof(lowType));
    pipe_->InitBuffer(outQueue_, BUFFER_NUM_ONE, outQueueSize);
    pipe_->InitBuffer(tmpBuff_,
                      (Std::max((uint64_t)chunkSize_, (uint64_t)Dv_) * curDk_ + BLOCK_FLOAT_NUM) * sizeof(float));
    uint32_t buffOffset = 0;
    tmpBuffer1_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(Dv_ * curDk_), buffOffset);
    buffOffset += Ceil(Dv_ * curDk_ * sizeof(float), BLOCK_SIZE) * BLOCK_SIZE;
    // buffer for caching the last fetched element
    lastGCum_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(NUM_ONE), buffOffset);
  }

  __aicore__ inline void Process() {
    int64_t coreId = GetBlockIdx();
    if ASCEND_IS_AIV {
      coreId /= AIC_AIV_1_1;
    }
    int64_t nvPerCore = (Nv_ + coreNum_ - 1) / coreNum_;
    int64_t nvStart = coreId * nvPerCore;
    int64_t nvEnd = nvStart + nvPerCore;
    nvEnd = nvEnd > Nv_ ? Nv_ : nvEnd;
    int64_t lastChunkSize = seqLength_ % chunkSize_ == 0 ? chunkSize_ : seqLength_ % chunkSize_;
    for (int64_t nvId = nvStart; nvId < nvEnd; nvId++) {
      curChunkSize_ = chunkSize_;
      for (int64_t cId = 0; cId < chunkNum_; cId++) {
        auto curState = (cId == 0) ? sTP_->curState[nvId * Dv_ * Dk_] : sTP_->finalState[nvId * Dv_ * Dk_];
        auto finalState = sTP_->finalState[nvId * Dv_ * Dk_];
        int64_t length = cId * chunkSize_;
        if (cId == chunkNum_ - 1) {
          curChunkSize_ = lastChunkSize;
        }
        if ASCEND_IS_AIV {
          if (GetSubBlockIdx() == 0) {
            CopyIn(curState, Dv_, Dk_);
            CalGCumExp(sTP_->gCum[nvId * Sp_ + length]);
          }
          CrossCoreWaitFlag(0x2);
          if (GetSubBlockIdx() == 0) {
            CopyOutState(finalState);
          }
          CrossCoreSetFlag<0x2, PIPE_MTE3>(0x5);
          CrossCoreWaitFlag(0x4);
        }
        if ASCEND_IS_AIC {
          uint64_t mm_offset0 = nvId * Sp_ * Dk_ + length * Dk_;
          uint64_t mm_offset1 = nvId * Sp_ * Dv_ + length * Dv_;
          CalVPrime(sTP_->kCumdecay[mm_offset0], curState, sTP_->vInner[mm_offset1]);
          CalAttnInter(sTP_->qPrime[mm_offset0], curState, sTP_->out[nvId * Dv_ + cId * chunkSize_ * Nv_ * Dv_]);
          CrossCoreSetFlag<0x2, PIPE_FIX>(0x2);  // AIV must not write until AIC finishes reading
          CrossCoreWaitFlag(0x5);
          CalStateNew(sTP_->vInner[mm_offset1], sTP_->kg[mm_offset0], finalState);
          SetFlag<HardEvent::FIX_MTE2>(FIX_MTE2_EVENT);
          WaitFlag<HardEvent::FIX_MTE2>(FIX_MTE2_EVENT);
          CrossCoreSetFlag<0x2, PIPE_FIX>(0x4);
        }
      }
    }
  }

  __aicore__ inline void CalGCumExp(GlobalTensor<float> gCum) {
    if (gOptional_) {
      // refresh cache
      DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(gCum[curChunkSize_ - 1]);
      float tmpFloat = gCum.GetValue(curChunkSize_ - 1);
      lastGCum_.SetValue(0, tmpFloat);
      SetFlag<HardEvent::S_V>(S_V_EVENT);
      WaitFlag<HardEvent::S_V>(S_V_EVENT);
      Exp<float, 0, true>(lastGCum_, lastGCum_, 1);
    } else {
      lastGCum_.SetValue(0, 1.0f);
    }
    float tmpFloat = lastGCum_.GetValue(0);
    auto stateIn = inQueue_.DeQue<lowType>();
    auto stateOut = outQueue_.AllocTensor<lowType>();
    SetFlag<HardEvent::MTE2_V>(MTE2_V_EVENT);
    WaitFlag<HardEvent::MTE2_V>(MTE2_V_EVENT);
    Cast(tmpBuffer1_, stateIn, RoundMode::CAST_NONE, Dv_ * curDk_);
    SetFlag<HardEvent::S_V>(S_V_EVENT);
    WaitFlag<HardEvent::S_V>(S_V_EVENT);
    Muls(tmpBuffer1_, tmpBuffer1_, tmpFloat, Dv_ * curDk_);
    Cast(stateOut, tmpBuffer1_, RoundMode::CAST_RINT, Dv_ * curDk_);
    SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
    WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
    outQueue_.EnQue(stateOut);
    inQueue_.FreeTensor(stateIn);
  }

  __aicore__ inline void CalAttnInter(GlobalTensor<lowType> qPrime, GlobalTensor<lowType> state,
                                      GlobalTensor<lowType> out) {
    // q_prime @ state.transpose(0, 1)
    sTP_->mm1->SetOrgShape(curChunkSize_, Dv_, Dk_, Dk_, Nv_ * Dv_);  // MNKaKbN
    sTP_->mm1->SetSingleShape(curChunkSize_, Dv_, Dk_);               // SingleCoreMNK
    sTP_->mm1->SetTensorA(qPrime, false);
    sTP_->mm1->SetTensorB(state, true);
    sTP_->mm1->IterateAll(out);
    sTP_->mm1->End();
  }

  __aicore__ inline void CalVPrime(GlobalTensor<lowType> kCumdecay, GlobalTensor<lowType> state,
                                   GlobalTensor<lowType> vPrime) {
    // v_inner += k_cumdecay @ state.transpose(0, 1)
    sTP_->mm1->SetOrgShape(curChunkSize_, Dv_, Dk_);     // MNK
    sTP_->mm1->SetSingleShape(curChunkSize_, Dv_, Dk_);  // SingleCoreMNK
    sTP_->mm1->SetTensorA(kCumdecay, false);
    sTP_->mm1->SetTensorB(state, true);
    sTP_->mm1->IterateAll(vPrime, 1);
    sTP_->mm1->End();
  }

  __aicore__ inline void CalStateNew(GlobalTensor<lowType> vInner, GlobalTensor<lowType> kg,
                                     GlobalTensor<lowType> state) {
    // state_out = v_new.transpose(0, 1) @ kg
    sTP_->mm1->SetOrgShape(Dv_, Dk_, curChunkSize_);     // MNK
    sTP_->mm1->SetSingleShape(Dv_, Dk_, curChunkSize_);  // SingleCoreMNK
    sTP_->mm1->SetTensorA(vInner, true);
    sTP_->mm1->SetTensorB(kg, false);
    sTP_->mm1->IterateAll(state, 1);
    sTP_->mm1->End();
  }

  template <typename inType>
  __aicore__ inline void CopyIn(GlobalTensor<inType> tmpGM, int32_t row, int32_t col) {
    LocalTensor<inType> inLocal = inQueue_.AllocTensor<inType>();
    DataCopyExtParams inParams{static_cast<uint16_t>(row),
                               static_cast<uint32_t>(col * sizeof(inType)),  // unaligned case requires zero padding
                               static_cast<uint32_t>(0), 0, 0};
    int padding = Ceil(col, BLOCK_SIZE / sizeof(inType)) * (BLOCK_SIZE / sizeof(inType)) - col;
    DataCopyPadExtParams<inType> copyPadParams{true, 0, static_cast<uint8_t>(padding), 0};
    DataCopyPad(inLocal, tmpGM, inParams, copyPadParams);
    inQueue_.EnQue(inLocal);
  }

  __aicore__ inline void CopyOutState(GlobalTensor<lowType> stateNew) { CopyOut<lowType>(stateNew, Dv_, Dk_, false); }

  template <typename outType>
  __aicore__ inline void CopyOut(GlobalTensor<outType> tmpGM, int32_t row, int32_t col, bool setAtomic = false) {
    auto outLocal = outQueue_.DeQue<outType>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(row);
    copyParams.blockLen = static_cast<uint32_t>(col * sizeof(outType));
    copyParams.srcStride = static_cast<uint32_t>(0);
    copyParams.dstStride = static_cast<uint32_t>(0);
    if (setAtomic) {
      SetAtomicAdd<outType>();
    }
    DataCopyPad(tmpGM, outLocal, copyParams);
    if (setAtomic) {
      SetAtomicNone();
    }
    outQueue_.FreeTensor(outLocal);
  }

 private:
  StageTwoParams<lowType> *sTP_;
  TPipe *pipe_;
  TQue<QuePosition::VECIN, BUFFER_NUM_ONE> inQueue_;
  TQue<QuePosition::VECOUT, BUFFER_NUM_ONE> outQueue_;
  TBuf<TPosition::VECCALC> tmpBuff_;
  LocalTensor<float> tmpBuffer1_;
  LocalTensor<float> lastGCum_;
  int64_t Nk_;
  int64_t Nv_;
  int64_t Dk_;
  int64_t Dv_;
  int64_t seqLength_;
  int32_t chunkSize_;
  int32_t curChunkSize_;
  int32_t curDk_;
  int64_t Sp_;
  int32_t chunkNum_;
  int32_t coreNum_;
  bool gOptional_;
};
}  // namespace ChunkGatedDeltaRule
#endif  // CHUNK_GATED_DELTA_RULE_STAGE2_H
