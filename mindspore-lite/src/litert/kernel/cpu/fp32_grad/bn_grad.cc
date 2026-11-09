/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/cpu/fp32_grad/bn_grad.h"
#include <algorithm>
#include <vector>
#include <string>
#include <thread>
#include <fstream>

#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "nnacl_c/fp32_grad/batch_norm_grad.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_BatchNormGrad;

namespace mindspore::kernel {
namespace {
constexpr int kMaxTaskNum = 3;
constexpr int kNumInputDim2 = 2;
constexpr int kNumInputDim4 = 4;
// BNGrad has 6 inputs; the FusedBatchNormGrad variant shifts mean/var to 5/6.
constexpr int kNumBNGradInputs = 6;
constexpr int kNumBNGradOutputs = 3;
constexpr int kBNGradYtIdx = 0;
constexpr int kBNGradXIdx = 1;
constexpr int kBNGradScaleIdx = 2;
constexpr int kBNGradMeanIdx = 3;
constexpr int kBNGradVarIdx = 4;
constexpr int kBNGradReserveIdx = 5;
constexpr int kFusedBnGradMeanIdx = 4;
constexpr int kFusedBnGradVarIdx = 5;
constexpr int kBNGradDxIdx = 0;
constexpr int kBNGradDscaleIdx = 1;
constexpr int kBNGradDbiasIdx = 2;
// Run() stage machine: init (var2Invar + zero) -> param-grad partials -> reduce -> dx.
constexpr int kStageInit = 0;
constexpr int kStageParamGrad = 1;
constexpr int kStageDxGrad = 2;
// Per-thread workspace partial buffers: dbias + dscale.
constexpr int kNumPartialBufs = 2;
}  // namespace

int BNGradCPUKernel::ReSize() {
  CHECK_NULL_RETURN(op_parameter_);
  CHECK_LESS_RETURN(in_tensors_.size(), kNumBNGradInputs);
  auto *input_x = in_tensors_.at(kBNGradXIdx);
  CHECK_NULL_RETURN(input_x);
  int32_t batch = input_x->Batch();
  int32_t channels = input_x->Channel();
  int32_t spatial = input_x->Height() * input_x->Width();
  int total = 0;
  if (input_x->shape().size() == kNumInputDim4) {
    total = spatial * batch;
  } else if (input_x->shape().size() == kNumInputDim2) {
    total = batch;
    auto *input_scale = in_tensors_.at(kBNGradScaleIdx);
    CHECK_NULL_RETURN(input_scale);
    channels = static_cast<int>(input_scale->ElementsNum());
  } else {
    MS_LOG(ERROR) << "Unsupported tensor shape: " << input_x->shape().size();
    return RET_ERROR;
  }
  channels_ = channels;
  total_ = total;
  thread_num_ = MSMAX(1, op_parameter_->thread_num_);
  // Per-thread private dbias + dscale partials (each `channels` floats), reduced once.
  // Only needed for multi-thread; single-thread keeps the original backwardAll path.
  use_partial_ = (thread_num_ > 1);
  if (use_partial_) {
    set_workspace_size(static_cast<size_t>(channels_) * static_cast<size_t>(thread_num_) * kNumPartialBufs *
                       sizeof(float));
  } else {
    set_workspace_size(0);
  }
  return RET_OK;
}

int BNGradCPUKernel::Prepare() {
  CHECK_NULL_RETURN(op_parameter_);
  CHECK_LESS_RETURN(in_tensors_.size(), kNumBNGradInputs);
  CHECK_LESS_RETURN(out_tensors_.size(), kNumBNGradOutputs);
  CHECK_NULL_RETURN(in_tensors_[kBNGradYtIdx]);
  CHECK_NULL_RETURN(in_tensors_[kBNGradXIdx]);
  CHECK_NULL_RETURN(in_tensors_[kBNGradScaleIdx]);
  CHECK_NULL_RETURN(in_tensors_[kBNGradMeanIdx]);
  CHECK_NULL_RETURN(in_tensors_[kBNGradVarIdx]);
  CHECK_NULL_RETURN(in_tensors_[kBNGradReserveIdx]);
  CHECK_NULL_RETURN(out_tensors_[kBNGradDxIdx]);
  CHECK_NULL_RETURN(out_tensors_[kBNGradDscaleIdx]);
  CHECK_NULL_RETURN(out_tensors_[kBNGradDbiasIdx]);
  return ReSize();
}

void BNGradCPUKernel::DoParamGradPartial(int task_id, const float *x, const float *yt, const float *save_mean) {
  // Split the sample range across threads into private partials (lock-free, reads the
  // input once), reduced later by DoReduce. dbias/dscale (shared) stay zeroed from
  // stage 0 until the reduction writes them.
  auto *ws = reinterpret_cast<float *>(workspace());
  if (ws == nullptr) {
    MS_LOG(ERROR) << "BNGrad workspace is null.";
    return;
  }
  float *dbias_priv_base = ws;
  float *dscale_priv_base = ws + static_cast<size_t>(channels_) * static_cast<size_t>(thread_num_);
  float *dbias_priv = dbias_priv_base + static_cast<size_t>(task_id) * static_cast<size_t>(channels_);
  float *dscale_priv = dscale_priv_base + static_cast<size_t>(task_id) * static_cast<size_t>(channels_);
  std::fill(dbias_priv, dbias_priv + channels_, 0.f);
  std::fill(dscale_priv, dscale_priv + channels_, 0.f);
  int stride = UP_DIV(total_, thread_num_);
  int start = stride * task_id;
  int count = MSMIN(stride, total_ - start);
  if (count > 0) {
    backwardP1Partial(x, yt, save_mean, start, count, channels_, dbias_priv, dscale_priv);
  }
}

int BNGradCPUKernel::DoExecute(int task_id) {
  auto *input_yt = in_tensors_.at(kBNGradYtIdx);
  auto *input_x = in_tensors_.at(kBNGradXIdx);
  auto *input_scale = in_tensors_.at(kBNGradScaleIdx);
  auto *input_mean = in_tensors_.at(kBNGradMeanIdx);
  auto *input_var = in_tensors_.at(kBNGradVarIdx);

  auto kernel_name = this->name();
  if (kernel_name.find("FusedBatchNormGradCPU") != std::string::npos) {
    input_mean = in_tensors_.at(kFusedBnGradMeanIdx);
    input_var = in_tensors_.at(kFusedBnGradVarIdx);
  }
  auto bn_param = reinterpret_cast<BNGradParameter *>(op_parameter_);
  CHECK_NULL_RETURN(bn_param);
  int stage = stage_;
  int thread_num = thread_num_;
  float *save_mean = reinterpret_cast<float *>(input_mean->MutableData());
  CHECK_NULL_RETURN(save_mean);
  float *save_var = reinterpret_cast<float *>(input_var->MutableData());
  CHECK_NULL_RETURN(save_var);

  auto *output_dx = out_tensors_.at(kBNGradDxIdx);
  auto *output_scale = out_tensors_.at(kBNGradDscaleIdx);
  auto *output_bias = out_tensors_.at(kBNGradDbiasIdx);
  int32_t batch = input_x->Batch();
  int32_t channels = input_x->Channel();
  int32_t spatial = input_x->Height() * input_x->Width();
  float *x = reinterpret_cast<float *>(input_x->MutableData());
  CHECK_NULL_RETURN(x);
  float *yt = reinterpret_cast<float *>(input_yt->MutableData());
  CHECK_NULL_RETURN(yt);
  float *scale = reinterpret_cast<float *>(input_scale->MutableData());
  CHECK_NULL_RETURN(scale);
  float *dx = reinterpret_cast<float *>(output_dx->MutableData());
  CHECK_NULL_RETURN(dx);
  float *dbias = reinterpret_cast<float *>(output_bias->MutableData());
  CHECK_NULL_RETURN(dbias);
  float *dscale = reinterpret_cast<float *>(output_scale->MutableData());
  int total = 0;
  if (in_tensors().at(kBNGradXIdx)->shape().size() == kNumInputDim4) {
    total = spatial * batch;
  } else if (in_tensors().at(kBNGradXIdx)->shape().size() == kNumInputDim2) {
    total = batch;
    channels = static_cast<int>(input_scale->ElementsNum());
  } else {
    MS_LOG(ERROR) << "Unsupported tensor shape: " << in_tensors().at(kBNGradXIdx)->shape().size();
    return RET_ERROR;
  }
  int stride = UP_DIV(total, thread_num);
  int count = MSMIN(stride, total - stride * task_id);
  count = (count < 0) ? 0 : count;
  switch (stage) {
    case kStageInit: {
      for (int job = task_id; job < kMaxTaskNum; job += thread_num) {
        switch (job) {
          case 0:
            var2Invar(save_var, input_var->ElementsNum(), bn_param->epsilon_);
            break;
          case 1:
            std::fill(dbias, dbias + channels, 0.f);
            break;
          case 2:
            std::fill(dscale, dscale + channels, 0.f);
            break;
          default:
            MS_LOG(ERROR) << "Exceeds the maximum thread";
            return RET_ERROR;
        }
      }
      if (thread_num == 1) {
        backwardAll(x, yt, save_mean, save_var, scale, total, channels, dbias, dscale, dx,
                    (IsTrain() && bn_param->is_training_));
      }
      break;
    }
    case kStageParamGrad: {
      if (use_partial_) {
        DoParamGradPartial(task_id, x, yt, save_mean);
      } else {
        backwardP1(x, yt, save_mean, save_var, scale, total, channels, dbias, dscale);
      }
      break;
    }
    case kStageDxGrad: {
      backwardP2(x + task_id * stride * channels, yt + task_id * stride * channels, save_mean, save_var, dscale, dbias,
                 scale, count, total, channels, dx + task_id * stride * channels,
                 (IsTrain() && bn_param->is_training_));
      break;
    }
    default:
      MS_LOG(ERROR) << "Unsupported stage";
      return RET_ERROR;
  }

  return RET_OK;
}

int BNGradRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  CHECK_NULL_RETURN(cdata);
  auto bn_kernel = reinterpret_cast<BNGradCPUKernel *>(cdata);
  auto error_code = bn_kernel->DoExecute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "BNGradRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int BNGradReduceRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  CHECK_NULL_RETURN(cdata);
  auto bn_kernel = reinterpret_cast<BNGradCPUKernel *>(cdata);
  auto error_code = bn_kernel->DoReduce(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "BNGradReduceRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int BNGradCPUKernel::DoReduce(int task_id) {
  int thread_num = thread_num_;
  int channels = channels_;
  // Split the per-channel reduction across threads (disjoint write ranges -> no lock).
  int stride = UP_DIV(channels, thread_num);
  int start = stride * task_id;
  int count = MSMIN(stride, channels - start);
  if (count <= 0) {
    return RET_OK;
  }
  auto *input_var = in_tensors_.at(kBNGradVarIdx);
  auto kernel_name = this->name();
  if (kernel_name.find("FusedBatchNormGradCPU") != std::string::npos) {
    input_var = in_tensors_.at(kFusedBnGradVarIdx);
  }
  float *save_var = reinterpret_cast<float *>(input_var->MutableData());
  CHECK_NULL_RETURN(save_var);
  float *dbias = reinterpret_cast<float *>(out_tensors_.at(kBNGradDbiasIdx)->MutableData());
  CHECK_NULL_RETURN(dbias);
  float *dscale = reinterpret_cast<float *>(out_tensors_.at(kBNGradDscaleIdx)->MutableData());
  CHECK_NULL_RETURN(dscale);
  auto *ws = reinterpret_cast<float *>(workspace());
  CHECK_NULL_RETURN(ws);
  float *dbias_priv_base = ws;
  float *dscale_priv_base = ws + static_cast<size_t>(channels) * thread_num;
  // dbias/dscale were zeroed in stage 0; sum each thread's partial into them,
  // then apply the trailing dscale *= invar that backwardP1 used to do inline.
  for (int c = start; c < start + count; c++) {
    float b = 0.f;
    float s = 0.f;
    for (int t = 0; t < thread_num; t++) {
      b += dbias_priv_base[t * channels + c];
      s += dscale_priv_base[t * channels + c];
    }
    dbias[c] = b;
    dscale[c] = s * save_var[c];  // save_var holds 1/sqrt(var+eps) after var2Invar
  }
  return RET_OK;
}

int BNGradCPUKernel::Run() {
  stage_ = kStageInit;
  const int thread_num = MSMAX(1, op_parameter_->thread_num_);
  // ReSize sized the workspace for the thread count seen at Prepare time. If it grew since
  // then, the per-thread partial buffers would overrun their allocation.
  const size_t needed =
    static_cast<size_t>(channels_) * static_cast<size_t>(thread_num) * kNumPartialBufs * sizeof(float);
  if (thread_num > 1 && workspace_size() < needed) {
    MS_LOG(ERROR) << "BNGrad workspace " << workspace_size() << " smaller than " << needed << " for " << thread_num
                  << " threads.";
    return RET_ERROR;
  }
  thread_num_ = thread_num;
  use_partial_ = (thread_num > 1);
  int error_code;
  if (thread_num_ == 1) {
    // Single-thread: original backwardAll path (bit-identical, no workspace).
    error_code = ParallelLaunch(this->ms_context_, BNGradRun, this, thread_num_);
    if (error_code != RET_OK) {
      MS_LOG(ERROR) << "BN function error error_code[" << error_code << "]";
      return RET_ERROR;
    }
    return RET_OK;
  }
  // Multi-thread: init -> param-grad partials -> reduce -> dx.
  error_code = ParallelLaunch(this->ms_context_, BNGradRun, this, thread_num_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "BN function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  stage_ = kStageParamGrad;
  error_code = ParallelLaunch(this->ms_context_, BNGradRun, this, thread_num_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "BN function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  error_code = ParallelLaunch(this->ms_context_, BNGradReduceRun, this, thread_num_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "BN reduce function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  stage_ = kStageDxGrad;
  error_code = ParallelLaunch(this->ms_context_, BNGradRun, this, thread_num_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "BN function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuBNGradFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                               const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                               const lite::InnerContext *ctx, const kernel::KernelKey &desc) {
  MS_ASSERT(desc.type == schema::PrimitiveType_BatchNormGrad);
  auto *kernel = new (std::nothrow) BNGradCPUKernel(opParameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new BNGradCPUKernel fail!";
    free(opParameter);
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_BatchNormGrad, CpuBNGradFp32KernelCreator)
}  // namespace mindspore::kernel
