/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_COMMON_OPS_OPS_FUNC_DECLARE_H_
#define MINDSPORE_LITE_SRC_COMMON_OPS_OPS_FUNC_DECLARE_H_

#ifdef PRIMITIVE_WRITEABLE
#include <memory>
#include "schema/inner/model_generated.h"
#include "mindspore/ops/op_def/auto_generate/gen_lite_ops.h"
#include "infer/adam.h"
#include "infer/adder.h"
#include "infer/all.h"
#include "infer/apply_momentum.h"
#include "infer/assert.h"
#include "infer/attention.h"
#include "infer/audio_spectrogram.h"
#include "infer/batch_to_space.h"
#include "infer/batch_to_space_nd.h"
#include "infer/broadcast.h"
#include "infer/clip.h"
#include "infer/custom.h"
#include "infer/custom_normalize.h"
#include "infer/custom_predict.h"
#include "infer/custom_extract_features.h"
#include "infer/constant_of_shape.h"
#include "infer/crop.h"
#include "infer/depth_to_space.h"
#include "infer/depend.h"
#include "infer/detection_post_process.h"
#include "infer/eltwise.h"
#include "infer/embedding_lookup.h"
#include "infer/fake_quant_with_min_max_vars.h"
#include "infer/fake_quant_with_min_max_vars_per_channel.h"
#include "infer/fft_imag.h"
#include "infer/fft_real.h"
#include "infer/fill.h"
#include "infer/fill_v2.h"
#include "infer/fused_batch_norm.h"
#include "infer/ops_func_impl/gather.h"
#include "infer/hashtable_lookup.h"
#include "infer/instance_norm.h"
#include "infer/l2_normalize.h"
#include "infer/leaky_relu.h"
#include "infer/lp_normalization.h"
#include "infer/lrn.h"
#include "infer/lsh_projection.h"
#include "infer/lstm.h"
#include "infer/cxx_api/mat_mul_fusion.h"
#include "infer/max_pool.h"
#include "infer/switch_layer.h"
#include "infer/mfcc.h"
#include "infer/mod.h"
#include "infer/non_max_suppression.h"
#include "infer/pad.h"
#include "infer/prior_box.h"
#include "infer/proposal.h"
#include "infer/quant_dtype_cast.h"
#include "infer/ragged_range.h"
#include "infer/reduce.h"
#include "infer/resize.h"
#include "infer/reverse_sequence.h"
#include "infer/rfft.h"
#include "infer/roi_pooling.h"
#include "infer/scale.h"
#include "infer/scatter_nd_update.h"
#include "infer/ops_func_impl/select.h"
#include "infer/sgd.h"
#include "infer/sigmoid_cross_entropy_with_logits.h"
#include "infer/skip_gram.h"
#include "infer/softmax_cross_entropy_with_logits.h"
#include "infer/space_to_batch.h"
#include "infer/space_to_batch_nd.h"
#include "infer/space_to_depth.h"
#include "infer/sparse_softmax_cross_entropy_with_logits.h"
#include "infer/sparse_to_dense.h"
#include "infer/sparse_fill_empty_rows.h"
#include "infer/sparse_reshape.h"
#include "infer/sparse_segment_sum.h"
#include "infer/squared_difference.h"
#include "infer/stack.h"
#include "infer/switch.h"
#include "infer/ops_func_impl/tan.h"
#include "infer/tensor_list_from_tensor.h"
#include "infer/tensor_list_get_item.h"
#include "infer/tensor_list_reserve.h"
#include "infer/tensor_list_set_item.h"
#include "infer/tensor_list_stack.h"
#include "infer/unique.h"
#include "infer/unstack.h"
#include "infer/unsqueeze.h"
#include "infer/where.h"
#include "infer/grad/activation_grad.h"
#include "infer/grad/add_grad.h"
#include "infer/grad/de_conv2d_grad_filter.h"
#include "infer/grad/div_grad.h"
#include "infer/grad/dropout_grad.h"
#include "infer/grad/flatten_grad.h"
#include "infer/grad/group_conv2d_grad_input.h"
#include "infer/grad/log_grad.h"
#include "infer/grad/lstm_grad.h"
#include "infer/grad/lstm_grad_data.h"
#include "infer/grad/lstm_grad_weight.h"
#include "infer/grad/max_pool_grad.h"
#include "infer/grad/mul_grad.h"
#include "infer/grad/neg_grad.h"
#include "infer/grad/pooling_grad.h"
#include "infer/grad/power_grad.h"
#include "infer/grad/resize_grad.h"
#include "infer/grad/sigmoid_cross_entropy_with_logits_grad.h"
#include "infer/grad/sub_grad.h"
#include "infer/cxx_api/activation.h"
#include "infer/cxx_api/add_fusion.h"
#include "infer/cxx_api/adder_fusion.h"
#include "infer/cxx_api/arg_max_fusion.h"
#include "infer/cxx_api/arg_min_fusion.h"
#include "infer/cxx_api/avg_pool_fusion.h"
#include "infer/cxx_api/conv2d_backprop_filter_fusion.h"
#include "infer/cxx_api/conv2d_backprop_input_fusion.h"
#include "infer/cxx_api/conv2d_fusion.h"
#include "infer/cxx_api/conv2d_transpose_fusion.h"
#include "infer/cxx_api/div_fusion.h"
#include "infer/cxx_api/embedding_lookup_fusion.h"
#include "infer/cxx_api/exp_fusion.h"
#include "infer/cxx_api/full_connection.h"
#include "infer/cxx_api/l2_normalize_fusion.h"
#include "infer/cxx_api/layer_norm_fusion.h"
#include "infer/cxx_api/max_pool_fusion.h"
#include "infer/cxx_api/mul_fusion.h"
#include "infer/cxx_api/pad_fusion.h"
#include "infer/cxx_api/partial_fusion.h"
#include "infer/cxx_api/pow_fusion.h"
#include "infer/cxx_api/prelu_fusion.h"
#include "infer/cxx_api/reduce_fusion.h"
#include "infer/cxx_api/scale_fusion.h"
#include "infer/cxx_api/slice_fusion.h"
#include "infer/cxx_api/sub_fusion.h"
#include "infer/cxx_api/tile_fusion.h"
#include "infer/cxx_api/topk_fusion.h"
#include "infer/cxx_api/groupnorm_fusion.h"
#include "infer/gru.h"
#include "infer/invert_permutation.h"
#include "infer/size.h"
#include "infer/random_standard_normal.h"
#include "infer/crop_and_resize.h"
#include "infer/grad/strided_slice_grad.h"
#include "infer/uniform_real.h"
#include "infer/ops_func_impl/abs_grad.h"
#include "infer/splice.h"
#include "infer/call.h"
#include "infer/split_with_overlap.h"
#include "infer/tensor_array.h"
#include "infer/tensor_array_read.h"
#include "infer/tensor_array_write.h"
#include "infer/affine.h"
#include "infer/all_gather.h"
#include "infer/reduce_scatter.h"
#include "infer/dynamic_quant.h"
#include "infer/random_normal.h"
#include "infer/format_transpose.h"
#include "infer/tensor_scatter_add.h"
#include "infer/decoder_layer.h"
#include "infer/encoder_layer.h"
#include "infer/scatter_elements.h"
#include "infer/ops_func_impl/triu.h"
#include "infer/tril.h"

namespace mindspore::lite::ops {
#define FUNC_MSOP2SCHEMAOP_DECLARE(OP) std::unique_ptr<schema::PrimitiveT> MSOp2SchemaOp(const mindspore::ops::OP *op);

#ifdef PRIMITIVE_WRITEABLE
FUNC_MSOP2SCHEMAOP_DECLARE(Abs)
FUNC_MSOP2SCHEMAOP_DECLARE(Activation)
FUNC_MSOP2SCHEMAOP_DECLARE(ActivationGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(Adam)
FUNC_MSOP2SCHEMAOP_DECLARE(AddFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(AdderFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(AddGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(AddN)
FUNC_MSOP2SCHEMAOP_DECLARE(All)
FUNC_MSOP2SCHEMAOP_DECLARE(ApplyMomentum)
FUNC_MSOP2SCHEMAOP_DECLARE(Argmax)
FUNC_MSOP2SCHEMAOP_DECLARE(ArgMaxFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(Argmin)
FUNC_MSOP2SCHEMAOP_DECLARE(ArgMinFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(Asin)
FUNC_MSOP2SCHEMAOP_DECLARE(Assert)
FUNC_MSOP2SCHEMAOP_DECLARE(Assign)
FUNC_MSOP2SCHEMAOP_DECLARE(AssignAdd)
FUNC_MSOP2SCHEMAOP_DECLARE(Atan)
FUNC_MSOP2SCHEMAOP_DECLARE(AudioSpectrogram)
FUNC_MSOP2SCHEMAOP_DECLARE(AvgPoolFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(AvgPoolGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(BatchNorm)
FUNC_MSOP2SCHEMAOP_DECLARE(BatchNormGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(BatchToSpace)
FUNC_MSOP2SCHEMAOP_DECLARE(BatchToSpaceND)
FUNC_MSOP2SCHEMAOP_DECLARE(BiasAdd)
FUNC_MSOP2SCHEMAOP_DECLARE(BiasAddGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(BinaryCrossEntropy)
FUNC_MSOP2SCHEMAOP_DECLARE(BinaryCrossEntropyGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(BroadcastTo)
FUNC_MSOP2SCHEMAOP_DECLARE(Cast)
FUNC_MSOP2SCHEMAOP_DECLARE(Ceil)
FUNC_MSOP2SCHEMAOP_DECLARE(Clip)
FUNC_MSOP2SCHEMAOP_DECLARE(Concat)
FUNC_MSOP2SCHEMAOP_DECLARE(Attention)
FUNC_MSOP2SCHEMAOP_DECLARE(ConstantOfShape)
FUNC_MSOP2SCHEMAOP_DECLARE(Conv2DBackpropFilterFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(Conv2DBackpropInputFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(Conv2DFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(Conv2dTransposeFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(Cos)
FUNC_MSOP2SCHEMAOP_DECLARE(Crop)
FUNC_MSOP2SCHEMAOP_DECLARE(CustomExtractFeatures)
FUNC_MSOP2SCHEMAOP_DECLARE(CustomNormalize)
FUNC_MSOP2SCHEMAOP_DECLARE(CustomPredict)
FUNC_MSOP2SCHEMAOP_DECLARE(DeConv2DGradFilter)
FUNC_MSOP2SCHEMAOP_DECLARE(Depend)
FUNC_MSOP2SCHEMAOP_DECLARE(DepthToSpace)
FUNC_MSOP2SCHEMAOP_DECLARE(DetectionPostProcess)
FUNC_MSOP2SCHEMAOP_DECLARE(DivFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(DivGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(Dropout)
FUNC_MSOP2SCHEMAOP_DECLARE(DropoutGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(Eltwise)
FUNC_MSOP2SCHEMAOP_DECLARE(Elu)
FUNC_MSOP2SCHEMAOP_DECLARE(EmbeddingLookupFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(Equal)
FUNC_MSOP2SCHEMAOP_DECLARE(ExpFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(ExpandDims)
FUNC_MSOP2SCHEMAOP_DECLARE(FakeQuantWithMinMaxVars)
FUNC_MSOP2SCHEMAOP_DECLARE(FakeQuantWithMinMaxVarsPerChannel)
FUNC_MSOP2SCHEMAOP_DECLARE(FftImag)
FUNC_MSOP2SCHEMAOP_DECLARE(FftReal)
FUNC_MSOP2SCHEMAOP_DECLARE(Fill)
FUNC_MSOP2SCHEMAOP_DECLARE(FillV2)
FUNC_MSOP2SCHEMAOP_DECLARE(Flatten)
FUNC_MSOP2SCHEMAOP_DECLARE(FlattenGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(Floor)
FUNC_MSOP2SCHEMAOP_DECLARE(FloorDiv)
FUNC_MSOP2SCHEMAOP_DECLARE(FloorMod)
FUNC_MSOP2SCHEMAOP_DECLARE(FullConnection)
FUNC_MSOP2SCHEMAOP_DECLARE(FusedBatchNorm)
FUNC_MSOP2SCHEMAOP_DECLARE(Gather)
FUNC_MSOP2SCHEMAOP_DECLARE(GatherNd)
FUNC_MSOP2SCHEMAOP_DECLARE(Greater)
FUNC_MSOP2SCHEMAOP_DECLARE(GreaterEqual)
FUNC_MSOP2SCHEMAOP_DECLARE(GroupConv2DGradInput)
FUNC_MSOP2SCHEMAOP_DECLARE(HashtableLookup)
FUNC_MSOP2SCHEMAOP_DECLARE(InstanceNorm)
FUNC_MSOP2SCHEMAOP_DECLARE(LayerNormFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(LeakyRelu)
FUNC_MSOP2SCHEMAOP_DECLARE(Less)
FUNC_MSOP2SCHEMAOP_DECLARE(LessEqual)
FUNC_MSOP2SCHEMAOP_DECLARE(Log)
FUNC_MSOP2SCHEMAOP_DECLARE(LogGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(LogicalAnd)
FUNC_MSOP2SCHEMAOP_DECLARE(LogicalNot)
FUNC_MSOP2SCHEMAOP_DECLARE(LogicalOr)
FUNC_MSOP2SCHEMAOP_DECLARE(LogicalXor)
FUNC_MSOP2SCHEMAOP_DECLARE(LpNormalization)
FUNC_MSOP2SCHEMAOP_DECLARE(LRN)
FUNC_MSOP2SCHEMAOP_DECLARE(LshProjection)
FUNC_MSOP2SCHEMAOP_DECLARE(LSTM)
FUNC_MSOP2SCHEMAOP_DECLARE(LSTMGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(LSTMGradData)
FUNC_MSOP2SCHEMAOP_DECLARE(LSTMGradWeight)
FUNC_MSOP2SCHEMAOP_DECLARE(L2NormalizeFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(MatMulFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(Maximum)
FUNC_MSOP2SCHEMAOP_DECLARE(MaximumGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(MaxPoolFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(MaxPoolGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(SwitchLayer)
FUNC_MSOP2SCHEMAOP_DECLARE(Mfcc)
FUNC_MSOP2SCHEMAOP_DECLARE(Minimum)
FUNC_MSOP2SCHEMAOP_DECLARE(MinimumGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(Mod)
FUNC_MSOP2SCHEMAOP_DECLARE(MulFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(MulGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(Neg)
FUNC_MSOP2SCHEMAOP_DECLARE(NegGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(NotEqual)
FUNC_MSOP2SCHEMAOP_DECLARE(NonMaxSuppression)
FUNC_MSOP2SCHEMAOP_DECLARE(OneHot)
FUNC_MSOP2SCHEMAOP_DECLARE(OnesLike)
FUNC_MSOP2SCHEMAOP_DECLARE(PadFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(PartialFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(PowFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(PowerGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(PReLUFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(PriorBox)
FUNC_MSOP2SCHEMAOP_DECLARE(Proposal)
FUNC_MSOP2SCHEMAOP_DECLARE(RaggedRange)
FUNC_MSOP2SCHEMAOP_DECLARE(Rank)
FUNC_MSOP2SCHEMAOP_DECLARE(Range)
FUNC_MSOP2SCHEMAOP_DECLARE(Rank)
FUNC_MSOP2SCHEMAOP_DECLARE(RealDiv)
FUNC_MSOP2SCHEMAOP_DECLARE(Reciprocal)
FUNC_MSOP2SCHEMAOP_DECLARE(ReduceFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(Reshape)
FUNC_MSOP2SCHEMAOP_DECLARE(Resize)
FUNC_MSOP2SCHEMAOP_DECLARE(ReverseSequence)
FUNC_MSOP2SCHEMAOP_DECLARE(ReverseV2)
FUNC_MSOP2SCHEMAOP_DECLARE(Rfft)
FUNC_MSOP2SCHEMAOP_DECLARE(ROIPooling)
FUNC_MSOP2SCHEMAOP_DECLARE(Round)
FUNC_MSOP2SCHEMAOP_DECLARE(Rsqrt)
FUNC_MSOP2SCHEMAOP_DECLARE(QuantDTypeCast)
FUNC_MSOP2SCHEMAOP_DECLARE(Scale)
FUNC_MSOP2SCHEMAOP_DECLARE(ScaleFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(ScatterNd)
FUNC_MSOP2SCHEMAOP_DECLARE(Select)
FUNC_MSOP2SCHEMAOP_DECLARE(SGD)
FUNC_MSOP2SCHEMAOP_DECLARE(Shape)
FUNC_MSOP2SCHEMAOP_DECLARE(SigmoidCrossEntropyWithLogits)
FUNC_MSOP2SCHEMAOP_DECLARE(SigmoidCrossEntropyWithLogitsGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(Sin)
FUNC_MSOP2SCHEMAOP_DECLARE(SkipGram)
FUNC_MSOP2SCHEMAOP_DECLARE(SliceFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(SmoothL1Loss)
FUNC_MSOP2SCHEMAOP_DECLARE(SmoothL1LossGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(Softmax)
FUNC_MSOP2SCHEMAOP_DECLARE(SoftmaxCrossEntropyWithLogits)
FUNC_MSOP2SCHEMAOP_DECLARE(SpaceToBatch)
FUNC_MSOP2SCHEMAOP_DECLARE(SpaceToBatchND)
FUNC_MSOP2SCHEMAOP_DECLARE(SpaceToDepth)
FUNC_MSOP2SCHEMAOP_DECLARE(SparseSoftmaxCrossEntropyWithLogits)
FUNC_MSOP2SCHEMAOP_DECLARE(SparseToDense)
FUNC_MSOP2SCHEMAOP_DECLARE(Split)
FUNC_MSOP2SCHEMAOP_DECLARE(Sqrt)
FUNC_MSOP2SCHEMAOP_DECLARE(Square)
FUNC_MSOP2SCHEMAOP_DECLARE(SquaredDifference)
FUNC_MSOP2SCHEMAOP_DECLARE(Squeeze)
FUNC_MSOP2SCHEMAOP_DECLARE(Stack)
FUNC_MSOP2SCHEMAOP_DECLARE(StridedSlice)
FUNC_MSOP2SCHEMAOP_DECLARE(Sub)
FUNC_MSOP2SCHEMAOP_DECLARE(SubFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(SubGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(Switch)
FUNC_MSOP2SCHEMAOP_DECLARE(Tan)
FUNC_MSOP2SCHEMAOP_DECLARE(TensorListFromTensor)
FUNC_MSOP2SCHEMAOP_DECLARE(TensorListGetItem)
FUNC_MSOP2SCHEMAOP_DECLARE(TensorListReserve)
FUNC_MSOP2SCHEMAOP_DECLARE(TensorListSetItem)
FUNC_MSOP2SCHEMAOP_DECLARE(TensorListStack)
FUNC_MSOP2SCHEMAOP_DECLARE(TileFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(TopKFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(Transpose)
FUNC_MSOP2SCHEMAOP_DECLARE(Unique)
FUNC_MSOP2SCHEMAOP_DECLARE(UnsortedSegmentSum)
FUNC_MSOP2SCHEMAOP_DECLARE(Unsqueeze)
FUNC_MSOP2SCHEMAOP_DECLARE(Unstack)
FUNC_MSOP2SCHEMAOP_DECLARE(Where)
FUNC_MSOP2SCHEMAOP_DECLARE(ZerosLike)
FUNC_MSOP2SCHEMAOP_DECLARE(Select)
FUNC_MSOP2SCHEMAOP_DECLARE(GRU)
FUNC_MSOP2SCHEMAOP_DECLARE(NonZero)
FUNC_MSOP2SCHEMAOP_DECLARE(InvertPermutation)
FUNC_MSOP2SCHEMAOP_DECLARE(Size)
FUNC_MSOP2SCHEMAOP_DECLARE(RandomStandardNormal)
FUNC_MSOP2SCHEMAOP_DECLARE(CropAndResize)
FUNC_MSOP2SCHEMAOP_DECLARE(Erf)
FUNC_MSOP2SCHEMAOP_DECLARE(StridedSliceGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(IsFinite)
FUNC_MSOP2SCHEMAOP_DECLARE(LinSpace)
FUNC_MSOP2SCHEMAOP_DECLARE(UniformReal)
FUNC_MSOP2SCHEMAOP_DECLARE(AbsGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(RsqrtGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(SqrtGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(LayerNormGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(ResizeGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(Splice)
FUNC_MSOP2SCHEMAOP_DECLARE(LogSoftmax)
FUNC_MSOP2SCHEMAOP_DECLARE(Call)
FUNC_MSOP2SCHEMAOP_DECLARE(CumSum)
FUNC_MSOP2SCHEMAOP_DECLARE(SplitWithOverlap)
FUNC_MSOP2SCHEMAOP_DECLARE(GLU)
FUNC_MSOP2SCHEMAOP_DECLARE(TensorArray)
FUNC_MSOP2SCHEMAOP_DECLARE(TensorArrayRead)
FUNC_MSOP2SCHEMAOP_DECLARE(TensorArrayWrite)
FUNC_MSOP2SCHEMAOP_DECLARE(Affine)
FUNC_MSOP2SCHEMAOP_DECLARE(ScatterNdUpdate)
FUNC_MSOP2SCHEMAOP_DECLARE(AllGather)
FUNC_MSOP2SCHEMAOP_DECLARE(ReduceScatter)
FUNC_MSOP2SCHEMAOP_DECLARE(DynamicQuant)
FUNC_MSOP2SCHEMAOP_DECLARE(RandomNormal)
FUNC_MSOP2SCHEMAOP_DECLARE(NLLLoss)
FUNC_MSOP2SCHEMAOP_DECLARE(NLLLossGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(FormatTranspose)
FUNC_MSOP2SCHEMAOP_DECLARE(GatherD)
FUNC_MSOP2SCHEMAOP_DECLARE(GroupNormFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(Log1p)
FUNC_MSOP2SCHEMAOP_DECLARE(TensorScatterAdd)
FUNC_MSOP2SCHEMAOP_DECLARE(SparseFillEmptyRows)
FUNC_MSOP2SCHEMAOP_DECLARE(SparseReshape)
FUNC_MSOP2SCHEMAOP_DECLARE(SparseSegmentSum)
FUNC_MSOP2SCHEMAOP_DECLARE(ScatterElements)
FUNC_MSOP2SCHEMAOP_DECLARE(Triu)
FUNC_MSOP2SCHEMAOP_DECLARE(Tril)
FUNC_MSOP2SCHEMAOP_DECLARE(AdamWeightDecay)
#endif
}  // namespace mindspore::lite::ops
#else
#define FUNC_MSOP2SCHEMAOP_DECLARE(OP)
#endif
#endif  // MINDSPORE_LITE_SRC_COMMON_OPS_OPS_FUNC_DECLARE_H_
