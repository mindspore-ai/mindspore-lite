#!/bin/bash
# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

set -e
CUR_DIR=$(
  cd "$(dirname $0)"
  pwd
)
BUILD_DIR=${CUR_DIR}/../../build

while getopts "e:r:" opt; do
  case ${opt} in
      e)
          backend=${OPTARG}
          echo "backend is ${OPTARG}"
          echo "WARNING: backend support ascend_a2 or ascend_300iduo, ascend_300iduo is reserved parameter."
          ;;
      r)
          release_path=${OPTARG}
          echo "release_path is ${OPTARG}"
          ;;
      ?)
      echo "unknown para"
      exit 1;; 
  esac
done
export GLOG_v=2

# prepare run directory for ut
mkdir -pv ${CUR_DIR}/do_test

# prepare data for ut
cd ${CUR_DIR}/do_test

if [[ "${backend}" == "ascend_a2" ]]; then
  export MSLITE_ENABLE_CLOUD_FUSION_INFERENCE=on
  mindspore_lite_whl=$(ls ${release_path}/linux_aarch64/cloud_fusion/*.whl) || true
else
  ls ${BUILD_DIR}/test/
  cp ${BUILD_DIR}/test/lite-test ./
  ENABLE_CONVERTER_TEST=false
  if [ -f "${BUILD_DIR}/test/lite-test-converter" ]; then
    cp ${BUILD_DIR}/test/lite-test-converter ./
    ENABLE_CONVERTER_TEST=true
  fi
  cp ${BUILD_DIR}/googletest/googlemock/gtest/libgtest.so ./
  cp ${BUILD_DIR}/googletest/googlemock/gtest/libgmock.so ./

  # prepare data for dataset
  TEST_DATA_DIR=${CUR_DIR}/../../mindspore/tests/ut/data/dataset/
  cp -fr $TEST_DATA_DIR/testPK ./data

  # check mslite whl pkg
  mindspore_lite_whl=$(ls ${CUR_DIR}/../../output/*.whl) || true
fi

ls -l *.so*
if [[ "X${CUDA_HOME}" != "X" ]]; then
  export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
fi
export LD_LIBRARY_PATH=./:${LD_LIBRARY_PATH}

cp -r ${CUR_DIR}/ut/test_data/* ./
cp -r ${CUR_DIR}/ut/src/runtime/kernel/arm/test_data/* ./
cp -r ${CUR_DIR}/ut/tools/converter/parser/tflite/test_data/* ./
cp -r ${CUR_DIR}/ut/tools/converter/registry/test_data/* ./
# MindrtRuntimeTest reads ./test_data/mindrt_parallel/parallel.ms; the source
# file lives under the opencl test_data tree, so expose it under test_data/.
mkdir -p ./test_data/mindrt_parallel
cp ${CUR_DIR}/ut/src/runtime/kernel/opencl/test_data/mindrt_parallel/parallel.ms ./test_data/mindrt_parallel/ 2>/dev/null || true
# some ut sources still reference data as ./test_data/<dir>; expose the copied
# arm kernel test_data subdirs under that prefix via symlinks (no duplicate copy)
mkdir -p ./test_data
for data_dir in ${CUR_DIR}/ut/src/runtime/kernel/arm/test_data/*/; do
  ln -sfn "$(pwd)/$(basename "${data_dir}")" "./test_data/$(basename "${data_dir}")"
done

if [[ -f "${mindspore_lite_whl}" || "$MSLITE_ENABLE_SERVER_INFERENCE" == on ]]; then
  echo "download mobilenetv2.ms..."
  # prepare model and inputdata for Python-API ut test
  if [ ! -e mobilenetv2.ms ]; then
    if [[ -e "${SHARE_LITE_DATASET_PATH}/quick_start/mobilenetv2.ms" ]]; then
        cp ${SHARE_LITE_DATASET_PATH}/quick_start/mobilenetv2.ms ./mobilenetv2.ms || exit 1
    else
        MODEL_DOWNLOAD_URL="https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.ms"
        wget -c -O mobilenetv2.ms --no-check-certificate ${MODEL_DOWNLOAD_URL}
    fi
  fi

  if [ ! -e mobilenetv2.ms.bin ]; then
    if [[ -e "${SHARE_LITE_DATASET_PATH}/quick_start/micro/mobilenetv2.tar.gz" ]]; then
        cp ${SHARE_LITE_DATASET_PATH}/quick_start/micro/mobilenetv2.tar.gz ./mobilenetv2.tar.gz || exit 1
    else
        BIN_DOWNLOAD_URL="https://download.mindspore.cn/model_zoo/official/lite/quick_start/micro/mobilenetv2.tar.gz"
        wget -c --no-check-certificate ${BIN_DOWNLOAD_URL}
    fi
    tar -zxf mobilenetv2.tar.gz
    cp mobilenetv2/*.tflite ./mobilenetv2.tflite
    cp mobilenetv2/*.ms.out ./mobilenetv2.ms.out
    cp mobilenetv2/*.ms.bin ./mobilenetv2.ms.bin
    rm -rf mobilenetv2.tar.gz mobilenetv2/
  fi
fi

echo 'run common ut tests'
# UT for mindspore lite cloud inference
if [[ "${MSLITE_ENABLE_CLOUD_FUSION_INFERENCE}" == "on" || "${MSLITE_ENABLE_CLOUD_FUSION_INFERENCE}" == "ON" || "${MSLITE_ENABLE_CLOUD_INFERENCE}" == "ON" || "${MSLITE_ENABLE_CLOUD_INFERENCE}" == "on" ]]; then
  echo 'run MSLITE_ENABLE_CLOUD_FUSION_INFERENCE ut test'
  set +e
  if [ -d "/usr/local/Ascend/ascend-toolkit" ]; then
    ascend_setenv_path=/usr/local/Ascend/ascend-toolkit/set_env.sh
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/devlib
  else
    ascend_setenv_path=/usr/local/Ascend/latest/bin/setenv.bash
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/Ascend/latest/lib64
  fi
  source ${ascend_setenv_path}
  set -e

  if [[ "${backend}" == "ascend_a2" ]]; then
    user_name=${USER}
    echo "Current user name is ${user_name}"
    benchmark_test_path=/home/${user_name}/benchmark_test/0
    cp ${benchmark_test_path}/ms_models/matmul_ops_for_ut.static.onnx.mindir ${CUR_DIR}/do_test
    cp ${benchmark_test_path}/ms_models/matmul_ops_for_ut.dynamic.onnx.mindir ${CUR_DIR}/do_test

    # Note: If a new UT script needs to be added, implement a ***.cc file. 
    # Refer to model_test.cc and model_parallel_runner_test.cc for implementation content.
    
    # Run ModelTest all ut testcases in Ascend.
    ./lite-test --gtest_filter="ModelTest.*"
    # Run ModelParallelRunnerTest ut testcases in Ascend and cpu.
    ./lite-test --gtest_filter="ModelParallelRunnerTest.*"
  else
    # mindspore lite converter
    ./lite-test-converter --gtest_filter="PassRegistryPositionAscendTest.*"
    # mapper
    ./lite-test-converter --gtest_filter="ArgminFusionMapperTest.*"
    ./lite-test-converter --gtest_filter="ActivationMapperTest.*"
    ./lite-test-converter --gtest_filter="ClipMapperTest.*"
    ./lite-test-converter --gtest_filter="ArithmeticMapperTest.*"
  fi
  exit 0
fi

echo 'run flatbuffers verifier ut test'
./lite-test --gtest_filter="FlatBuffersVerifierTest.RejectModelExceedingTableLimit"

# test cases of Converter
# TFLite per-op parser tests (~85 suites), ONNX parsers, fusion passes, mappers
# and SVD. These were previously commented out; re-enabling lifts tools/converter
# parser coverage. Any failure aborts the run (set -e) — keep an eye on flaky nets.
# OnnxLayerNormParserTest/OnnxPoolParserTest/ArgmaxFusionMapperTest/FusedBatchNormMapperTest
# are only compiled when MSLITE_ENABLE_CLOUD_FUSION_INFERENCE or
# MSLITE_ENABLE_CLOUD_INFERENCE is on (conditional glob in test/CMakeLists.txt);
# in other builds these filters match 0 tests and are harmless no-ops.
if [ "$ENABLE_CONVERTER_TEST" = true ]; then
  ./lite-test-converter --gtest_filter="TestTfliteParser*"
  ./lite-test-converter --gtest_filter="OnnxLayerNormParserTest*"
  ./lite-test-converter --gtest_filter="OnnxPoolParserTest*"
  ./lite-test-converter --gtest_filter="ConstantFoldingFusionTest*"
  ./lite-test-converter --gtest_filter="ConvActivationFusionTest*"
  ./lite-test-converter --gtest_filter="ConvBiasAddFusionTest*"
  ./lite-test-converter --gtest_filter="ConvBNFusionTest*"
  ./lite-test-converter --gtest_filter="ConvScaleFusionTest*"
  ./lite-test-converter --gtest_filter="MatMulAddFusionTest*"
  ./lite-test-converter --gtest_filter="TransMatMulFusionTest*"
  ./lite-test-converter --gtest_filter="ActivationFusionTest*"
  ./lite-test-converter --gtest_filter="AddConcatActivationFusionTest*"
  ./lite-test-converter --gtest_filter="ArgmaxFusionMapperTest*"
  ./lite-test-converter --gtest_filter="FusedBatchNormMapperTest*"
  ./lite-test-converter --gtest_filter="SVDTest*"
fi
# The *InoutTest suites are compiled into the converter build tree; guard them
# so a CONVERTER=off build (no lite-test-converter binary) doesn't abort under
# `set -e` with "No such file or directory".
if [ "$ENABLE_CONVERTER_TEST" = true ]; then
  ./lite-test-converter --gtest_filter="ConvActFusionInoutTest*"
  ./lite-test-converter --gtest_filter="ConvBiasFusionInoutTest*"
  ./lite-test-converter --gtest_filter="ConcatActFusionInoutTest*"
  ./lite-test-converter --gtest_filter="MatmulMulFusionInoutTest*"
  ./lite-test-converter --gtest_filter="MatMulActivationFusionInoutTest*"
  ./lite-test-converter --gtest_filter="ActivationFusionInoutTest*"
  ./lite-test-converter --gtest_filter="TransMatMulFusionInoutTest*"
fi
# test cases of framework

# test cases of FP32 OP
./lite-test --gtest_filter=TestBatchnormFp32*
./lite-test --gtest_filter=TestBatchToSpaceFp32*
./lite-test --gtest_filter=TestConv1x1Fp32*
./lite-test --gtest_filter=TestConvolutionFp32*
./lite-test --gtest_filter=CropTestFp32*
./lite-test --gtest_filter=TestDeConvolutionFp32*
./lite-test --gtest_filter=DepthToSpaceTestFp32*
./lite-test --gtest_filter=TestFcFp32*
./lite-test --gtest_filter=TestLogicalOrFp32*
./lite-test --gtest_filter=TestNLLLossFp32*
./lite-test --gtest_filter=TestOneHotFp32*
./lite-test --gtest_filter=TestPowerFp32*
./lite-test --gtest_filter=TestReduceFp32*
./lite-test --gtest_filter=TestRaggedRangeFp32*
./lite-test --gtest_filter=TestScaleFp32*
./lite-test --gtest_filter=TestTileFp32*
./lite-test --gtest_filter=TileFp32Test.*
./lite-test --gtest_filter=ResizeFp32Test.*
./lite-test --gtest_filter=LstmFp32Test.*
./lite-test --gtest_filter=EluFp32Test.*
./lite-test --gtest_filter=ErfFp32Test.*
./lite-test --gtest_filter=StackFp32Test.*
./lite-test --gtest_filter=UnstackFp32Test.*
./lite-test --gtest_filter=StackInferTest.*
./lite-test --gtest_filter=GeluFp32Test.*
./lite-test --gtest_filter=TriuTrilFp32Test.*
./lite-test --gtest_filter=DepthToSpaceFp32Test.*
./lite-test --gtest_filter=SpaceToDepthFp32Test.*
./lite-test --gtest_filter=BroadcastFp32Test.*
./lite-test --gtest_filter=ArgMinFp32Test.*
./lite-test --gtest_filter=HSigmoidFp32Test.*
./lite-test --gtest_filter=CeluFp32Test.*
./lite-test --gtest_filter=ConstantOfShapeFp32Test.*
./lite-test --gtest_filter=ArithmeticFp32Test.*
./lite-test --gtest_filter=ReduceFp32Test.*

# test cases of FP32 OP (supplementary, compiled but previously not filtered)
# These raise kernel/cpu/fp32 coverage. Any failure aborts the run (set -e).
./lite-test --gtest_filter=TestActivationFp32*
./lite-test --gtest_filter=TestMatMulFp32*
./lite-test --gtest_filter=TestSoftmaxFp32*
./lite-test --gtest_filter=TestTransposeFp32*
./lite-test --gtest_filter=TestTopKFp32*
./lite-test --gtest_filter=TestResizeBilinearFp32*
./lite-test --gtest_filter=TestResizeNearestNeighborFp32*
./lite-test --gtest_filter=TestConvolutionDwFp32*
./lite-test --gtest_filter=TestCumsum*
./lite-test --gtest_filter=TestConstantOfShapeFp32*
./lite-test --gtest_filter=TestL2NormFp32*
./lite-test --gtest_filter=TestDetectionPostProcessFp32*
./lite-test --gtest_filter=TestEmbeddingLookupFp32*
./lite-test --gtest_filter=TestLshProjectionFp32*
./lite-test --gtest_filter=TestNMSFp32*
./lite-test --gtest_filter=TestROIPoolingFp32*
./lite-test --gtest_filter=TestReverseSequenceFp32*
./lite-test --gtest_filter=TestScatterNdAdd*
./lite-test --gtest_filter=TestScatterNdFp32*
./lite-test --gtest_filter=TestSkipGramFp32*
./lite-test --gtest_filter=TestSparseToDenseFp32*
./lite-test --gtest_filter=TestUniformRealFp32*
./lite-test --gtest_filter=TestUniqueFp32*
./lite-test --gtest_filter=TestUnstackFp32*
./lite-test --gtest_filter=StackTestFp32*
./lite-test --gtest_filter=SpaceToBatchTestFp32*
./lite-test --gtest_filter=SpaceToDepthTestFp32*
./lite-test --gtest_filter=LstmFp32*

# test cases of INT8 OP
./lite-test --gtest_filter=TestBatchnormInt8.*
./lite-test --gtest_filter=TestDeconvInt8.*
./lite-test --gtest_filter=TestPadInt8.*
./lite-test --gtest_filter=MulInt8Test.*
./lite-test --gtest_filter=MatmulInt8Test.*
./lite-test --gtest_filter=ResizeInt8Test.*
./lite-test --gtest_filter=AvgPoolInt8Test.*
./lite-test --gtest_filter=MaxPoolInt8Test.*
./lite-test --gtest_filter=ArithmeticSelfInt8Test.*
./lite-test --gtest_filter=GatherInt8Test.*
./lite-test --gtest_filter=SoftmaxInt8Test.*
./lite-test --gtest_filter=DivInt8Test.*
./lite-test --gtest_filter=SubInt8Test.*
./lite-test --gtest_filter=EluInt8Test.*
./lite-test --gtest_filter=ErfInt8Test.*
./lite-test --gtest_filter=StackInt8Test.*
./lite-test --gtest_filter=UnstackInt8Test.*
./lite-test --gtest_filter=GeluInt8Test.*
./lite-test --gtest_filter=TriuTrilInt8Test.*
./lite-test --gtest_filter=DepthToSpaceInt8Test.*
./lite-test --gtest_filter=SpaceToDepthInt8Test*
./lite-test --gtest_filter=AddInt8Test.*
./lite-test --gtest_filter=SpaceToBatchInt8Test.*
./lite-test --gtest_filter=HardSigmoidInt8Test.*
./lite-test --gtest_filter=CeluInt8Test.*
./lite-test --gtest_filter=ConstantOfShapeInt8Test.*
./lite-test --gtest_filter=MaximumMinimumInt8Test.*
./lite-test --gtest_filter=ReduceInt8Test.*

# test cases of INT8 OP (supplementary, compiled but previously not filtered)
# These raise kernel/cpu/int8 coverage. Any failure aborts the run (set -e).
./lite-test --gtest_filter=TestConcatInt8*
./lite-test --gtest_filter=TestMatmulInt8*
./lite-test --gtest_filter=TestConv1x1Int8*
./lite-test --gtest_filter=TestReduceInt8*
./lite-test --gtest_filter=TestSoftmaxInt8*
./lite-test --gtest_filter=TestQuantizedAdd*
./lite-test --gtest_filter=QuantCastInt8Test*
./lite-test --gtest_filter=QuantDTypeCastTestFp32*
./lite-test --gtest_filter=TestArithmeticSelfInt8*
./lite-test --gtest_filter=TestCropInt8*
./lite-test --gtest_filter=TestFcInt8*
./lite-test --gtest_filter=TestGatherInt8*
./lite-test --gtest_filter=TestGatherNdInt8*
./lite-test --gtest_filter=TestHSwishInt8*
./lite-test --gtest_filter=TestL2NormInt8*
./lite-test --gtest_filter=TestMulInt8*
./lite-test --gtest_filter=TestPowerInt8*
./lite-test --gtest_filter=TestPreluInt8*
./lite-test --gtest_filter=TestReluXInt8*
./lite-test --gtest_filter=TestReshapeInt8*
./lite-test --gtest_filter=TestResizeBilinearInt8*
./lite-test --gtest_filter=TestResizeNearestNeighborInt8*
./lite-test --gtest_filter=TestScaleInt8*
./lite-test --gtest_filter=TestSigmoidInt8*
./lite-test --gtest_filter=TestSliceInt8*
./lite-test --gtest_filter=TestSplitInt8*
./lite-test --gtest_filter=TestSqueezeInt8*
./lite-test --gtest_filter=TestSubInt8*
./lite-test --gtest_filter=TestTopKInt8*
./lite-test --gtest_filter=TestUnsqueezeInt8*
./lite-test --gtest_filter=SpaceToBatchTestInt8*

# test cases of generic api
# GenericApiTest source (ut/src/api/generic_api_test.cc) is not referenced by
# any GLOB in test/CMakeLists.txt; only included if a build flag adds it.
# Filter is a no-op in current builds.
./lite-test --gtest_filter="GenericApiTest*"

# Compiled-but-previously-unfiltered runtime/util suites. All pure CPU.
# Skipped MultipleDeviceTest (needs GPU/NPU), NetworkTest (stale .ms schema),
# MindrtRuntimeTest (parallel.ms schema incompatible with current runtime),
# TestNormalize (CustomNormalize creator not registered in this build),
# SchedulerTest/UtilsTest (coredump in this build — investigated, root cause
# is runtime init ordering, not the test itself).
./lite-test --gtest_filter="LiteMindRtTest.*"
./lite-test --gtest_filter="ModelObfuscationDeprecatedTest.*"
./lite-test --gtest_filter="OptimizeAllocator.*"
./lite-test --gtest_filter="RandomStandardNormalTest.*"
./lite-test --gtest_filter="ReduceMaxFp32Test.*"
./lite-test --gtest_filter="TestPack.*"
./lite-test --gtest_filter="TestStridedSlice.*"

# NNACL shape-inference tests: suites compiled from ut/nnacl/infer/*.cc,
# previously never filtered. Pure CPU, very fast. Lifts nnacl_c/infer coverage
# (and acts as a safety net for the infer path).
echo 'run nnacl infer ut tests'
./lite-test --gtest_filter="*InferTest*"
# The *InferTest* filter matches the 98 nnacl/infer suites whose names end in
# InferTest, plus the end-to-end InferTest suite in ut/src/infer_test.cc. The 6
# nnacl/infer suites below are named without the "InferTest" substring, so add
# them explicitly to cover the full ut/nnacl/infer/*.cc set.
./lite-test --gtest_filter="AdamWeightDecayInfer*"
./lite-test --gtest_filter="InferManagerTest*"
./lite-test --gtest_filter="TestNLLLossGradInfer*"
./lite-test --gtest_filter="TestNLLLossInfer*"
./lite-test --gtest_filter="TestScatterNdAddInfer*"
./lite-test --gtest_filter="TestStridedSliceFp32*"

if [ "$ENABLE_CONVERTER_TEST" = true ]; then
  ./lite-test-converter --gtest_filter="ModelParserRegistryTest.TestRegistry"
  ./lite-test-converter --gtest_filter="NodeParserRegistryTest.TestRegistry"
  ./lite-test-converter --gtest_filter="PassRegistryTest.TestRegistry"
  ./lite-test-converter --gtest_filter="TestConverterAPI.*"
  ./lite-test-converter --gtest_filter="SpecifyGraphOutputFormatTest*"
fi
./lite-test --gtest_filter="TestRegistry.TestAdd"
./lite-test --gtest_filter="TestRegistryCustomOp.TestCustomAdd"

if [ -f "$BUILD_DIR/src/libmindspore-lite-train.so" ]; then
  echo 'run cxx_api ut tests'
  ./lite-test --gtest_filter="TestCxxApiLiteModel*"
  ./lite-test --gtest_filter="TestCxxApiLiteSerialization*"

  echo 'run train ut tests'
  ./lite-test --gtest_filter="TestActGradFp32*"
  ./lite-test --gtest_filter="TestSoftmaxGradFp32*"
  ./lite-test --gtest_filter="TestSoftmaxCrossEntropyFp32*"
  ./lite-test --gtest_filter="TestBiasGradFp32*"
  # Grad tests below were previously commented out; re-enable to lift
  # kernel/cpu/fp32_grad coverage (currently 19.8%).
  ./lite-test --gtest_filter="TestConvolutionGradFp32*"
  ./lite-test --gtest_filter="TestDeConvolutionGradFp32*"
  ./lite-test --gtest_filter="TestArithmeticGradFp32*"
  ./lite-test --gtest_filter="TestPoolingGradFp32*"
  ./lite-test --gtest_filter="TestBNGradFp32*"
  ./lite-test --gtest_filter="TestNLLLossGradFp32*"
fi

echo 'run inference ut tests'
# ControlFlowTest.TestMergeWhileModel and GraphTest.UserSetGraphOutput* are
# orphan filters — the suites are not defined anywhere in test/. Removed.

echo 'run mindrt parallel ut test'
# MindrtParallelTest.* and BenchmarkTest.* are compiled only in cloud builds
# (st/mindrt_parallel_test.cc and st/benchmark_test.cc are REMOVE_ITEM'd in
# non-cloud builds via test/CMakeLists.txt:161-163). Kept as no-op here.
if [ "$ENABLE_CONVERTER_TEST" = true ]; then
  ./lite-test-converter --gtest_filter="MindrtParallelTest.*"
fi
./lite-test --gtest_filter="BenchmarkTest.mindrtParallelOffline*"

# DelegateTest.CustomDelegate removed — st/delegate_test.cc no longer exists.

echo 'runtime pass'
./lite-test --gtest_filter="RuntimePass.*"

echo 'runtime convert'
./lite-test --gtest_filter="RuntimeConvert.*"
./lite-test --gtest_filter="BenchmarkTest.runtimeConvert1"

echo 'Optimize Allocator'
./lite-test --gtest_filter="OptimizeAllocator.*"

echo 'Runtime config file test'
./lite-test --gtest_filter="MixDataTypeTest.Config1"

echo 'run c api ut test'
./lite-test --gtest_filter="TensorCTest.*"
./lite-test --gtest_filter="ContextCTest.*"
./lite-test --gtest_filter="ModelCApiTest.*"

echo 'run bfc memory ut test'
# DynamicMemManagerTest is only compiled in cloud builds (test/CMakeLists.txt
# includes ut/src/runtime/dynamic_mem_manager_test.cc inside the
# MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE branch).
# In other builds this filter matches 0 tests and is a harmless no-op.
./lite-test --gtest_filter="DynamicMemManagerTest.*"

echo "lite Python API ut test"
if [ ! -f "${mindspore_lite_whl}" ]; then
  echo -e "\e[31mPython-API Whl not found, so lite Python API ut test will not be run. \e[0m"
else
  export PYTHONPATH=${BUILD_DIR}/package/:${PYTHONPATH}

  # run converter Python-API ut test
  if [[ ! "${MSLITE_ENABLE_CONVERTER}" || "${MSLITE_ENABLE_CONVERTER}" == "ON" || "${MSLITE_ENABLE_CONVERTER}" == "on" ]]; then
    echo "run converter Python API ut test"
    pytest ${CUR_DIR}/ut/python/test_converter_api.py -s
    RET=$?
    if [ ${RET} -ne 0 ]; then
      exit ${RET}
    fi
  fi

  # run inference Python-API ut test
  echo "run inference Python API ut test"
  pytest ${CUR_DIR}/ut/python/test_inference_api.py -s
  RET=$?
  if [ ${RET} -ne 0 ]; then
    exit ${RET}
  fi

  # run LLMEngine Python-API ut test
  # echo "run LLMEngine Python API ut test"
  # pytest ${CUR_DIR}/ut/python/test_lite_llm_engine_api.py -s
  # RET=$?
  # if [ ${RET} -ne 0 ]; then
  #   exit ${RET}
  # fi

  # run inference CPU Python-API st test
  echo "run inference CPU Python API st test"
  pytest ${CUR_DIR}/st/python/test_inference.py::test_cpu_inference_01 -s
  RET=$?
  if [ ${RET} -ne 0 ]; then
    exit ${RET}
  fi
fi

if [ "$MSLITE_ENABLE_KERNEL_EXECUTOR" = on ]; then
  echo 'run kernel executor api ut test'
  ./lite-test --gtest_filter="KernelExecutorTest.*"
fi
