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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_GENERATOR_COMPONENT_CONST_BLOCKS_BENCHMARK_COMMON_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_GENERATOR_COMPONENT_CONST_BLOCKS_BENCHMARK_COMMON_H_

#include <string>
#include <sstream>

namespace mindspore {
namespace lite {
namespace micro {
namespace benchmark_common {

// Common header includes for generated benchmark code
inline std::string GetBenchmarkIncludes() {
  return R"(
#include "load_input.h"
#include "calib_output.h"
#include "c_api/types_c.h"
#include "c_api/model_c.h"
#include "c_api/context_c.h"
#include "src/tensor.h"
#include <time.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef ENABLE_FP16
#include <arm_neon.h>
#endif
)";
}

// Common GetTimeUs function
inline std::string GetGetTimeUs() {
  return R"(
uint64_t GetTimeUs() {
  const int USEC = 1000000;
  const int MSEC = 1000;
  struct timespec ts = {0, 0};
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
    return 0;
  }
  uint64_t retval = (uint64_t)((ts.tv_sec * USEC) + (ts.tv_nsec / MSEC));
  return retval;
}
)";
}

// Common PrintTensorHandle function
inline std::string GetPrintTensorHandle(bool enable_fp16 = true) {
  std::ostringstream oss;
  oss << R"(
void PrintTensorHandle(MSTensorHandle tensor) {
  if (tensor == NULL) {
    printf("input tensor is null");
    return;
  }
  printf("name: %s, ", MSTensorGetName(tensor));
  MSDataType data_type = MSTensorGetDataType(tensor);
  printf("DataType: %d, ", data_type);
  size_t element_num = (size_t)(MSTensorGetElementNum(tensor));
  printf("Elements: %zu, ", element_num);
  printf("Shape: [");
  size_t shape_num = 0;
  const int64_t *dims = MSTensorGetShape(tensor, &shape_num);
  for (size_t i = 0; i < shape_num; i++) {
    printf("%d ", (int)dims[i]);
  }
  printf("], Data: \n");
  void *data = MSTensorGetMutableData(tensor);
  const size_t MAX_ELEMENT_NUM = 10;
  element_num = element_num > MAX_ELEMENT_NUM ? MAX_ELEMENT_NUM : element_num;
  switch (data_type) {
    case kMSDataTypeNumberTypeFloat32: {
      for (size_t i = 0; i < element_num; i++) {
        printf("%.6f, ", ((float *)data)[i]);
      }
      printf("\n");
    } break;
)";

  if (enable_fp16) {
    oss << R"(    case kMSDataTypeNumberTypeFloat16:
#ifdef ENABLE_FP16
    {
      for (size_t i = 0; i < element_num; i++) {
        printf("%.6f, ", ((float16_t *)data)[i]);
      }
      printf("\n");
    } break;
#endif
)";
  } else {
    oss << R"(    case kMSDataTypeNumberTypeFloat16:
    case kMSDataTypeNumberTypeInt16: {
      for (size_t i = 0; i < element_num; i++) {
        printf("%" PRId16, ((int16_t *)data)[i]);
      }
      printf("\n");
    } break;
)";
  }

  oss << R"(    case kMSDataTypeNumberTypeInt32: {
      for (size_t i = 0; i < element_num; i++) {
        printf("%" PRId32, ((int32_t *)data)[i]);
      }
      printf("\n");
    } break;
    case kMSDataTypeNumberTypeInt8: {
      for (size_t i = 0; i < element_num; i++) {
        printf("%" PRIi8, ((int8_t *)data)[i]);
      }
      printf("\n");
    } break;
    case kMSDataTypeNumberTypeUInt8: {
      for (size_t i = 0; i < element_num; i++) {
        printf("%u", ((uint8_t *)data)[i]);
      }
      printf("\n");
    } break;
    default:
      printf("Unsupported data type to print");
      break;
  }
}
)";
  return oss.str();
}

// Common model loading logic
inline std::string GetModelLoadLogic(const std::string &model_source = "file") {
  std::ostringstream oss;
  if (model_source == "file") {
    oss << R"(
  void *model_buffer = NULL;
  int model_size = 0;
  // read .bin file by ReadBinaryFile;
  if (argc >= 3) {
    model_buffer = ReadInputData(argv[2], &model_size);
    if (model_buffer == NULL) {
      printf("Read model file failed.");
      return kMSStatusLiteParamInvalid;
    }
  }
  MSModelHandle model_handle = MSModelCreate();
  int ret = MSModelBuild(model_handle, model_buffer, model_size, kMSModelTypeMindIR, ms_context_handle);
  MSContextDestroy(&ms_context_handle);
  if (ret != kMSStatusSuccess) {
    printf("MSModelBuildFromFile failed, ret: %d\n", ret);
    free(model_buffer);
    model_buffer = NULL;
    return ret;
  }
  if (model_buffer) {
    free(model_buffer);
    model_buffer = NULL;
  }
)";
  } else {
    oss << R"(
  MSModelHandle model_handle = MSModelCreate();
  if (model_handle == NULL) {
    printf("MSModelCreate failed.\n");
    return kMSStatusLiteNullptr;
  }
  int ret = MSModelBuild(model_handle, NULL, 0, kMSModelTypeMindIR, NULL);
  if (ret != kMSStatusSuccess) {
    printf("MSModelBuildFromFile failed, ret: %d\n", ret);
    MSModelDestroy(&model_handle);
    return ret;
  }
)";
  }
  return oss.str();
}

// Common inputs loading logic
inline std::string GetInputsLoadLogic() {
  return R"(
  // set model inputs tensor data
  MSTensorHandleArray inputs_handle = MSModelGetInputs(model_handle);
  if (inputs_handle.handle_list == NULL) {
    printf("MSModelGetInputs failed, ret: %d", ret);
    return ret;
  }
  size_t inputs_num = inputs_handle.handle_num;
  void *inputs_binbuf[inputs_num];
  int inputs_size[inputs_num];
  for (size_t i = 0; i < inputs_num; ++i) {
    MSTensorHandle tensor = inputs_handle.handle_list[i];
    inputs_size[i] = (int)MSTensorGetDataSize(tensor);
  }
  ret = ReadInputsFile((char *)(argv[1]), inputs_binbuf, inputs_size, (int)inputs_num);
  if (ret != 0) {
    MSModelDestroy(&model_handle);
    return ret;
  }
  for (size_t i = 0; i < inputs_num; ++i) {
    void *input_data = MSTensorGetMutableData(inputs_handle.handle_list[i]);
    memcpy(input_data, inputs_binbuf[i], inputs_size[i]);
    free(inputs_binbuf[i]);
    inputs_binbuf[i] = NULL;
  }
)";
}

// Common warmup and benchmark loop
inline std::string GetWarmupAndBenchmark(const std::string &predict_call = "MSModelPredict") {
  std::ostringstream oss;
  oss << R"(
  int warm_up_loop_count = 3;
  if (argc >= 8) {
      warm_up_loop_count = atoi(argv[7]);
      if (warm_up_loop_count < 0) {
        printf("The warm up loop count error! Cannot be less than 0.\n");
        return kMSStatusLiteParamInvalid;
      }
  }
  printf("Running warm up loops...");
  for (int i = 0; i < warm_up_loop_count; ++i) {
    ret = )"
      << predict_call
      << R"((model_handle, inputs_handle, &outputs_handle, NULL, NULL);)"
         R"(
    if (ret != kMSStatusSuccess) {
      MSModelDestroy(&model_handle);
      printf(")"
      << predict_call
      << R"( failed, ret: %d", ret);)"
         R"(
      return ret;
    }
  }

  if (argc >= 4) {
    int loop_count = atoi(argv[3]);
    printf("\nloop count: %d\n", loop_count);
    uint64_t start_time = GetTimeUs();
    for (int i = 0; i < loop_count; ++i) {
      ret = )"
      << predict_call
      << R"((model_handle, inputs_handle, &outputs_handle, NULL, NULL);)"
         R"(
      if (ret != kMSStatusSuccess) {
        MSModelDestroy(&model_handle);
        printf(")"
      << predict_call
      << R"( failed, ret: %d", ret);)"
         R"(
        return ret;
      }
    }
    uint64_t end_time = GetTimeUs();
    float total_time = (float)(end_time - start_time) / 1000.0f;
    printf("total time: %.5fms, per time: %.5fms\n", total_time, total_time / loop_count);
  }
  ret = )"
      << predict_call
      << R"((model_handle, inputs_handle, &outputs_handle, NULL, NULL);)"
         R"(
  if (ret != kMSStatusSuccess) {
    MSModelDestroy(&model_handle);
    printf(")"
      << predict_call
      << R"( failed, ret: %d", ret);)"
         R"(
    return ret;
  }
)";
  return oss.str();
}

// Common calibration and output comparison
inline std::string GetCalibrationLogic() {
  return R"(
  if (argc >= 5) {
    CalibTensor *calib_tensors;
    int calib_num = 0;
    ret = ReadCalibData(argv[4], &calib_tensors, &calib_num);
    if (ret != kMSStatusSuccess) {
      MSModelDestroy(&model_handle);
      return ret;
    }
    float cosine_distance_threshold = 0.9999;
    if (argc >= 9) {
      cosine_distance_threshold = atof(argv[8]);
    }
    ret = CompareOutputs(outputs_handle, &calib_tensors, calib_num, cosine_distance_threshold);
    if (ret != kMSStatusSuccess) {
      MSModelDestroy(&model_handle);
      return ret;
    }
    FreeCalibTensors(&calib_tensors, calib_num);
  }
)";
}

// Common context setup logic
inline std::string GetContextSetup() {
  return R"(
  MSContextHandle ms_context_handle = MSContextCreate();
  if (argc >= 6) {
    int thread_num = atoi(argv[5]);
    if (thread_num < 1 || thread_num > kMaxThreadNum) {
      printf("Thread number error! It should be greater than 0 and less than 5\n");
      return kMSStatusLiteParamInvalid;
    }
    MSContextSetThreadNum(ms_context_handle, thread_num);
  }
  printf("ThreadNum: %d.\n", MSContextGetThreadNum(ms_context_handle));

  int bind_mode = kBindDefault;
  if (argc >= 7) {
    bind_mode = atoi(argv[6]);
    if (bind_mode < 0 || bind_mode > 2) {
      printf("Thread bind mode error! 0: No bind, 1: Bind hign cpu, 2: Bind mid cpu.\n");
      return kMSStatusLiteParamInvalid;
    }
  }
  MSContextSetThreadAffinityMode(ms_context_handle, bind_mode);
  printf("BindMode: %d.\n", MSContextGetThreadAffinityMode(ms_context_handle));
)";
}

}  // namespace benchmark_common
}  // namespace micro
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_GENERATOR_COMPONENT_CONST_BLOCKS_BENCHMARK_COMMON_H_
