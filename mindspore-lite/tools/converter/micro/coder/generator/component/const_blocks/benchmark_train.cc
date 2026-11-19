/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "coder/generator/component/const_blocks/benchmark_train.h"

namespace mindspore::lite::micro {
const char benchmark_train_source[] = R"RAW(/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "load_input.h"
#include "calib_output.h"
#include "c_api/types_c.h"
#include "c_api/model_c.h"
#include "c_api/context_c.h"
#include "src/tensor.h"
#include "src/model.h"
#include <time.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define kMaxThreadNum 4
#define kDefaultThreadNum 1
#define kDefaultBindMode 1

void usage() {
  printf(
    "-- mindspore benchmark_train params usage:\n"
    "args[0]: executable file\n"
    "args[1]: comma separated input binary files, for example input.bin,label.bin\n"
    "args[2]: loop count for train steps, default is 1\n"
    "args[3]: warm up loop count, default is 0\n"
    "args[4]: runtime thread num, default is 1\n"
    "args[5]: runtime thread bind mode, 0: No bind, 1: Bind high cpu, 2: Bind mid cpu, default is 1\n"
    "args[6]: comma separated eval input binary files, optional\n"
    "args[7]: calibration file for eval output, optional\n"
    "args[8]: cosine distance threshold, default is 0.9999\n\n");
}

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
    case kMSDataTypeNumberTypeFloat16:
    case kMSDataTypeNumberTypeInt16: {
      for (size_t i = 0; i < element_num; i++) {
        printf("%" PRId16 ", ", ((int16_t *)data)[i]);
      }
      printf("\n");
    } break;
    case kMSDataTypeNumberTypeInt32: {
      for (size_t i = 0; i < element_num; i++) {
        printf("%" PRId32 ", ", ((int32_t *)data)[i]);
      }
      printf("\n");
    } break;
    case kMSDataTypeNumberTypeInt8: {
      for (size_t i = 0; i < element_num; i++) {
        printf("%" PRIi8 ", ", ((int8_t *)data)[i]);
      }
      printf("\n");
    } break;
    case kMSDataTypeNumberTypeBool:
    case kMSDataTypeNumberTypeUInt8: {
      for (size_t i = 0; i < element_num; i++) {
        printf("%u, ", ((uint8_t *)data)[i]);
      }
      printf("\n");
    } break;
    default:
      printf("Unsupported data type to print");
      break;
  }
}

int LoadInputFiles(MSTensorHandleArray inputs_handle, const char *input_files) {
  if (inputs_handle.handle_list == NULL || input_files == NULL) {
    return kMSStatusLiteParamInvalid;
  }
  size_t inputs_num = inputs_handle.handle_num;
  void *inputs_binbuf[inputs_num];
  int inputs_size[inputs_num];
  for (size_t i = 0; i < inputs_num; ++i) {
    MSTensorHandle tensor = inputs_handle.handle_list[i];
    inputs_binbuf[i] = NULL;
    inputs_size[i] = (int)MSTensorGetDataSize(tensor);
  }
  char *input_paths = strdup(input_files);
  if (input_paths == NULL) {
    return kMSStatusLiteError;
  }
  int ret = ReadInputsFile(input_paths, inputs_binbuf, inputs_size, (int)inputs_num);
  free(input_paths);
  if (ret != 0) {
    return ret;
  }
  for (size_t i = 0; i < inputs_num; ++i) {
    void *input_data = MSTensorGetMutableData(inputs_handle.handle_list[i]);
    memcpy(input_data, inputs_binbuf[i], inputs_size[i]);
    free(inputs_binbuf[i]);
    inputs_binbuf[i] = NULL;
  }
  return kMSStatusSuccess;
}

char *FirstInputFile(const char *input_files) {
  if (input_files == NULL) {
    return NULL;
  }
  char *first_input = strdup(input_files);
  if (first_input == NULL) {
    return NULL;
  }
  char *comma = strchr(first_input, ',');
  if (comma != NULL) {
    *comma = '\0';
  }
  return first_input;
}

char *InputFilesSegment(const char *input_files, size_t start, size_t count) {
  if (input_files == NULL) {
    return NULL;
  }
  char *paths = strdup(input_files);
  if (paths == NULL) {
    return NULL;
  }
  char *segment = malloc(strlen(input_files) + 1);
  if (segment == NULL) {
    free(paths);
    return NULL;
  }
  segment[0] = '\0';
  char *save_ptr = NULL;
  char *token = strtok_r(paths, ",", &save_ptr);
  size_t index = 0;
  size_t copied = 0;
  while (token != NULL) {
    if (index >= start && copied < count) {
      if (copied > 0) {
        strcat(segment, ",");
      }
      strcat(segment, token);
      copied++;
    }
    index++;
    token = strtok_r(NULL, ",", &save_ptr);
  }
  free(paths);
  if (copied != count) {
    free(segment);
    return NULL;
  }
  return segment;
}

int LoadTrainInputAndLabelFiles(MSTensorHandleArray inputs_handle, MSTensorHandleArray labels_handle,
                                const char *input_files) {
  char *input_segment = InputFilesSegment(input_files, 0, inputs_handle.handle_num);
  if (input_segment == NULL) {
    printf("get train input files failed\n");
    return kMSStatusLiteParamInvalid;
  }
  int ret = LoadInputFiles(inputs_handle, input_segment);
  free(input_segment);
  if (ret != kMSStatusSuccess) {
    return ret;
  }
  char *label_segment = InputFilesSegment(input_files, inputs_handle.handle_num, labels_handle.handle_num);
  if (label_segment == NULL) {
    printf("get train label files failed\n");
    return kMSStatusLiteParamInvalid;
  }
  ret = LoadInputFiles(labels_handle, label_segment);
  free(label_segment);
  return ret;
}

int main(int argc, const char **argv) {
  if (argc < 2) {
    printf("input command is invalid\n");
    usage();
    return kMSStatusLiteError;
  }
  printf("=======run benchmark_train======\n");

  int thread_num = kDefaultThreadNum;
  if (argc >= 5) {
    thread_num = atoi(argv[4]);
    if (thread_num < 1 || thread_num > kMaxThreadNum) {
      printf("Thread number error! It should be greater than 0 and less than 5\n");
      return kMSStatusLiteParamInvalid;
    }
  }
  int bind_mode = kDefaultBindMode;
  if (argc >= 6) {
    bind_mode = atoi(argv[5]);
    if (bind_mode < 0 || bind_mode > 2) {
      printf("Thread bind mode error! 0: No bind, 1: Bind high cpu, 2: Bind mid cpu.\n");
      return kMSStatusLiteParamInvalid;
    }
  }
  MSContextHandle ms_context_handle = MSContextCreate();
  if (ms_context_handle == NULL) {
    printf("MSContextCreate failed\n");
    return kMSStatusLiteNullptr;
  }
  MSContextSetThreadNum(ms_context_handle, thread_num);
  MSContextSetThreadAffinityMode(ms_context_handle, bind_mode);
  printf("context: ThreadNum: %d, BindMode: %d\n", thread_num, bind_mode);

  MSModelHandle model_handle = MSModelCreate();
  // Micro train models use generated built-in weights. External model data is not passed through benchmark_train.
  int ret = MSModelBuild(model_handle, NULL, 0, kMSModelTypeMindIR, ms_context_handle);
  MSContextDestroy(&ms_context_handle);
  if (ret != kMSStatusSuccess) {
    printf("MSModelBuild failed, ret: %d\n", ret);
    return ret;
  }
  // Build only creates the RAM weight area. Load built-in default weights before predict/train/eval.
  ret = MSModelLoadWeight(model_handle, NULL);
  if (ret != kMSStatusSuccess) {
    MSModelDestroy(&model_handle);
    printf("MSModelLoadWeight failed, ret: %d\n", ret);
    return ret;
  }
  // set model inputs tensor data
  MSTensorHandleArray inputs_handle = MSModelGetInputs(model_handle);
  if (inputs_handle.handle_list == NULL) {
    printf("MSModelGetInputs failed, ret: %d", ret);
    return ret;
  }
  MSTensorHandleArray labels_handle = MSModelGetLabels(model_handle);
  if (labels_handle.handle_list == NULL) {
    printf("MSModelGetLabels failed, ret: %d", ret);
    return ret;
  }

  MSTensorHandleArray outputs_handle = MSModelGetOutputs(model_handle);
  if (!outputs_handle.handle_list) {
    printf("MSModelGetOutputs failed, ret: %d", ret);
    return ret;
  }

  MSTensorHandleArray predict_inputs_handle = {1, inputs_handle.handle_list};
  if (outputs_handle.handle_num < 2) {
    MSModelDestroy(&model_handle);
    printf("train outputs should contain graph outputs and one trailing loss tensor\n");
    return kMSStatusLiteError;
  }
  // Predict is pure inference, so it uses only graph outputs and hides the trailing loss tensor.
  MSTensorHandleArray predict_outputs_handle = {outputs_handle.handle_num - 1, outputs_handle.handle_list};
  char *predict_input_files = FirstInputFile(argv[1]);
  if (predict_input_files == NULL) {
    MSModelDestroy(&model_handle);
    printf("get predict input file failed\n");
    return kMSStatusLiteError;
  }
  ret = LoadInputFiles(predict_inputs_handle, predict_input_files);
  free(predict_input_files);
  if (ret != kMSStatusSuccess) {
    MSModelDestroy(&model_handle);
    return ret;
  }

  ret = MSModelPredict(model_handle, predict_inputs_handle, &predict_outputs_handle, NULL, NULL);
  if (ret != kMSStatusSuccess) {
    MSModelDestroy(&model_handle);
    printf("MSModelPredict failed, ret: %d", ret);
    return ret;
  }
  printf("\n========run predict mode success=======\n");
  printf("outputs: \n");
  for (size_t i = 0; i < predict_outputs_handle.handle_num; i++) {
    MSTensorHandle output = predict_outputs_handle.handle_list[i];
    PrintTensorHandle(output);
  }

  ret = LoadTrainInputAndLabelFiles(inputs_handle, labels_handle, argv[1]);
  if (ret != kMSStatusSuccess) {
    MSModelDestroy(&model_handle);
    return ret;
  }

  ret = MSModelSetTrainMode(model_handle, true);
  if (ret != kMSStatusSuccess) {
    MSModelDestroy(&model_handle);
    printf("MSModelSetTrainMode failed, ret: %d", ret);
    return ret;
  }

  int warm_up_loop_count = 0;
  if (argc >= 4) {
    warm_up_loop_count = atoi(argv[3]);
    if (warm_up_loop_count < 0) {
      printf("The warm up loop count error! Cannot be less than 0.\n");
      return kMSStatusLiteParamInvalid;
    }
  }
  printf("Running warm up loops...\n");
  for (int i = 0; i < warm_up_loop_count; ++i) {
    ret = MSModelRunStep(model_handle, NULL, NULL);
    if (ret != kMSStatusSuccess) {
      MSModelDestroy(&model_handle);
      printf("MSModelRunStep failed, ret: %d", ret);
      return ret;
    }
  }

  int loop_count = 1;
  if (argc >= 3) {
    loop_count = atoi(argv[2]);
    if (loop_count < 1) {
      printf("The train loop count error! It should be greater than 0.\n");
      return kMSStatusLiteParamInvalid;
    }
  }
  printf("\ntrain loop count: %d\n", loop_count);
  uint64_t start_time = GetTimeUs();
  for (int i = 0; i < loop_count; ++i) {
    ret = MSModelRunStep(model_handle, NULL, NULL);
    if (ret != kMSStatusSuccess) {
      MSModelDestroy(&model_handle);
      printf("MSModelRunStep failed, ret: %d", ret);
      return ret;
    }
  }
  uint64_t end_time = GetTimeUs();
  float total_time = (float)(end_time - start_time) / 1000.0f;
  printf("total train time: %.5fms, per step: %.5fms\n", total_time, total_time / loop_count);
  printf("========run train mode success=======\n");
  printf("outputs: \n");
  for (size_t i = 0; i < outputs_handle.handle_num; i++) {
    MSTensorHandle output = outputs_handle.handle_list[i];
    PrintTensorHandle(output);
  }

  ret = MSModelSetTrainMode(model_handle, false);
  if (ret != kMSStatusSuccess) {
    MSModelDestroy(&model_handle);
    printf("MSModelSetTrainMode failed, ret: %d", ret);
    return ret;
  }
  if (argc >= 7) {
    // Eval can use a separate test sample after the train step updates weights.
    ret = LoadTrainInputAndLabelFiles(inputs_handle, labels_handle, argv[6]);
    if (ret != kMSStatusSuccess) {
      MSModelDestroy(&model_handle);
      return ret;
    }
  }
  ret = MSModelRunStep(model_handle, NULL, NULL);
  if (ret != kMSStatusSuccess) {
    MSModelDestroy(&model_handle);
    printf("MSModelRunStep failed, ret: %d", ret);
    return ret;
  }
  printf("\n========run eval mode success=======\n");
  printf("outputs: \n");
  for (size_t i = 0; i < outputs_handle.handle_num; i++) {
    MSTensorHandle output = outputs_handle.handle_list[i];
    PrintTensorHandle(output);
  }

  if (argc >= 8) {
    CalibTensor *calib_tensors;
    int calib_num = 0;
    ret = ReadCalibData(argv[7], &calib_tensors, &calib_num);
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

  ret = MSModelExportWeight(model_handle, "./export.bin");
  if (ret != kMSStatusSuccess) {
    MSModelDestroy(&model_handle);
    printf("MSModelExportWeight failed, ret: %d", ret);
    return ret;
  }
  printf("========export weight success=======\n");

  printf("========run success=======\n");
  MSModelDestroy(&model_handle);
  return kMSStatusSuccess;
}
)RAW";

const char *GetBenchmarkTrainSource() { return benchmark_train_source; }
}  // namespace mindspore::lite::micro
