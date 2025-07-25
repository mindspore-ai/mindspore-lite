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

#include <algorithm>
#include <random>
#include <iostream>
#include <fstream>
#include <cstring>
#include <memory>
#include "include/api/model.h"
#include "include/api/context.h"
#include "include/api/status.h"
#include "include/api/types.h"

#ifdef ENABLE_ASCEND
#include "./mem_ascend.h"
#else
#include "./mem_gpu.h"
#endif
bool g_set_data = true;
std::vector<std::vector<uint8_t>> g_cmp_data;

static std::string ShapeToString(const std::vector<int64_t> &shape) {
  std::string result = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    result += std::to_string(shape[i]);
    if (i + 1 < shape.size()) {
      result += ", ";
    }
  }
  result += "]";
  return result;
}

template <typename T, typename Distribution>
void GenerateRandomData(int size, void *data, Distribution distribution) {
  std::random_device rd{};
  std::mt19937 random_engine{rd()};
  int elements_num = size / sizeof(T);
  (void)std::generate_n(static_cast<T *>(data), elements_num,
                        [&distribution, &random_engine]() { return static_cast<T>(distribution(random_engine)); });
}

int GenerateRandomInputData(std::vector<mindspore::MSTensor> inputs, std::vector<uint8_t *> *host_data_buffer) {
  for (auto tensor : inputs) {
    auto data_size = tensor.DataSize();
    if (data_size == 0) {
      std::cerr << "Data size cannot be 0, tensor shape: " << ShapeToString(tensor.Shape()) << std::endl;
      return -1;
    }
    auto host_data = new uint8_t[data_size];
    host_data_buffer->push_back(host_data);
    GenerateRandomData<float>(data_size, host_data, std::normal_distribution<float>(0.0f, 1.0f));
  }
  return 0;
}

int SetHostData(std::vector<mindspore::MSTensor> tensors, const std::vector<uint8_t *> &host_data_buffer) {
  for (size_t i = 0; i < tensors.size(); i++) {
    tensors[i].SetData(host_data_buffer[i], false);
    tensors[i].SetDeviceData(nullptr);
  }
  return 0;
}

int GetDeviceId() {
#ifdef ENABLE_ASCEND
  uint32_t device_id = 0;
  auto device_id_env = std::getenv("ASCEND_DEVICE_ID");
  if (device_id_env != nullptr) {
    try {
      device_id = static_cast<uint32_t>(std::stoul(device_id_env));
    } catch (std::invalid_argument &e) {
      std::cerr << "Invalid device id env:" << device_id_env << ". Set default device id 0.";
    }
    std::cerr << "Ascend device_id = " << device_id;
  }
#else
  uint32_t = device_id = 0;
#endif
  return device_id;
}

int SetDeviceData(std::vector<mindspore::MSTensor> tensors, const std::vector<uint8_t *> &host_data_buffer,
                  std::vector<mindspore::MSTensor *> *device_buffers) {
  uint32_t device_id = GetDeviceId();
  for (size_t i = 0; i < tensors.size(); i++) {
    auto &tensor = tensors[i];
    auto host_data = host_data_buffer[i];
    auto data_size = tensor.DataSize();
    if (data_size == 0) {
      std::cerr << "Data size cannot be 0, tensor shape: " << ShapeToString(tensor.Shape()) << std::endl;
      return -1;
    }
    auto device_data = mindspore::MSTensor::CreateTensor(tensor.Name(), tensor.DataType(), tensor.Shape(), host_data,
                                                         data_size, "ascend", device_id);
    device_buffers->push_back(device_data);
    tensors[i] = *device_data;
    tensors[i].SetData(nullptr, false);
  }
  return 0;
}

int SetOutputHostData(std::vector<mindspore::MSTensor> tensors, std::vector<uint8_t *> *host_buffers) {
  for (size_t i = 0; i < tensors.size(); i++) {
    auto &tensor = tensors[i];
    auto data_size = tensor.DataSize();
    if (data_size == 0) {
      std::cerr << "Data size cannot be 0, tensor shape: " << ShapeToString(tensor.Shape()) << std::endl;
      return -1;
    }
    auto host_data = new uint8_t[data_size];
    host_buffers->push_back(host_data);
    tensor.SetData(host_data, false);
    tensor.SetDeviceData(nullptr);
  }
  return 0;
}

int SetOutputDeviceData(std::vector<mindspore::MSTensor> tensors, std::vector<mindspore::MSTensor *> *device_buffers) {
  uint32_t device_id = GetDeviceId();
  for (size_t i = 0; i < tensors.size(); i++) {
    auto &tensor = tensors[i];
    auto data_size = tensor.DataSize();
    if (data_size == 0) {
      std::cerr << "Data size cannot be 0, tensor shape: " << ShapeToString(tensor.Shape()) << std::endl;
      return -1;
    }
    char host_data[data_size] = {0};
    auto device_data = mindspore::MSTensor::CreateTensor(tensor.Name(), tensor.DataType(), tensor.Shape(), host_data,
                                                         data_size, "ascend", device_id);
    tensors[i] = *device_data;
    tensor.SetData(nullptr, false);
  }
  return 0;
}

template <class T>
void PrintBuffer(const void *buffer, size_t elem_count) {
  auto data = reinterpret_cast<const T *>(buffer);
  constexpr size_t max_print_count = 50;
  for (size_t i = 0; i < elem_count && i <= max_print_count; i++) {
    std::cout << data[i] << " ";
  }
  std::cout << std::endl;
}

void PrintInfosByDataType(mindspore::DataType data_type, const void *print_data, size_t elem_num) {
  if (data_type == mindspore::DataType::kNumberTypeFloat32) {
    PrintBuffer<float>(print_data, elem_num);
  } else if (data_type == mindspore::DataType::kNumberTypeFloat64) {
    PrintBuffer<double>(print_data, elem_num);
  } else if (data_type == mindspore::DataType::kNumberTypeInt64) {
    PrintBuffer<int64_t>(print_data, elem_num);
  } else if (data_type == mindspore::DataType::kNumberTypeInt32) {
    PrintBuffer<int32_t>(print_data, elem_num);
  } else if (data_type == mindspore::DataType::kNumberTypeInt16) {
    PrintBuffer<int16_t>(print_data, elem_num);
  } else if (data_type == mindspore::DataType::kNumberTypeInt8) {
    PrintBuffer<int8_t>(print_data, elem_num);
  } else if (data_type == mindspore::DataType::kNumberTypeUInt64) {
    PrintBuffer<uint64_t>(print_data, elem_num);
  } else if (data_type == mindspore::DataType::kNumberTypeUInt32) {
    PrintBuffer<uint32_t>(print_data, elem_num);
  } else if (data_type == mindspore::DataType::kNumberTypeUInt16) {
    PrintBuffer<uint16_t>(print_data, elem_num);
  } else if (data_type == mindspore::DataType::kNumberTypeUInt8) {
    PrintBuffer<uint8_t>(print_data, elem_num);
  } else if (data_type == mindspore::DataType::kNumberTypeBool) {
    PrintBuffer<bool>(print_data, elem_num);
  } else {
    std::cout << "Unsupported data type " << static_cast<int>(data_type) << std::endl;
  }
}

bool PrintOutputsTensor(std::vector<mindspore::MSTensor> outputs) {
  if (g_set_data) {
    g_cmp_data.clear();
  } else {
    if (g_cmp_data.size() != outputs.size()) {
      std::cout << "Output size " << outputs.size() << " != output size last time " << g_cmp_data.size() << std::endl;
      return false;
    }
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    auto &tensor = outputs[i];
    auto elem_num = tensor.ElementNum();
    auto data_size = tensor.DataSize();
    std::vector<uint8_t> host_data;
    const void *print_data;
    if (tensor.GetDeviceData() != nullptr) {
      host_data.resize(data_size);
      CopyMemoryDevice2Host(host_data.data(), host_data.size(), tensor.GetDeviceData(), data_size);
      print_data = host_data.data();
      std::cout << "Device data, tensor name is:" << tensor.Name() << " tensor size is:" << data_size
                << " tensor elements num is:" << elem_num << std::endl;
    } else {
      print_data = tensor.Data().get();
      std::cout << "Host data, tensor name is:" << tensor.Name() << " tensor size is:" << data_size
                << " tensor elements num is:" << elem_num << std::endl;
    }
    if (print_data == nullptr) {
      std::cerr << "Invalid output data" << std::endl;
      return false;
    }
    auto data_type = tensor.DataType();
    PrintInfosByDataType(data_type, print_data, elem_num);
    if (g_set_data) {
      if (host_data.empty()) {
        host_data.resize(data_size);
        memcpy(host_data.data(), print_data, host_data.size());
      }
      g_cmp_data.emplace_back(std::move(host_data));
    } else {
      auto &cmp_data = g_cmp_data[i];
      if (cmp_data.size() != data_size) {
        std::cout << "Output " << i << " data size " << data_size << " != data size last time " << cmp_data.size()
                  << std::endl;
        return false;
      }
      auto host_uint8 = reinterpret_cast<const uint8_t *>(print_data);
      for (size_t k = 0; k < cmp_data.size(); k++) {
        if (cmp_data[k] != host_uint8[k]) {
          std::cout << "Output " << i << " data as uint8_t " << (uint32_t)host_uint8[k] << " != that last time "
                    << (uint32_t)host_uint8[k] << std::endl;
          return false;
        }
      }
    }
  }
  return true;
}

int Predict(mindspore::Model *model, const std::vector<mindspore::MSTensor> &inputs,
            std::vector<mindspore::MSTensor> *outputs) {
  auto ret = model->Predict(inputs, outputs);
  if (ret != mindspore::kSuccess) {
    std::cerr << "Predict error " << ret << std::endl;
    return -1;
  }
  if (!PrintOutputsTensor(*outputs)) {
    return -1;
  }
  return 0;
}

class ResourceGuard {
 public:
  explicit ResourceGuard(std::function<void()> rel_func) : rel_func_(rel_func) {}
  ~ResourceGuard() {
    if (rel_func_) {
      rel_func_();
    }
  }

 private:
  std::function<void()> rel_func_ = nullptr;
};

int TestHostDeviceInput(mindspore::Model *model, uint32_t batch_size) {
  // Get Input
  auto inputs = model->GetInputs();
  std::vector<std::vector<int64_t>> input_shapes;
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(input_shapes), [batch_size](auto &item) {
    auto shape = item.Shape();
    shape[0] = batch_size;
    return shape;
  });
  if (model->Resize(inputs, input_shapes) != mindspore::kSuccess) {
    std::cerr << "Failed to resize model batch size to " << batch_size << std::endl;
    return -1;
  }
  std::cout << "Success resize model batch size to " << batch_size << std::endl;

  // Generate random data as input data.
  std::vector<uint8_t *> host_buffers;
  ResourceGuard host_rel([&host_buffers]() {
    for (auto &item : host_buffers) {
      delete[] item;
    }
  });

  auto ret = GenerateRandomInputData(inputs, &host_buffers);
  if (ret != 0) {
    std::cerr << "Generate Random Input Data failed." << std::endl;
    return -1;
  }
  // empty outputs
  std::vector<mindspore::MSTensor> outputs;
  // Model Predict, input host memory
  SetHostData(inputs, host_buffers);
  g_set_data = true;
  if (Predict(model, inputs, &outputs) != 0) {
    return -1;
  }
  g_set_data = false;  // compare data next time
  // Model Predict, input device memory
  outputs.clear();
  std::vector<mindspore::MSTensor *> device_buffers;
  ResourceGuard device_rel([&device_buffers]() {
    for (auto &item : device_buffers) {
      delete item;
    }
  });
  SetDeviceData(inputs, host_buffers, &device_buffers);
  if (Predict(model, inputs, &outputs) != 0) {
    return -1;
  }
  return 0;
}

int TestHostDeviceOutput(mindspore::Model *model, uint32_t batch_size) {
  // Get Input
  auto inputs = model->GetInputs();
  std::vector<std::vector<int64_t>> input_shapes;
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(input_shapes), [batch_size](auto &item) {
    auto shape = item.Shape();
    shape[0] = batch_size;
    return shape;
  });
  if (model->Resize(inputs, input_shapes) != mindspore::kSuccess) {
    std::cerr << "Failed to resize model batch size to " << batch_size << std::endl;
    return -1;
  }
  std::cout << "Success resize model batch size to " << batch_size << std::endl;

  // Generate random data as input data.
  std::vector<uint8_t *> host_buffers;
  ResourceGuard host_rel([&host_buffers]() {
    for (auto &item : host_buffers) {
      delete[] item;
    }
  });

  auto ret = GenerateRandomInputData(inputs, &host_buffers);
  if (ret != 0) {
    std::cerr << "Generate Random Input Data failed." << std::endl;
    return -1;
  }
  // Get Output from model
  auto outputs = model->GetOutputs();
  // ---------------------- output host data
  std::vector<uint8_t *> output_host_buffers;
  ResourceGuard output_host_rel([&output_host_buffers]() {
    for (auto &item : output_host_buffers) {
      delete[] item;
    }
  });
  if (SetOutputHostData(outputs, &output_host_buffers) != 0) {
    std::cerr << "Failed to set output host data" << std::endl;
    return -1;
  }
  // Model Predict, input host memory
  SetHostData(inputs, host_buffers);
  g_set_data = true;
  if (Predict(model, inputs, &outputs) != 0) {
    return -1;
  }
  g_set_data = false;  // compare data next time
  std::vector<mindspore::MSTensor *> device_buffers;
  ResourceGuard device_rel([&device_buffers]() {
    for (auto &item : device_buffers) {
      delete item;
    }
  });
  // Model Predict, input device memory
  if (SetDeviceData(inputs, host_buffers, &device_buffers) != 0) {
    std::cerr << "Failed to set input device data" << std::endl;
    return -1;
  }
  if (Predict(model, inputs, &outputs) != 0) {
    return -1;
  }
  // ---------------------- output device data
  std::vector<mindspore::MSTensor *> output_device_buffers;
  ResourceGuard output_device_rel([&output_device_buffers]() {
    for (auto &item : output_device_buffers) {
      delete item;
    }
  });
  if (SetOutputDeviceData(outputs, &output_device_buffers) != 0) {
    std::cerr << "Failed to set output device data" << std::endl;
    return -1;
  }
  // Model Predict, input host memory
  SetHostData(inputs, host_buffers);
  if (Predict(model, inputs, &outputs) != 0) {
    return -1;
  }
  // Model Predict, input device memory
  if (SetDeviceData(inputs, host_buffers, &device_buffers) != 0) {
    std::cerr << "Failed to set input device data" << std::endl;
    return -1;
  }
  if (Predict(model, inputs, &outputs) != 0) {
    return -1;
  }
  return 0;
}

int QuickStart(int argc, const char **argv) {
  if (argc < 2) {
    std::cerr << "Model file must be provided.\n";
    return -1;
  }
  // Read model file.
  std::string model_path = argv[1];
  if (model_path.empty()) {
    std::cerr << "Model path " << model_path << " is invalid.";
    return -1;
  }
  std::cerr << "model path:" << model_path << std::endl;
  // Create and init context, add CPU device info
  auto context = std::make_shared<mindspore::Context>();
  if (context == nullptr) {
    std::cerr << "New context failed." << std::endl;
    return -1;
  }
  auto &device_list = context->MutableDeviceInfo();

#ifdef ENABLE_ASCEND
  uint32_t device_id = 0;
  auto device_info = std::make_shared<mindspore::AscendDeviceInfo>();
  auto device_id_env = std::getenv("ASCEND_DEVICE_ID");
  if (device_id_env != nullptr) {
    try {
      device_id = static_cast<uint32_t>(std::stoul(device_id_env));
    } catch (std::invalid_argument &e) {
      std::cerr << "Invalid device id env:" << device_id_env << ". Set default device id 0.";
    }
    std::cerr << "Ascend device_id = " << device_id << std::endl;
  }
  device_info->SetDeviceID(device_id);
#else
  auto device_info = std::make_shared<mindspore::GPUDeviceInfo>();
  device_info->SetDeviceID(0);
#endif
  if (device_info == nullptr) {
    std::cerr << "New CPUDeviceInfo failed." << std::endl;
    return -1;
  }
  device_list.push_back(device_info);

  mindspore::Model model;
  // Build model
  auto build_ret = model.Build(model_path, mindspore::kMindIR, context);
  if (build_ret != mindspore::kSuccess) {
    std::cerr << "Build model error " << build_ret << std::endl;
    return -1;
  }
  TestHostDeviceInput(&model, 1);
  TestHostDeviceOutput(&model, 1);
  mindspore::Status finalize_ret = model.Finalize();
  if (!finalize_ret == mindspore::kSuccess) {
    std::cerr << "finalize executed success.";
  }
  return 0;
}

int main(int argc, const char **argv) { return QuickStart(argc, argv); }
