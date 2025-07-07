/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_GENERATOR_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_GENERATOR_OP_H_

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "mindspore-lite/minddata/dataset/api/python/python_mp.h"
#include "mindspore-lite/minddata/dataset/core/data_type.h"
#include "mindspore-lite/minddata/dataset/core/tensor.h"
#include "mindspore-lite/minddata/dataset/engine/data_schema.h"
#include "mindspore-lite/minddata/dataset/engine/datasetops/pipeline_op.h"
#include "mindspore-lite/minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "mindspore-lite/minddata/dataset/util/wait_post.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace mindspore {
namespace dataset {
#ifndef _MSC_VER
#pragma GCC visibility push(hidden)
#endif

constexpr int32_t kGetItemTimeOutMilliSeconds = 60000;

class GeneratorOp : public PipelineOp, public RandomAccessOp {
 public:
  GeneratorOp(const py::function &generator_function, std::vector<std::string> column_names,
              std::vector<DataType> column_types, int32_t prefetch_size, int32_t connector_size,
              std::shared_ptr<SamplerRT> sampler, int32_t num_parallel_workers, bool has_batch_sampler);

  ~GeneratorOp() override;

  /// A print method typically used for debugging
  /// \param out - The output stream to write output to
  /// \param show_all - A bool to control if you want to show all info or just a summary
  void Print(std::ostream &out, bool show_all) const override;

  /// << Stream output operator overload
  /// \notes This allows you to write the debug print info using stream operators
  /// \param out - reference to the output stream being overloaded
  /// \param generator_op - reference to the GeneratorOp to display
  /// \return - the output stream must be returned
  friend std::ostream &operator<<(std::ostream &out, const GeneratorOp &generator_op) {
    generator_op.Print(out, false);
    return out;
  }

  /// Class functor operator () override.
  /// All DatasetOps operate by launching a thread (see ExecutionTree). This class functor will
  /// provide the master loop that drives the logic for performing the work.
  /// \return Status The status code returned
  Status operator()() override;

  /// Method of calculating sample ids
  /// @return Status The status code returned
  Status CalculatedSampleId();

  /// Get sample ids
  /// @return Status The status code returned
  Status GetMappedIndex(int64_t index, int64_t *out_index);

  /// Overrides base class reset method.  When an operator does a reset, it cleans up any state
  /// info from it's previous execution and then initializes itself so that it can be executed
  /// again.
  /// \return Status The status code returned
  Status Reset() override;

  /// Op name getter
  /// \return Name of the current Op
  std::string Name() const override { return "GeneratorOp"; }

  bool IsPython() const override { return true; }

  /// Number of parallel workers getter
  /// \return Number of parallel workers of the current Op
  int32_t NumWorkers() const override { return num_parallel_workers_; }

  /// Set the instance of python multiprocessing which will passed from python
  /// \param python_multiprocessing_runtime PythonMultiprocessingRuntime
  void SetPythonMp(std::shared_ptr<PythonMultiprocessingRuntime> python_multiprocessing_runtime);

  /// \brief In pull mode, gets the next row
  /// \param row[out] - Fetched TensorRow
  /// \return Status The status code returned
  Status GetNextRowPullMode(TensorRow *const row) override;

  /// Used by independent dataset mode to stop the subprocess
  /// \return Status The status code returned
  Status Terminate() override;

 protected:
  /// \brief Launch subprocesses in multiprocessing mode.
  /// \return Status Status code.
  Status Launch() override;

  /// \brief Gets the implementation status for operator in pull mode
  /// \return implementation status
  ImplementedPullMode PullModeImplementationStatus() const override { return ImplementedPullMode::Implemented; }

 private:
  py::function generator_function_;
  std::vector<std::string> column_names_;
  std::vector<DataType> column_types_;
  int32_t prefetch_size_;
  int64_t generator_counter_;
  int32_t num_parallel_workers_;
  int64_t num_rows_sampled_;
  bool has_batch_sampler_;
  int64_t num_samples_collected_;
  int64_t size_of_this_batch_;
  std::deque<int64_t> batch_sizes_of_epoch_;

  py::object generator_;
  std::vector<int64_t> sample_ids_;

  WaitPost wp_;

  bool prepared_data_{false};  // flag to indicate whether the data is prepared before taking for pull mode
  bool eof_received_{false};   // flag to indicate whether end of epoch signal is reached in pull mode

  std::shared_ptr<PythonMultiprocessingRuntime> python_multiprocessing_runtime_;  // python multiprocessing instance

  Status PyRowToTensorRow(py::object py_data, TensorRow *tensor_row);

  /// Private function for computing the assignment of the column name map.
  /// \return - Status
  Status ComputeColMap() override;

  /// Initialize Sampler, calls sampler->Init() within
  /// \return Status The status code returned
  Status InitSampler();

  /// Create new Generator object from the generator function
  /// \return Status The status code returned
  Status CreateGeneratorObject();

  /// Initialize GeneratorOp
  /// \return Status The status code returned
  Status Init();

  /// Check whether the target number of samples has been retrieved when eoe is hit.
  /// \return Status The status code returned
  Status CheckNumSamples() const;

  /// \brief Get the size for next batch.
  /// \return Status The status code returned.
  Status GetNextBatchSize();

  /// \brief Get the sizes of each batch for next epoch.
  /// \return Status The status code returned.
  Status GetNextEpochBatchSizes();

  /// \brief Check whether the number of samples collected and send EOB.
  /// \return Status The status code returned.
  Status CheckAndSendEOB();
};

#ifndef _MSC_VER
#pragma GCC visibility pop
#endif
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_GENERATOR_OP_H_
