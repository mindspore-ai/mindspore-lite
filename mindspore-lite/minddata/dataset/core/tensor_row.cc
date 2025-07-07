/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "mindspore-lite/minddata/dataset/core/tensor_row.h"

#include <iomanip>
#include <utility>

#include "mindspore-lite/minddata/dataset/core/config_manager.h"

namespace mindspore {
namespace dataset {

TensorRow::TensorRow() noexcept
    : id_(kDefaultRowId), path_({}), tensor_row_flag_(kFlagNone), timer_(std::make_shared<RowTimer>()) {}

TensorRow::TensorRow(size_type n, const TensorRow::value_type &t) noexcept
    : id_(kDefaultRowId), path_({}), row_(n, t), tensor_row_flag_(kFlagNone), timer_(std::make_shared<RowTimer>()) {}

TensorRow::TensorRow(const TensorRow::vector_type &v)
    : id_(kDefaultRowId), path_({}), row_(v), tensor_row_flag_(kFlagNone), timer_(std::make_shared<RowTimer>()) {}

TensorRow::TensorRow(row_id_type id, const std::initializer_list<value_type> &lst)
    : id_(id), path_({}), row_(lst), tensor_row_flag_(kFlagNone), timer_(std::make_shared<RowTimer>()) {}

TensorRow::TensorRow(const TensorRow &tr)
    : id_(tr.id_), path_(tr.path_), row_(tr.row_), tensor_row_flag_(tr.tensor_row_flag_), timer_(tr.timer_) {}

TensorRow::TensorRow(TensorRow::TensorRowFlags flag)
    : id_(kDefaultRowId), path_({}), tensor_row_flag_(flag), timer_(std::make_shared<RowTimer>()) {}

TensorRow &TensorRow::operator=(const TensorRow &tr) {
  if (this == &tr) {
    return *this;
  }
  row_ = tr.row_;
  id_ = tr.id_;
  path_ = tr.path_;
  tensor_row_flag_ = tr.tensor_row_flag_;
  timer_ = tr.timer_;
  return *this;
}

Status TensorRow::Clone(TensorRow *new_tr) const {
  RETURN_UNEXPECTED_IF_NULL(new_tr);
  new_tr->row_.clear();
  for (const std::shared_ptr<Tensor> &s : row_) {
    std::shared_ptr<Tensor> d;
    RETURN_IF_NOT_OK(Tensor::CreateFromTensor(s, &d));
    (void)new_tr->row_.emplace_back(std::move(d));
  }
  new_tr->id_ = id_;
  new_tr->path_ = path_;
  new_tr->tensor_row_flag_ = tensor_row_flag_;
  new_tr->timer_ = timer_;
  return Status::OK();
}

TensorRow &TensorRow::operator=(const std::initializer_list<TensorRow::value_type> &lst) {
  row_ = lst;
  tensor_row_flag_ = kFlagNone;
  return *this;
}

TensorRow::TensorRow(TensorRow::vector_type &&v) noexcept
    : id_(kDefaultRowId),
      path_({}),
      row_(std::move(v)),
      tensor_row_flag_(kFlagNone),
      timer_(std::make_shared<RowTimer>()) {}

TensorRow::TensorRow(row_id_type id, std::initializer_list<value_type> &&lst) noexcept
    : id_(id), path_({}), row_(std::move(lst)), tensor_row_flag_(kFlagNone), timer_(std::make_shared<RowTimer>()) {}

TensorRow::TensorRow(TensorRow &&tr) noexcept {
  id_ = tr.id_;
  path_ = std::move(tr.path_);
  row_ = std::move(tr.row_);
  tensor_row_flag_ = tr.tensor_row_flag_;
  timer_ = std::move(tr.timer_);
}

TensorRow &TensorRow::operator=(TensorRow &&tr) noexcept {
  if (this == &tr) {
    return *this;
  }
  row_ = std::move(tr.row_);
  id_ = tr.id_;
  tr.id_ = kDefaultRowId;
  path_ = std::move(tr.path_);
  tensor_row_flag_ = tr.tensor_row_flag_;
  timer_ = std::move(tr.timer_);
  return *this;
}

TensorRow &TensorRow::operator=(std::initializer_list<TensorRow::value_type> &&lst) noexcept {
  row_ = std::move(lst);
  tensor_row_flag_ = kFlagNone;
  return *this;
}

Status TensorRow::ValidateTensorRow(const TensorRow &input, const DataType &data_type) {
  if (data_type == DataType::DE_UNKNOWN) {
    RETURN_STATUS_UNEXPECTED("ConvertFromTensorRow: Data type was not recognized.");
  }
  if (data_type.IsString()) {
    RETURN_STATUS_UNEXPECTED("ConvertFromTensorRow: Data type string and bytes are not supported.");
  }
  if (input.size() != 1) {
    RETURN_STATUS_UNEXPECTED("ConvertFromTensorRow: The input TensorRow must have exactly one tensor.");
  }
  return Status::OK();
}

// TODO(ly): need support in independent mode
void TensorRow::CopyTimerTo(TensorRow *out) const { out->timer_ = timer_; }

void TensorRow::TimerRecord(const std::string &op_name, const std::string &info_name,
                            const std::vector<double> &duration, TensorRow *copy_from) {
  if (!timer_->Enabled()) {
    return;
  }

  // since some op(map/batch/zip) will create new TensorRow, need to copy original info
  if (copy_from) {
    copy_from->CopyTimerTo(this);
  }
  timer_->Record(op_name, info_name, duration);
}

const char RowTimer::kMaxTime[] = " WorkerTimeMax";
const char RowTimer::kMinTime[] = " WorkerTimeMin";
const char RowTimer::kWorkerTime[] = "WorkerTime";
const char RowTimer::kIOTime[] = "IOTime";
const char RowTimer::kMaxIOTime[] = "IOTimeMax";
const char RowTimer::kMinIOTime[] = "IOTimeMin";
const char RowTimer::kLoadTensorTime[] = "LoadTensorTime";
const char RowTimer::kThroughputTime[] = " PipelineTime";
const char RowTimer::kPushToDeviceTime[] = " PushToAscendTime";
const char RowTimer::kRowCount[] = "kRowCount";

bool RowTimer::Enabled() {
#ifndef ENABLE_ANDROID
  return IS_VLOG_ON(VL_MD);
#else
  return false;
#endif
}

void RowTimer::Record(const std::string &op_name, const std::string &info_name, const std::vector<double> &duration) {
  time_table_[op_name][info_name] = std::move(duration);
  if (std::find(op_order_.begin(), op_order_.end(), op_name) == op_order_.end()) {
    op_order_.push_back(op_name);
  }
}

std::string RowTimer::Summary(const std::vector<std::string> &specified_op) {
  std::stringstream ss;
  std::string title = "\n[PROF Dataset]\n";
  std::string title2 =
    "worker_time: The worker time in OP, include max, min, avg; "
    "io_time: The I/O time of in OP, which is part of worker_time; "
    "throughput_time: The overall throughput time of the data pipeline; "
    "push_device_time: The time taken for the host to push data to the device.\n";
  ss << title << title2;
  for (auto &op : op_order_) {
    // only print specified op
    if (!specified_op.empty()) {
      auto it = std::find_if(specified_op.begin(), specified_op.end(),
                             [&](const std::string &str) { return op.find(str) != std::string::npos; });
      if (it == specified_op.end()) {
        continue;
      }
    }
    ss << "-- " << std::left << std::setw(20) << op << ": ";

    // has worker time
    if (time_table_[op].find(RowTimer::kWorkerTime) != time_table_[op].end()) {
      ss << "worker_time (";
      ss << "avg:" << std::fixed << std::setprecision(2) << time_table_[op][RowTimer::kWorkerTime].front() << "ms, ";
      if (time_table_[op].find(RowTimer::kMaxTime) != time_table_[op].end()) {
        ss << "max:" << std::fixed << std::setprecision(2) << time_table_[op][RowTimer::kMaxTime].front() << "ms, ";
      } else {
        ss << "max: /, ";
      }
      if (time_table_[op].find(RowTimer::kMinTime) != time_table_[op].end()) {
        ss << "min:" << std::fixed << std::setprecision(2) << time_table_[op][RowTimer::kMinTime].front() << "ms, ";
      } else {
        ss << "min: /, ";
      }
      if (time_table_[op].find(RowTimer::kRowCount) != time_table_[op].end()) {
        ss << "count:" << std::fixed << std::setprecision(0) << time_table_[op][RowTimer::kRowCount].front() << ")";
      } else {
        ss << "count:" << 1 << ") ";
      }
    }

    // has io time
    if (time_table_[op].find(RowTimer::kIOTime) != time_table_[op].end()) {
      ss << ", io_time (";
      ss << "avg:" << std::fixed << std::setprecision(2) << time_table_[op][RowTimer::kIOTime].front() << "ms, ";
      if (time_table_[op].find(RowTimer::kMaxIOTime) != time_table_[op].end()) {
        ss << "max:" << std::fixed << std::setprecision(2) << time_table_[op][RowTimer::kMaxIOTime].front() << "ms, ";
      } else {
        ss << "max: /, ";
      }
      if (time_table_[op].find(RowTimer::kMinIOTime) != time_table_[op].end()) {
        ss << "min:" << std::fixed << std::setprecision(2) << time_table_[op][RowTimer::kMinIOTime].front() << "ms, ";
      } else {
        ss << "min: /, ";
      }
      if (time_table_[op].find(RowTimer::kRowCount) != time_table_[op].end()) {
        ss << "count:" << std::fixed << std::setprecision(0) << time_table_[op][RowTimer::kRowCount].front() << ") ";
      } else {
        ss << "count:" << 1 << ") ";
      }
    }

    // has kThroughputTime time
    if (time_table_[op].find(RowTimer::kThroughputTime) != time_table_[op].end()) {
      ss << "throughput_time: " << time_table_[op][RowTimer::kThroughputTime].front() << "ms";
      // has kPushToDeviceTime time
      if (time_table_[op].find(RowTimer::kPushToDeviceTime) != time_table_[op].end()) {
        ss << ", push_device_time: " << time_table_[op][RowTimer::kPushToDeviceTime].front() << "ms";
      }
    }
    ss << "\n";
  }
  return ss.str();
}
}  // namespace dataset
}  // namespace mindspore
