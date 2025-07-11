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

#include "mindspore-lite/minddata/dataset/engine/ir/datasetops/source/iwslt2016_node.h"

#include <algorithm>
#include <utility>

#include "include/common/debug/common.h"
#include "mindspore-lite/minddata/dataset/engine/datasetops/source/iwslt_op.h"
#include "utils/file_utils.h"

namespace mindspore {
namespace dataset {
// Constructor for IWSLT2016Node.
IWSLT2016Node::IWSLT2016Node(const std::string &dataset_dir, const std::string &usage,
                             const std::vector<std::string> &language_pair, const std::string &valid_set,
                             const std::string &test_set, int64_t num_samples, ShuffleMode shuffle, int32_t num_shards,
                             int32_t shard_id, std::shared_ptr<DatasetCache> cache)
    : NonMappableSourceNode(std::move(cache)),
      dataset_dir_(dataset_dir),
      usage_(usage),
      language_pair_(language_pair),
      valid_set_(valid_set),
      test_set_(test_set),
      num_samples_(num_samples),
      shuffle_(shuffle),
      num_shards_(num_shards),
      shard_id_(shard_id) {
  // Update the num_shards_ in global context. this number is only used for now by auto_num_worker_pass.
  // User discretion is advised. Auto_num_worker_pass is currently an experimental feature which can still work
  // if the num_shards_ isn't 100% correct. The reason behind is for now, PreBuildSampler doesn't offer a way to
  // return num_shards. Once PreBuildSampler is phased out, this can be cleaned up.
  GlobalContext::config_manager()->set_num_shards_for_auto_num_workers(num_shards_);
  support_language_pair_map_["en"] = {"ar", "de", "fr", "cs"};
  support_language_pair_map_["ar"] = {"en"};
  support_language_pair_map_["fr"] = {"en"};
  support_language_pair_map_["de"] = {"en"};
  support_language_pair_map_["cs"] = {"en"};
}

std::shared_ptr<DatasetNode> IWSLT2016Node::Copy() {
  auto node = std::make_shared<IWSLT2016Node>(dataset_dir_, usage_, language_pair_, valid_set_, test_set_, num_samples_,
                                              shuffle_, num_shards_, shard_id_, cache_);
  (void)node->SetNumWorkers(num_workers_);
  (void)node->SetConnectorQueueSize(connector_que_size_);
  return node;
}

void IWSLT2016Node::Print(std::ostream &out) const {
  out << (Name() + "(cache: " + ((cache_ != nullptr) ? "true" : "false") +
          ", num_shards: " + std::to_string(num_shards_) + ", shard_id: " + std::to_string(shard_id_) + ")");
}

Status IWSLT2016Node::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  RETURN_IF_NOT_OK(ValidateDatasetDirParam("IWSLT2016Node", dataset_dir_));
  RETURN_IF_NOT_OK(ValidateStringValue("IWSLT2016Node", usage_, {"train", "valid", "test", "all"}));
  const int kLanguagePairSize = 2;
  if (language_pair_.size() != kLanguagePairSize) {
    std::string err_msg =
      "IWSLT2016Node: language_pair expecting size 2, but got: " + std::to_string(language_pair_.size());
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  RETURN_IF_NOT_OK(ValidateMapKey("IWSLT2016Node", language_pair_[0], support_language_pair_map_));
  RETURN_IF_NOT_OK(ValidateMapValue("IWSLT2016Node", language_pair_[1], support_language_pair_map_[language_pair_[0]]));
  RETURN_IF_NOT_OK(ValidateStringValue("IWSLT2016Node", valid_set_,
                                       {"dev2010", "tst2010", "tst2011", "tst2012", "tst2013", "tst2014"}));
  RETURN_IF_NOT_OK(ValidateStringValue("IWSLT2016Node", test_set_,
                                       {"dev2010", "tst2010", "tst2011", "tst2012", "tst2013", "tst2014"}));
  RETURN_IF_NOT_OK(ValidateEnum("IWSLT2016Node", "ShuffleMode", shuffle_,
                                {ShuffleMode::kFalse, ShuffleMode::kFiles, ShuffleMode::kGlobal}));

  if (num_samples_ < 0) {
    std::string err_msg = "IWSLT2016Node: Invalid number of samples: " + std::to_string(num_samples_);
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  RETURN_IF_NOT_OK(ValidateDatasetShardParams("IWSLT2016Node", num_shards_, shard_id_));
  return Status::OK();
}

Status IWSLT2016Node::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  bool shuffle_files = (shuffle_ == ShuffleMode::kGlobal || shuffle_ == ShuffleMode::kFiles);

  // Do internal Schema generation.
  auto schema = std::make_unique<DataSchema>();
  RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("text", DataType(DataType::DE_STRING), TensorImpl::kFlexible, 1)));
  RETURN_IF_NOT_OK(
    schema->AddColumn(ColDescriptor("translation", DataType(DataType::DE_STRING), TensorImpl::kFlexible, 1)));

  std::shared_ptr<IWSLTOp> iwslt_op = std::make_shared<IWSLTOp>(
    num_workers_, num_samples_, worker_connector_size_, connector_que_size_, shuffle_files, num_shards_, shard_id_,
    std::move(schema), IWSLTOp::IWSLTType::kIWSLT2016, dataset_dir_, usage_, language_pair_, valid_set_, test_set_);
  RETURN_IF_NOT_OK(iwslt_op->Init());

  // If a global shuffle is used for IWSLT, it will inject a shuffle op over the IWSLT.
  // But, if there is a cache in the tree, we do not need the global shuffle and the shuffle op should not be
  // built. This is achieved in the cache transform pass where we call MakeSimpleProducer to reset IWSLT's
  // shuffle option to false.
  if (shuffle_ == ShuffleMode::kGlobal) {
    // Inject ShuffleOp.
    std::shared_ptr<ShuffleOp> shuffle_op = nullptr;
    int64_t num_rows = 0;

    // First, get the number of rows in the dataset.
    RETURN_IF_NOT_OK(IWSLTOp::CountTotalRows(IWSLTOp::IWSLTType::kIWSLT2016, dataset_dir_, usage_, language_pair_,
                                             valid_set_, test_set_, &num_rows));
    // Add the shuffle op after this op.
    RETURN_IF_NOT_OK(
      AddShuffleOp(iwslt_op->FileNames().size(), num_shards_, num_rows, 0, connector_que_size_, &shuffle_op));
    shuffle_op->SetTotalRepeats(GetTotalRepeats());
    shuffle_op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
    shuffle_op->Skip(skip_steps_);
    node_ops->push_back(shuffle_op);
  }
  iwslt_op->SetTotalRepeats(GetTotalRepeats());
  iwslt_op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(iwslt_op);
  return Status::OK();
}

Status IWSLT2016Node::GetShardId(int32_t *shard_id) {
  *shard_id = shard_id_;
  return Status::OK();
}

Status IWSLT2016Node::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                                     int64_t *dataset_size) {
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }
  int64_t num_rows, sample_size;

  RETURN_IF_NOT_OK(IWSLTOp::CountTotalRows(IWSLTOp::IWSLTType::kIWSLT2016, dataset_dir_, usage_, language_pair_,
                                           valid_set_, test_set_, &num_rows));
  sample_size = num_samples_;
  num_rows = static_cast<int64_t>(ceil(num_rows / (1.0 * num_shards_)));
  *dataset_size = sample_size > 0 ? std::min(num_rows, sample_size) : num_rows;
  dataset_size_ = *dataset_size;
  return Status::OK();
}

Status IWSLT2016Node::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["num_parallel_workers"] = num_workers_;
  args["connector_queue_size"] = connector_que_size_;
  args["dataset_dir"] = dataset_dir_;
  args["usage"] = usage_;
  args["language_pair"] = language_pair_;
  args["valid_set"] = valid_set_;
  args["test_set"] = test_set_;
  args["num_samples"] = num_samples_;
  args["shuffle"] = shuffle_;
  args["num_shards"] = num_shards_;
  args["shard_id"] = shard_id_;
  if (cache_ != nullptr) {
    nlohmann::json cache_args;
    RETURN_IF_NOT_OK(cache_->to_json(&cache_args));
    args["cache"] = cache_args;
  }
  *out_json = args;
  return Status::OK();
}

Status IWSLT2016Node::SetupSamplerForCache(std::shared_ptr<SamplerObj> *sampler) {
  *sampler = SelectSampler(num_samples_, shuffle_, num_shards_, shard_id_);
  return Status::OK();
}

Status IWSLT2016Node::MakeSimpleProducer() {
  shard_id_ = 0;
  num_shards_ = 1;
  shuffle_ = ShuffleMode::kFalse;
  num_samples_ = 0;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
