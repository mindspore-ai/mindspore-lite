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
#include "tools/optimizer/common/pass_manager_extends.h"
#ifndef _MSC_VER
#include <sys/time.h>
#endif
#include <deque>
#include <string>
#include <algorithm>
#include <map>
#include <set>
#include "ir/anf.h"
#include "tools/converter/ir_dump.h"
#include "tools/optimizer/common/fusion_trace_utils.h"

namespace mindspore {
namespace opt {
namespace {
// Plan-B: the graph-diff transport is the ONLY automatic carrier. It runs for every pass
// while tracing is enabled, EXCEPT the passes below - the fan-out / split fusions that trace
// explicitly inside Process (a single replacement target cannot express their mapping) and the
// cleanup passes whose deletions nullify identity by design (records must die with the node).
const std::set<std::string> &NoDiffTransportPasses() {
  static const std::set<std::string> passes = {
    "SqueezeFusion", "AddConcatActivationFusion", "SqueezeExpandDimsFusion", "TensorDotFusion", "FlashAttentionFusion",
    "ConstFoldPass", "RemoveTransitivityOp"};
  return passes;
}

struct GraphSnapshot {
  std::vector<AnfNodePtr> order;  // topo-ordered reachable CNodes
  std::set<AnfNodePtr> cnodes;
  std::map<AnfNodePtr, std::set<AnfNodePtr>> users;   // cnode -> its user cnodes
  std::map<AnfNodePtr, std::set<AnfNodePtr>> inputs;  // cnode -> its input nodes
};

GraphSnapshot SnapshotGraph(const FuncGraphPtr &func_graph) {
  GraphSnapshot snap;
  for (const auto &cnode : func_graph->GetOrderedCnodes()) {
    snap.order.push_back(cnode);
    snap.cnodes.insert(cnode);
    for (const auto &input : cnode->inputs()) {
      if (input == nullptr) {
        continue;
      }
      snap.inputs[cnode].insert(input);
      if (input->isa<CNode>()) {
        snap.users[input].insert(cnode);
      }
    }
  }
  return snap;
}

// Unified receiver rule, resolved in REVERSE topo order with memoization: the vanished node's
// identity goes to whoever now stands in its place. h-replace: the surviving (or already
// resolved) consumers rewired to a common CNode replacement - covers tail-absorb (conv+add ->
// conv), new-node fusions (helper nodes are irrelevant) and chained collapses (a vanished
// consumer's receiver is that consumer's own replacement). h-consumer: rewired to a non-CNode
// (conv_conv bypasses to the plain input tensor) - then the surviving consumers absorbed the
// vanished node themselves. Anything else is left to the explicit calls.
void DiffCarryRecords(const GraphSnapshot &before, const FuncGraphPtr &func_graph) {
  GraphSnapshot after = SnapshotGraph(func_graph);
  std::map<AnfNodePtr, AnfNodePtr> receiver_of;
  for (auto it = before.order.rbegin(); it != before.order.rend(); ++it) {
    const auto &vanished = *it;
    if (after.cnodes.count(vanished) != 0 || !HasOriginRecords(vanished)) {
      continue;
    }
    // Consumers: surviving former users, or vanished users' already-resolved receivers.
    std::set<AnfNodePtr> consumers;
    auto uit = before.users.find(vanished);
    if (uit != before.users.end()) {
      for (const auto &user : uit->second) {
        if (after.cnodes.count(user) != 0) {
          consumers.insert(user);
        } else {
          auto mit = receiver_of.find(user);
          if (mit != receiver_of.end()) {
            consumers.insert(mit->second);
          }
        }
      }
    }
    if (consumers.empty()) {
      continue;
    }
    // The CNode replacements those consumers now consume.
    std::set<AnfNodePtr> replacements;
    for (const auto &consumer : consumers) {
      auto after_it = after.inputs.find(consumer);
      auto before_it = before.inputs.find(consumer);
      if (after_it == after.inputs.end()) {
        continue;
      }
      for (const auto &input : after_it->second) {
        bool is_new = before_it == before.inputs.end() || before_it->second.count(input) == 0;
        if (is_new && input->isa<CNode>()) {
          replacements.insert(input);
        }
      }
    }
    AnfNodePtr receiver = nullptr;
    if (replacements.size() == 1) {
      receiver = *replacements.begin();
      CarryOriginRecords(receiver->cast<CNodePtr>(), vanished);
    } else if (replacements.empty()) {
      for (const auto &consumer : consumers) {
        CarryOriginRecords(consumer->cast<CNodePtr>(), vanished);
      }
      receiver = consumers.size() == 1 ? *consumers.begin() : nullptr;
    }
    receiver_of[vanished] = receiver;
  }
}
}  // namespace
constexpr size_t kMaxRepassTimes = 12;
constexpr uint64_t kUSecondInSecond = 1000000;

PassManager::PassManager(const std::string &name, bool run_only_once)
    : name_(name), passes_{}, run_only_once_(run_only_once) {}

void PassManager::AddPass(const PassPtr &pass) {
  if (pass != nullptr) {
    passes_.push_back(pass);
  }
}

bool PassManager::RunPass(const FuncGraphPtr &func_graph, size_t pass_id, const PassPtr &pass) const {
  MS_LOG(ERROR) << "stub func";
  return false;
}

std::string PassManager::GetPassFullname(size_t pass_id, const PassPtr &pass) const {
  return std::string("hwopt_") + name() + "_" + std::to_string(pass_id) + "_" + pass->name();
}

void PassManager::DumpPassIR(const FuncGraphPtr &func_graph, const std::string &pass_fullname) const {
  MS_LOG(ERROR) << "stub func";
}

bool PassManager::Run(const FuncGraphPtr &func_graph, const std::vector<PassPtr> &passes) const {
  MS_LOG(ERROR) << "stub func";
  return false;
}

bool PassManager::Run(const FuncGraphPtr &func_graph) const {
  MS_LOG(ERROR) << "stub func";
  return false;
}

void PassManager::AddFusionPass(const PassPtr &pass, bool condition) { MS_LOG(ERROR) << "stub func"; }

void LitePassManager::AddPass(const PassPtr &pass) {
  if (pass != nullptr) {
    passes_.push_back(pass);
  }
}

bool LitePassManager::RunPass(const FuncGraphPtr &func_graph, size_t pass_id, const PassPtr &pass) const {
  bool changed = false;
#if defined(_WIN32) || defined(_WIN64)
  auto start_time = std::chrono::steady_clock::now();
#else
  struct timeval start_time {};
  struct timeval end_time {};
  (void)gettimeofday(&start_time, nullptr);
#endif
  if (pass->Run(func_graph)) {
    MS_LOG(DEBUG) << "Run pass and find change";
    changed = true;
  }
#if defined(_WIN32) || defined(_WIN64)
  auto end_time = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::ratio<1, kUSecondInSecond>> cost = end_time - start_time;
  MS_LOG(INFO) << "Run pass " << GetPassFullname(pass_id, pass) << " in " << cost.count() << " us.";
#else
  (void)gettimeofday(&end_time, nullptr);
  uint64_t cost = kUSecondInSecond * static_cast<uint64_t>(end_time.tv_sec - start_time.tv_sec);
  cost += static_cast<uint64_t>(end_time.tv_usec - start_time.tv_usec);
  MS_LOG(INFO) << "Run pass " << GetPassFullname(pass_id, pass) << " in " << cost << " us.";
#endif
  return changed;
}

std::string LitePassManager::GetPassFullname(size_t pass_id, const PassPtr &pass) const {
  return "hwopt_" + name() + "_" + std::to_string(pass_id) + "_" + pass->name();
}

bool LitePassManager::Run(const FuncGraphPtr &func_graph, const std::vector<PassPtr> &passes) const {
  if (func_graph == nullptr) {
    return false;
  }
  bool changed = false;
  size_t num = 0;
  for (const auto &pass : passes) {
    if (pass != nullptr) {
      const bool diff_transport = IsFusionTraceEnabled() && NoDiffTransportPasses().count(pass->name()) == 0;
      GraphSnapshot snapshot_before;
      if (diff_transport) {
        snapshot_before = SnapshotGraph(func_graph);
      }
      auto ret = RunPass(func_graph, num, pass);
      if (diff_transport) {
        DiffCarryRecords(snapshot_before, func_graph);
      }
      if (ret) {
        auto status = lite::DumpGraph(pass->name(), func_graph);
        if (status != kSuccess) {
          MS_LOG(WARNING) << "dump mindspore lite graph ir failed in " << pass->name();
        }
      }
      changed = ret || changed;
    } else {
      MS_LOG(INFO) << "pass is null";
    }
    num++;
  }
  return changed;
}

bool LitePassManager::Run(const FuncGraphPtr &func_graph) const {
  if (func_graph == nullptr) {
    return false;
  }
  bool changed = false;
  size_t count = 0;
  // run all passes
  bool change = true;
  while (change) {
    change = Run(func_graph, passes_);
    changed = change || changed;
    if (run_only_once_ || count > kMaxRepassTimes) {
      break;
    }
    count++;
    MS_LOG(INFO) << "Run pass counts:" << count;
  }
  return changed;
}
}  // namespace opt
}  // namespace mindspore
