// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <queue>
#include <unordered_map>
#include <vector>

#include "runners/blaze_runner/blaze_utils.hpp"
#include "runners/blaze_runner/i_pipeline_manager.hpp"

namespace tt::testing {

/**
 * Deterministic fake PipelineManager for BlazeRunner unit tests.
 *
 * The test pre-programmes allocations and outputs, then drives the runner.
 * No real tt-blaze backend is required.
 */
class FakePipelineManager : public tt::runners::IPipelineManager {
 public:
  struct OutputEntry {
    uint32_t slotId;
    uint64_t tokenId;
    bool isComplete;
  };

  struct AllocEntry {
    uint32_t requestId;
    uint32_t slotId;  // pm::INVALID_SLOT to simulate failure
  };

  // Control API --------------------------------------------------------------

  /** Pre-programme an ALLOCATE response. */
  void whenAllocate(uint32_t requestId, uint32_t slotId) {
    allocResponses_[requestId] = slotId;
  }

  /** Pre-programme a sequence of output tokens for a slot. */
  void enqueueOutputs(uint32_t slotId, const std::vector<uint64_t>& tokenIds,
                      bool finalIsComplete = true) {
    for (size_t i = 0; i < tokenIds.size(); ++i) {
      outputs_.push({slotId, tokenIds[i],
                     finalIsComplete && (i + 1 == tokenIds.size())});
    }
  }

  /** Pre-programme a single output token. */
  void enqueueOutput(uint32_t slotId, uint64_t tokenId, bool isComplete) {
    outputs_.push({slotId, tokenId, isComplete});
  }

  /** Clear all pre-programmed state. */
  void reset() {
    allocResponses_.clear();
    while (!responses_.empty()) responses_.pop();
    while (!outputs_.empty()) outputs_.pop();
    submittedSlots_.clear();
  }

  // Inspection API -----------------------------------------------------------

  bool wasSubmitted(uint32_t slotId) const {
    return submittedSlots_.count(slotId) > 0;
  }

  size_t submitCount() const { return submittedSlots_.size(); }

  // IPipelineManager implementation ------------------------------------------

  void start() override {}
  void stop() override {}

  bool push_request(const pm::ISRequest& req) override {
    switch (req.type) {
      case pm::RequestType::ALLOCATE: {
        auto it = allocResponses_.find(req.request_id);
        uint32_t slot = (it != allocResponses_.end())
                            ? it->second
                            : pm::INVALID_SLOT;
        responses_.push({.request_id = req.request_id, .slot_id = slot});
        return true;
      }
      case pm::RequestType::SUBMIT:
      case pm::RequestType::CONTINUE: {
        submittedSlots_.insert(req.slot_id);
        return true;
      }
      case pm::RequestType::CANCEL: {
        // Evict ack
        responses_.push(
            {.request_id = req.request_id, .slot_id = req.slot_id});
        return true;
      }
      default:
        return false;
    }
  }

  bool try_pop_response(pm::PMResponse& out) override {
    if (responses_.empty()) return false;
    out = responses_.front();
    responses_.pop();
    return true;
  }

  bool try_pop_output(pm::OutputMessage& out) override {
    if (outputs_.empty()) return false;
    out = outputs_.front();
    outputs_.pop();
    return true;
  }

  uint32_t get_spec_accepts(uint32_t /*slot_id*/) override { return 0; }

  uint32_t get_spec_rejects(uint32_t /*slot_id*/) override { return 0; }

 private:
  std::unordered_map<uint32_t, uint32_t> allocResponses_;
  std::queue<pm::PMResponse> responses_;
  std::queue<pm::OutputMessage> outputs_;
  std::unordered_set<uint32_t> submittedSlots_;
};

}  // namespace tt::testing
