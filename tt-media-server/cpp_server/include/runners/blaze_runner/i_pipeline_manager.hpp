// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include "pipeline_manager/pipeline_manager_types.hpp"

namespace tt::runners {

namespace pm = tt_blaze::pipeline_manager;

/**
 * Abstract interface for the subset of PipelineManager operations used by
 * BlazeRunner and BlazeMemoryManager.
 *
 * Enables unit-test doubles that don't require the real tt-blaze backend.
 */
class IPipelineManager {
 public:
  virtual ~IPipelineManager() = default;

  virtual void start() = 0;
  virtual void stop() = 0;

  virtual bool push_request(const pm::ISRequest& req) = 0;
  virtual bool try_pop_response(pm::PMResponse& out) = 0;
  virtual bool try_pop_output(pm::OutputMessage& out) = 0;

  virtual uint32_t get_spec_accepts(uint32_t slot_id) = 0;
  virtual uint32_t get_spec_rejects(uint32_t slot_id) = 0;
};

}  // namespace tt::runners
