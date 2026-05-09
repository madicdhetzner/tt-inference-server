// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>

#include "i_pipeline_manager.hpp"
#include "pipeline_manager/pipeline_manager.hpp"

namespace tt::runners {

/**
 * Production adapter: wraps the real pm::PipelineManager and exposes it
 * through the IPipelineManager interface.
 */
class PipelineManagerAdapter : public IPipelineManager {
 public:
  PipelineManagerAdapter(pm::PipelineConfig config, pm::ManagerParams params);

  void start() override;
  void stop() override;

  bool push_request(const pm::ISRequest& req) override;
  bool try_pop_response(pm::PMResponse& out) override;
  bool try_pop_output(pm::OutputMessage& out) override;

  uint32_t get_spec_accepts(uint32_t slot_id) override;
  uint32_t get_spec_rejects(uint32_t slot_id) override;

 private:
  std::unique_ptr<pm::PipelineManager> pm_;
};

}  // namespace tt::runners
