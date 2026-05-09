// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runners/blaze_runner/pipeline_manager_adapter.hpp"

namespace tt::runners {

PipelineManagerAdapter::PipelineManagerAdapter(pm::PipelineConfig config,
                                               pm::ManagerParams params)
    : pm_(std::make_unique<pm::PipelineManager>(config, params)) {}

void PipelineManagerAdapter::start() { pm_->start(); }

void PipelineManagerAdapter::stop() { pm_->stop(); }

bool PipelineManagerAdapter::push_request(const pm::ISRequest& req) {
  return pm_->push_request(req);
}

bool PipelineManagerAdapter::try_pop_response(pm::PMResponse& out) {
  return pm_->try_pop_response(out);
}

bool PipelineManagerAdapter::try_pop_output(pm::OutputMessage& out) {
  return pm_->try_pop_output(out);
}

uint32_t PipelineManagerAdapter::get_spec_accepts(uint32_t slot_id) {
  return pm_->get_spec_accepts(slot_id);
}

uint32_t PipelineManagerAdapter::get_spec_rejects(uint32_t slot_id) {
  return pm_->get_spec_rejects(slot_id);
}

}  // namespace tt::runners
