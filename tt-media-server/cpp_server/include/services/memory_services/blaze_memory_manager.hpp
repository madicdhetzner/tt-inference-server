// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <unordered_set>

#include "runners/blaze_runner/i_pipeline_manager.hpp"
#include "services/memory_services/memory_manager.hpp"

namespace tt::services {

class BlazeMemoryManager : public MemoryManager {
  using onEvictCb = std::function<void(uint32_t slotId)>;

 public:
  BlazeMemoryManager(tt::runners::IPipelineManager& pipelineManager,
                     onEvictCb onEvict);
  ~BlazeMemoryManager() = default;

  std::optional<domain::ManageMemoryTask> getRequest() override;

  void handleRequest(const domain::ManageMemoryTask& request) override;

  void handleResponse(uint32_t taskId, uint32_t slotId) override;

 private:
  tt::runners::IPipelineManager& pipelineManager;
  std::unordered_set<uint32_t> allocating;
  std::unordered_map</*taskId*/ uint32_t, /*slotId*/ uint32_t> evicting;
  onEvictCb onEvict;
  std::optional<domain::ManageMemoryTask> pendingRetry;
};

}  // namespace tt::services
