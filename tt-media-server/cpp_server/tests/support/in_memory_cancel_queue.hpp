// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <mutex>
#include <queue>
#include <vector>

#include "ipc/cancel_queue.hpp"

namespace tt::testing {

/**
 * In-memory ICancelQueue for unit tests.
 */
class InMemoryCancelQueue : public tt::ipc::ICancelQueue {
 public:
  void push(uint32_t taskId) override {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(taskId);
  }

  void tryPopAll(std::vector<uint32_t>& out) override {
    std::lock_guard<std::mutex> lock(mutex_);
    while (!queue_.empty()) {
      out.push_back(queue_.front());
      queue_.pop();
    }
  }

  void remove() override {}

  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

 private:
  mutable std::mutex mutex_;
  std::queue<uint32_t> queue_;
};

}  // namespace tt::testing
