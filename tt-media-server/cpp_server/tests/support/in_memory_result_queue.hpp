// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>

#include "ipc/result_queue.hpp"

namespace tt::testing {

/**
 * In-memory IResultQueue for unit tests.
 * No IPC overhead; fully deterministic.
 */
class InMemoryResultQueue : public tt::ipc::IResultQueue {
 public:
  bool push(const tt::ipc::SharedToken& token) override {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(token);
    cv_.notify_one();
    return true;
  }

  bool tryPop(tt::ipc::SharedToken& out) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) return false;
    out = queue_.front();
    queue_.pop();
    return true;
  }

  bool blockingPop(tt::ipc::SharedToken& out) override {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return !queue_.empty() || shutdown_; });
    if (shutdown_ && queue_.empty()) return false;
    out = queue_.front();
    queue_.pop();
    return true;
  }

  bool empty() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  void shutdown() override {
    std::lock_guard<std::mutex> lock(mutex_);
    shutdown_ = true;
    cv_.notify_all();
  }

  bool isShutdown() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return shutdown_;
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

 private:
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::queue<tt::ipc::SharedToken> queue_;
  bool shutdown_ = false;
};

}  // namespace tt::testing
