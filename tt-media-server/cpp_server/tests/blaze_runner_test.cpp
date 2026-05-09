// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runners/blaze_runner/blaze_runner.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <thread>
#include <vector>

#include "config/runner_config.hpp"
#include "domain/llm/sequence.hpp"
#include "ipc/boost_ipc_queue.hpp"
#include "runners/llm_runner/in_memory_task_queue.hpp"
#include "support/fake_pipeline_manager.hpp"
#include "support/in_memory_cancel_queue.hpp"
#include "support/in_memory_result_queue.hpp"

namespace tt::testing {

using namespace tt::domain::llm;

class BlazeRunnerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Unique IPC queue names so parallel tests don't collide.
    requestQueueName_ = "blaze_test_mem_req_" + std::to_string(++counter_);
    resultQueueName_ = "blaze_test_mem_res_" + std::to_string(counter_);
    setenv("TT_MEMORY_REQUEST_QUEUE", requestQueueName_.c_str(), 1);
    setenv("TT_MEMORY_RESULT_QUEUE", resultQueueName_.c_str(), 1);

    // Create the IPC queues *before* BlazeRunner opens them.
    memReqQueue_ = std::make_unique<tt::ipc::MemoryRequestQueue>(
        requestQueueName_, tt::ipc::MEMORY_QUEUE_CAPACITY);
    memResQueue_ = std::make_unique<tt::ipc::MemoryResultQueue>(
        resultQueueName_, tt::ipc::MEMORY_QUEUE_CAPACITY);

    taskQueue_ = std::make_unique<tt::runners::llm_engine::InMemoryTaskQueue>();
    resultQueue_ = std::make_unique<InMemoryResultQueue>();
    cancelQueue_ = std::make_unique<InMemoryCancelQueue>();
    fakePm_ = std::make_unique<FakePipelineManager>();

    tt::config::LLMConfig cfg;
    cfg.runner_type = tt::config::ModelRunnerType::MOCK_PIPELINE;
    // Large enough to avoid accidental stop-token hits.
    cfg.stop_token_ids = {999999};

    runner_ = std::make_unique<tt::runners::BlazeRunner>(
        cfg, resultQueue_.get(), taskQueue_.get(), cancelQueue_.get(),
        std::move(fakePm_));
  }

  void TearDown() override {
    runner_.reset();
    // Clean up IPC resources.
    tt::ipc::MemoryRequestQueue(requestQueueName_, 1).remove();
    tt::ipc::MemoryResultQueue(resultQueueName_, 1).remove();
  }

  /** Push an ALLOCATE request and block until the success response arrives. */
  uint32_t allocateSlot() {
    tt::domain::ManageMemoryTask task;
    task.taskId = nextTaskId_++;
    task.action = tt::domain::MemoryManagementAction::ALLOCATE;

    ASSERT_TRUE(memReqQueue_->tryPush(task));

    // Spin the runner until the allocation response lands.
    tt::domain::ManageMemoryResult result;
    auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
      runner_->step();
      if (memResQueue_->tryPop(result)) {
        EXPECT_EQ(result.status, tt::domain::ManageMemoryStatus::SUCCESS);
        EXPECT_NE(result.slotId, tt::domain::INVALID_SLOT_ID);
        return result.slotId;
      }
    }
    ADD_FAILURE() << "Timed out waiting for slot allocation";
    return tt::domain::INVALID_SLOT_ID;
  }

  /** Create a simple Sequence and push it to the task queue. */
  uint32_t submitRequest(uint32_t slotId,
                         const std::vector<int64_t>& promptTokens,
                         int maxTokens) {
    uint32_t taskId = nextTaskId_++;
    tt::domain::llm::SamplingParams sp;
    sp.max_tokens = maxTokens;
    sp.ignore_eos = true;

    auto seq = std::make_unique<tt::domain::llm::Sequence>(
        taskId, 8, promptTokens, sp);
    seq->setNumPromptTokens(promptTokens.size());
    seq->setKVCacheSlot(slotId);

    taskQueue_->push(*seq);
    return taskId;
  }

  /** Drive the runner until the requested number of tokens arrive. */
  std::vector<tt::ipc::SharedToken> collectTokens(uint32_t taskId,
                                                   size_t minCount,
                                                   int timeoutMs = 5000) {
    std::vector<tt::ipc::SharedToken> tokens;
    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(timeoutMs);

    while (std::chrono::steady_clock::now() < deadline) {
      runner_->step();

      tt::ipc::SharedToken t;
      while (resultQueue_->tryPop(t)) {
        if (t.task_id == taskId) {
          tokens.push_back(t);
          if (t.isFinal()) return tokens;
        }
      }
    }
    return tokens;
  }

  static uint32_t counter_;
  uint32_t nextTaskId_ = 1000;

  std::string requestQueueName_;
  std::string resultQueueName_;
  std::unique_ptr<tt::ipc::MemoryRequestQueue> memReqQueue_;
  std::unique_ptr<tt::ipc::MemoryResultQueue> memResQueue_;
  std::unique_ptr<tt::runners::llm_engine::InMemoryTaskQueue> taskQueue_;
  std::unique_ptr<InMemoryResultQueue> resultQueue_;
  std::unique_ptr<InMemoryCancelQueue> cancelQueue_;
  std::unique_ptr<FakePipelineManager> fakePm_;
  std::unique_ptr<tt::runners::BlazeRunner> runner_;
};

uint32_t BlazeRunnerTest::counter_ = 0;

TEST_F(BlazeRunnerTest, HappyPath_AllocateSubmitAndReceiveTokens) {
  // --- Arrange: programme the fake pipeline manager -------------------------
  const uint32_t expectedSlot = 7;
  const uint32_t allocTaskId = nextTaskId_++;
  const std::vector<uint64_t> expectedTokenIds = {42, 43, 44};

  fakePm_->whenAllocate(allocTaskId, expectedSlot);
  fakePm_->enqueueOutputs(expectedSlot, expectedTokenIds,
                          /*finalIsComplete=*/true);

  // --- Act 1: send ALLOCATE and collect the slot ----------------------------
  tt::domain::ManageMemoryTask allocTask;
  allocTask.taskId = allocTaskId;
  allocTask.action = tt::domain::MemoryManagementAction::ALLOCATE;
  ASSERT_TRUE(memReqQueue_->tryPush(allocTask));

  uint32_t slot = allocateSlot();
  ASSERT_EQ(slot, expectedSlot);

  // --- Act 2: submit a request with that slot -------------------------------
  const std::vector<int64_t> prompt = {1, 2, 3};
  uint32_t taskId = submitRequest(slot, prompt, /*maxTokens=*/10);

  // --- Act 3: drive the runner and collect output tokens --------------------
  auto tokens = collectTokens(taskId, expectedTokenIds.size());

  // --- Assert ---------------------------------------------------------------
  ASSERT_EQ(tokens.size(), expectedTokenIds.size());

  for (size_t i = 0; i < expectedTokenIds.size(); ++i) {
    EXPECT_EQ(tokens[i].token_id, expectedTokenIds[i])
        << "Token mismatch at index " << i;
  }

  // Last token must be marked final.
  EXPECT_TRUE(tokens.back().isFinal())
      << "Last token should have FLAG_FINAL set";

  // The fake should have seen the SUBMIT.
  EXPECT_TRUE(fakePm_->wasSubmitted(expectedSlot));
}

//------------------------------------------------------------------------------
// Bug-reproduction tests for cancel edge cases
//------------------------------------------------------------------------------

TEST(CancelTombstonesTest, DequeGrowsUnbounded) {
  tt::runners::blaze_utils::CancelTombstones tombstones;
  for (uint32_t i = 0; i < 100; ++i) {
    tombstones.rememberCancelTombstone(i);
    EXPECT_TRUE(tombstones.consumeCancelTombstone(i));
  }
  // The set is correctly cleared, but the deque still holds 100 stale entries.
  // This is the bug pointed out in the review: lazy removal leaves the deque
  // growing without bound when the set never hits MAX_CANCEL_TOMBSTONES.
  EXPECT_EQ(tombstones.cancelTombstoneSet.size(), 0u);
  EXPECT_EQ(tombstones.cancelTombstoneOrder.size(), 0u)
      << "Bug: cancelTombstoneOrder leaked "
      << tombstones.cancelTombstoneOrder.size() << " stale entries";
}

TEST_F(BlazeRunnerTest, CancelBeforeAllocateRequestSeen_NoFailureSent) {
  // Scenario: cancel arrives before BlazeRunner sees the ALLOCATE request.
  // The tombstone consumes the request, but handleResponse is called for a
  // taskId that was never inserted into BlazeMemoryManager::allocating.
  // Result: no FAILURE response is sent back → SessionManager hangs.
  const uint32_t taskId = nextTaskId_++;

  // 1. Cancel arrives first.
  cancelQueue_->push(taskId);
  runner_->step();  // tombstone is remembered

  // 2. ALLOCATE request arrives later.
  tt::domain::ManageMemoryTask allocTask;
  allocTask.taskId = taskId;
  allocTask.action = tt::domain::MemoryManagementAction::ALLOCATE;
  ASSERT_TRUE(memReqQueue_->tryPush(allocTask));

  // 3. Runner processes the ALLOCATE request.
  runner_->step();

  // 4. Bug: memory result queue should contain FAILURE but it doesn't.
  tt::domain::ManageMemoryResult result;
  bool gotResult = memResQueue_->tryPop(result);
  EXPECT_TRUE(gotResult) << "Bug: FAILURE response was never sent to "
                            "SessionManager for cancelled task";
  if (gotResult) {
    EXPECT_EQ(result.status, tt::domain::ManageMemoryStatus::FAILURE);
    EXPECT_EQ(result.taskId, taskId);
  }
}

TEST_F(BlazeRunnerTest,
       CancelBetweenAllocateRequestAndPmResponse_DropsResponse) {
  // Scenario: ALLOCATE is forwarded to PipelineManager, cancel arrives before
  // the PM response comes back. The tombstone should consume the late response.
  const uint32_t taskId = nextTaskId_++;
  const uint32_t expectedSlot = 3;

  fakePm_->whenAllocate(taskId, expectedSlot);

  // 1. Send ALLOCATE request.
  tt::domain::ManageMemoryTask allocTask;
  allocTask.taskId = taskId;
  allocTask.action = tt::domain::MemoryManagementAction::ALLOCATE;
  ASSERT_TRUE(memReqQueue_->tryPush(allocTask));
  runner_->step();  // ALLOCATE forwarded to PM

  // 2. Cancel arrives before PM responds.
  cancelQueue_->push(taskId);
  runner_->step();  // tombstone remembered

  // 3. PM response arrives.
  runner_->step();  // handleMemoryResponse sees tombstone, drops it

  // 4. Result queue must be empty — SUCCESS was suppressed.
  tt::domain::ManageMemoryResult result;
  EXPECT_FALSE(memResQueue_->tryPop(result))
      << "Bug: late ALLOCATE response should have been dropped";
}

TEST_F(BlazeRunnerTest,
       CancelBetweenPmResponseAndHandleRequest_DropsSequence) {
  // Scenario: ALLOCATE succeeds, PM responds, MemoryManager sends SUCCESS,
  // SessionManager pushes Sequence to taskQueue, but cancel arrives before
  // handleRequest runs. The tombstone should drop the Sequence.
  const uint32_t taskId = nextTaskId_++;
  const uint32_t expectedSlot = 4;

  fakePm_->whenAllocate(taskId, expectedSlot);

  // 1. Full allocate cycle.
  tt::domain::ManageMemoryTask allocTask;
  allocTask.taskId = taskId;
  allocTask.action = tt::domain::MemoryManagementAction::ALLOCATE;
  ASSERT_TRUE(memReqQueue_->tryPush(allocTask));
  uint32_t slot = allocateSlot();
  ASSERT_EQ(slot, expectedSlot);

  // 2. SessionManager sends Sequence.
  submitRequest(slot, {1, 2, 3}, 10);

  // 3. Cancel arrives before the runner handles the Sequence.
  cancelQueue_->push(taskId);
  runner_->step();

  // 4. Next step: handleRequest sees tombstone, drops the Sequence.
  runner_->step();

  // 5. The fake should NOT have seen a SUBMIT.
  EXPECT_FALSE(fakePm_->wasSubmitted(expectedSlot))
      << "Bug: cancelled Sequence was still submitted to PipelineManager";
}

TEST_F(BlazeRunnerTest, CancelAfterHandleRequestWhileRunning) {
  // Scenario: Sequence is already running when cancel arrives.
  // handleCancelRequest should call memoryManager->evict(slotId).
  const uint32_t taskId = nextTaskId_++;
  const uint32_t expectedSlot = 5;

  fakePm_->whenAllocate(taskId, expectedSlot);
  fakePm_->enqueueOutputs(expectedSlot, {10, 11, 12},
                          /*finalIsComplete=*/true);

  uint32_t slot = allocateSlot();
  ASSERT_EQ(slot, expectedSlot);

  submitRequest(slot, {1, 2, 3}, 10);
  runner_->step();  // SUBMIT
  EXPECT_TRUE(fakePm_->wasSubmitted(expectedSlot));

  // Cancel while running.
  cancelQueue_->push(taskId);
  runner_->step();

  // The slot should have been evicted.
  EXPECT_TRUE(fakePm_->wasEvicted(expectedSlot))
      << "Bug: running task was not evicted on cancel";
}

TEST_F(BlazeRunnerTest, CancelRequestToRetry_ReleasesSlot) {
  // Scenario: push_request fails, requestToRetry is set. The slot is allocated
  // but taskIdToSlotId doesn't have the mapping yet (it's added only after a
  // successful push). Cancelling must still release the slot.
  const uint32_t taskId = nextTaskId_++;
  const uint32_t expectedSlot = 6;

  fakePm_->whenAllocate(taskId, expectedSlot);
  fakePm_->whenPushFails(expectedSlot);

  uint32_t slot = allocateSlot();
  ASSERT_EQ(slot, expectedSlot);

  // Submit request — push will fail, creating requestToRetry.
  submitRequest(slot, {1, 2, 3}, 10);
  runner_->step();

  // Cancel the task.
  cancelQueue_->push(taskId);
  runner_->step();

  // Bug: slot was never evicted because taskIdToSlotId didn't contain taskId.
  EXPECT_TRUE(fakePm_->wasEvicted(expectedSlot))
      << "Bug: requestToRetry slot was not released on cancel";
}

TEST_F(BlazeRunnerTest, CancelFinishedTask_IsHarmlessNoOp) {
  // Scenario: task runs to completion, then a late cancel arrives.
  // Must be a harmless no-op + tombstone GC.
  const uint32_t taskId = nextTaskId_++;
  const uint32_t expectedSlot = 8;

  fakePm_->whenAllocate(taskId, expectedSlot);
  fakePm_->enqueueOutputs(expectedSlot, {20, 21},
                          /*finalIsComplete=*/true);

  uint32_t slot = allocateSlot();
  ASSERT_EQ(slot, expectedSlot);

  submitRequest(slot, {1, 2}, 10);
  auto tokens = collectTokens(taskId, 2);
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_TRUE(tokens.back().isFinal());

  // Slot context should be gone.
  EXPECT_EQ(fakePm_->submitCount(), 1u);

  // Late cancel.
  cancelQueue_->push(taskId);
  runner_->step();

  // Should not crash or evict.
  EXPECT_FALSE(fakePm_->wasEvicted(expectedSlot))
      << "Bug: finished task was incorrectly evicted";
}

}  // namespace tt::testing
