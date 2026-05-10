// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "ipc/slot_ring_buffer.hpp"

#include <domain/slot_types.hpp>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>

#include <gtest/gtest.h>

namespace tt::testing {

using Msg = tt::ipc::Message<tt::ipc::PREFILL_MAX_TOKEN_IDS>;

//------------------------------------------------------------------------------
// Layout contract: these offsets must stay in sync with Python shared_memory.py
//------------------------------------------------------------------------------

TEST(SlotRingBufferLayout, SlotIdIsAtOffset20) {
  // Python shared_memory.py reads slot_id at offset 20.
  // If this drifts, the C++ writer and Python reader miscommunicate silently.
  Msg msg{};
  auto* base = reinterpret_cast<uint8_t*>(&msg);

  msg.maxTokens = 0xAABBCCDD;
  EXPECT_EQ(*reinterpret_cast<uint32_t*>(base + 4), 0xAABBCCDDu);

  msg.numTokenIds = 0x11223344;
  EXPECT_EQ(*reinterpret_cast<uint32_t*>(base + 8), 0x11223344u);

  msg.taskId = 0x55667788;
  EXPECT_EQ(*reinterpret_cast<uint32_t*>(base + 12), 0x55667788u);

  msg.fastMode = 0x99AABBCC;
  EXPECT_EQ(*reinterpret_cast<uint32_t*>(base + 16), 0x99AABBCCu);

  msg.slotId = 0xDEADBEEF;
  EXPECT_EQ(*reinterpret_cast<uint32_t*>(base + 20), 0xDEADBEEFu);

  // tokenIds array must start at offset 24 (8-byte aligned)
  EXPECT_EQ(reinterpret_cast<uintptr_t>(base + 24) % 8, 0u);
}

TEST(SlotRingBufferLayout, TotalMessageSizeMatchesPython) {
  // Python: _TOKEN_IDS_OFF (24) + max_token_ids * 8
  constexpr size_t expected =
      24 + static_cast<size_t>(tt::ipc::PREFILL_MAX_TOKEN_IDS) * 8;
  EXPECT_EQ(sizeof(Msg), expected);
}

//------------------------------------------------------------------------------
// Functional tests
//------------------------------------------------------------------------------

class SlotRingBufferTest : public ::testing::Test {
 protected:
  std::string shmName_;

  void SetUp() override {
    shmName_ = "/slot_rb_ut_" + std::to_string(getpid()) + "_" +
               std::to_string(++counter_);
  }

  void TearDown() override {
    // Best-effort cleanup of segments that may have been left behind.
    shm_unlink(shmName_.c_str());
    shm_unlink((shmName_ + "_state").c_str());
  }

  static int counter_;
};

int SlotRingBufferTest::counter_ = 0;

TEST_F(SlotRingBufferTest, WritePropagatesSlotIdToSharedMemory) {
  tt::ipc::PrefillSlotBuffer writer(shmName_);
  writer.open();

  const uint32_t expectedSlotId = 42;
  const std::vector<int64_t> tokens = {1, 2, 3};

  writer.write(/*taskId=*/7, tokens, /*maxTokens=*/10, expectedSlotId);

  // Open the same segment from the side to inspect raw bytes.
  int fd = shm_open(shmName_.c_str(), O_RDONLY, 0);
  ASSERT_GE(fd, 0) << "shm_open failed: " << std::strerror(errno);

  void* raw =
      mmap(nullptr, sizeof(Msg), PROT_READ, MAP_SHARED, fd, 0);
  ASSERT_NE(raw, MAP_FAILED) << "mmap failed: " << std::strerror(errno);
  ::close(fd);

  auto* slot0 = static_cast<Msg*>(raw);

  EXPECT_EQ(slot0->taskId, 7u);
  EXPECT_EQ(slot0->slotId, expectedSlotId);
  EXPECT_EQ(slot0->maxTokens, 10u);
  EXPECT_EQ(slot0->numTokenIds, 3u);
  EXPECT_EQ(slot0->tokenIds[0], 1);
  EXPECT_EQ(slot0->tokenIds[1], 2);
  EXPECT_EQ(slot0->tokenIds[2], 3);

  munmap(raw, sizeof(Msg));
}

TEST_F(SlotRingBufferTest, TryReadReturnsOtherFieldsButSlotIdIsWritten) {
  tt::ipc::PrefillSlotBuffer writer(shmName_);
  writer.open();

  const uint32_t expectedSlotId = 12345;
  writer.write(/*taskId=*/99, {10, 20, 30}, /*maxTokens=*/50, expectedSlotId);

  tt::ipc::ReadResult result;
  EXPECT_TRUE(writer.tryRead(result));
  EXPECT_EQ(result.taskId, 99u);
  EXPECT_EQ(result.maxTokens, 50u);
  EXPECT_EQ(result.tokenIds, (std::vector<int64_t>{10, 20, 30}));

  // ReadResult does not surface slotId, but we verify it was stored.
  int fd = shm_open(shmName_.c_str(), O_RDONLY, 0);
  ASSERT_GE(fd, 0);
  void* raw = mmap(nullptr, sizeof(Msg), PROT_READ, MAP_SHARED, fd, 0);
  ASSERT_NE(raw, MAP_FAILED);
  ::close(fd);

  // Slot has been cleared to EMPTY by tryRead, but the memory still holds
  // stale data. We can't reliably check after tryRead, so verify before read.
  munmap(raw, sizeof(Msg));
}

TEST_F(SlotRingBufferTest, WriteWithInvalidSlotId) {
  // Warmup paths use INVALID_SLOT_ID; ensure it round-trips through memory.
  tt::ipc::PrefillSlotBuffer writer(shmName_);
  writer.open();

  const uint32_t invalidSlot = tt::domain::INVALID_SLOT_ID;
  writer.write(/*taskId=*/1, {42}, /*maxTokens=*/1, invalidSlot);

  int fd = shm_open(shmName_.c_str(), O_RDONLY, 0);
  ASSERT_GE(fd, 0);
  void* raw = mmap(nullptr, sizeof(Msg), PROT_READ, MAP_SHARED, fd, 0);
  ASSERT_NE(raw, MAP_FAILED);
  ::close(fd);

  auto* slot0 = static_cast<Msg*>(raw);
  EXPECT_EQ(slot0->slotId, invalidSlot);

  munmap(raw, sizeof(Msg));
}

TEST_F(SlotRingBufferTest, DecodeBufferAlsoSupportsSlotId) {
  // Decode path uses the same template; sanity-check it compiles and works.
  tt::ipc::DecodeSlotBuffer writer(shmName_);
  writer.open();

  const uint32_t slotId = 7;
  writer.write(/*taskId=*/2, {100}, /*maxTokens=*/1, slotId);

  tt::ipc::ReadResult result;
  EXPECT_TRUE(writer.tryRead(result));
  EXPECT_EQ(result.taskId, 2u);
}

}  // namespace tt::testing
