// Unit tests for src/engine/worker_thread.hpp.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <vector>

#include "engine/worker_thread.hpp"

using digitalcurling::client::WorkerThread;

TEST(WorkerThread, ReturnsResults) {
    WorkerThread worker;
    auto a = worker.Submit([] { return 6 * 7; });
    auto b = worker.Submit([] { return std::vector<int>{1, 2, 3}; });
    EXPECT_EQ(a.get(), 42);
    EXPECT_EQ(b.get().size(), 3u);
}

TEST(WorkerThread, RunsEveryTaskOnTheSameThread) {
    WorkerThread worker;
    std::vector<std::thread::id> ids;
    for (int i = 0; i < 20; ++i) {
        ids.push_back(worker.Submit([] { return std::this_thread::get_id(); }).get());
    }
    for (auto const& id : ids) {
        EXPECT_EQ(id, worker.GetId());
        EXPECT_NE(id, std::this_thread::get_id());
    }
}

TEST(WorkerThread, RunsTasksInOrder) {
    WorkerThread worker;
    std::vector<int> order;
    std::vector<std::future<void>> done;
    for (int i = 0; i < 100; ++i) {
        done.push_back(worker.Submit([&order, i] { order.push_back(i); }));
    }
    for (auto& f : done) f.get();
    ASSERT_EQ(order.size(), 100u);
    for (int i = 0; i < 100; ++i) EXPECT_EQ(order[i], i);
}

TEST(WorkerThread, PassesExceptionsToTheFuture) {
    WorkerThread worker;
    auto f = worker.Submit([]() -> int { throw std::runtime_error("boom"); });
    EXPECT_THROW(f.get(), std::runtime_error);
    // The worker keeps running after a task throws.
    EXPECT_EQ(worker.Submit([] { return 1; }).get(), 1);
}

TEST(WorkerThread, DestructorFinishesQueuedTasks) {
    std::atomic<int> ran{0};
    {
        WorkerThread worker;
        for (int i = 0; i < 10; ++i) {
            worker.Submit([&ran] {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                ++ran;
            });
        }
    }
    EXPECT_EQ(ran.load(), 10);
}
