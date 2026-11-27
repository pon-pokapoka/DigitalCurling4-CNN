// Copyright (c) 2023-2026 Takehiro Yoshioka
// SPDX-License-Identifier: MIT

#ifndef WORKER_THREAD_HPP
#define WORKER_THREAD_HPP

#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>

namespace digitalcurling::client {

// One long-lived thread that runs submitted tasks in order.
//
// The search evaluates each batch on this thread instead of a new std::async
// thread per batch: libtorch keeps per-thread state for the model (cuDNN and
// cuBLAS handles, TorchScript interpreter state) and doesn't free all of it
// when a thread exits, so a new thread per batch leaked about 110 KB a batch.
class WorkerThread {
public:
    WorkerThread() : thread_([this] { Run(); }) {}

    WorkerThread(WorkerThread const&) = delete;
    WorkerThread& operator=(WorkerThread const&) = delete;

    // Runs the tasks already submitted, then joins the thread.
    ~WorkerThread() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        cv_.notify_one();
        thread_.join();
    }

    // Queues `f` and returns a future of its result; an exception thrown by `f`
    // is rethrown by the future's get().
    template <typename F>
    auto Submit(F&& f) -> std::future<std::invoke_result_t<std::decay_t<F>>> {
        using R = std::invoke_result_t<std::decay_t<F>>;
        auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
        auto result = task->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks_.emplace_back([task] { (*task)(); });
        }
        cv_.notify_one();
        return result;
    }

    std::thread::id GetId() const { return thread_.get_id(); }

private:
    void Run() {
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
                if (tasks_.empty()) return;  // stopping and nothing left
                task = std::move(tasks_.front());
                tasks_.pop_front();
            }
            task();
        }
    }

    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::function<void()>> tasks_;
    bool stopping_ = false;
    std::thread thread_;  // last: starts after the members it uses
};

} // namespace digitalcurling::client

#endif // WORKER_THREAD_HPP
