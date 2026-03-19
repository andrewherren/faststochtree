#pragma once
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace bart {

// Persistent thread pool: workers sleep between calls; caller participates as
// worker 0 so no core is wasted.  Workers are woken via a generation counter;
// the caller spin-waits for completion (tasks complete in milliseconds).
struct ThreadPool {
    explicit ThreadPool(int num_threads)
        : n_workers(num_threads), gen(0), shutdown(false)
    {
        for (int t = 1; t < num_threads; t++)
            workers.emplace_back([this] { worker_loop(); });
    }

    ~ThreadPool() {
        { std::lock_guard<std::mutex> lk(mtx); shutdown = true; gen++; }
        cv.notify_all();
        for (auto& w : workers) w.join();
    }

    // Runs fn(i) for i in [begin, end). Caller participates; blocks until done.
    template<typename F>
    void parallel_for(int begin, int end, F&& fn) {
        if (begin >= end) return;
        if (n_workers == 1) {
            for (int i = begin; i < end; i++) fn(i);
            return;
        }
        // Store work state before waking workers.  Workers are sleeping (in
        // cv.wait) so it's safe to write func/next/work_end without a mutex.
        func     = fn;
        next.store(begin,   std::memory_order_relaxed);
        work_end.store(end, std::memory_order_relaxed);
        finished.store(0,   std::memory_order_relaxed);

        { std::lock_guard<std::mutex> lk(mtx); gen++; }
        cv.notify_all();

        run();  // caller participates

        // Spin-wait: tasks finish in O(ms); avoids condvar overhead.
        while (finished.load(std::memory_order_acquire) < n_workers)
            std::this_thread::yield();
        // Post-condition: all workers are back in cv.wait — func is safe to overwrite.
    }

    int size() const { return n_workers; }

private:
    void worker_loop() {
        uint64_t my_gen = 0;
        while (true) {
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv.wait(lk, [&]{ return gen != my_gen || shutdown; });
                if (shutdown) return;
                my_gen = gen;
            }
            run();
        }
    }

    void run() {
        while (true) {
            int i = next.fetch_add(1, std::memory_order_relaxed);
            if (i >= work_end.load(std::memory_order_relaxed)) break;
            func(i);
        }
        finished.fetch_add(1, std::memory_order_release);
    }

    int                         n_workers;
    std::vector<std::thread>    workers;
    std::mutex                  mtx;
    std::condition_variable     cv;
    std::function<void(int)>    func;
    std::atomic<int>            next{0};
    std::atomic<int>            work_end{0};
    std::atomic<int>            finished{0};
    uint64_t                    gen;
    bool                        shutdown;
};

} // namespace bart
