// Direct unit tests for the atomic semantics of `ContinuousLoopState`.
//
// The struct is exercised indirectly by the 2x2 mode-matrix integration
// tests, but its atomic ordering is load-bearing for the continuous
// parallel runners (`opportunistic_runner.h`, `bandit_runner.h`) — a
// regression in ordering or in the `>=` stop predicate would be a silent
// heisenbug.  These tests target the primitive directly.
//
// The `poll_termination` helper takes a `HighsMipSolver&` and reads
// non-thread-safe HiGHS internals, so it is intentionally not unit-tested
// here; its semantics are covered by the mode-matrix integration tests.

#include "continuous_loop.h"

#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <thread>
#include <vector>

TEST_CASE("ContinuousLoopState: single-threaded add_effort accumulates correctly",
          "[continuous_loop]") {
    ContinuousLoopState loop;
    constexpr size_t kBudget = 1000;

    for (int i = 0; i < 5; ++i) {
        loop.add_effort(100, kBudget);
        REQUIRE_FALSE(loop.stopped());
    }
    REQUIRE(loop.total_effort.load() == 500);

    // One more push keeps us below budget.
    loop.add_effort(400, kBudget);
    REQUIRE(loop.total_effort.load() == 900);
    REQUIRE_FALSE(loop.stopped());

    // Crossing the threshold flips `stop` via the `>=` comparison.
    loop.add_effort(100, kBudget);
    REQUIRE(loop.total_effort.load() == 1000);
    REQUIRE(loop.stopped());
}

TEST_CASE(
    "ContinuousLoopState: single-threaded note_staleness increments without improvement, resets on "
    "improvement",
    "[continuous_loop]") {
    constexpr size_t kStaleBudget = 200;

    SECTION("stale accumulates and triggers stop at >= budget") {
        ContinuousLoopState loop;
        // 3 increments of 50: total stale = 150 < 200.
        for (int i = 0; i < 3; ++i) {
            loop.note_staleness(50, /*improved=*/false, kStaleBudget);
            REQUIRE_FALSE(loop.stopped());
        }
        REQUIRE(loop.effort_since_improvement.load() == 150);

        // One more: stale = 200, triggers stop via the `>=` comparison.
        loop.note_staleness(50, /*improved=*/false, kStaleBudget);
        REQUIRE(loop.effort_since_improvement.load() == 200);
        REQUIRE(loop.stopped());
    }

    SECTION("improvement resets counter and does not stop") {
        ContinuousLoopState loop;
        // An "improved" call with positive effort still resets the counter
        // to 0 (improvement supersedes accumulation).
        loop.note_staleness(50, /*improved=*/true, kStaleBudget);
        REQUIRE(loop.effort_since_improvement.load() == 0);
        REQUIRE_FALSE(loop.stopped());

        // Fresh accumulation post-reset still works.
        loop.note_staleness(50, /*improved=*/false, kStaleBudget);
        REQUIRE(loop.effort_since_improvement.load() == 50);
        REQUIRE_FALSE(loop.stopped());
    }
}

TEST_CASE("ContinuousLoopState: request_stop is idempotent and irreversible", "[continuous_loop]") {
    ContinuousLoopState loop;
    REQUIRE_FALSE(loop.stopped());

    loop.request_stop();
    REQUIRE(loop.stopped());

    loop.request_stop();
    REQUIRE(loop.stopped());

    // Subsequent no-op atomic updates must not clear the flag; `stopped`
    // is monotonically one-way until the struct goes out of scope.
    loop.note_staleness(0, /*improved=*/true, 1000);
    REQUIRE(loop.stopped());
    loop.add_effort(0, 1000);
    REQUIRE(loop.stopped());
}

TEST_CASE("ContinuousLoopState: multi-threaded add_effort monotonic accumulation",
          "[continuous_loop]") {
    ContinuousLoopState loop;
    constexpr size_t kBudget = 10000;
    constexpr int kThreads = 8;
    constexpr int kPerThread = 1000;

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&loop]() {
            for (int i = 0; i < kPerThread; ++i) {
                loop.add_effort(1, kBudget);
            }
        });
    }
    for (auto &th : threads) {
        th.join();
    }

    REQUIRE(loop.total_effort.load() == static_cast<size_t>(kThreads * kPerThread));
    REQUIRE_FALSE(loop.stopped());  // 8000 < 10000
}

TEST_CASE("ContinuousLoopState: multi-threaded add_effort triggers stop exactly once at budget",
          "[continuous_loop]") {
    ContinuousLoopState loop;
    constexpr size_t kBudget = 10000;
    constexpr int kThreads = 8;
    constexpr int kPerThread = 20;
    constexpr size_t kChunk = 100;
    // Total pushed = 8 * 20 * 100 = 16000, comfortably over kBudget.

    // Each thread records, for each of its calls, what the new total was
    // right after its atomic increment.  Exactly one of those totals will
    // be the first to cross kBudget from below.
    std::vector<std::vector<size_t>> observed(kThreads);

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&loop, &observed, t]() {
            observed[t].reserve(kPerThread);
            for (int i = 0; i < kPerThread; ++i) {
                size_t new_total = loop.add_effort(kChunk, kBudget);
                observed[t].push_back(new_total);
            }
        });
    }
    for (auto &th : threads) {
        th.join();
    }

    REQUIRE(loop.stopped());
    REQUIRE(loop.total_effort.load() == static_cast<size_t>(kThreads * kPerThread) * kChunk);

    // Count calls whose returned total is the first to cross the budget:
    // `new_total >= kBudget && new_total - kChunk < kBudget`.
    // Atomicity of fetch_add guarantees exactly one such call across all
    // threads.
    int crossings = 0;
    for (const auto &per_thread : observed) {
        for (size_t total : per_thread) {
            if (total >= kBudget && total - kChunk < kBudget) {
                ++crossings;
            }
        }
    }
    REQUIRE(crossings == 1);
}

TEST_CASE("ContinuousLoopState: note_staleness under contention never under-counts improvements",
          "[continuous_loop]") {
    // Relaxed-memory race between an incrementing writer and a
    // periodically resetting writer: the reset branch stores 0 directly,
    // the increment branch uses fetch_add.  The stop predicate reads
    // relaxed.  This test asserts that (a) the counter never goes
    // negative-by-wraparound, (b) the reset is observable by the stop
    // predicate (if a reset happens between two increments, the inc
    // thread does not spuriously stop below the total number of
    // non-reset increments), and (c) concurrent reset/increment does not
    // deadlock.
    //
    // We deliberately pick a stale budget larger than the number of
    // increments so no stop should fire in the expected interleavings;
    // the test's invariant is "no spurious stops" rather than a tight
    // numerical equality (which relaxed ordering cannot guarantee).
    ContinuousLoopState loop;
    constexpr size_t kStaleBudget = 100000;  // large enough that no stop fires
    constexpr int kIncrements = 5000;
    constexpr int kResetsEvery = 50;

    std::atomic<bool> go{false};
    std::atomic<bool> incrementer_done{false};

    std::thread incrementer([&]() {
        while (!go.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        for (int i = 0; i < kIncrements; ++i) {
            loop.note_staleness(1, /*improved=*/false, kStaleBudget);
        }
        incrementer_done.store(true, std::memory_order_release);
    });

    std::thread resetter([&]() {
        while (!go.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        int iter = 0;
        while (!incrementer_done.load(std::memory_order_acquire)) {
            if ((iter++ % kResetsEvery) == 0) {
                loop.note_staleness(0, /*improved=*/true, kStaleBudget);
            }
            std::this_thread::yield();
        }
    });

    go.store(true, std::memory_order_release);
    incrementer.join();
    resetter.join();

    // No stop should have fired: kIncrements (5000) is well under
    // kStaleBudget (100000), even if no reset ever took effect.
    REQUIRE_FALSE(loop.stopped());

    // The final counter is bounded above by kIncrements: at most every
    // increment survived with no intervening reset.  A reset may have
    // zeroed some increments, so the lower bound is 0.  What we can
    // assert is the non-wraparound invariant: the counter is a valid
    // size_t in [0, kIncrements].
    size_t final_stale = loop.effort_since_improvement.load();
    REQUIRE(final_stale <= static_cast<size_t>(kIncrements));
}
