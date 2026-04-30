// Unit tests for the ContestedPdlp try_lock + stale-snapshot plumbing
// added in issue #76.  These exercise the concurrency plumbing without
// spinning up a real PDLP solve — a test-double subclass overrides
// `solve_locked()` to fake a bounded-duration "solve" with canned
// output, so the tests run in milliseconds and stay deterministic on
// CPU-only build servers (no cuPDLP dependency).
//
// The tests drive three things:
//   (a) Stale readers see the last published snapshot while a peer
//       holds the mutex (try_solve_or_snapshot returns fresh=false).
//   (b) The one-solve-in-flight invariant holds under contention —
//       peak_in_flight() == 1 regardless of how many workers hammer
//       the API.
//   (c) Fresh solves advance snapshot_generation() exactly once per
//       publication; stale rounds do not.

#include "contested_pdlp.h"

#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

namespace {

// Test double: exposes the protected ForTesting constructor + overrides
// `solve_locked` with a controllable fake.  The fake sleeps for a
// configurable duration so tests can deterministically land workers in
// the "peer is solving" state; it fills in canned `col_value` /
// `row_dual` so `publish_snapshot_locked` actually publishes.
class FakePdlp : public ContestedPdlp {
public:
    FakePdlp() : ContestedPdlp(ContestedPdlp::ForTesting{}) {}

    // Sleep inserted at the start of the fake solve to widen the
    // critical-section window so other threads are guaranteed to hit
    // `try_lock` contention during the test.  Default 0 — most tests
    // set it explicitly.
    std::atomic<int> solve_sleep_ms{0};
    // Counts completed fake solves for post-hoc assertions.
    std::atomic<int> solve_count{0};

    using ContestedPdlp::acquire_for_test;
    using ContestedPdlp::publish_snapshot_for_test;

protected:
    SolveResult solve_locked(const std::vector<double> & /*modified_cost*/,
                             const std::vector<double> & /*warm_start_col_value*/,
                             const std::vector<double> & /*warm_start_row_dual*/,
                             bool /*warm_start_valid*/, double /*epsilon*/,
                             double /*time_limit*/) override {
        int sleep_ms = solve_sleep_ms.load(std::memory_order_relaxed);
        if (sleep_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        }
        SolveResult r;
        r.col_value = {1.0, 2.0, 3.0};
        r.row_dual = {0.5};
        r.pdlp_iters = 7;
        r.status = HighsStatus::kOk;
        r.model_status = HighsModelStatus::kOptimal;
        r.value_valid = true;
        r.dual_valid = true;
        solve_count.fetch_add(1, std::memory_order_relaxed);
        return r;
    }
};

}  // namespace

TEST_CASE("ContestedPdlp: try_solve_or_snapshot returns fresh when uncontended",
          "[contested_pdlp][overlap]") {
    FakePdlp pdlp;
    std::vector<double> cost, ws_col, ws_row;
    auto res = pdlp.try_solve_or_snapshot(cost, ws_col, ws_row, false, 1e-4, 1.0);
    REQUIRE(res.fresh);
    REQUIRE(res.solve.status == HighsStatus::kOk);
    REQUIRE(pdlp.peak_in_flight() == 1);
    REQUIRE(pdlp.snapshot_generation() == 1);
    // After a successful solve, a subsequent latest_snapshot() read is
    // non-null and points at the newly-published snapshot.
    auto snap = pdlp.latest_snapshot();
    REQUIRE(snap != nullptr);
    REQUIRE(snap->col_value.size() == 3);
    REQUIRE(snap->value_valid);
    // Snapshot carries its own generation stamp matching the global
    // counter — consumers should compare by generation, not pointer.
    REQUIRE(snap->generation == 1);
}

TEST_CASE("ContestedPdlp: snapshot generation increases monotonically across solves",
          "[contested_pdlp][overlap]") {
    // Regression for R3-5: each fresh publish must stamp the Snapshot
    // with a strictly-monotonic generation, so consumers (e.g.
    // ScyllaWorker) can detect "is this a new snapshot?" without
    // relying on `shared_ptr` address identity (heap addresses can be
    // recycled).
    FakePdlp pdlp;
    std::vector<double> cost, ws_col, ws_row;

    auto r1 = pdlp.try_solve_or_snapshot(cost, ws_col, ws_row, false, 1e-4, 1.0);
    REQUIRE(r1.fresh);
    auto s1 = pdlp.latest_snapshot();
    REQUIRE(s1 != nullptr);
    REQUIRE(s1->generation == 1);

    auto r2 = pdlp.try_solve_or_snapshot(cost, ws_col, ws_row, false, 1e-4, 1.0);
    REQUIRE(r2.fresh);
    auto s2 = pdlp.latest_snapshot();
    REQUIRE(s2 != nullptr);
    REQUIRE(s2->generation == 2);
    REQUIRE(s2->generation > s1->generation);

    // The blocking `solve()` path must publish too and use the same
    // counter (no separate channel for fresh-via-solve vs
    // fresh-via-try).
    (void)pdlp.solve(cost, ws_col, ws_row, false, 1e-4, 1.0);
    auto s3 = pdlp.latest_snapshot();
    REQUIRE(s3 != nullptr);
    REQUIRE(s3->generation == 3);
    REQUIRE(pdlp.snapshot_generation() == 3);
}

TEST_CASE("ContestedPdlp: stale readers see last snapshot while peer holds mutex",
          "[contested_pdlp][overlap]") {
    FakePdlp pdlp;

    // Publish an initial snapshot so the stale path has something to
    // return (mirrors "one solve has already completed in normal use").
    {
        ContestedPdlp::Snapshot seed;
        seed.col_value = {9.0, 8.0, 7.0};
        seed.row_dual = {0.1};
        seed.pdlp_iters = 42;
        seed.value_valid = true;
        seed.dual_valid = true;
        auto guard = pdlp.acquire_for_test();
        pdlp.publish_snapshot_for_test(std::move(seed));
    }
    REQUIRE(pdlp.snapshot_generation() == 1);

    // Take the mutex on the main thread to simulate "peer is solving".
    auto lock = pdlp.acquire_for_test();

    std::vector<double> cost, ws_col, ws_row;
    auto res = pdlp.try_solve_or_snapshot(cost, ws_col, ws_row, false, 1e-4, 1.0);
    REQUIRE_FALSE(res.fresh);
    REQUIRE(res.stale_snapshot != nullptr);
    REQUIRE(res.stale_snapshot->col_value.size() == 3);
    REQUIRE(res.stale_snapshot->col_value[0] == 9.0);
    // Stale path must not advance snapshot generation.
    REQUIRE(pdlp.snapshot_generation() == 1);
    // And must not run a solve.
    REQUIRE(pdlp.solve_count.load() == 0);
}

TEST_CASE("ContestedPdlp: cold try returns null snapshot before any solve",
          "[contested_pdlp][overlap]") {
    FakePdlp pdlp;
    auto lock = pdlp.acquire_for_test();  // block the solve path
    std::vector<double> cost, ws_col, ws_row;
    auto res = pdlp.try_solve_or_snapshot(cost, ws_col, ws_row, false, 1e-4, 1.0);
    REQUIRE_FALSE(res.fresh);
    REQUIRE(res.stale_snapshot == nullptr);
}

TEST_CASE("ContestedPdlp: concurrent workers preserve one-solve-in-flight invariant",
          "[contested_pdlp][overlap]") {
    FakePdlp pdlp;
    // Wide enough window that multiple worker threads will definitely
    // contend on the lock.
    pdlp.solve_sleep_ms.store(10);

    constexpr int kWorkers = 8;
    constexpr int kIters = 20;
    std::atomic<int> total_fresh{0};
    std::atomic<int> total_stale{0};

    std::vector<std::thread> threads;
    threads.reserve(kWorkers);
    for (int w = 0; w < kWorkers; ++w) {
        threads.emplace_back([&pdlp, &total_fresh, &total_stale]() {
            std::vector<double> cost, ws_col, ws_row;
            for (int i = 0; i < kIters; ++i) {
                auto res = pdlp.try_solve_or_snapshot(cost, ws_col, ws_row, false, 1e-4, 1.0);
                if (res.fresh) {
                    total_fresh.fetch_add(1);
                } else {
                    total_stale.fetch_add(1);
                }
            }
        });
    }
    for (auto &t : threads) {
        t.join();
    }

    // One-solve-in-flight invariant: never exceeded 1 concurrent solve,
    // even though up to kWorkers threads hit the API simultaneously.
    REQUIRE(pdlp.peak_in_flight() == 1);
    // Every fresh attempt corresponded to a completed solve and bumped
    // the generation exactly once.
    REQUIRE(static_cast<int>(pdlp.snapshot_generation()) == total_fresh.load());
    // Some work actually overlapped: at least one worker got stale.
    // With 8 workers × 20 iters and a 10ms critical section, this is
    // effectively guaranteed; the REQUIRE is the actual regression
    // guard against a future refactor that accidentally serialises.
    REQUIRE(total_stale.load() > 0);
    REQUIRE(total_fresh.load() + total_stale.load() == kWorkers * kIters);
    // Sanity: solve_count equals fresh count (each fresh path runs one
    // solve, stale path runs zero).
    REQUIRE(pdlp.solve_count.load() == total_fresh.load());
}

TEST_CASE("ContestedPdlp: blocking solve() always serialises but never dead-locks",
          "[contested_pdlp][overlap]") {
    FakePdlp pdlp;
    pdlp.solve_sleep_ms.store(5);

    constexpr int kWorkers = 4;
    constexpr int kIters = 10;
    std::vector<std::thread> threads;
    threads.reserve(kWorkers);
    for (int w = 0; w < kWorkers; ++w) {
        threads.emplace_back([&pdlp]() {
            std::vector<double> cost, ws_col, ws_row;
            for (int i = 0; i < kIters; ++i) {
                (void)pdlp.solve(cost, ws_col, ws_row, false, 1e-4, 1.0);
            }
        });
    }
    for (auto &t : threads) {
        t.join();
    }
    REQUIRE(pdlp.peak_in_flight() == 1);
    // Every blocking solve completed → generation == total invocations.
    REQUIRE(static_cast<int>(pdlp.snapshot_generation()) == kWorkers * kIters);
}

TEST_CASE("ContestedPdlp: stale workers can round while one worker solves",
          "[contested_pdlp][overlap]") {
    // This is the scenario-level regression test for issue #76: one
    // worker is inside a (faked) solve, and other workers must be
    // able to read the *previous* snapshot concurrently — no waiting
    // on the mutex, no accidental serialisation.
    FakePdlp pdlp;

    // Seed one snapshot so stale readers have something to return.
    {
        ContestedPdlp::Snapshot seed;
        seed.col_value = {0.25, 0.75};
        seed.row_dual = {};
        seed.value_valid = true;
        auto guard = pdlp.acquire_for_test();
        pdlp.publish_snapshot_for_test(std::move(seed));
    }

    pdlp.solve_sleep_ms.store(50);

    std::atomic<int> stale_hits{0};
    std::atomic<bool> solver_done{false};

    // Start one worker that holds the mutex for ~50ms doing the fake
    // solve.
    std::thread solver([&pdlp, &solver_done]() {
        std::vector<double> cost, ws_col, ws_row;
        (void)pdlp.solve(cost, ws_col, ws_row, false, 1e-4, 1.0);
        solver_done.store(true);
    });

    // Give the solver thread time to actually enter the critical
    // section — otherwise the busy loop below races ahead and may
    // itself win the lock first.
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    // While the solver is inside, hammer try_solve_or_snapshot from
    // this thread and verify we repeatedly come back with fresh=false
    // and a usable stale snapshot (no blocking).
    while (!solver_done.load()) {
        std::vector<double> cost, ws_col, ws_row;
        auto res = pdlp.try_solve_or_snapshot(cost, ws_col, ws_row, false, 1e-4, 1.0);
        if (!res.fresh && res.stale_snapshot) {
            stale_hits.fetch_add(1);
        }
        // Brief yield to not pegging a whole core for the 50ms window.
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    solver.join();

    REQUIRE(stale_hits.load() > 0);
    REQUIRE(pdlp.peak_in_flight() == 1);
}
