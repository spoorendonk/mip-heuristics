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
#include "pump_common.h"
#include "test_common.h"

#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstdint>
#include <mutex>
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
    // `deadline` defaults to one that never expires, which is what every
    // case but the two deadline ones wants (issue #117).
    explicit FakePdlp(Deadline deadline = {})
        : ContestedPdlp(ContestedPdlp::ForTesting{}, deadline) {}

    // Sleep inserted at the start of the fake solve to widen the
    // critical-section window so other threads are guaranteed to hit
    // `try_lock` contention during the test.  Default 0 — most tests
    // set it explicitly.
    std::atomic<int> solve_sleep_ms{0};
    // Counts completed fake solves for post-hoc assertions.
    std::atomic<int> solve_count{0};
    // Every time limit `solve_locked` was handed, in call order.  The
    // mutex serialises the fake solves, so ordinary push_back under a
    // mutex of our own is enough and the order is the solve order.
    std::mutex limits_mtx;
    std::vector<double> time_limits;

    using ContestedPdlp::acquire_for_test;
    using ContestedPdlp::publish_snapshot_for_test;

protected:
    SolveResult solve_locked(const std::vector<double>& /*modified_cost*/,
                             const std::vector<double>& /*warm_start_col_value*/,
                             const std::vector<double>& /*warm_start_row_dual*/,
                             bool /*warm_start_valid*/, double /*epsilon*/,
                             double time_limit) override {
        {
            std::scoped_lock lock(limits_mtx);
            time_limits.push_back(time_limit);
        }
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
    std::vector<double> cost;
    std::vector<double> ws_col;
    std::vector<double> ws_row;
    auto res = pdlp.try_solve_or_snapshot(cost, ws_col, ws_row, false, 1e-4);
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
    std::vector<double> cost;
    std::vector<double> ws_col;
    std::vector<double> ws_row;

    auto r1 = pdlp.try_solve_or_snapshot(cost, ws_col, ws_row, false, 1e-4);
    REQUIRE(r1.fresh);
    auto s1 = pdlp.latest_snapshot();
    REQUIRE(s1 != nullptr);
    REQUIRE(s1->generation == 1);

    auto r2 = pdlp.try_solve_or_snapshot(cost, ws_col, ws_row, false, 1e-4);
    REQUIRE(r2.fresh);
    auto s2 = pdlp.latest_snapshot();
    REQUIRE(s2 != nullptr);
    REQUIRE(s2->generation == 2);
    REQUIRE(s2->generation > s1->generation);

    // The blocking `solve()` path must publish too and use the same
    // counter (no separate channel for fresh-via-solve vs
    // fresh-via-try).
    (void)pdlp.solve(cost, ws_col, ws_row, false, 1e-4);
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

    std::vector<double> cost;
    std::vector<double> ws_col;
    std::vector<double> ws_row;
    auto res = pdlp.try_solve_or_snapshot(cost, ws_col, ws_row, false, 1e-4);
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
    std::vector<double> cost;
    std::vector<double> ws_col;
    std::vector<double> ws_row;
    auto res = pdlp.try_solve_or_snapshot(cost, ws_col, ws_row, false, 1e-4);
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
            std::vector<double> cost;
            std::vector<double> ws_col;
            std::vector<double> ws_row;
            for (int i = 0; i < kIters; ++i) {
                auto res = pdlp.try_solve_or_snapshot(cost, ws_col, ws_row, false, 1e-4);
                if (res.fresh) {
                    total_fresh.fetch_add(1);
                } else {
                    total_stale.fetch_add(1);
                }
            }
        });
    }
    for (auto& t : threads) {
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
            std::vector<double> cost;
            std::vector<double> ws_col;
            std::vector<double> ws_row;
            for (int i = 0; i < kIters; ++i) {
                (void)pdlp.solve(cost, ws_col, ws_row, false, 1e-4);
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }
    REQUIRE(pdlp.peak_in_flight() == 1);
    // Every blocking solve completed → generation == total invocations.
    REQUIRE(static_cast<int>(pdlp.snapshot_generation()) == kWorkers * kIters);
}

// `[serial]` (issue #146), and it is the same class as the deadline case
// further down even though it asserts a race outcome rather than an
// elapsed time.  The probe loop below only runs while the solver thread is
// inside its ~50 ms faked solve, and it is entered after a 5 ms sleep of
// its own; deschedule this thread across that whole window — reachable at
// `ctest -j$(nproc)`, where every other case is spawning its own thread
// pool — and the loop body never executes, leaving `stale_hits` at 0.
//
// The deterministic alternative was considered and rejected: holding `mu_`
// from a helper thread via `acquire_for_test()` and probing against a
// signalled ready-flag would remove both sleeps, but no solve would then
// run through `run_locked_with_accounting`, so the `peak_in_flight() == 1`
// assertion below — the half that proves stale reads do not accidentally
// serialise a *real* solve — would be testing an idle mutex.  Buying
// determinism with that is a bad trade, so the window stays and the tag
// protects it.
TEST_CASE("ContestedPdlp: stale workers can round while one worker solves",
          "[contested_pdlp][overlap][serial]") {
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
        std::vector<double> cost;
        std::vector<double> ws_col;
        std::vector<double> ws_row;
        (void)pdlp.solve(cost, ws_col, ws_row, false, 1e-4);
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
        std::vector<double> cost;
        std::vector<double> ws_col;
        std::vector<double> ws_row;
        auto res = pdlp.try_solve_or_snapshot(cost, ws_col, ws_row, false, 1e-4);
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

// Regression guard for the bug class `set_option_or_die` exists to catch.
// HiGHS renames `advanced` options across minor versions with no
// deprecation shim (`pdlp_scaling` -> `pdlp_scaling_mode` and
// `pdlp_e_restart_method` -> `pdlp_cupdlpc_restart_method` at v1.14.0), and
// because every `Highs` instance we build sets `output_flag=false` first, a
// rejected write is reported only through the discarded return status.  Both
// options above sat dead in ContestedPdlp's constructor from the v1.14.0
// bump until they were removed.  Pin the names the production PDLP path
// writes so the next HiGHS bump fails here instead of silently reverting the
// solver to its defaults and invalidating every benchmark.
TEST_CASE("ContestedPdlp: every PDLP option name we write exists in HiGHS",
          "[contested_pdlp][options]") {
    Highs highs;
    REQUIRE(highs.setOptionValue("output_flag", false) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("solver", "pdlp") == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("pdlp_iteration_limit", HighsInt{1000}) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("time_limit", 1.0) == HighsStatus::kOk);
    // Presolve is written `off` on the shared instance (#153).  Both the
    // name and the value string are pinned here: a rename or a re-spelling
    // of the "off" level would otherwise leave `set_option_or_die` aborting
    // a benchmark run rather than failing a test, and "ContestedPdlp: the
    // shared instance never presolves" can only see the consequence.
    REQUIRE(highs.setOptionValue("presolve", "off") == HighsStatus::kOk);

    // The option the epsilon schedule drives (#140).  cuPDLP-C resolves it
    // into all three of its termination tolerances, which is what makes it
    // the single write; both ends of the schedule must be in domain --
    // `[kMinimumKktTolerance = 1e-10, kHighsInf]` -- since a rejected
    // write would silently leave the tolerance wherever the last solve
    // left it.
    REQUIRE(highs.setOptionValue("kkt_tolerance", pump::kEpsilonInit) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("kkt_tolerance", pump::kEpsilonFloor) == HighsStatus::kOk);

    // The three options we deliberately do NOT write, kept here as named
    // negative controls: cuPDLP-C reads them too, but so does `HPresolve`,
    // so driving them from epsilon presolves a different LP.
    //
    // This loop does two jobs, and neither is guarding against a stale
    // read — `getOptionValue` leaves its out-parameter untouched on a bad
    // name and every reader here zero-initialises, so a renamed-away
    // option fails its comparison rather than passing.  What it does
    // establish, on a *fresh* `Highs`, is that these three options'
    // HiGHS defaults really are `kDefaultKktTolerance` — the constant
    // "ContestedPdlp: epsilon drives kkt_tolerance alone on every solve"
    // asserts against.  And because it runs two lines after a
    // `kkt_tolerance` write, it incidentally shows that writing
    // `kkt_tolerance` does not resolve into them at write time, which is
    // the whole premise of taking that route.
    for (const char* name : {"primal_feasibility_tolerance", "dual_feasibility_tolerance",
                             "pdlp_optimality_tolerance"}) {
        INFO("option: " << name);
        double value = 0.0;
        REQUIRE(highs.getOptionValue(name, value) == HighsStatus::kOk);
        REQUIRE(value == kDefaultKktTolerance);
    }

    // Negative control: the two dropped names must stay absent.  If a future
    // HiGHS reintroduces either, revisit the rationale comment in the
    // ContestedPdlp constructor before wiring them back up.
    REQUIRE(highs.setOptionValue("pdlp_scaling", true) != HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("pdlp_e_restart_method", HighsInt{2}) != HighsStatus::kOk);
}

// ===================================================================
// The wrapper owns the deadline, and reads it inside the lock (#117)
// ===================================================================

TEST_CASE("ContestedPdlp: a solve entered past the deadline does not run",
          "[contested_pdlp][deadline]") {
    // A `HighsTimer` of the test's own, standing in for the MIP solver's:
    // `Deadline` is two words and cares only that the clock is running.
    HighsTimer timer;
    timer.start();
    FakePdlp pdlp{make_deadline(timer, 1e-6)};
    std::vector<double> cost;
    std::vector<double> ws_col;
    std::vector<double> ws_row;

    auto res = pdlp.solve(cost, ws_col, ws_row, false, 1e-4);

    // No solve at all, and the empty result keeps `SolveResult`'s `kError`
    // default — which is what `ScyllaWorker::absorb_fresh_solve` reads as
    // "this chain is done".  Handing the sub-solver a zero or negative
    // limit instead would be worse than useless: HiGHS reads
    // `time_limit == 0` as *no limit*.
    CHECK(pdlp.solve_count.load() == 0);
    CHECK(res.status == HighsStatus::kError);
    CHECK(res.col_value.empty());
    // Nothing was published either, so a peer on the stale path still sees
    // whatever the last real solve left.
    CHECK(pdlp.snapshot_generation() == 0);
}

// `[serial]` (issue #146): the fixture spends a fixed 250 ms of a 500 ms
// deadline holding the mutex, and the assertions below need the other half
// still to be there when the waiter is scheduled.  Under `ctest -j$(nproc)`
// on a saturated host the 250 ms sleep, the thread launch and the wake
// after the unlock can together outrun the deadline, at which point
// `Deadline::remaining()` clamps to 0, `ContestedPdlp::solve` declines
// outright, and `REQUIRE(solve_count == 1)` fails on a build with nothing
// wrong with it.  Widening `kLimit` would weaken exactly the half the case
// exists to check — that the limit is read *after* the wait — so the fix
// is an unloaded machine instead.
TEST_CASE("ContestedPdlp: a solve that waited for the mutex gets the time that is left",
          "[contested_pdlp][deadline][serial]") {
    // The bug this pins: `ScyllaWorker` used to compute `time_limit -
    // timer.read()` at the top of its pump iteration and pass it here.
    // Every `kMaxStaleRounds` rounds it takes the *blocking* path, where
    // that value then sits queued behind a peer's entire PDLP solve — so
    // the solve that had waited longest was the one running on the most
    // stale limit, and could overrun the deadline by the length of the
    // wait.  The limit is now read inside the critical section, so it is
    // the time left when the solve actually starts.
    constexpr double kLimit = 0.5;
    constexpr auto kHold = std::chrono::milliseconds(250);

    HighsTimer timer;
    timer.start();
    FakePdlp pdlp{make_deadline(timer, kLimit)};
    std::vector<double> cost;
    std::vector<double> ws_col;
    std::vector<double> ws_row;

    std::thread waiter;
    {
        // Deterministic contention: hold the lock ourselves rather than
        // racing a peer solve, the same fixture the stale-snapshot cases
        // above use.
        auto guard = pdlp.acquire_for_test();
        waiter = std::thread([&] { (void)pdlp.solve(cost, ws_col, ws_row, false, 1e-4); });
        std::this_thread::sleep_for(kHold);
    }
    waiter.join();

    REQUIRE(pdlp.solve_count.load() == 1);
    std::scoped_lock lock(pdlp.limits_mtx);
    REQUIRE(pdlp.time_limits.size() == 1);
    INFO("time limit handed to the solve: " << pdlp.time_limits.front());
    // The wait was 250 ms of a 500 ms deadline, so a limit read after it
    // cannot be more than half the deadline; a limit read *before* it
    // would be the whole thing.  The bound is loose on the side that
    // machine load moves — a slower runner spends more of the deadline
    // waiting, not less.
    CHECK(pdlp.time_limits.front() < kLimit / 2.0);
    CHECK(pdlp.time_limits.front() > 0.0);
}

// ===================================================================
// The epsilon schedule reaches every tolerance cuPDLP-C terminates on,
// and reaches nothing else (#140)
// ===================================================================

// Two defects, one test case each below.
//
// The original: `solve_locked` wrote the caller's epsilon to
// `pdlp_optimality_tolerance` alone, so cuPDLP-C's D_GAP_TOL followed the
// schedule while D_PRIMAL_TOL and D_DUAL_TOL -- the two the paper's Sect.
// 2.2 actually names -- sat at `kDefaultKktTolerance` (1e-7).  cuPDLP-C
// terminates on a conjunction over all three, so a relaxed epsilon bought
// almost nothing.
//
// The first fix's: writing those two options explicitly *does* reach
// cuPDLP-C, but they are not private to it.  `Highs::run` always presolves
// this instance, and `HPresolve` reads both verbatim -- at epsilon=1e-2 it
// fixes weakly dominated columns to bounds and returns a smaller, different
// LP (measured: afiro 7/10/28 -> 8/11/30, 25fv47 666/1434/9659 ->
// 663/1427/9623).  Driving `kkt_tolerance` instead reaches exactly the same
// three cuPDLP parameters and leaves presolve bit-identical.
//
// Both cases drive the *real* `ContestedPdlp`, not the `FakePdlp` double:
// the double overrides `solve_locked`, which is the function under test.
TEST_CASE("ContestedPdlp: epsilon drives kkt_tolerance alone on every solve",
          "[contested_pdlp][options][scylla]") {
    // `ContestedPdlp`'s constructor needs a real `mipdata_`, whose `init`
    // reads `parallel::num_threads()`; see the note at the other
    // `build_bare_mipsolver` call sites.  A no-op once started.
    highs::parallel::initialize_scheduler();

    Highs highs;
    highs.setOptionValue("output_flag", false);
    HighsCallback cb(&highs);
    auto mipsolver = build_bare_mipsolver(highs, cb, "flugpl.mps");

    ContestedPdlp pdlp(*mipsolver, 200);
    REQUIRE(pdlp.initialized());

    const std::vector<double> cost(static_cast<size_t>(pdlp.num_col()), 0.0);
    const std::vector<double> empty;

    // Two solves at two epsilons, because the schedule needs the write to
    // be per-solve: a construction-time write would pass a single-solve
    // check and still never follow the decay.  Both ends of the schedule
    // are covered for domain by the option-name pin above.
    for (const double epsilon : {pump::kEpsilonInit, 5e-4}) {
        INFO("epsilon: " << epsilon);
        static_cast<void>(pdlp.solve(cost, empty, empty, false, epsilon));
        const auto tol = pdlp.tolerances_for_test();
        CHECK(tol.kkt == epsilon);
        // And the half that keeps LP presolve out of it: the three
        // options `HPresolve` also reads must never move.
        CHECK(tol.primal_feasibility == kDefaultKktTolerance);
        CHECK(tol.dual_feasibility == kDefaultKktTolerance);
        CHECK(tol.pdlp_optimality == kDefaultKktTolerance);
    }
}

// The option-value check above pins *what we write*.  This pins *what the
// write does*, which is the half no option-name check can reach: the whole
// route depends on `getUserParamsFromOptions`'s `if (kkt_tolerance !=
// kDefaultKktTolerance)` override, an upstream "changed from its default"
// branch.  If a HiGHS bump removes, renames or re-conditions it, epsilon
// stops reaching cuPDLP-C's termination check entirely and every option
// name we write still exists -- so only an effect test catches it.
//
// Each epsilon gets its own `ContestedPdlp`, so neither solve can inherit
// the other's iterate through the wrapped instance; both are cold starts
// against the same LP with the same costs, and the only difference is the
// tolerance.
TEST_CASE("ContestedPdlp: a looser epsilon really does terminate PDLP sooner",
          "[contested_pdlp][options][scylla]") {
    highs::parallel::initialize_scheduler();

    Highs highs;
    highs.setOptionValue("output_flag", false);
    HighsCallback cb(&highs);
    auto mipsolver = build_bare_mipsolver(highs, cb, "flugpl.mps");

    const std::vector<double> empty;

    auto iters_at = [&](double epsilon) {
        ContestedPdlp pdlp(*mipsolver, 100000);
        REQUIRE(pdlp.initialized());
        // A non-zero objective, so the gap term is a live part of the
        // termination check rather than trivially satisfied at zero.
        const std::vector<double> cost(static_cast<size_t>(pdlp.num_col()), 1.0);
        const auto result = pdlp.solve(cost, empty, empty, false, epsilon);
        return result.pdlp_iters;
    };

    const HighsInt loose = iters_at(pump::kEpsilonInit);
    const HighsInt tight = iters_at(pump::kEpsilonFloor);
    // Third arm, and it is the one that pins the *loose* end.  At exactly
    // `kDefaultKktTolerance` the override does not fire, so this is the
    // solve cuPDLP-C would do with `kkt_tolerance` untouched.
    const HighsInt untouched = iters_at(kDefaultKktTolerance);
    // Recorded for the next reader: 200 / 840 / 760 on flugpl as of #153.
    // These moved when `presolve=off` landed on the shared instance — they
    // were 80 / 1160 / 840, taken on the *reduced* LP, and the ordering
    // this case asserts is what is stable, not the magnitudes.  Do not
    // treat a change in them as a regression on its own.
    INFO("iterations at kEpsilonInit=" << pump::kEpsilonInit << ": " << loose);
    INFO("iterations at kEpsilonFloor=" << pump::kEpsilonFloor << ": " << tight);
    INFO("iterations at the untouched default: " << untouched);

    // Strict: the schedule's whole purpose is that its first solves are
    // cheaper than its last ones.  Equality would mean epsilon reaches
    // nothing that terminates the solver — which is what an outright
    // removal of `getUserParamsFromOptions`'s override branch looks like, since
    // that branch is the only path by which `kkt_tolerance` reaches
    // cuPDLP-C: with it gone both epsilons produce identical parameters
    // on a deterministic solver and this fails.
    CHECK(loose < tight);
    CHECK(loose > 0);
    // Monotone dependence alone would not be enough.  A bump that
    // re-conditioned the override to, say, `if (kkt_tolerance <
    // kDefaultKktTolerance)` would silently stop the *loose* end from
    // firing — epsilon=0.01 would fall back to the 1e-7 defaults while
    // epsilon=1e-8 still fired, `tight` would still exceed `loose`, and
    // the schedule's cheap early solves would quietly stop happening with
    // the two checks above still green.  This closes that: the loose end
    // must beat the untouched default, which it cannot do by falling back
    // to it.
    CHECK(loose < untouched);
}

// The cause behind #152's half-deadline ratio, pinned where the ratio
// cannot pin it.
//
// `deadline_.remaining()` says "this solve may run for at most this long",
// and two things downstream of `options_.time_limit` charge against it:
// LP presolve, from the *wrapped instance's accumulated* run time
// (`runPresolve`'s `left = options_.time_limit - timer_.read()`), and
// cuPDLP-C, from *this solve's* elapsed time (`dSolvingBeg` is set at
// entry to `PDHG_Solve`).  Reusing one `Highs` across a whole dispatch
// made the first of those a growing number compared against a shrinking
// one; measured directly on `bell5` at a 4 s limit, the accumulated run
// time reached 1.944 s while the remaining limit fell to 1.943, and from
// that solve on every `run()` returned `Time limit reached` with 0 PDLP
// iterations and `value_valid=0` — which `ScyllaWorker::absorb_fresh_solve`
// retires the chain on.
//
// `test_deadline.cpp`'s "a clock-bound Scylla dispatch spends its whole
// limit" measures the *symptom*.  It cannot tell this fix from the one the
// issue explicitly rules out — inflating the limit handed to the
// sub-solver, e.g. writing `remaining + accumulated` — which would restore
// the ratio while leaving LP presolve budgeting against an ever-growing
// origin, and would over-grant cuPDLP-C's `D_TIME_LIM` by that same
// growing amount on top.  This case fails for that fix and passes only for
// one that resets the origin.
//
// The assertion is exact rather than a threshold, which is why it needs no
// slack and no `[serial]`: with the clock zeroed per solve,
// `run_time_for_test()` covers a strict sub-interval of the wall time
// measured around the enclosing `solve()` call, so `reported <= wall` holds
// by construction on any machine at any load.  Without the reset the
// reported value is the *sum* over all `kSolves` runs, which exceeds one
// call's wall time by roughly `kSolves`x.  A starved runner stretches both
// sides together.
TEST_CASE("ContestedPdlp: the wrapped instance's clock does not accumulate across solves",
          "[contested_pdlp][deadline][scylla]") {
    highs::parallel::initialize_scheduler();

    Highs highs;
    highs.setOptionValue("output_flag", false);
    HighsCallback cb(&highs);
    auto mipsolver = build_bare_mipsolver(highs, cb, "flugpl.mps");

    // A cap high enough that each solve does real work — the point is for
    // the accumulated total to be visibly larger than any one solve.
    ContestedPdlp pdlp(*mipsolver, 100000);
    REQUIRE(pdlp.initialized());

    // Non-zero costs so the gap term is live rather than trivially
    // satisfied at zero, and a tight epsilon so a solve is not over in one
    // iteration.
    const std::vector<double> cost(static_cast<size_t>(pdlp.num_col()), 1.0);
    const std::vector<double> empty;

    constexpr int kSolves = 6;
    double last_wall = 0.0;
    double total_wall = 0.0;
    for (int i = 0; i < kSolves; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        const auto result = pdlp.solve(cost, empty, empty, false, pump::kEpsilonFloor);
        const auto t1 = std::chrono::steady_clock::now();
        // Every solve must actually have run: a run that did nothing would
        // make both sides of the comparison below zero and the case
        // vacuous.  This is also the assertion that would have caught the
        // defect's *downstream* symptom directly — after the crossing,
        // `pdlp_iters` went to 0 and stayed there.
        INFO("solve " << i << " iters " << result.pdlp_iters);
        REQUIRE(result.pdlp_iters > 0);
        last_wall = std::chrono::duration<double>(t1 - t0).count();
        total_wall += last_wall;
    }

    const double reported = pdlp.run_time_for_test();
    // Slack expressed as a fraction of the measured total rather than as an
    // absolute constant, so the bound says the same thing on a machine of
    // any speed.  It absorbs the one way `reported <= last_wall` can fail
    // without the bug — `HighsTimer` bottoms out in `high_resolution_clock`,
    // which libstdc++ aliases to the non-monotonic `system_clock`, so a
    // forward step inside the last solve can inflate `reported` against the
    // `steady_clock` measured around it.  It does not blunt the assertion:
    // the accumulating value overshoots `last_wall` by (kSolves-1)/kSolves
    // of the total, four times this slack.
    //
    // Measured on the development machine: `reported` 0.479 ms against a
    // `last_wall` of 0.502 ms and a `total_wall` of 3.20 ms, i.e. a bound of
    // 0.822 ms met with 1.7x room, where the accumulating value is 3.04 ms
    // and misses it by 3.7x.
    const double slack = 0.1 * total_wall;
    INFO("reported run time " << reported << ", last solve wall " << last_wall << ", total wall "
                              << total_wall << " over " << kSolves << " solves, bound "
                              << (last_wall + slack));
    // The hook reads something: a stub returning 0.0, or a `Highs` whose
    // clock never ran, would satisfy the bound below without the reset
    // doing anything.
    CHECK(reported > 0.0);
    // The bound itself.  One solve's worth, not six.
    CHECK(reported <= last_wall + slack);
}

// The per-solve deadline actually reaches the wrapped instance, and it is
// written per solve rather than once.
//
// This closes a hole that predates #152 but that #152 is the moment to
// close, because #152 is *about* that write.  Deleting
// `set_option_or_die(highs_, "time_limit", time_limit)` from `solve_locked`
// left the entire suite green on a build that handed cuPDLP-C no deadline
// at all: every wall-clock assertion in `test_deadline.cpp` is an *upper*
// bound, which an unlimited sub-solve does not violate, and #152's own
// ratio assertion is a *lower* bound, which an unlimited sub-solve makes
// easier rather than harder.  `a solve that waited for the mutex gets the
// time that is left` looks like it covers this and does not: it drives
// `FakePdlp`, which overrides `solve_locked` and records the argument
// instead of performing the option write on a real `Highs`.
//
// Why a readback rather than an effect, against #140's precedent.  The
// observable effect of a *shorter* limit is a truncated solve, and every
// way of provoking one here is a race against a live clock — unlike #140's
// tolerance, whose effect (iteration count at a fixed tolerance) is
// deterministic on a deterministic solver.  What is deterministic is the
// value itself and, more usefully, the fact that it *moves*: `time_limit`
// is `Deadline::remaining()`, so under a finite deadline consecutive solves
// must see a strictly smaller one.  That is what separates the write being
// present from it being hoisted to the constructor, and it is also what an
// "inflate the limit" fix — `remaining + accumulated`, the one #152's issue
// rules out — fails, since those two move in opposite directions by
// construction.
TEST_CASE("ContestedPdlp: every solve is given the deadline's remaining time",
          "[contested_pdlp][deadline][scylla]") {
    highs::parallel::initialize_scheduler();

    Highs highs;
    highs.setOptionValue("output_flag", false);
    HighsCallback cb(&highs);
    // A finite limit, generous enough that it cannot expire mid-case: the
    // point is that `remaining()` shrinks, not that it runs out.
    constexpr double kDeadlineSeconds = 30.0;
    auto mipsolver = build_bare_mipsolver(highs, cb, "flugpl.mps", kDeadlineSeconds);

    ContestedPdlp pdlp(*mipsolver, 100000);
    REQUIRE(pdlp.initialized());

    const std::vector<double> cost(static_cast<size_t>(pdlp.num_col()), 1.0);
    const std::vector<double> empty;

    static_cast<void>(pdlp.solve(cost, empty, empty, false, pump::kEpsilonFloor));
    const double first = pdlp.tolerances_for_test().time_limit;
    static_cast<void>(pdlp.solve(cost, empty, empty, false, pump::kEpsilonFloor));
    const double second = pdlp.tolerances_for_test().time_limit;

    INFO("time_limit after solve 1: " << first << ", after solve 2: " << second);
    // A limit was written at all.  HiGHS's own default is `kHighsInf`, so
    // this is exactly what a deleted write reads back as.
    CHECK(first < kHighsInf);
    // It is the deadline's remaining time, not some larger constant.
    CHECK(first > 0.0);
    CHECK(first <= kDeadlineSeconds);
    // And it is written per solve.  A write hoisted to the constructor, or
    // any value that does not track `remaining()`, fails here.
    CHECK(second < first);
}

// ===================================================================
// LP presolve is off on the shared instance (#153)
// ===================================================================

// The wrapped instance must never presolve, and the reason is the warm
// start rather than the cost of presolving.
//
// `Highs::optimizeModel` cannot take its presolve-skip branch under
// `solver=pdlp` (the `solver_will_use_basis` conjunct decides it), so at
// default options every pump solve presolved.  On `kReduced` HiGHS then
// hands `solveLp(reduced_lp, ...)` the *full-model* solution object, and
// `solveLpCupdlp` resizes it to the reduced LP's dimensions while passing
// `value_valid` / `dual_valid` through unchanged — so `PDHG_PreSolve` reads
// a truncated prefix of our warm start as its hot start, on columns and
// rows that are not the ones those values belong to.
//
// This case pins the status and the shape.  The *effect* — that a warm
// start now arrives intact — is the case below it; both are needed,
// because the status alone would survive a HiGHS that stopped honouring
// `presolve=off`'s consequences and the effect alone would not say which
// mechanism produced it.
//
// The status readback is sound after `solve()` returns: `clearPresolve`
// runs at the head of the next `run`/`presolve` or on a model-modifying
// call, and `changeColsCost` precedes `run` inside `solve_locked`, so the
// value stands for the last solve.  What is deliberately *not* asserted is
// the presolve *time* — under `kNotPresolved` HiGHS emits no presolve log
// line and `run_data_.presolve_time` is ~0, but neither is a sound
// assertion about work not done.
TEST_CASE("ContestedPdlp: the shared instance never presolves", "[contested_pdlp][options]") {
    highs::parallel::initialize_scheduler();

    Highs highs;
    highs.setOptionValue("output_flag", false);
    HighsCallback cb(&highs);
    auto mipsolver = build_bare_mipsolver(highs, cb, "flugpl.mps");

    ContestedPdlp pdlp(*mipsolver, 100000);
    REQUIRE(pdlp.initialized());

    const std::vector<double> cost(static_cast<size_t>(pdlp.num_col()), 1.0);
    const std::vector<double> empty;
    const auto result = pdlp.solve(cost, empty, empty, false, pump::kEpsilonInit);

    // The solve ran, so the status below is about a real `run()` rather
    // than an instance that never solved anything.
    REQUIRE(result.pdlp_iters > 0);

    // With the constructor's `presolve=off` write deleted this reads
    // `kReduced` on this instance — `build_bare_mipsolver` turns HiGHS's
    // *MIP* presolve off, so the LP the pump wraps is the raw relaxation
    // and every bundled instance reduces.
    CHECK(pdlp.presolve_status_for_test() == HighsPresolveStatus::kNotPresolved);

    // And the solver's output is in the full model's column space, with no
    // postsolve step between: this is what `absorb_fresh_solve` stores and
    // what `ScyllaWorker` asserts on.
    CHECK(result.col_value.size() == static_cast<size_t>(pdlp.num_col()));
    CHECK(result.row_dual.size() == static_cast<size_t>(mipsolver->model_->num_row_));
}

// The effect the status above exists for: a warm start survives the trip
// into cuPDLP-C.
//
// Solve cold, then solve again from the point that solve returned, with the
// same costs and the same epsilon.  cuPDLP-C runs its termination check on
// every one of the first ten iterations, so a hot start already inside
// epsilon terminates at `nIter == 0`.  That is an exact number, not a
// threshold, which is why this case needs no slack and no `[serial]`.
//
// With presolve back on, the second count is *not* zero: the previous
// solve's `x_bar` is truncated onto the reduced LP's first `num_col_`
// columns, which are not the columns those values belong to, so the warm
// start is a worse starting point than a cold one and the solver has to
// work its way back.  Measured on this fixture by deleting the
// constructor's `presolve=off` write: cold 80 / warm 120 — the warm start
// costs *more* than the cold one — against cold 200 / warm 0 as shipped.
TEST_CASE("ContestedPdlp: a warm start at the previous optimum reaches cuPDLP-C intact",
          "[contested_pdlp][scylla]") {
    highs::parallel::initialize_scheduler();

    Highs highs;
    highs.setOptionValue("output_flag", false);
    HighsCallback cb(&highs);
    auto mipsolver = build_bare_mipsolver(highs, cb, "flugpl.mps");

    ContestedPdlp pdlp(*mipsolver, 100000);
    REQUIRE(pdlp.initialized());

    // The same non-zero cost vector the "looser epsilon" case uses, so the
    // gap term is a live part of the termination check rather than
    // trivially satisfied at zero.
    const std::vector<double> cost(static_cast<size_t>(pdlp.num_col()), 1.0);
    const std::vector<double> empty;

    const auto cold = pdlp.solve(cost, empty, empty, false, pump::kEpsilonInit);
    REQUIRE(cold.value_valid);
    REQUIRE(cold.dual_valid);
    // A cold solve that already terminated at 0 would make the comparison
    // below vacuous.
    REQUIRE(cold.pdlp_iters > 0);
    REQUIRE(cold.col_value.size() == static_cast<size_t>(pdlp.num_col()));
    REQUIRE(cold.row_dual.size() == static_cast<size_t>(mipsolver->model_->num_row_));

    const auto warm = pdlp.solve(cost, cold.col_value, cold.row_dual, true, pump::kEpsilonInit);

    INFO("cold iterations: " << cold.pdlp_iters << ", warm iterations: " << warm.pdlp_iters);
    // The exact assertion: a start already within epsilon is recognised on
    // the very first termination check.
    CHECK(warm.pdlp_iters == 0);
    // Stated separately so a future cuPDLP-C that checked termination less
    // eagerly would still be held to the point of the case — the warm start
    // must be *worth* something.
    CHECK(warm.pdlp_iters < cold.pdlp_iters);
}
