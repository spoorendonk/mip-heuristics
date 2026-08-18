#include "mip/HighsMipSolverData.h"  // for kSolutionSource* constants
#include "rng.h"
#include "solution_pool.h"

#include <atomic>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

TEST_CASE("SolutionPool: basic operations", "[pool]") {
    SolutionPool pool(3, true);  // minimize, capacity 3

    // Empty pool
    auto snap = pool.snapshot();
    REQUIRE_FALSE(snap.has_solution);
    REQUIRE(pool.size() == 0);

    // Add solutions
    REQUIRE(pool.try_add(10.0, {1.0, 2.0}, kSolutionSourceFPR));
    REQUIRE(pool.try_add(5.0, {3.0, 4.0}, kSolutionSourceFJ));
    REQUIRE(pool.try_add(8.0, {5.0, 6.0}, kSolutionSourceLocalMIP));

    snap = pool.snapshot();
    REQUIRE(snap.has_solution);
    REQUIRE(snap.best_objective == Catch::Approx(5.0));
    REQUIRE(pool.size() == 3);

    // Adding worse solution when full should fail
    REQUIRE_FALSE(pool.try_add(15.0, {7.0, 8.0}, kSolutionSourceScylla));

    // Adding better solution when full should replace worst
    REQUIRE(pool.try_add(3.0, {9.0, 10.0}, kSolutionSourceScylla));
    snap = pool.snapshot();
    REQUIRE(snap.best_objective == Catch::Approx(3.0));

    // Sorted entries should be best-first
    auto entries = pool.sorted_entries();
    REQUIRE(entries.size() == 3);
    REQUIRE(entries[0].objective <= entries[1].objective);
    REQUIRE(entries[1].objective <= entries[2].objective);

    // Per-entry source tags must be preserved across inserts.  Best (obj=3)
    // was the Scylla replacement; second-best (obj=5) came in tagged FJ;
    // third (obj=8) came in tagged LocalMIP.  obj=10 was dropped when the
    // better obj=3 entry replaced the worst.
    REQUIRE(entries[0].source == kSolutionSourceScylla);
    REQUIRE(entries[1].source == kSolutionSourceFJ);
    REQUIRE(entries[2].source == kSolutionSourceLocalMIP);
}

TEST_CASE("SolutionPool: diversity replacement preserves source tag", "[pool]") {
    // Regression guard for #73's "source round-trips through diversity
    // replacements" claim.  Under the shared-pool model from #72, one
    // heuristic can diversity-replace another's entry; the replacement's
    // source tag must be the new entry's (not the evicted entry's).
    //
    // Triggering the *diversity-replacement* branch (not worst-replacement)
    // requires: pool full, new obj does not improve on worst, new obj within
    // kDiversityObjTolerance * |best_obj| of best, and new solution Hamming-
    // distant from the most similar existing entry by at least
    // kDiversityMinHammingFrac of the integer-var count.
    SolutionPool pool(kPoolCapacity, /*minimize=*/true);

    // 20 integer vars: 1 flip = 5% Hamming fraction exactly, which satisfies
    // the `min_frac < kDiversityMinHammingFrac` rejection (strict <).  Use
    // 2+ flips for a comfortable margin.
    constexpr int kNumIntVars = 20;
    std::vector<bool> mask(kNumIntVars, true);
    pool.set_integer_mask(mask);

    // Fill to capacity with FJ-tagged entries.  Each entry is the zero
    // vector with a single distinct bit set (positions 0..kPoolCapacity-1),
    // so any two existing entries differ by Hamming distance 2.  All share
    // the same objective so best == worst; this guarantees the new entry's
    // obj can be made to (a) not improve on worst yet (b) stay within the
    // diversity objective tolerance of best.
    constexpr double kObj = 10.0;
    for (int i = 0; i < kPoolCapacity; ++i) {
        std::vector<double> sol(kNumIntVars, 0.0);
        sol[i] = 1.0;
        REQUIRE(pool.try_add(kObj, sol, kSolutionSourceFJ));
    }
    REQUIRE(pool.size() == kPoolCapacity);

    // Build the challenger:
    //   - obj slightly worse than worst (so standard worst-replacement path
    //     is skipped: `obj >= worst.objective` -> dominated)
    //   - obj within kDiversityObjTolerance * |best_obj| (= 1.0) of best
    //   - solution flips bits at positions kPoolCapacity (=10) and 11, so
    //     Hamming distance to every existing entry is 3 (differ at entry's
    //     set bit + positions 10 and 11) = 15% > 5% threshold
    //   - tagged kSolutionSourceFPR -- the round-trip probe
    const double new_obj = kObj + 0.5;  // gap 0.5, threshold = 0.10*10 = 1.0
    std::vector<double> new_sol(kNumIntVars, 0.0);
    new_sol[kPoolCapacity] = 1.0;
    new_sol[kPoolCapacity + 1] = 1.0;

    REQUIRE(pool.try_add(new_obj, new_sol, kSolutionSourceFPR));

    // The new entry must land in the pool with its own source tag intact.
    auto entries = pool.sorted_entries();
    REQUIRE(entries.size() == kPoolCapacity);

    bool found_fpr = false;
    for (const auto& e : entries) {
        if (e.objective == Catch::Approx(new_obj)) {
            REQUIRE(e.source == kSolutionSourceFPR);
            found_fpr = true;
        } else {
            // Surviving pre-existing entries keep their original FJ tag.
            REQUIRE(e.source == kSolutionSourceFJ);
        }
    }
    REQUIRE(found_fpr);
}

TEST_CASE("SolutionPool: restart strategies", "[pool]") {
    SolutionPool pool(5, true);
    pool.try_add(10.0, {0.0, 1.0, 0.0}, kSolutionSourceFPR);
    pool.try_add(5.0, {1.0, 0.0, 1.0}, kSolutionSourceFPR);
    pool.try_add(7.0, {0.0, 0.0, 1.0}, kSolutionSourceFPR);

    Rng rng(42);
    std::vector<double> restart;

    // Simple restart (crossover or copy)
    REQUIRE(pool.get_restart(rng, restart));
    REQUIRE(restart.size() == 3);

    // Multiple restarts should not crash
    for (int i = 0; i < 50; ++i) {
        REQUIRE(pool.get_restart(rng, restart));
    }
}

// ── SolutionPool: thread-safety stress test ──

TEST_CASE("SolutionPool: concurrent try_add and get_restart", "[pool][thread-safety]") {
    SolutionPool pool(10, true);
    pool.try_add(100.0, {1.0, 2.0, 3.0}, kSolutionSourceFPR);

    constexpr int kNumThreads = 4;
    constexpr int kOpsPerThread = 200;
    std::vector<std::thread> threads;

    for (int t = 0; t < kNumThreads; ++t) {
        threads.emplace_back([&pool, t]() {
            Rng rng(42 + t);
            for (int i = 0; i < kOpsPerThread; ++i) {
                double obj = std::uniform_real_distribution<double>(1.0, 200.0)(rng);
                pool.try_add(obj, {obj, obj + 1.0, obj + 2.0}, kSolutionSourceFPR);

                std::vector<double> restart;
                pool.get_restart(rng, restart);

                pool.snapshot();
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Pool should be internally consistent
    auto entries = pool.sorted_entries();
    REQUIRE(entries.size() <= 10);
    REQUIRE(!entries.empty());
    for (size_t i = 1; i < entries.size(); ++i) {
        REQUIRE(entries[i - 1].objective <= entries[i].objective);
    }
}

// ── SolutionPool: set_on_accept callback ──

TEST_CASE("SolutionPool: on_accept fires for accepted solutions only", "[pool]") {
    SolutionPool pool(3, true);  // minimize, capacity 3

    std::vector<std::pair<std::vector<double>, int>> fired;
    pool.set_on_accept(
        [&](const std::vector<double>& sol, int src) { fired.push_back({sol, src}); });

    // Three insertions into an empty pool — all accepted.
    REQUIRE(pool.try_add(10.0, {10.0}, kSolutionSourceFPR));
    REQUIRE(pool.try_add(8.0, {8.0}, kSolutionSourceLocalMIP));
    REQUIRE(pool.try_add(6.0, {6.0}, kSolutionSourceFJ));
    REQUIRE(fired.size() == 3);
    REQUIRE(fired[0].second == kSolutionSourceFPR);
    REQUIRE(fired[1].second == kSolutionSourceLocalMIP);
    REQUIRE(fired[2].second == kSolutionSourceFJ);
    REQUIRE(fired[2].first == std::vector<double>{6.0});

    // Pool is full (capacity 3). Inserting a dominated solution must not fire.
    REQUIRE_FALSE(pool.try_add(999.0, {999.0}, kSolutionSourceFPR));
    REQUIRE(fired.size() == 3);  // unchanged

    // Inserting an improving solution fires the callback.
    REQUIRE(pool.try_add(4.0, {4.0}, kSolutionSourceFPR));
    REQUIRE(fired.size() == 4);
    REQUIRE(fired[3].first == std::vector<double>{4.0});
}

TEST_CASE("SolutionPool: on_accept callback under concurrent try_add", "[pool][thread-safety]") {
    SolutionPool pool(50, true);

    std::mutex cb_mtx;
    std::atomic<int> cb_count{0};
    std::atomic<int> accepted_count{0};

    // Set callback before spawning workers (happens-before satisfied).
    pool.set_on_accept([&](const std::vector<double>& sol, int /*src*/) {
        // Acquiring cb_mtx while the pool spin-lock is NOT held proves
        // no lock inversion: callback fires outside the pool lock.
        std::lock_guard<std::mutex> guard(cb_mtx);
        REQUIRE_FALSE(sol.empty());
        cb_count.fetch_add(1, std::memory_order_relaxed);
    });

    constexpr int kNumThreads = 4;
    constexpr int kOpsPerThread = 50;
    std::vector<std::thread> threads;
    for (int t = 0; t < kNumThreads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < kOpsPerThread; ++i) {
                double obj = static_cast<double>((t * kOpsPerThread) + i);
                if (pool.try_add(obj, {obj}, kSolutionSourceFPR)) {
                    accepted_count.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    for (auto& thr : threads) {
        thr.join();
    }

    // Every accepted insertion must have triggered exactly one callback.
    REQUIRE(cb_count.load() == accepted_count.load());
}

TEST_CASE("SolutionPool: empty pool restart returns false", "[pool][edge]") {
    SolutionPool pool(5, true);
    Rng rng(42);
    std::vector<double> out;
    REQUIRE_FALSE(pool.get_restart(rng, out));
    REQUIRE(out.empty());
}

// ── SolutionPool: single-entry pool always returns copy ──

TEST_CASE("SolutionPool: single-entry restart is always copy", "[pool][edge]") {
    SolutionPool pool(5, true);
    pool.try_add(10.0, {1.0, 2.0, 3.0}, kSolutionSourceFPR);

    Rng rng(42);
    for (int i = 0; i < 20; ++i) {
        std::vector<double> out;
        REQUIRE(pool.get_restart(rng, out));
        REQUIRE(out.size() == 3);
        // With only 1 entry, crossover can't pick 2 different entries,
        // so it falls through to copy
        REQUIRE(out[0] == Catch::Approx(1.0));
        REQUIRE(out[1] == Catch::Approx(2.0));
        REQUIRE(out[2] == Catch::Approx(3.0));
    }
}
