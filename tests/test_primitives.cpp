#include "solution_pool.h"
#include "thompson_sampler.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <random>
#include <thread>
#include <vector>

TEST_CASE("ThompsonSampler: basic operation", "[bandit]") {
    double priors[] = {2.0, 3.0, 2.5};
    ThompsonSampler sampler(3, priors, false);

    std::mt19937 rng(42);

    // Select should return valid arm indices
    for (int i = 0; i < 100; ++i) {
        int arm = sampler.select(rng);
        REQUIRE(arm >= 0);
        REQUIRE(arm < 3);
    }

    // Update should not crash
    sampler.update(0, 0);  // infeasible
    sampler.update(1, 1);  // stale
    sampler.update(2, 2);  // first feasible
    sampler.update(0, 3);  // improved

    // Stats should reflect updates
    auto s0 = sampler.stats(0);
    REQUIRE(s0.pulls == 2);
    REQUIRE(s0.alpha == Catch::Approx(3.5));  // 2.0 + 1.5
    REQUIRE(s0.beta == Catch::Approx(2.0));   // 1.0 + 1.0

    auto s1 = sampler.stats(1);
    REQUIRE(s1.pulls == 1);
    REQUIRE(s1.beta == Catch::Approx(1.25));  // 1.0 + 0.25

    auto s2 = sampler.stats(2);
    REQUIRE(s2.alpha == Catch::Approx(3.5));  // 2.5 + 1.0
}

TEST_CASE("ThompsonSampler: thread-safe mode", "[bandit]") {
    double priors[] = {2.0, 2.0};
    ThompsonSampler sampler(2, priors, true);

    std::mt19937 rng(123);
    int arm = sampler.select(rng);
    REQUIRE(arm >= 0);
    REQUIRE(arm < 2);
    sampler.update(arm, 2);
}

TEST_CASE("ThompsonSampler: effort tracking", "[bandit]") {
    double priors[] = {2.0, 2.0, 2.0};
    ThompsonSampler sampler(3, priors, false);

    // Initially no effort recorded
    auto s0 = sampler.stats(0);
    REQUIRE(s0.avg_effort == Catch::Approx(0.0));

    // First observation sets the average directly
    sampler.record_effort(0, 1000);
    s0 = sampler.stats(0);
    REQUIRE(s0.avg_effort == Catch::Approx(1000.0));

    // Subsequent observations use EMA (alpha=0.3)
    sampler.record_effort(0, 2000);
    s0 = sampler.stats(0);
    // 0.3 * 2000 + 0.7 * 1000 = 1300
    REQUIRE(s0.avg_effort == Catch::Approx(1300.0));

    // Arm 1 still has no effort
    auto s1 = sampler.stats(1);
    REQUIRE(s1.avg_effort == Catch::Approx(0.0));
}

TEST_CASE("ThompsonSampler: effort-aware select falls back without effort", "[bandit]") {
    double priors[] = {2.0, 2.0};
    ThompsonSampler sampler(2, priors, false);

    std::mt19937 rng(42);

    // Without effort observations, select_effort_aware behaves like select
    for (int i = 0; i < 50; ++i) {
        int arm = sampler.select_effort_aware(rng);
        REQUIRE(arm >= 0);
        REQUIRE(arm < 2);
    }
}

TEST_CASE("ThompsonSampler: effort-aware select prefers cheap arms", "[bandit]") {
    double priors[] = {2.0, 2.0};
    ThompsonSampler sampler(2, priors, false);

    // Give both arms equal reward history
    for (int i = 0; i < 20; ++i) {
        sampler.update(0, 2);
        sampler.update(1, 2);
    }

    // Arm 0 is 100x cheaper than arm 1
    sampler.record_effort(0, 100);
    sampler.record_effort(1, 10000);

    std::mt19937 rng(42);
    int arm0_count = 0;
    constexpr int kTrials = 200;
    for (int i = 0; i < kTrials; ++i) {
        int arm = sampler.select_effort_aware(rng);
        REQUIRE(arm >= 0);
        REQUIRE(arm < 2);
        if (arm == 0) {
            arm0_count++;
        }
    }

    // With equal reward and 100x cost difference, arm 0 should be selected
    // much more often
    REQUIRE(arm0_count > kTrials / 2);
}

TEST_CASE("SolutionPool: basic operations", "[pool]") {
    SolutionPool pool(3, true);  // minimize, capacity 3

    // Empty pool
    auto snap = pool.snapshot();
    REQUIRE_FALSE(snap.has_solution);
    REQUIRE(pool.size() == 0);

    // Add solutions
    REQUIRE(pool.try_add(10.0, {1.0, 2.0}));
    REQUIRE(pool.try_add(5.0, {3.0, 4.0}));
    REQUIRE(pool.try_add(8.0, {5.0, 6.0}));

    snap = pool.snapshot();
    REQUIRE(snap.has_solution);
    REQUIRE(snap.best_objective == Catch::Approx(5.0));
    REQUIRE(pool.size() == 3);

    // Adding worse solution when full should fail
    REQUIRE_FALSE(pool.try_add(15.0, {7.0, 8.0}));

    // Adding better solution when full should replace worst
    REQUIRE(pool.try_add(3.0, {9.0, 10.0}));
    snap = pool.snapshot();
    REQUIRE(snap.best_objective == Catch::Approx(3.0));

    // Sorted entries should be best-first
    auto entries = pool.sorted_entries();
    REQUIRE(entries.size() == 3);
    REQUIRE(entries[0].objective <= entries[1].objective);
    REQUIRE(entries[1].objective <= entries[2].objective);
}

TEST_CASE("SolutionPool: restart strategies", "[pool]") {
    SolutionPool pool(5, true);
    pool.try_add(10.0, {0.0, 1.0, 0.0});
    pool.try_add(5.0, {1.0, 0.0, 1.0});
    pool.try_add(7.0, {0.0, 0.0, 1.0});

    std::mt19937 rng(42);
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
    pool.try_add(100.0, {1.0, 2.0, 3.0});

    constexpr int kNumThreads = 4;
    constexpr int kOpsPerThread = 200;
    std::vector<std::thread> threads;

    for (int t = 0; t < kNumThreads; ++t) {
        threads.emplace_back([&pool, t]() {
            std::mt19937 rng(42 + t);
            for (int i = 0; i < kOpsPerThread; ++i) {
                double obj = std::uniform_real_distribution<double>(1.0, 200.0)(rng);
                pool.try_add(obj, {obj, obj + 1.0, obj + 2.0});

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
    REQUIRE(entries.size() > 0);
    for (size_t i = 1; i < entries.size(); ++i) {
        REQUIRE(entries[i - 1].objective <= entries[i].objective);
    }
}

// ── ThompsonSampler: concurrent select/update stress ──

TEST_CASE("ThompsonSampler: concurrent select and update", "[bandit][thread-safety]") {
    double priors[] = {2.0, 2.5, 3.0};
    ThompsonSampler sampler(3, priors, true);

    constexpr int kNumThreads = 4;
    constexpr int kOpsPerThread = 500;
    std::vector<std::thread> threads;

    for (int t = 0; t < kNumThreads; ++t) {
        threads.emplace_back([&sampler, t]() {
            std::mt19937 rng(123 + t);
            for (int i = 0; i < kOpsPerThread; ++i) {
                int arm = sampler.select(rng);
                REQUIRE(arm >= 0);
                REQUIRE(arm < 3);
                int reward = std::uniform_int_distribution<int>(0, 3)(rng);
                sampler.update(arm, reward);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // All arms should have been pulled
    int total_pulls = 0;
    for (int a = 0; a < 3; ++a) {
        auto s = sampler.stats(a);
        REQUIRE(s.pulls >= 0);
        total_pulls += s.pulls;
    }
    REQUIRE(total_pulls == kNumThreads * kOpsPerThread);
}

// ── SolutionPool: empty pool restart returns false ──

TEST_CASE("SolutionPool: empty pool restart returns false", "[pool][edge]") {
    SolutionPool pool(5, true);
    std::mt19937 rng(42);
    std::vector<double> out;
    REQUIRE_FALSE(pool.get_restart(rng, out));
    REQUIRE(out.empty());
}

// ── SolutionPool: single-entry pool always returns copy ──

TEST_CASE("SolutionPool: single-entry restart is always copy", "[pool][edge]") {
    SolutionPool pool(5, true);
    pool.try_add(10.0, {1.0, 2.0, 3.0});

    std::mt19937 rng(42);
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
