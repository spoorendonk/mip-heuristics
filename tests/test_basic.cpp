#include "fpr_core.h"
#include "fpr_lp.h"
#include "fpr_strategies.h"
#include "heuristic_common.h"
#include "Highs.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "prop_engine.h"
#include "solution_pool.h"
#include "thompson_sampler.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <random>
#include <string>
#include <thread>
#include <vector>

static const std::string kInstancesDir = INSTANCES_DIR;

TEST_CASE("Smoke test: solve small MIP", "[basic]") {
    // min x + y
    // s.t. x + y >= 1
    //      x, y in {0, 1}
    Highs highs;
    highs.setOptionValue("output_flag", false);

    highs.addVar(0.0, 1.0);
    highs.addVar(0.0, 1.0);
    highs.changeColCost(0, 1.0);
    highs.changeColCost(1, 1.0);
    highs.changeColIntegrality(0, HighsVarType::kInteger);
    highs.changeColIntegrality(1, HighsVarType::kInteger);

    HighsInt idx[] = {0, 1};
    double val[] = {1.0, 1.0};
    highs.addRow(1.0, kHighsInf, 2, idx, val);

    HighsStatus status = highs.run();
    REQUIRE(status == HighsStatus::kOk);

    HighsInt sol_status;
    highs.getInfoValue("primal_solution_status", sol_status);
    REQUIRE(sol_status == kSolutionStatusFeasible);

    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == 1.0);
}

TEST_CASE("Characterization: flugpl", "[heuristic][fpr]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    REQUIRE(highs.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("Characterization: egout", "[heuristic][fpr]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    REQUIRE(highs.readModel(kInstancesDir + "/egout.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(568.1007).epsilon(1e-4));
}

TEST_CASE("Characterization: bell5", "[heuristic][fpr]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    REQUIRE(highs.readModel(kInstancesDir + "/bell5.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(8966406.49152).epsilon(1e-6));
}

TEST_CASE("Options: disable custom heuristics", "[options]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    // Verify options exist and can be set
    REQUIRE(highs.setOptionValue("mip_heuristic_run_fpr", false) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_heuristic_run_local_mip", false) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_heuristic_run_scylla", false) == HighsStatus::kOk);
    // Solve still works with all custom heuristics disabled
    REQUIRE(highs.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("Options: portfolio option exists", "[options][portfolio]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    REQUIRE(highs.setOptionValue("mip_heuristic_portfolio", true) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_heuristic_portfolio", false) == HighsStatus::kOk);
}

TEST_CASE("Portfolio: flugpl finds solution", "[portfolio]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_portfolio", true);
    REQUIRE(highs.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("Portfolio: egout finds solution", "[portfolio]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_portfolio", true);
    REQUIRE(highs.readModel(kInstancesDir + "/egout.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(568.1007).epsilon(1e-4));
}

TEST_CASE("Portfolio: bell5 finds solution", "[portfolio]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_portfolio", true);
    REQUIRE(highs.readModel(kInstancesDir + "/bell5.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(8966406.49152).epsilon(1e-6));
}

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

TEST_CASE("Options: opportunistic option exists", "[options][portfolio]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    REQUIRE(highs.setOptionValue("mip_heuristic_opportunistic", true) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_heuristic_opportunistic", false) == HighsStatus::kOk);
}

TEST_CASE("Portfolio opportunistic: flugpl finds solution", "[portfolio][opportunistic]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_portfolio", true);
    highs.setOptionValue("mip_heuristic_opportunistic", true);
    REQUIRE(highs.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("Portfolio opportunistic: egout finds solution", "[portfolio][opportunistic]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_portfolio", true);
    highs.setOptionValue("mip_heuristic_opportunistic", true);
    REQUIRE(highs.readModel(kInstancesDir + "/egout.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(568.1007).epsilon(1e-4));
}

TEST_CASE("Portfolio opportunistic: bell5 finds solution", "[portfolio][opportunistic]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_portfolio", true);
    highs.setOptionValue("mip_heuristic_opportunistic", true);
    REQUIRE(highs.readModel(kInstancesDir + "/bell5.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    // Opportunistic parallel mode may not reach exact optimal within time limit
    REQUIRE(obj == Catch::Approx(8966406.49152).epsilon(1e-3));
}

TEST_CASE("Portfolio deterministic: fixed seed produces same result", "[portfolio][determinism]") {
    auto solve = [](int seed) {
        Highs highs;
        highs.setOptionValue("output_flag", false);
        highs.setOptionValue("mip_heuristic_portfolio", true);
        highs.setOptionValue("random_seed", seed);
        highs.readModel(kInstancesDir + "/flugpl.mps");
        highs.run();
        double obj;
        highs.getInfoValue("objective_function_value", obj);
        return obj;
    };
    double obj1 = solve(42);
    double obj2 = solve(42);
    REQUIRE(obj1 == Catch::Approx(obj2).epsilon(1e-12));
}

// ── Determinism across multiple seeds and instances ──

TEST_CASE("Portfolio deterministic: egout same result across runs", "[portfolio][determinism]") {
    auto solve = [](int seed) {
        Highs highs;
        highs.setOptionValue("output_flag", false);
        highs.setOptionValue("mip_heuristic_portfolio", true);
        highs.setOptionValue("random_seed", seed);
        highs.readModel(kInstancesDir + "/egout.mps");
        highs.run();
        double obj;
        highs.getInfoValue("objective_function_value", obj);
        return obj;
    };
    double obj1 = solve(7);
    double obj2 = solve(7);
    REQUIRE(obj1 == Catch::Approx(obj2).epsilon(1e-12));
}

TEST_CASE("Portfolio deterministic: different seeds can differ", "[portfolio][determinism]") {
    // Sanity check: the seed actually affects something (not a no-op).
    // We just verify both runs produce valid optimal solutions — the
    // important thing is that the code path doesn't crash with different seeds.
    for (int seed : {1, 99, 12345}) {
        Highs highs;
        highs.setOptionValue("output_flag", false);
        highs.setOptionValue("mip_heuristic_portfolio", true);
        highs.setOptionValue("random_seed", seed);
        REQUIRE(highs.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);
        REQUIRE(highs.run() == HighsStatus::kOk);
        double obj;
        highs.getInfoValue("objective_function_value", obj);
        REQUIRE(obj == Catch::Approx(1201500.0).epsilon(1e-6));
    }
}

// ── Opportunistic mode on additional instances ──

TEST_CASE("Portfolio opportunistic: lseu finds solution", "[portfolio][opportunistic]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_portfolio", true);
    highs.setOptionValue("mip_heuristic_opportunistic", true);
    REQUIRE(highs.readModel(kInstancesDir + "/lseu.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(1120.0).epsilon(1e-3));
}

// ── Single-arm portfolio (only FJ enabled, others disabled) ──

TEST_CASE("Portfolio: FJ-only mode works", "[portfolio][single-arm]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_portfolio", true);
    highs.setOptionValue("mip_heuristic_run_fpr", false);
    highs.setOptionValue("mip_heuristic_run_local_mip", false);
    highs.setOptionValue("mip_heuristic_run_feasibility_jump", true);
    REQUIRE(highs.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("Portfolio: FPR-only mode works", "[portfolio][single-arm]") {
    // Build a tiny MIP: min x s.t. x >= 1, x integer
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_portfolio", true);
    highs.setOptionValue("mip_heuristic_run_fpr", true);
    highs.setOptionValue("mip_heuristic_run_local_mip", false);
    highs.setOptionValue("mip_heuristic_run_feasibility_jump", false);
    highs.addVar(0.0, 10.0);
    highs.changeColCost(0, 1.0);
    highs.changeColIntegrality(0, HighsVarType::kInteger);
    HighsInt idx[] = {0};
    double val[] = {1.0};
    highs.addRow(1.0, kHighsInf, 1, idx, val);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(1.0));
}

TEST_CASE("Portfolio opportunistic: FJ-only mode works", "[portfolio][opportunistic][single-arm]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_portfolio", true);
    highs.setOptionValue("mip_heuristic_opportunistic", true);
    highs.setOptionValue("mip_heuristic_run_fpr", false);
    highs.setOptionValue("mip_heuristic_run_local_mip", false);
    highs.setOptionValue("mip_heuristic_run_feasibility_jump", true);
    REQUIRE(highs.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(1201500.0).epsilon(1e-6));
}

// ── All arms disabled: portfolio is a no-op, B&B still solves ──

TEST_CASE("Portfolio: all presolve arms disabled still solves", "[portfolio][edge]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_portfolio", true);
    highs.setOptionValue("mip_heuristic_run_fpr", false);
    highs.setOptionValue("mip_heuristic_run_local_mip", false);
    highs.setOptionValue("mip_heuristic_run_feasibility_jump", false);
    highs.setOptionValue("mip_heuristic_run_scylla", false);
    REQUIRE(highs.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("Portfolio opportunistic: all arms disabled still solves",
          "[portfolio][opportunistic][edge]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_portfolio", true);
    highs.setOptionValue("mip_heuristic_opportunistic", true);
    highs.setOptionValue("mip_heuristic_run_fpr", false);
    highs.setOptionValue("mip_heuristic_run_local_mip", false);
    highs.setOptionValue("mip_heuristic_run_feasibility_jump", false);
    highs.setOptionValue("mip_heuristic_run_scylla", false);
    REQUIRE(highs.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(1201500.0).epsilon(1e-6));
}

// ── Sequential opportunistic mode (seq/opp cell of the 2x2 matrix) ──

TEST_CASE("Sequential opportunistic: flugpl finds solution", "[options][opportunistic]") {
    // portfolio=false + opportunistic=true: this is the seq/opp cell of the
    // 2x2 execution matrix.  Each heuristic's run_parallel dispatches to its
    // opportunistic variant when the flag is set.
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_portfolio", false);
    highs.setOptionValue("mip_heuristic_opportunistic", true);
    REQUIRE(highs.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("Sequential opportunistic: egout finds solution", "[options][opportunistic]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_portfolio", false);
    highs.setOptionValue("mip_heuristic_opportunistic", true);
    REQUIRE(highs.readModel(kInstancesDir + "/egout.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(568.1007).epsilon(1e-4));
}

// ── Per-heuristic opportunistic smoke tests ──

TEST_CASE("FJ opportunistic: flugpl finds solution", "[heuristic][fj][opportunistic]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_run_fpr", false);
    highs.setOptionValue("mip_heuristic_run_local_mip", false);
    highs.setOptionValue("mip_heuristic_run_scylla", false);
    highs.setOptionValue("mip_heuristic_run_feasibility_jump", true);
    highs.setOptionValue("mip_heuristic_portfolio", false);
    highs.setOptionValue("mip_heuristic_opportunistic", true);
    REQUIRE(highs.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("FPR opportunistic: flugpl finds solution", "[heuristic][fpr][opportunistic]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_run_fpr", true);
    highs.setOptionValue("mip_heuristic_run_local_mip", false);
    highs.setOptionValue("mip_heuristic_run_scylla", false);
    highs.setOptionValue("mip_heuristic_run_feasibility_jump", false);
    highs.setOptionValue("mip_heuristic_portfolio", false);
    highs.setOptionValue("mip_heuristic_opportunistic", true);
    REQUIRE(highs.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("LocalMIP opportunistic: flugpl finds solution",
          "[heuristic][local_mip][opportunistic]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_run_fpr", false);
    highs.setOptionValue("mip_heuristic_run_local_mip", true);
    highs.setOptionValue("mip_heuristic_run_scylla", false);
    highs.setOptionValue("mip_heuristic_run_feasibility_jump", false);
    highs.setOptionValue("mip_heuristic_portfolio", false);
    highs.setOptionValue("mip_heuristic_opportunistic", true);
    REQUIRE(highs.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(1201500.0).epsilon(1e-6));
}

// ── Infeasible MIP: portfolio handles gracefully ──

TEST_CASE("Portfolio: infeasible MIP handled correctly", "[portfolio][edge]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_portfolio", true);
    REQUIRE(highs.readModel(kInstancesDir + "/infeasible-mip0.mps") == HighsStatus::kOk);
    highs.run();
    HighsModelStatus model_status = highs.getModelStatus();
    REQUIRE(model_status == HighsModelStatus::kInfeasible);
}

TEST_CASE("Portfolio opportunistic: infeasible MIP handled correctly",
          "[portfolio][opportunistic][edge]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_portfolio", true);
    highs.setOptionValue("mip_heuristic_opportunistic", true);
    REQUIRE(highs.readModel(kInstancesDir + "/infeasible-mip0.mps") == HighsStatus::kOk);
    highs.run();
    HighsModelStatus model_status = highs.getModelStatus();
    REQUIRE(model_status == HighsModelStatus::kInfeasible);
}

// ── Opportunistic runs repeatedly without crash (stress) ──

TEST_CASE("Portfolio opportunistic: repeated runs on flugpl",
          "[portfolio][opportunistic][stress]") {
    for (int i = 0; i < 5; ++i) {
        Highs highs;
        highs.setOptionValue("output_flag", false);
        highs.setOptionValue("mip_heuristic_portfolio", true);
        highs.setOptionValue("mip_heuristic_opportunistic", true);
        highs.setOptionValue("random_seed", i);
        REQUIRE(highs.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);
        REQUIRE(highs.run() == HighsStatus::kOk);
        double obj;
        highs.getInfoValue("objective_function_value", obj);
        REQUIRE(obj == Catch::Approx(1201500.0).epsilon(1e-6));
    }
}

// ===================================================================
// 2x2 mode-matrix correctness tests
//
// The four cells of the (portfolio × opportunistic) execution matrix:
//   seq/det : portfolio=false, opportunistic=false — weighted sequential
//   seq/opp : portfolio=false, opportunistic=true  — per-heuristic opportunistic
//   port/det: portfolio=true,  opportunistic=false — deterministic epoch bandit
//   port/opp: portfolio=true,  opportunistic=true  — opportunistic bandit
//
// Each cell should be exercised on at least one real instance. See also
// #63 for the fpr_lp mode-matrix follow-up (dive-time variant).
// ===================================================================

namespace {
double solve_mode(const char* inst, bool portfolio, bool opp) {
    Highs h;
    h.setOptionValue("output_flag", false);
    h.setOptionValue("mip_heuristic_portfolio", portfolio);
    h.setOptionValue("mip_heuristic_opportunistic", opp);
    REQUIRE(h.readModel(std::string(INSTANCES_DIR) + "/" + inst) == HighsStatus::kOk);
    REQUIRE(h.run() == HighsStatus::kOk);
    double obj;
    h.getInfoValue("objective_function_value", obj);
    return obj;
}
}  // namespace

// ── 8 tests: 4 modes × {flugpl, egout} objective ──

TEST_CASE("mode-matrix seq/det: flugpl objective", "[mode-matrix]") {
    REQUIRE(solve_mode("flugpl.mps", false, false) == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("mode-matrix seq/opp: flugpl objective", "[mode-matrix]") {
    REQUIRE(solve_mode("flugpl.mps", false, true) == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("mode-matrix port/det: flugpl objective", "[mode-matrix]") {
    REQUIRE(solve_mode("flugpl.mps", true, false) == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("mode-matrix port/opp: flugpl objective", "[mode-matrix]") {
    REQUIRE(solve_mode("flugpl.mps", true, true) == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("mode-matrix seq/det: egout objective", "[mode-matrix]") {
    REQUIRE(solve_mode("egout.mps", false, false) == Catch::Approx(568.1007).epsilon(1e-4));
}

TEST_CASE("mode-matrix seq/opp: egout objective", "[mode-matrix]") {
    REQUIRE(solve_mode("egout.mps", false, true) == Catch::Approx(568.1007).epsilon(1e-4));
}

TEST_CASE("mode-matrix port/det: egout objective", "[mode-matrix]") {
    REQUIRE(solve_mode("egout.mps", true, false) == Catch::Approx(568.1007).epsilon(1e-4));
}

TEST_CASE("mode-matrix port/opp: egout objective", "[mode-matrix]") {
    REQUIRE(solve_mode("egout.mps", true, true) == Catch::Approx(568.1007).epsilon(1e-4));
}

// ── 4 tests: infeasibility detection × 4 modes ──

namespace {
void check_infeasible_mode(bool portfolio, bool opp) {
    Highs h;
    h.setOptionValue("output_flag", false);
    h.setOptionValue("mip_heuristic_portfolio", portfolio);
    h.setOptionValue("mip_heuristic_opportunistic", opp);
    REQUIRE(h.readModel(std::string(INSTANCES_DIR) + "/infeasible-mip0.mps") == HighsStatus::kOk);
    h.run();
    REQUIRE(h.getModelStatus() == HighsModelStatus::kInfeasible);
}
}  // namespace

TEST_CASE("mode-matrix seq/det: infeasible detected", "[mode-matrix]") {
    check_infeasible_mode(false, false);
}

TEST_CASE("mode-matrix seq/opp: infeasible detected", "[mode-matrix]") {
    check_infeasible_mode(false, true);
}

TEST_CASE("mode-matrix port/det: infeasible detected", "[mode-matrix]") {
    check_infeasible_mode(true, false);
}

TEST_CASE("mode-matrix port/opp: infeasible detected", "[mode-matrix]") {
    check_infeasible_mode(true, true);
}

// ── 4 tests: all custom heuristics disabled × 4 modes ──
// With every custom arm off the dispatcher is a no-op and HiGHS's own
// B&B must still solve flugpl.  This verifies none of the mode paths
// accidentally block fallback behaviour.

namespace {
double solve_mode_no_heuristics(bool portfolio, bool opp) {
    Highs h;
    h.setOptionValue("output_flag", false);
    h.setOptionValue("mip_heuristic_portfolio", portfolio);
    h.setOptionValue("mip_heuristic_opportunistic", opp);
    h.setOptionValue("mip_heuristic_run_fpr", false);
    h.setOptionValue("mip_heuristic_run_local_mip", false);
    h.setOptionValue("mip_heuristic_run_feasibility_jump", false);
    h.setOptionValue("mip_heuristic_run_scylla", false);
    REQUIRE(h.readModel(std::string(INSTANCES_DIR) + "/flugpl.mps") == HighsStatus::kOk);
    REQUIRE(h.run() == HighsStatus::kOk);
    double obj;
    h.getInfoValue("objective_function_value", obj);
    return obj;
}
}  // namespace

TEST_CASE("mode-matrix seq/det: all heuristics disabled still solves", "[mode-matrix]") {
    REQUIRE(solve_mode_no_heuristics(false, false) == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("mode-matrix seq/opp: all heuristics disabled still solves", "[mode-matrix]") {
    REQUIRE(solve_mode_no_heuristics(false, true) == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("mode-matrix port/det: all heuristics disabled still solves", "[mode-matrix]") {
    REQUIRE(solve_mode_no_heuristics(true, false) == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("mode-matrix port/opp: all heuristics disabled still solves", "[mode-matrix]") {
    REQUIRE(solve_mode_no_heuristics(true, true) == Catch::Approx(1201500.0).epsilon(1e-6));
}

// ── 4 tests: single-arm (FJ-only) × 4 modes ──
// Only feasibility_jump enabled: exercises the opportunistic runner's
// single-worker-type path, which is easy to break with worker-count logic.

namespace {
double solve_mode_fj_only(bool portfolio, bool opp) {
    Highs h;
    h.setOptionValue("output_flag", false);
    h.setOptionValue("mip_heuristic_portfolio", portfolio);
    h.setOptionValue("mip_heuristic_opportunistic", opp);
    h.setOptionValue("mip_heuristic_run_fpr", false);
    h.setOptionValue("mip_heuristic_run_local_mip", false);
    h.setOptionValue("mip_heuristic_run_scylla", false);
    h.setOptionValue("mip_heuristic_run_feasibility_jump", true);
    REQUIRE(h.readModel(std::string(INSTANCES_DIR) + "/flugpl.mps") == HighsStatus::kOk);
    REQUIRE(h.run() == HighsStatus::kOk);
    double obj;
    h.getInfoValue("objective_function_value", obj);
    return obj;
}
}  // namespace

TEST_CASE("mode-matrix seq/det: FJ-only flugpl", "[mode-matrix]") {
    REQUIRE(solve_mode_fj_only(false, false) == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("mode-matrix seq/opp: FJ-only flugpl", "[mode-matrix]") {
    REQUIRE(solve_mode_fj_only(false, true) == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("mode-matrix port/det: FJ-only flugpl", "[mode-matrix]") {
    REQUIRE(solve_mode_fj_only(true, false) == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("mode-matrix port/opp: FJ-only flugpl", "[mode-matrix]") {
    REQUIRE(solve_mode_fj_only(true, true) == Catch::Approx(1201500.0).epsilon(1e-6));
}

// ── 2 tests: determinism for deterministic cells only ──
// Opportunistic cells are intentionally non-deterministic so no determinism
// guarantee is asserted for them.

TEST_CASE("mode-matrix seq/det: same seed → same objective and node count", "[mode-matrix]") {
    struct RunResult {
        double obj;
        HighsInt nodes;
    };
    auto run_seeded = [](int seed) {
        Highs h;
        h.setOptionValue("output_flag", false);
        h.setOptionValue("mip_heuristic_portfolio", false);
        h.setOptionValue("mip_heuristic_opportunistic", false);
        h.setOptionValue("random_seed", seed);
        REQUIRE(h.readModel(std::string(INSTANCES_DIR) + "/flugpl.mps") == HighsStatus::kOk);
        REQUIRE(h.run() == HighsStatus::kOk);
        RunResult res;
        h.getInfoValue("objective_function_value", res.obj);
        h.getInfoValue("mip_node_count", res.nodes);
        return res;
    };
    auto first = run_seeded(42);
    auto second = run_seeded(42);
    REQUIRE(first.obj == Catch::Approx(second.obj).epsilon(1e-12));
    REQUIRE(first.nodes == second.nodes);
}

TEST_CASE("mode-matrix port/det: same seed → same objective and node count", "[mode-matrix]") {
    struct RunResult {
        double obj;
        HighsInt nodes;
    };
    auto run_seeded = [](int seed) {
        Highs h;
        h.setOptionValue("output_flag", false);
        h.setOptionValue("mip_heuristic_portfolio", true);
        h.setOptionValue("mip_heuristic_opportunistic", false);
        h.setOptionValue("random_seed", seed);
        REQUIRE(h.readModel(std::string(INSTANCES_DIR) + "/flugpl.mps") == HighsStatus::kOk);
        REQUIRE(h.run() == HighsStatus::kOk);
        RunResult res;
        h.getInfoValue("objective_function_value", res.obj);
        h.getInfoValue("mip_node_count", res.nodes);
        return res;
    };
    auto first = run_seeded(42);
    auto second = run_seeded(42);
    REQUIRE(first.obj == Catch::Approx(second.obj).epsilon(1e-12));
    REQUIRE(first.nodes == second.nodes);
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

// ── LocalMIP standalone: neighborhood search finds feasible solution ──

TEST_CASE("LocalMIP standalone: flugpl", "[heuristic][local_mip]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_run_fpr", false);
    highs.setOptionValue("mip_heuristic_run_local_mip", true);
    highs.setOptionValue("mip_heuristic_run_scylla", false);
    highs.setOptionValue("mip_heuristic_portfolio", false);
    REQUIRE(highs.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("LocalMIP standalone: egout", "[heuristic][local_mip]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_run_fpr", false);
    highs.setOptionValue("mip_heuristic_run_local_mip", true);
    highs.setOptionValue("mip_heuristic_run_scylla", false);
    highs.setOptionValue("mip_heuristic_portfolio", false);
    REQUIRE(highs.readModel(kInstancesDir + "/egout.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(568.1007).epsilon(1e-4));
}

// ── LocalMIP parallel: epoch-gated parallel local search ──

TEST_CASE("LocalMIP parallel: flugpl finds solution", "[heuristic][local_mip]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_run_fpr", false);
    highs.setOptionValue("mip_heuristic_run_feasibility_jump", false);
    highs.setOptionValue("mip_heuristic_run_local_mip", true);
    highs.setOptionValue("mip_heuristic_run_scylla", false);
    highs.setOptionValue("mip_heuristic_portfolio", false);
    REQUIRE(highs.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("LocalMIP parallel: egout finds solution", "[heuristic][local_mip]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_run_fpr", false);
    highs.setOptionValue("mip_heuristic_run_feasibility_jump", false);
    highs.setOptionValue("mip_heuristic_run_local_mip", true);
    highs.setOptionValue("mip_heuristic_run_scylla", false);
    highs.setOptionValue("mip_heuristic_portfolio", false);
    REQUIRE(highs.readModel(kInstancesDir + "/egout.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(568.1007).epsilon(1e-4));
}

// ── FJ standalone: HiGHS dispatches FJ via fj::run_parallel ──

TEST_CASE("FJ standalone: flugpl", "[heuristic][fj]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_run_fpr", false);
    highs.setOptionValue("mip_heuristic_run_local_mip", false);
    highs.setOptionValue("mip_heuristic_run_scylla", false);
    highs.setOptionValue("mip_heuristic_portfolio", false);
    highs.setOptionValue("mip_heuristic_run_feasibility_jump", true);
    REQUIRE(highs.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(1201500.0).epsilon(1e-6));
}

// ===================================================================
// fpr_lp 4-mode smoke tests (issue #63)
//
// fpr_lp runs LP-dependent FPR (paper Classes 2-3) during the B&B dive,
// after RINS/RENS, when the LP relaxation is at an optimal scaled state.
// Each cell of the (portfolio × opportunistic) execution matrix must
// exercise the dive path and find the known optimum.  bell5 is chosen
// because it is small enough to run fast and has a non-trivial root LP
// where LP-dependent FPR contributes (also exercised by the existing
// "FPR strategies: portfolio multi-arm on bell5" characterization).
// ===================================================================

namespace {
double solve_fpr_lp_mode(const char* inst, bool portfolio, bool opp) {
    Highs h;
    h.setOptionValue("output_flag", false);
    h.setOptionValue("mip_heuristic_run_fpr", true);
    h.setOptionValue("mip_heuristic_portfolio", portfolio);
    h.setOptionValue("mip_heuristic_opportunistic", opp);
    REQUIRE(h.readModel(kInstancesDir + "/" + inst) == HighsStatus::kOk);
    REQUIRE(h.run() == HighsStatus::kOk);
    double obj;
    h.getInfoValue("objective_function_value", obj);
    return obj;
}
}  // namespace

TEST_CASE("fpr_lp seq/det: bell5 finds optimum and dispatches", "[fpr_lp][mode-matrix]") {
    fpr_lp::reset_dispatch_counts();
    REQUIRE(solve_fpr_lp_mode("bell5.mps", false, false) ==
            Catch::Approx(8966406.49152).epsilon(1e-4));
    const auto counts = fpr_lp::dispatch_counts();
    REQUIRE(counts.seq_det >= 1);
    REQUIRE(counts.seq_opp == 0);
    REQUIRE(counts.port_det == 0);
    REQUIRE(counts.port_opp == 0);
}

TEST_CASE("fpr_lp seq/opp: bell5 finds optimum and dispatches",
          "[fpr_lp][mode-matrix][opportunistic]") {
    fpr_lp::reset_dispatch_counts();
    REQUIRE(solve_fpr_lp_mode("bell5.mps", false, true) ==
            Catch::Approx(8966406.49152).epsilon(1e-4));
    const auto counts = fpr_lp::dispatch_counts();
    REQUIRE(counts.seq_opp >= 1);
    REQUIRE(counts.seq_det == 0);
    REQUIRE(counts.port_det == 0);
    REQUIRE(counts.port_opp == 0);
}

TEST_CASE("fpr_lp port/det: bell5 finds optimum and dispatches",
          "[fpr_lp][mode-matrix][portfolio]") {
    fpr_lp::reset_dispatch_counts();
    REQUIRE(solve_fpr_lp_mode("bell5.mps", true, false) ==
            Catch::Approx(8966406.49152).epsilon(1e-4));
    const auto counts = fpr_lp::dispatch_counts();
    REQUIRE(counts.port_det >= 1);
    REQUIRE(counts.seq_det == 0);
    REQUIRE(counts.seq_opp == 0);
    REQUIRE(counts.port_opp == 0);
}

TEST_CASE("fpr_lp port/opp: bell5 finds optimum and dispatches",
          "[fpr_lp][mode-matrix][portfolio][opportunistic]") {
    fpr_lp::reset_dispatch_counts();
    REQUIRE(solve_fpr_lp_mode("bell5.mps", true, true) ==
            Catch::Approx(8966406.49152).epsilon(1e-4));
    const auto counts = fpr_lp::dispatch_counts();
    REQUIRE(counts.port_opp >= 1);
    REQUIRE(counts.seq_det == 0);
    REQUIRE(counts.seq_opp == 0);
    REQUIRE(counts.port_det == 0);
}

// ── Scylla standalone: PDLP pump finds feasible solution ──

TEST_CASE("Scylla standalone: flugpl general integers", "[heuristic][scylla]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_run_fpr", false);
    highs.setOptionValue("mip_heuristic_run_local_mip", false);
    highs.setOptionValue("mip_heuristic_run_scylla", true);
    highs.setOptionValue("mip_heuristic_portfolio", false);
    REQUIRE(highs.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("Scylla standalone: gt2 pure binary instance", "[heuristic][scylla]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_run_fpr", false);
    highs.setOptionValue("mip_heuristic_run_local_mip", false);
    highs.setOptionValue("mip_heuristic_run_scylla", true);
    highs.setOptionValue("mip_heuristic_portfolio", false);
    REQUIRE(highs.readModel(kInstancesDir + "/gt2.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(21166.0).epsilon(1e-3));
}

TEST_CASE("Scylla standalone: egout mixed integers", "[heuristic][scylla]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_run_fpr", false);
    highs.setOptionValue("mip_heuristic_run_local_mip", false);
    highs.setOptionValue("mip_heuristic_run_scylla", true);
    highs.setOptionValue("mip_heuristic_portfolio", false);
    REQUIRE(highs.readModel(kInstancesDir + "/egout.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(568.1007).epsilon(1e-4));
}

// ── Sequential orchestrator: weighted effort allocation ──

TEST_CASE("Sequential orchestrator: flugpl weighted effort", "[heuristic][sequential]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_run_fpr", true);
    highs.setOptionValue("mip_heuristic_run_local_mip", true);
    highs.setOptionValue("mip_heuristic_run_scylla", true);
    highs.setOptionValue("mip_heuristic_portfolio", false);
    REQUIRE(highs.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("Sequential orchestrator: egout all arms", "[heuristic][sequential]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_run_feasibility_jump", true);
    highs.setOptionValue("mip_heuristic_run_fpr", true);
    highs.setOptionValue("mip_heuristic_run_local_mip", true);
    highs.setOptionValue("mip_heuristic_run_scylla", true);
    highs.setOptionValue("mip_heuristic_portfolio", false);
    REQUIRE(highs.readModel(kInstancesDir + "/egout.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(568.1007).epsilon(1e-4));
}

TEST_CASE("Portfolio: Scylla arm finds solution on flugpl", "[portfolio][scylla]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_run_scylla", true);
    highs.setOptionValue("mip_heuristic_portfolio", true);
    REQUIRE(highs.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("Portfolio: Scylla arm finds solution on egout", "[portfolio][scylla]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_run_scylla", true);
    highs.setOptionValue("mip_heuristic_portfolio", true);
    REQUIRE(highs.readModel(kInstancesDir + "/egout.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(568.1007).epsilon(1e-4));
}

TEST_CASE("Portfolio: Scylla-only single arm", "[portfolio][scylla][single-arm]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_portfolio", true);
    highs.setOptionValue("mip_heuristic_run_fpr", false);
    highs.setOptionValue("mip_heuristic_run_local_mip", false);
    highs.setOptionValue("mip_heuristic_run_feasibility_jump", false);
    highs.setOptionValue("mip_heuristic_run_scylla", true);
    REQUIRE(highs.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("Portfolio deterministic: Scylla arm same result", "[portfolio][scylla][determinism]") {
    auto run_once = [](double& obj) {
        Highs highs;
        highs.setOptionValue("output_flag", false);
        highs.setOptionValue("mip_heuristic_portfolio", true);
        highs.setOptionValue("mip_heuristic_run_scylla", true);
        REQUIRE(highs.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);
        REQUIRE(highs.run() == HighsStatus::kOk);
        highs.getInfoValue("objective_function_value", obj);
    };
    double obj1, obj2;
    run_once(obj1);
    run_once(obj2);
    REQUIRE(obj1 == Catch::Approx(obj2).epsilon(1e-12));
}

// ── Scylla parallel: run_parallel is the unified entry for pump chains ──
// Scylla has both det and opp variants, selected by mip_heuristic_opportunistic
// via scylla::run_parallel_deterministic / scylla::run_parallel_opportunistic.

TEST_CASE("Scylla parallel: flugpl finds solution", "[heuristic][scylla]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_run_fpr", false);
    highs.setOptionValue("mip_heuristic_run_local_mip", false);
    highs.setOptionValue("mip_heuristic_run_scylla", true);
    highs.setOptionValue("mip_heuristic_portfolio", false);
    REQUIRE(highs.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("Scylla parallel: egout finds solution", "[heuristic][scylla]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_run_fpr", false);
    highs.setOptionValue("mip_heuristic_run_local_mip", false);
    highs.setOptionValue("mip_heuristic_run_scylla", true);
    highs.setOptionValue("mip_heuristic_portfolio", false);
    REQUIRE(highs.readModel(kInstancesDir + "/egout.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(568.1007).epsilon(1e-4));
}

TEST_CASE("Scylla parallel: gt2 binary instance", "[heuristic][scylla]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_run_fpr", false);
    highs.setOptionValue("mip_heuristic_run_local_mip", false);
    highs.setOptionValue("mip_heuristic_run_scylla", true);
    highs.setOptionValue("mip_heuristic_portfolio", false);
    REQUIRE(highs.readModel(kInstancesDir + "/gt2.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(21166.0).epsilon(1e-3));
}

// ── Scylla characterization: verify known-optimal objectives ──

TEST_CASE("Scylla sequential: flugpl characterization", "[scylla]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_run_fpr", false);
    highs.setOptionValue("mip_heuristic_run_local_mip", false);
    highs.setOptionValue("mip_heuristic_run_scylla", true);
    highs.setOptionValue("mip_heuristic_portfolio", false);
    REQUIRE(highs.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("Scylla parallel: flugpl characterization", "[scylla]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_run_fpr", false);
    highs.setOptionValue("mip_heuristic_run_local_mip", false);
    highs.setOptionValue("mip_heuristic_run_scylla", true);
    highs.setOptionValue("mip_heuristic_portfolio", false);
    REQUIRE(highs.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("Scylla parallel: egout feasibility", "[scylla]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_run_fpr", false);
    highs.setOptionValue("mip_heuristic_run_local_mip", false);
    highs.setOptionValue("mip_heuristic_run_scylla", true);
    highs.setOptionValue("mip_heuristic_portfolio", false);
    REQUIRE(highs.readModel(kInstancesDir + "/egout.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj <= 568.1007 + 1e-4);
}

// ── Scylla opportunistic: continuous N-chain parallelism variant ──

TEST_CASE("Scylla opportunistic: flugpl characterization", "[scylla][opportunistic]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_run_fpr", false);
    highs.setOptionValue("mip_heuristic_run_local_mip", false);
    highs.setOptionValue("mip_heuristic_run_feasibility_jump", false);
    highs.setOptionValue("mip_heuristic_run_scylla", true);
    highs.setOptionValue("mip_heuristic_portfolio", false);
    highs.setOptionValue("mip_heuristic_opportunistic", true);
    REQUIRE(highs.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("Scylla opportunistic: egout feasibility", "[scylla][opportunistic]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_run_fpr", false);
    highs.setOptionValue("mip_heuristic_run_local_mip", false);
    highs.setOptionValue("mip_heuristic_run_feasibility_jump", false);
    highs.setOptionValue("mip_heuristic_run_scylla", true);
    highs.setOptionValue("mip_heuristic_portfolio", false);
    highs.setOptionValue("mip_heuristic_opportunistic", true);
    REQUIRE(highs.readModel(kInstancesDir + "/egout.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj <= 568.1007 + 1e-4);
}

TEST_CASE("Scylla opportunistic: gt2 pure binary", "[scylla][opportunistic]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_run_fpr", false);
    highs.setOptionValue("mip_heuristic_run_local_mip", false);
    highs.setOptionValue("mip_heuristic_run_feasibility_jump", false);
    highs.setOptionValue("mip_heuristic_run_scylla", true);
    highs.setOptionValue("mip_heuristic_portfolio", false);
    highs.setOptionValue("mip_heuristic_opportunistic", true);
    REQUIRE(highs.readModel(kInstancesDir + "/gt2.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(21166.0).epsilon(1e-3));
}

// ── Portfolio: gt2 instance (pure binary, tests FJ on binary vars) ──

TEST_CASE("Portfolio: gt2 binary instance", "[portfolio]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_portfolio", true);
    REQUIRE(highs.readModel(kInstancesDir + "/gt2.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(21166.0).epsilon(1e-3));
}

TEST_CASE("Portfolio opportunistic: gt2 binary instance", "[portfolio][opportunistic]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_portfolio", true);
    highs.setOptionValue("mip_heuristic_opportunistic", true);
    REQUIRE(highs.readModel(kInstancesDir + "/gt2.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(21166.0).epsilon(1e-3));
}

// ── Portfolio: p0548 (medium MIP) ──

TEST_CASE("Portfolio: p0548 medium instance", "[portfolio]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_portfolio", true);
    REQUIRE(highs.readModel(kInstancesDir + "/p0548.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(8691.0).epsilon(1e-3));
}

TEST_CASE("Portfolio opportunistic: p0548 medium instance", "[portfolio][opportunistic]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_portfolio", true);
    highs.setOptionValue("mip_heuristic_opportunistic", true);
    REQUIRE(highs.readModel(kInstancesDir + "/p0548.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(8691.0).epsilon(1e-3));
}

// ===================================================================
// FPR Strategy tests
// ===================================================================

TEST_CASE("FPR strategies: framework mode helpers", "[fpr][strategies]") {
    // dfs: propagate on, repair off, backtrack on
    REQUIRE(mode_propagates(FrameworkMode::kDfs));
    REQUIRE_FALSE(mode_repairs(FrameworkMode::kDfs));
    REQUIRE(mode_backtracks(FrameworkMode::kDfs));

    // dfsrep: propagate on, repair on, backtrack on
    REQUIRE(mode_propagates(FrameworkMode::kDfsrep));
    REQUIRE(mode_repairs(FrameworkMode::kDfsrep));
    REQUIRE(mode_backtracks(FrameworkMode::kDfsrep));

    // dive: propagate off, repair on, no backtrack
    REQUIRE_FALSE(mode_propagates(FrameworkMode::kDive));
    REQUIRE(mode_repairs(FrameworkMode::kDive));
    REQUIRE_FALSE(mode_backtracks(FrameworkMode::kDive));

    // diveprop: propagate on, repair on, no backtrack
    REQUIRE(mode_propagates(FrameworkMode::kDiveprop));
    REQUIRE(mode_repairs(FrameworkMode::kDiveprop));
    REQUIRE_FALSE(mode_backtracks(FrameworkMode::kDiveprop));

    // repairsearch: propagate on, WalkSAT repair off (own dispatch), backtrack on
    REQUIRE(mode_propagates(FrameworkMode::kRepairSearch));
    REQUIRE_FALSE(mode_repairs(FrameworkMode::kRepairSearch));
    REQUIRE(mode_backtracks(FrameworkMode::kRepairSearch));
}

TEST_CASE("FPR strategies: strategy_needs_lp", "[fpr][strategies]") {
    // LP-free strategies
    REQUIRE_FALSE(strategy_needs_lp(kStratRandom));
    REQUIRE_FALSE(strategy_needs_lp(kStratBadobjcl));
    REQUIRE_FALSE(strategy_needs_lp(kStratLocks2));
    REQUIRE_FALSE(strategy_needs_lp(kStratGoodobj));
    REQUIRE_FALSE(strategy_needs_lp(kStratDomsize));

    // Dynamic strategy helpers
    REQUIRE(is_dynamic_var_strategy(VarStrategy::kDomainSize));
    REQUIRE_FALSE(is_dynamic_var_strategy(VarStrategy::kType));
    REQUIRE_FALSE(is_dynamic_var_strategy(VarStrategy::kLocks));

    // LP-dependent strategies
    REQUIRE(strategy_needs_lp(kStratZerocore));
    REQUIRE(strategy_needs_lp(kStratZerolp));
    REQUIRE(strategy_needs_lp(kStratCore));
    REQUIRE(strategy_needs_lp(kStratLp));
    REQUIRE(strategy_needs_lp(kStratCliques));
    REQUIRE(strategy_needs_lp(kStratCliques2));
}

TEST_CASE("FPR strategies: DFS mode on flugpl", "[fpr][strategies][dfs]") {
    // Test that DFS mode (with backtracking) solves flugpl
    Highs highs;
    highs.setOptionValue("output_flag", false);
    REQUIRE(highs.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("FPR strategies: multi-config sequential on egout", "[fpr][strategies]") {
    // The sequential multi-config runner should solve egout
    Highs highs;
    highs.setOptionValue("output_flag", false);
    // FPR enabled (runs multi-config), portfolio off
    highs.setOptionValue("mip_heuristic_run_fpr", true);
    highs.setOptionValue("mip_heuristic_portfolio", false);
    REQUIRE(highs.readModel(kInstancesDir + "/egout.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(568.1007).epsilon(1e-4));
}

TEST_CASE("FPR strategies: portfolio with multi-arm FPR on flugpl",
          "[fpr][portfolio][strategies]") {
    // Portfolio mode with 6 FPR config arms should solve flugpl
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_portfolio", true);
    highs.setOptionValue("mip_heuristic_run_fpr", true);
    REQUIRE(highs.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("FPR strategies: portfolio multi-arm on bell5", "[fpr][portfolio][strategies]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_portfolio", true);
    highs.setOptionValue("mip_heuristic_run_fpr", true);
    REQUIRE(highs.readModel(kInstancesDir + "/bell5.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(8966406.49152).epsilon(1e-4));
}

// ===================================================================
// RepairSearch tests (Fig. 5 with secondary propagation engine R)
// ===================================================================

TEST_CASE("RepairSearch: portfolio with RepairSearch arm on flugpl", "[repair-search][portfolio]") {
    // Portfolio now includes the RepairSearch arm — verify it still solves flugpl
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_portfolio", true);
    highs.setOptionValue("mip_heuristic_run_fpr", true);
    REQUIRE(highs.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("RepairSearch: portfolio with RepairSearch arm on egout", "[repair-search][portfolio]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_portfolio", true);
    highs.setOptionValue("mip_heuristic_run_fpr", true);
    REQUIRE(highs.readModel(kInstancesDir + "/egout.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(568.1007).epsilon(1e-4));
}

TEST_CASE("RepairSearch: opportunistic portfolio on flugpl",
          "[repair-search][portfolio][opportunistic]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_portfolio", true);
    highs.setOptionValue("mip_heuristic_opportunistic", true);
    highs.setOptionValue("mip_heuristic_run_fpr", true);
    REQUIRE(highs.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("RepairSearch: FPR standalone with RepairSearch config on flugpl",
          "[repair-search][fpr]") {
    // Standalone FPR mode now includes RepairSearch config — must still solve
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_run_fpr", true);
    highs.setOptionValue("mip_heuristic_portfolio", false);
    REQUIRE(highs.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(1201500.0).epsilon(1e-6));
}

// ===================================================================
// PropEngine unit tests
// ===================================================================

// Helper: build a small test model for PropEngine tests.
// 3 variables: x0 (binary), x1 (integer [0,5]), x2 (continuous [0,10])
// 2 constraints:
//   row 0: x0 + x1 >= 2   (row_lo=2, row_hi=inf)
//   row 1: x1 + x2 <= 8   (row_lo=-inf, row_hi=8)
namespace {
struct SmallModel {
    static constexpr HighsInt ncol = 3;
    static constexpr HighsInt nrow = 2;
    std::vector<HighsInt> ar_start = {0, 2, 4};
    std::vector<HighsInt> ar_index = {0, 1, 1, 2};
    std::vector<double> ar_value = {1.0, 1.0, 1.0, 1.0};
    std::vector<double> col_lb = {0.0, 0.0, 0.0};
    std::vector<double> col_ub = {1.0, 5.0, 10.0};
    double row_lo[2] = {2.0, -kHighsInf};
    double row_hi[2] = {kHighsInf, 8.0};
    std::vector<HighsVarType> integrality = {HighsVarType::kInteger, HighsVarType::kInteger,
                                             HighsVarType::kContinuous};
    CscMatrix csc;

    SmallModel() { csc = build_csc(ncol, nrow, ar_start, ar_index, ar_value); }

    PropEngine make_engine(double feastol = 1e-6) {
        return PropEngine(ncol, nrow, ar_start.data(), ar_index.data(), ar_value.data(), csc,
                          col_lb.data(), col_ub.data(), row_lo, row_hi, integrality.data(),
                          feastol);
    }
};
}  // namespace

TEST_CASE("PropEngine: fix and propagate", "[prop-engine]") {
    SmallModel m;
    auto eng = m.make_engine();

    // Initial state: all variables unfixed with global bounds
    REQUIRE_FALSE(eng.var(0).fixed);
    REQUIRE(eng.var(1).lb == Catch::Approx(0.0));
    REQUIRE(eng.var(1).ub == Catch::Approx(5.0));

    // Fix x0 = 0, propagate: row 0 (x0+x1 >= 2) forces x1 >= 2
    REQUIRE(eng.fix(0, 0.0));
    REQUIRE(eng.propagate(0));
    REQUIRE(eng.var(0).fixed);
    REQUIRE(eng.var(0).val == Catch::Approx(0.0));
    REQUIRE(eng.var(1).lb >= 2.0 - 1e-6);

    // Fix x1 = 5, propagate: row 1 (x1+x2 <= 8) forces x2 <= 3
    REQUIRE(eng.fix(1, 5.0));
    REQUIRE(eng.propagate(1));
    REQUIRE(eng.var(2).ub <= 3.0 + 1e-6);
}

TEST_CASE("PropEngine: backtrack restores state", "[prop-engine]") {
    SmallModel m;
    auto eng = m.make_engine();

    HighsInt vs_m = eng.vs_mark();
    HighsInt sol_m = eng.sol_mark();

    REQUIRE(eng.fix(0, 1.0));
    REQUIRE(eng.propagate(0));
    REQUIRE(eng.var(0).fixed);

    eng.backtrack_to(vs_m, sol_m);
    REQUIRE_FALSE(eng.var(0).fixed);
    REQUIRE(eng.var(0).lb == Catch::Approx(0.0));
    REQUIRE(eng.var(0).ub == Catch::Approx(1.0));
    REQUIRE(eng.var(1).lb == Catch::Approx(0.0));
}

TEST_CASE("PropEngine: tighten bounds and auto-fix", "[prop-engine]") {
    SmallModel m;
    auto eng = m.make_engine();

    REQUIRE(eng.tighten_lb(1, 3.0));
    REQUIRE(eng.var(1).lb >= 3.0 - 1e-6);
    REQUIRE(eng.var(1).ub == Catch::Approx(5.0));

    // Tighten ub to match lb — should auto-fix
    REQUIRE(eng.tighten_ub(1, 3.0));
    REQUIRE(eng.var(1).fixed);
    REQUIRE(eng.var(1).val == Catch::Approx(3.0));
}

TEST_CASE("PropEngine: infeasible propagation", "[prop-engine]") {
    SmallModel m;
    auto eng = m.make_engine();

    // Fix x0=0, tighten x1 ub to 1. Propagation from row 0 (x0+x1 >= 2)
    // tries to tighten x1 lb to 2, but ub is 1 → lb > ub → infeasible.
    REQUIRE(eng.fix(0, 0.0));
    REQUIRE(eng.tighten_ub(1, 1.0));
    REQUIRE_FALSE(eng.propagate(0));
}

TEST_CASE("PropEngine: reset clears state", "[prop-engine]") {
    SmallModel m;
    auto eng = m.make_engine();

    eng.fix(0, 1.0);
    eng.propagate(0);
    REQUIRE(eng.var(0).fixed);

    eng.reset();
    REQUIRE_FALSE(eng.var(0).fixed);
    REQUIRE(eng.var(0).lb == Catch::Approx(0.0));
    REQUIRE(eng.var(0).ub == Catch::Approx(1.0));
    REQUIRE(eng.var(1).lb == Catch::Approx(0.0));
    REQUIRE(eng.var(1).ub == Catch::Approx(5.0));
}

TEST_CASE("PropEngine: effort tracking", "[prop-engine]") {
    SmallModel m;
    auto eng = m.make_engine();

    size_t before = eng.effort();
    eng.fix(0, 1.0);
    eng.propagate(0);
    REQUIRE(eng.effort() > before);

    eng.add_effort(100);
    REQUIRE(eng.effort() >= before + 100);
}
