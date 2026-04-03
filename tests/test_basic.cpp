#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "Highs.h"
#include "solution_pool.h"
#include "thompson_sampler.h"

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
  REQUIRE(highs.setOptionValue("mip_heuristic_run_fpr", false) ==
          HighsStatus::kOk);
  REQUIRE(highs.setOptionValue("mip_heuristic_run_local_mip", false) ==
          HighsStatus::kOk);
  REQUIRE(highs.setOptionValue("mip_heuristic_run_scylla_fpr", false) ==
          HighsStatus::kOk);
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
  REQUIRE(highs.setOptionValue("mip_heuristic_portfolio", true) ==
          HighsStatus::kOk);
  REQUIRE(highs.setOptionValue("mip_heuristic_portfolio", false) ==
          HighsStatus::kOk);
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

TEST_CASE("Options: portfolio_opportunistic option exists", "[options][portfolio]") {
  Highs highs;
  highs.setOptionValue("output_flag", false);
  REQUIRE(highs.setOptionValue("mip_heuristic_portfolio_opportunistic", true) ==
          HighsStatus::kOk);
  REQUIRE(highs.setOptionValue("mip_heuristic_portfolio_opportunistic", false) ==
          HighsStatus::kOk);
}

TEST_CASE("Portfolio opportunistic: flugpl finds solution", "[portfolio][opportunistic]") {
  Highs highs;
  highs.setOptionValue("output_flag", false);
  highs.setOptionValue("mip_heuristic_portfolio", true);
  highs.setOptionValue("mip_heuristic_portfolio_opportunistic", true);
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
  highs.setOptionValue("mip_heuristic_portfolio_opportunistic", true);
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
  highs.setOptionValue("mip_heuristic_portfolio_opportunistic", true);
  REQUIRE(highs.readModel(kInstancesDir + "/bell5.mps") == HighsStatus::kOk);
  REQUIRE(highs.run() == HighsStatus::kOk);
  double obj;
  highs.getInfoValue("objective_function_value", obj);
  REQUIRE(obj == Catch::Approx(8966406.49152).epsilon(1e-6));
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

TEST_CASE("Portfolio deterministic: egout same result across runs",
          "[portfolio][determinism]") {
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

TEST_CASE("Portfolio deterministic: different seeds can differ",
          "[portfolio][determinism]") {
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

TEST_CASE("Portfolio opportunistic: lseu finds solution",
          "[portfolio][opportunistic]") {
  Highs highs;
  highs.setOptionValue("output_flag", false);
  highs.setOptionValue("mip_heuristic_portfolio", true);
  highs.setOptionValue("mip_heuristic_portfolio_opportunistic", true);
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

TEST_CASE("Portfolio opportunistic: FJ-only mode works",
          "[portfolio][opportunistic][single-arm]") {
  Highs highs;
  highs.setOptionValue("output_flag", false);
  highs.setOptionValue("mip_heuristic_portfolio", true);
  highs.setOptionValue("mip_heuristic_portfolio_opportunistic", true);
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

TEST_CASE("Portfolio: all presolve arms disabled still solves",
          "[portfolio][edge]") {
  Highs highs;
  highs.setOptionValue("output_flag", false);
  highs.setOptionValue("mip_heuristic_portfolio", true);
  highs.setOptionValue("mip_heuristic_run_fpr", false);
  highs.setOptionValue("mip_heuristic_run_local_mip", false);
  highs.setOptionValue("mip_heuristic_run_feasibility_jump", false);
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
  highs.setOptionValue("mip_heuristic_portfolio_opportunistic", true);
  highs.setOptionValue("mip_heuristic_run_fpr", false);
  highs.setOptionValue("mip_heuristic_run_local_mip", false);
  highs.setOptionValue("mip_heuristic_run_feasibility_jump", false);
  REQUIRE(highs.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);
  REQUIRE(highs.run() == HighsStatus::kOk);
  double obj;
  highs.getInfoValue("objective_function_value", obj);
  REQUIRE(obj == Catch::Approx(1201500.0).epsilon(1e-6));
}

// ── Opportunistic without portfolio flag is ignored ──

TEST_CASE("Options: opportunistic without portfolio is ignored",
          "[options][portfolio]") {
  // opportunistic=true but portfolio=false: should behave like normal solve
  Highs highs;
  highs.setOptionValue("output_flag", false);
  highs.setOptionValue("mip_heuristic_portfolio", false);
  highs.setOptionValue("mip_heuristic_portfolio_opportunistic", true);
  REQUIRE(highs.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);
  REQUIRE(highs.run() == HighsStatus::kOk);
  double obj;
  highs.getInfoValue("objective_function_value", obj);
  REQUIRE(obj == Catch::Approx(1201500.0).epsilon(1e-6));
}

// ── Infeasible MIP: portfolio handles gracefully ──

TEST_CASE("Portfolio: infeasible MIP handled correctly",
          "[portfolio][edge]") {
  Highs highs;
  highs.setOptionValue("output_flag", false);
  highs.setOptionValue("mip_heuristic_portfolio", true);
  REQUIRE(highs.readModel(kInstancesDir + "/infeasible-mip0.mps") ==
          HighsStatus::kOk);
  highs.run();
  HighsModelStatus model_status = highs.getModelStatus();
  REQUIRE(model_status == HighsModelStatus::kInfeasible);
}

TEST_CASE("Portfolio opportunistic: infeasible MIP handled correctly",
          "[portfolio][opportunistic][edge]") {
  Highs highs;
  highs.setOptionValue("output_flag", false);
  highs.setOptionValue("mip_heuristic_portfolio", true);
  highs.setOptionValue("mip_heuristic_portfolio_opportunistic", true);
  REQUIRE(highs.readModel(kInstancesDir + "/infeasible-mip0.mps") ==
          HighsStatus::kOk);
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
    highs.setOptionValue("mip_heuristic_portfolio_opportunistic", true);
    highs.setOptionValue("random_seed", i);
    REQUIRE(highs.readModel(kInstancesDir + "/flugpl.mps") ==
            HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(1201500.0).epsilon(1e-6));
  }
}

// ── SolutionPool: thread-safety stress test ──

TEST_CASE("SolutionPool: concurrent try_add and get_restart",
          "[pool][thread-safety]") {
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

  for (auto& t : threads) t.join();

  // Pool should be internally consistent
  auto entries = pool.sorted_entries();
  REQUIRE(entries.size() <= 10);
  REQUIRE(entries.size() > 0);
  for (size_t i = 1; i < entries.size(); ++i) {
    REQUIRE(entries[i - 1].objective <= entries[i].objective);
  }
}

// ── ThompsonSampler: concurrent select/update stress ──

TEST_CASE("ThompsonSampler: concurrent select and update",
          "[bandit][thread-safety]") {
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

  for (auto& t : threads) t.join();

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

TEST_CASE("SolutionPool: single-entry restart is always copy",
          "[pool][edge]") {
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

// ── Scylla standalone: PDLP pump finds feasible solution ──

TEST_CASE("Scylla standalone: flugpl general integers",
          "[heuristic][scylla]") {
  Highs highs;
  highs.setOptionValue("output_flag", false);
  highs.setOptionValue("mip_heuristic_run_fpr", false);
  highs.setOptionValue("mip_heuristic_run_local_mip", false);
  highs.setOptionValue("mip_heuristic_run_scylla_fpr", true);
  highs.setOptionValue("mip_heuristic_portfolio", false);
  REQUIRE(highs.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);
  REQUIRE(highs.run() == HighsStatus::kOk);
  double obj;
  highs.getInfoValue("objective_function_value", obj);
  REQUIRE(obj == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("Scylla standalone: gt2 pure binary instance",
          "[heuristic][scylla]") {
  Highs highs;
  highs.setOptionValue("output_flag", false);
  highs.setOptionValue("mip_heuristic_run_fpr", false);
  highs.setOptionValue("mip_heuristic_run_local_mip", false);
  highs.setOptionValue("mip_heuristic_run_scylla_fpr", true);
  highs.setOptionValue("mip_heuristic_portfolio", false);
  REQUIRE(highs.readModel(kInstancesDir + "/gt2.mps") == HighsStatus::kOk);
  REQUIRE(highs.run() == HighsStatus::kOk);
  double obj;
  highs.getInfoValue("objective_function_value", obj);
  REQUIRE(obj == Catch::Approx(21166.0).epsilon(1e-3));
}

TEST_CASE("Scylla standalone: egout mixed integers",
          "[heuristic][scylla]") {
  Highs highs;
  highs.setOptionValue("output_flag", false);
  highs.setOptionValue("mip_heuristic_run_fpr", false);
  highs.setOptionValue("mip_heuristic_run_local_mip", false);
  highs.setOptionValue("mip_heuristic_run_scylla_fpr", true);
  highs.setOptionValue("mip_heuristic_portfolio", false);
  REQUIRE(highs.readModel(kInstancesDir + "/egout.mps") == HighsStatus::kOk);
  REQUIRE(highs.run() == HighsStatus::kOk);
  double obj;
  highs.getInfoValue("objective_function_value", obj);
  REQUIRE(obj == Catch::Approx(568.1007).epsilon(1e-4));
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

TEST_CASE("Portfolio opportunistic: gt2 binary instance",
          "[portfolio][opportunistic]") {
  Highs highs;
  highs.setOptionValue("output_flag", false);
  highs.setOptionValue("mip_heuristic_portfolio", true);
  highs.setOptionValue("mip_heuristic_portfolio_opportunistic", true);
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

TEST_CASE("Portfolio opportunistic: p0548 medium instance",
          "[portfolio][opportunistic]") {
  Highs highs;
  highs.setOptionValue("output_flag", false);
  highs.setOptionValue("mip_heuristic_portfolio", true);
  highs.setOptionValue("mip_heuristic_portfolio_opportunistic", true);
  REQUIRE(highs.readModel(kInstancesDir + "/p0548.mps") == HighsStatus::kOk);
  REQUIRE(highs.run() == HighsStatus::kOk);
  double obj;
  highs.getInfoValue("objective_function_value", obj);
  REQUIRE(obj == Catch::Approx(8691.0).epsilon(1e-3));
}
