#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <random>
#include <string>

#include "Highs.h"
#include "adaptive/solution_pool.h"
#include "adaptive/thompson_sampler.h"

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
