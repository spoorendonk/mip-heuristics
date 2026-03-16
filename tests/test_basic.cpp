#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <string>

#include "Highs.h"

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
