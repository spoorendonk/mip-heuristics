#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "Highs.h"
#include "heuristic_common.h"
#include "test_common.h"

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

TEST_CASE("Options: disable custom heuristics", "[options]") {
  Highs highs;
  highs.setOptionValue("output_flag", false);
  // Verify options exist and can be set
  REQUIRE(highs.setOptionValue("mip_heuristic_run_fpr", false) ==
          HighsStatus::kOk);
  REQUIRE(highs.setOptionValue("mip_heuristic_run_local_mip", false) ==
          HighsStatus::kOk);
  REQUIRE(highs.setOptionValue("mip_heuristic_run_scylla", false) ==
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

TEST_CASE("Options: opportunistic option exists", "[options][portfolio]") {
  Highs highs;
  highs.setOptionValue("output_flag", false);
  REQUIRE(highs.setOptionValue("mip_heuristic_opportunistic", true) ==
          HighsStatus::kOk);
  REQUIRE(highs.setOptionValue("mip_heuristic_opportunistic", false) ==
          HighsStatus::kOk);
}

TEST_CASE("Options: preset option exists and accepts all valid values",
          "[options][preset]") {
  Highs highs;
  highs.setOptionValue("output_flag", false);
  REQUIRE(highs.setOptionValue("mip_heuristic_preset", std::string("")) ==
          HighsStatus::kOk);
  REQUIRE(highs.setOptionValue("mip_heuristic_preset", std::string("off")) ==
          HighsStatus::kOk);
  REQUIRE(highs.setOptionValue("mip_heuristic_preset", std::string("fpr")) ==
          HighsStatus::kOk);
  REQUIRE(highs.setOptionValue("mip_heuristic_preset",
                               std::string("all_det")) == HighsStatus::kOk);
  REQUIRE(highs.setOptionValue("mip_heuristic_preset",
                               std::string("all_opp")) == HighsStatus::kOk);
  REQUIRE(highs.setOptionValue("mip_heuristic_preset", std::string("scylla")) ==
          HighsStatus::kOk);
  REQUIRE(highs.setOptionValue("mip_heuristic_preset",
                               std::string("portfolio")) == HighsStatus::kOk);
}
