#include "heuristic_common.h"
#include "Highs.h"
#include "test_common.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

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

TEST_CASE("heuristic_base_seed: propagates random_seed", "[primitive][seed]") {
    // Regression test for a previously-silent bug: every heuristic's worker
    // RNG ignored HiGHS's `random_seed` option and seeded only from an
    // internal counter.  heuristic_base_seed is the single point through
    // which `random_seed` now flows; if someone reverts it to a constant
    // or drops `random_seed` from the argument, this test fails.

    // Default (random_seed == 0, HiGHS's default) reproduces the old
    // constant-42 base so prior characterization tests stay stable.
    REQUIRE(heuristic_base_seed(0) == kBaseSeedOffset);

    // Non-zero random_seed changes the base by that amount.
    REQUIRE(heuristic_base_seed(1) != heuristic_base_seed(0));
    REQUIRE(heuristic_base_seed(12345) != heuristic_base_seed(12346));
    REQUIRE(heuristic_base_seed(12345) == kBaseSeedOffset + 12345u);
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

TEST_CASE("Options: opportunistic option exists", "[options][portfolio]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    REQUIRE(highs.setOptionValue("mip_heuristic_opportunistic", true) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_heuristic_opportunistic", false) == HighsStatus::kOk);
}
