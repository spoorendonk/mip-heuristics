#include "Highs.h"
#include "test_common.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

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
