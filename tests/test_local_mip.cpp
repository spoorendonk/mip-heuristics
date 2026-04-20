#include "Highs.h"
#include "test_common.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

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
