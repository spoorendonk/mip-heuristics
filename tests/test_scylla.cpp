#include "Highs.h"
#include "test_common.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

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
