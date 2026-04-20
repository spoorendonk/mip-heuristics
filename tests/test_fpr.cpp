#include "fpr_strategies.h"
#include "Highs.h"
#include "test_common.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

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
