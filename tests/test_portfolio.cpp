#include "Highs.h"
#include "test_common.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

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
