#include "heuristic_common.h"
#include "Highs.h"
#include "local_mip_construction.h"
#include "rng.h"
#include "test_common.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mutex>
#include <string>
#include <vector>

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

// ── LocalMIP cold-start construction phase (issue #75) ────────────────

// Build a tiny feasibility MIP by hand to drive the construction phase
// directly.  Two binary variables x0, x1 and one row:
//
//     x0 + x1 >= 1,   x0 ∈ {0,1},   x1 ∈ {0,1}
//
// The paper's zero-start (value closest to 0 in global bounds) sets
// both to 0, yielding an infeasible assignment.  The greedy sweep
// then picks at least one variable and flips it to 1 because the
// row is violated.  This tests both the zero-start and the
// feasibility-first greedy refinement.
TEST_CASE("LocalMIP construction: feasibility-first sweep repairs tiny MIP",
          "[heuristic][local_mip][construction]") {
    using local_mip_detail::construct_initial_solution;
    using local_mip_detail::ConstructionInputs;

    const HighsInt ncol = 2;
    const HighsInt nrow = 1;
    // Row-major: one row with (x0 + x1) >= 1.
    std::vector<HighsInt> ARstart = {0, 2};
    std::vector<HighsInt> ARindex = {0, 1};
    std::vector<double> ARvalue = {1.0, 1.0};
    std::vector<double> col_lb = {0.0, 0.0};
    std::vector<double> col_ub = {1.0, 1.0};
    std::vector<double> row_lo = {1.0};
    std::vector<double> row_hi = {kHighsInf};
    std::vector<HighsVarType> integrality = {HighsVarType::kInteger, HighsVarType::kInteger};
    CscMatrix csc = build_csc(ncol, nrow, ARstart, ARindex, ARvalue);

    ConstructionInputs inputs;
    inputs.ncol = ncol;
    inputs.nrow = nrow;
    inputs.ARstart = &ARstart;
    inputs.ARindex = &ARindex;
    inputs.ARvalue = &ARvalue;
    inputs.col_lb = &col_lb;
    inputs.col_ub = &col_ub;
    inputs.row_lo = &row_lo;
    inputs.row_hi = &row_hi;
    inputs.integrality = &integrality;
    inputs.csc = &csc;
    inputs.feastol = 1e-6;

    std::vector<double> solution;
    Rng rng(42);
    // Generous budget: this is a 2-variable model so any positive
    // budget is more than enough.
    construct_initial_solution(inputs, rng, /*max_effort=*/1000, solution);

    REQUIRE(solution.size() == 2);
    // Bounds respected.
    REQUIRE(solution[0] >= 0.0);
    REQUIRE(solution[0] <= 1.0);
    REQUIRE(solution[1] >= 0.0);
    REQUIRE(solution[1] <= 1.0);
    // Integer-valued.
    REQUIRE(solution[0] == Catch::Approx(std::round(solution[0])));
    REQUIRE(solution[1] == Catch::Approx(std::round(solution[1])));
    // Feasibility-first rule: the greedy sweep should have satisfied
    // the one violated row (x0 + x1 >= 1) by flipping at least one
    // variable to 1.
    REQUIRE((solution[0] + solution[1]) >= 1.0 - 1e-9);
}

// Construction starting point is inside bounds even when zero-start
// is infeasible (lb > 0 or ub < 0 forces a non-zero start).
TEST_CASE("LocalMIP construction: zero-start respects bounds with lb > 0 / ub < 0",
          "[heuristic][local_mip][construction]") {
    using local_mip_detail::construct_initial_solution;
    using local_mip_detail::ConstructionInputs;

    const HighsInt ncol = 3;
    const HighsInt nrow = 0;  // no constraints → only zero-start phase runs
    std::vector<HighsInt> ARstart = {0};
    std::vector<HighsInt> ARindex;
    std::vector<double> ARvalue;
    std::vector<double> col_lb = {2.0, -5.0, -3.0};  // lb>0 / lb<0 / ub<0
    std::vector<double> col_ub = {5.0, -1.0, -1.0};
    std::vector<double> row_lo;
    std::vector<double> row_hi;
    std::vector<HighsVarType> integrality = {HighsVarType::kInteger, HighsVarType::kContinuous,
                                             HighsVarType::kInteger};
    CscMatrix csc = build_csc(ncol, nrow, ARstart, ARindex, ARvalue);

    ConstructionInputs inputs;
    inputs.ncol = ncol;
    inputs.nrow = nrow;
    inputs.ARstart = &ARstart;
    inputs.ARindex = &ARindex;
    inputs.ARvalue = &ARvalue;
    inputs.col_lb = &col_lb;
    inputs.col_ub = &col_ub;
    inputs.row_lo = &row_lo;
    inputs.row_hi = &row_hi;
    inputs.integrality = &integrality;
    inputs.csc = &csc;
    inputs.feastol = 1e-6;

    std::vector<double> solution;
    Rng rng(7);
    construct_initial_solution(inputs, rng, /*max_effort=*/1000, solution);

    REQUIRE(solution.size() == 3);
    // x0: lb=2 (>0), value-closest-to-0 = lb = 2.
    REQUIRE(solution[0] == Catch::Approx(2.0));
    // x1: bounds straddle 0? actually lb=-5, ub=-1 → ub<0, so value closest to 0 = ub = -1.
    REQUIRE(solution[1] == Catch::Approx(-1.0));
    // x2: lb=-3, ub=-1, integer → closest-to-0 = ub = -1.
    REQUIRE(solution[2] == Catch::Approx(-1.0));
}

// End-to-end integration test: drive a full HiGHS solve with FJ / FPR /
// Scylla disabled and only LocalMIP enabled, on a small feasibility
// MIP.  Exercises the cold-start pathway: even if upstream HiGHS
// presolve doesn't find an incumbent, our LocalMIP construction +
// search phase should progress (the key acceptance criterion of issue
// #75).  Using flugpl (a small MIPLIB-like MIP) as the vehicle — the
// optimal is known and the solver chain still has to find it.
TEST_CASE("LocalMIP cold-start: finds flugpl optimum with all upstream heuristics off",
          "[heuristic][local_mip][construction][cold-start]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    // Disable every non-LocalMIP custom heuristic so LocalMIP is the
    // only primal-source in the presolve dispatch.
    highs.setOptionValue("mip_heuristic_run_fpr", false);
    highs.setOptionValue("mip_heuristic_run_feasibility_jump", false);
    highs.setOptionValue("mip_heuristic_run_scylla", false);
    highs.setOptionValue("mip_heuristic_run_local_mip", true);
    highs.setOptionValue("mip_heuristic_portfolio", false);
    REQUIRE(highs.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(1201500.0).epsilon(1e-6));
}

// Regression guard for `local_mip::run_parallel`'s warm-start path
// (issue #74).  The presolve chain in `mode_dispatch::run_sequential`
// flushes the shared `SolutionPool` into `mipdata->incumbent` only after
// all four heuristics have run.  Before #74 (and before the #75
// construction cold-start), `local_mip::run_parallel` bailed out on
// `mipdata->incumbent.empty()`, so an FJ solution sitting in the pool
// was invisible and local_mip's `[Sequential]` line read
// `effort=0 wall_ms=0`.  After the fix, `resolve_worker_start` prefers
// the pool's best entry over `mipdata->incumbent`, so local_mip sees
// FJ's fresh primal as its warm-start base.  The test runs FJ +
// LocalMIP (FPR + Scylla disabled), captures the developer-level log
// via HiGHS's logging callback, and asserts that both the `heur=fj`
// and `heur=local_mip` `[Sequential]` lines report non-zero effort.
// `lseu.mps` is chosen because FJ reliably produces a feasible for
// it inside the presolve budget.
TEST_CASE("LocalMIP: warm-starts from pool when FJ finds feasible before it (#74)",
          "[heuristic][local_mip][pool-aware]") {
    struct LogCapture {
        std::mutex mtx;
        std::vector<std::string> lines;
    };
    LogCapture capture;

    Highs h;
    h.setOptionValue("output_flag", true);
    h.setOptionValue("log_to_console", false);
    h.setOptionValue("log_dev_level", 3);
    h.setOptionValue("mip_heuristic_portfolio", false);
    h.setOptionValue("mip_heuristic_opportunistic", false);
    h.setOptionValue("mip_heuristic_run_feasibility_jump", true);
    h.setOptionValue("mip_heuristic_run_fpr", false);
    h.setOptionValue("mip_heuristic_run_local_mip", true);
    h.setOptionValue("mip_heuristic_run_scylla", false);
    // Force HiGHS to run the full root-presolve chain (fj → local_mip)
    // before any branching, so the [Sequential] lines are guaranteed
    // to appear regardless of whether HiGHS would otherwise shortcut
    // into B&B.  Reviewers (R3) flagged that the test would silently
    // fail if the chain never ran.
    h.setOptionValue("mip_root_presolve_only", true);

    auto log_cb = [](int callback_type, const std::string& message,
                     const HighsCallbackOutput* /*out*/, HighsCallbackInput* /*in*/,
                     void* user_data) {
        if (callback_type != kCallbackLogging) {
            return;
        }
        auto* cap = static_cast<LogCapture*>(user_data);
        std::lock_guard<std::mutex> lock(cap->mtx);
        cap->lines.emplace_back(message);
    };

    REQUIRE(h.setCallback(HighsCallbackFunctionType(log_cb), &capture) == HighsStatus::kOk);
    REQUIRE(h.startCallback(kCallbackLogging) == HighsStatus::kOk);
    REQUIRE(h.readModel(kInstancesDir + "/lseu.mps") == HighsStatus::kOk);
    REQUIRE(h.run() == HighsStatus::kOk);

    bool fj_ran = false;
    bool local_mip_ran = false;
    std::lock_guard<std::mutex> lock(capture.mtx);
    for (const auto& line : capture.lines) {
        if (line.find("[Sequential] heur=fj") != std::string::npos &&
            line.find("effort=0 ") == std::string::npos) {
            fj_ran = true;
        }
        if (line.find("[Sequential] heur=local_mip") != std::string::npos &&
            line.find("effort=0 ") == std::string::npos) {
            local_mip_ran = true;
        }
    }
    REQUIRE(fj_ran);
    REQUIRE(local_mip_ran);
}
