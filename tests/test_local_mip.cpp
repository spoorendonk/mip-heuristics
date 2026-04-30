#include "heuristic_common.h"
#include "Highs.h"
#include "local_mip.h"
#include "local_mip_construction.h"
#include "mip/HighsMipSolverData.h"  // for kSolutionSource* constants
#include "rng.h"
#include "solution_pool.h"
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
TEST_CASE("LocalMIP cold-start: emits non-zero [Sequential] when upstream heuristics off",
          "[heuristic][local_mip][construction][cold-start]") {
    // Reviewer R3-2 (round-3) flagged that the previous version of
    // this test ran full B&B and asserted `obj == 1201500.0` on
    // flugpl — which HiGHS finds via its own LP-driven branching even
    // if `construct_initial_solution` is a no-op.  Here we constrain
    // the solve to the presolve chain via `mip_root_presolve_only` and
    // assert directly on the `[Sequential] heur=local_mip effort=…`
    // line with non-zero effort.  That's the real signal that #75's
    // cold-start construction kicked in.
    struct LogCapture {
        std::mutex mtx;
        std::vector<std::string> lines;
    };
    LogCapture capture;

    Highs h;
    h.setOptionValue("output_flag", true);
    h.setOptionValue("log_to_console", false);
    h.setOptionValue("log_dev_level", 3);
    h.setOptionValue("mip_root_presolve_only", true);
    h.setOptionValue("mip_heuristic_run_fpr", false);
    h.setOptionValue("mip_heuristic_run_feasibility_jump", false);
    h.setOptionValue("mip_heuristic_run_scylla", false);
    h.setOptionValue("mip_heuristic_run_local_mip", true);
    h.setOptionValue("mip_heuristic_portfolio", false);
    h.setOptionValue("mip_heuristic_opportunistic", false);

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
    REQUIRE(h.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);
    REQUIRE(h.run() == HighsStatus::kOk);

    bool local_mip_ran = false;
    std::lock_guard<std::mutex> lock(capture.mtx);
    for (const auto& line : capture.lines) {
        if (line.find("[Sequential] heur=local_mip") != std::string::npos &&
            line.find("effort=0 ") == std::string::npos) {
            local_mip_ran = true;
            break;
        }
    }
    REQUIRE(local_mip_ran);
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

// Unit-level regression for #74's pool-aware helper (complements the
// log-based integration test above).  `resolve_worker_start` prefers
// the pool's best over `mipdata->incumbent` and over the cold-start
// construction; the reasoning delegates to `SolutionPool::copy_best`
// for that first branch.  Round-2 reviewers flagged that the
// integration test can't distinguish pool-warm-start from cold-start
// construction (both produce non-zero effort); testing `copy_best`
// directly proves the pool-first branch returns exactly the seeded
// vector, which is the cheap half of #74 to pin down.  The full
// integration-level distinction (did the worker start from the pool
// or construct fresh?) still relies on the `lseu.mps` test above.
TEST_CASE("SolutionPool::copy_best returns exactly the seeded best entry (#74 unit)",
          "[heuristic][local_mip][pool-aware][unit]") {
    SolutionPool pool(/*capacity=*/4, /*minimize=*/true);
    std::vector<double> probe;
    // Empty pool: no best, copy_best returns false and leaves `probe`
    // untouched.
    probe.assign(3, 9.9);  // sentinel to confirm no write
    REQUIRE_FALSE(pool.copy_best(probe));
    REQUIRE(probe == std::vector<double>{9.9, 9.9, 9.9});

    // Seed a worse and a better entry; copy_best must return the better.
    const std::vector<double> worse_sol{1.0, 2.0, 3.0};
    const std::vector<double> better_sol{4.0, 5.0, 6.0};
    REQUIRE(pool.try_add(/*obj=*/100.0, worse_sol, kSolutionSourceFJ));
    REQUIRE(pool.try_add(/*obj=*/10.0, better_sol, kSolutionSourceLocalMIP));
    probe.clear();
    REQUIRE(pool.copy_best(probe));
    REQUIRE(probe == better_sol);
}

// ── Distinguish #74 (pool warm-start) vs #75 (cold-start construction) ──
//
// Reviewers R1-8, R2-7, R3-3 (round-3) flagged that the existing
// integration tests assert "local_mip ran with non-zero effort", which
// is true regardless of whether the warm-start came from the shared
// pool (#74) or the paper's cold-start construction (#75).  These two
// tests use the warm-start branch counters in `local_mip.h` to pin
// down which path actually fired in each scenario.
//
// Counter contract (from `resolve_worker_start`):
//   - `pool`: SolutionPool::copy_best returned a vector (warm).
//   - `incumbent`: pool empty, mipdata->incumbent picked up.
//   - `construction`: both empty → paper's Phase A/B ran.

// Scenario A: nothing populates the pool/incumbent before LocalMIP, so
// the cold-start construction must fire on every worker (#75 active,
// #74 unreachable).
//
// State the test asserts on entry: with FJ, FPR, and Scylla disabled
// and `mip_root_presolve_only` set, no upstream heuristic populates
// either the shared pool or `mipdata->incumbent` before LocalMIP runs;
// the construction branch is therefore the only reachable warm-start
// path.  R2-6 / R3-4 round-4 review: assert *both* `pool == 0` AND
// `incumbent == 0` so a future HiGHS presolve change that pre-populates
// the incumbent surfaces as a clean test failure rather than silently
// reaching cold-start through a different (no-op) branch.  The
// `construction >= 1` assertion alone can't tell those apart; the
// purpose of this scenario is the cold-start path is *reachable*, not
// just "construction fired".
TEST_CASE("LocalMIP: cold-start construction fires when pool and incumbent are empty (#75)",
          "[heuristic][local_mip][cold-start][warm-start-counters]") {
    local_mip::reset_warm_start_counters();
    Highs h;
    h.setOptionValue("output_flag", false);
    h.setOptionValue("mip_root_presolve_only", true);
    h.setOptionValue("mip_heuristic_run_fpr", false);
    h.setOptionValue("mip_heuristic_run_feasibility_jump", false);
    h.setOptionValue("mip_heuristic_run_scylla", false);
    h.setOptionValue("mip_heuristic_run_local_mip", true);
    h.setOptionValue("mip_heuristic_portfolio", false);
    h.setOptionValue("mip_heuristic_opportunistic", false);
    REQUIRE(h.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);
    REQUIRE(h.run() == HighsStatus::kOk);

    auto counters = local_mip::warm_start_counters();
    // Cold-start construction must have run at least once (one per
    // worker, modulo the cold-start cache de-duplication).
    REQUIRE(counters.construction >= 1);
    // Pool was empty before LocalMIP fired (no upstream heuristic
    // populated it), so the pool branch must not have triggered.
    REQUIRE(counters.pool == 0);
    // Likewise, `mipdata->incumbent` must have been empty: a future
    // HiGHS presolve change that pre-populates the incumbent would
    // otherwise let the warm-start fall into the (different) incumbent
    // branch and silently bypass the cold-start construction this
    // scenario is meant to exercise.
    REQUIRE(counters.incumbent == 0);
}

// Scenario B: FJ runs first and populates the pool with a feasible
// solution; LocalMIP must then warm-start from the pool (#74 active,
// #75 unreachable for the worker setup paths).
TEST_CASE("LocalMIP: pool warm-start fires when FJ pre-populates pool (#74)",
          "[heuristic][local_mip][pool-aware][warm-start-counters]") {
    local_mip::reset_warm_start_counters();
    Highs h;
    h.setOptionValue("output_flag", false);
    h.setOptionValue("mip_root_presolve_only", true);
    h.setOptionValue("mip_heuristic_run_fpr", false);
    h.setOptionValue("mip_heuristic_run_feasibility_jump", true);
    h.setOptionValue("mip_heuristic_run_scylla", false);
    h.setOptionValue("mip_heuristic_run_local_mip", true);
    h.setOptionValue("mip_heuristic_portfolio", false);
    h.setOptionValue("mip_heuristic_opportunistic", false);
    // `lseu.mps` is the same instance the existing #74 regression test
    // uses — FJ reliably finds a feasible inside the presolve budget,
    // so the pool is non-empty by the time LocalMIP fires.
    REQUIRE(h.readModel(kInstancesDir + "/lseu.mps") == HighsStatus::kOk);
    REQUIRE(h.run() == HighsStatus::kOk);

    auto counters = local_mip::warm_start_counters();
    // The decisive assertion: at least one worker's start came from the
    // pool, not construction.  Without #74's pool-aware lookup the only
    // path that produces non-zero warm-start counts is `construction`.
    REQUIRE(counters.pool >= 1);
}
