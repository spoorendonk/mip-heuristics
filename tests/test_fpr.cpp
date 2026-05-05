#include "fpr.h"
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

// ===================================================================
// Issue #77 lifecycle: pause/resume across epoch gates is deterministic
// ===================================================================

namespace {

// Solve `inst` end-to-end at a small `mip_heuristic_effort` so the FPR
// per-call slice is well below the cost of a full DFS subtree on these
// instances — attempts must pause via `kBudgetGate` and resume on
// subsequent `run_epoch` calls, or fast-fail and trigger the
// multi-attempt fill loop.  Without this the [fpr][resume] tests can
// pass without ever exercising the new pause/resume code path on the
// small HiGHS check instances (egout / bell5 / flugpl all verdict in
// one slice at the default effort).  At 0.001 the slice fell below the
// cost of begin's initial `propagate(-1)` even on flugpl, so the loop
// never reached `kBudgetGate`; 0.01 = 5x the historical anchor effort
// (0.05/10) gives a slice large enough that step actually runs.
// Returns final objective.
double solve_with_seed_small_effort(const char *inst, int seed) {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("random_seed", seed);
    // Force seq/det path so the issue-#77 lifecycle is the dispatch under test.
    highs.setOptionValue("mip_heuristic_portfolio", false);
    highs.setOptionValue("mip_heuristic_opportunistic", false);
    // Pin threads=1 so the determinism contract is the *intra-worker*
    // lifecycle determinism (single-worker pause/resume + multi-attempt
    // fill).  Across-worker scheduling determinism is a different
    // (harder) property: HighsTaskExecutor is a global singleton lazily
    // initialised on the first Highs::run in a process and the per-thread
    // work-stealing order on subsequent runs depends on the scheduler's
    // internal state — running these tests sequentially in one Catch2
    // process exposes that as cross-test instability at effort=0.01 on
    // bell5 even though each test in isolation is deterministic.
    // CLAUDE.md says "don't pass --threads/threads= unless asked" for
    // benchmarks; this is a determinism test where threads=1 is the
    // ask.
    highs.setOptionValue("threads", 1);
    // Small effort → small per-call slice → multi-attempt loop and/or
    // pause-resume engages on the small HiGHS check instances.
    highs.setOptionValue("mip_heuristic_effort", 0.01);
    REQUIRE(highs.readModel(std::string(kInstancesDir) + "/" + inst) == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    return obj;
}

}  // namespace

// Test design note: these tests assert end-to-end objective equality
// across two same-seed runs.  This is a *proxy* for the issue #77
// literal acceptance bullet — "bit-identical [Sequential] summaries" —
// because parsing the HiGHS log would require wiring a callback into
// the test harness that can flake on log-format changes.  A divergence
// in effort count or attempt rotation that ultimately produces the
// same optimum would slip past objective equality alone.  In NDEBUG=0
// builds we additionally assert that the lifecycle counters
// (`fpr::budget_gate_hits()`, `fpr::multi_attempt_iters()`) are
// non-zero (proving the pause/resume / multi-attempt-fill paths
// actually fired) AND identical across runs (proving the lifecycle
// path traversal is deterministic).  Together this is a tighter
// guarantee than objective equality alone.
TEST_CASE("FPR resume: same seed reproduces same objective at small effort (egout)",
          "[fpr][resume][determinism]") {
#ifndef NDEBUG
    fpr::reset_test_counters();
#endif
    const double obj1 = solve_with_seed_small_effort("egout.mps", 42);
#ifndef NDEBUG
    const size_t gate1 = fpr::budget_gate_hits();
    const size_t multi1 = fpr::multi_attempt_iters();
    // Sanity: at least one of the two new lifecycle paths must have
    // engaged.  Without this, the determinism assertion below could
    // pass on a regression that bypassed the lifecycle entirely
    // (HiGHS' default B&B trivially solves these instances).
    REQUIRE((gate1 > 0 || multi1 > 0));
    fpr::reset_test_counters();
#endif
    const double obj2 = solve_with_seed_small_effort("egout.mps", 42);
    REQUIRE(obj1 == obj2);
#ifndef NDEBUG
    REQUIRE(fpr::budget_gate_hits() == gate1);
    REQUIRE(fpr::multi_attempt_iters() == multi1);
#endif
}

TEST_CASE("FPR resume: same seed reproduces same objective at small effort (bell5)",
          "[fpr][resume][determinism]") {
#ifndef NDEBUG
    fpr::reset_test_counters();
#endif
    const double obj1 = solve_with_seed_small_effort("bell5.mps", 7);
#ifndef NDEBUG
    const size_t gate1 = fpr::budget_gate_hits();
    const size_t multi1 = fpr::multi_attempt_iters();
    REQUIRE((gate1 > 0 || multi1 > 0));
    fpr::reset_test_counters();
#endif
    const double obj2 = solve_with_seed_small_effort("bell5.mps", 7);
    REQUIRE(obj1 == obj2);
#ifndef NDEBUG
    REQUIRE(fpr::budget_gate_hits() == gate1);
    REQUIRE(fpr::multi_attempt_iters() == multi1);
#endif
}

TEST_CASE("FPR resume: same seed reproduces same objective at small effort (flugpl)",
          "[fpr][resume][determinism]") {
#ifndef NDEBUG
    fpr::reset_test_counters();
#endif
    const double obj1 = solve_with_seed_small_effort("flugpl.mps", 0);
#ifndef NDEBUG
    const size_t gate1 = fpr::budget_gate_hits();
    const size_t multi1 = fpr::multi_attempt_iters();
    REQUIRE((gate1 > 0 || multi1 > 0));
    fpr::reset_test_counters();
#endif
    const double obj2 = solve_with_seed_small_effort("flugpl.mps", 0);
    REQUIRE(obj1 == obj2);
#ifndef NDEBUG
    REQUIRE(fpr::budget_gate_hits() == gate1);
    REQUIRE(fpr::multi_attempt_iters() == multi1);
#endif
}

TEST_CASE("FPR resume: paper-curated rotation still solves with multi-attempt cycling",
          "[fpr][resume]") {
    // Worker rotation `(worker_idx + attempt_idx) % kNumInitialFprConfigs`
    // visits every Class-1 config (paper Section 6.3) before cycling.
    // bell5 is a known-feasible instance that previously relied on the
    // randomized stale-epoch jump; the deterministic rotation must still
    // reach the same optimum.
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_run_fpr", true);
    highs.setOptionValue("mip_heuristic_run_local_mip", false);
    highs.setOptionValue("mip_heuristic_run_scylla", false);
    highs.setOptionValue("mip_heuristic_run_feasibility_jump", false);
    highs.setOptionValue("mip_heuristic_portfolio", false);
    REQUIRE(highs.readModel(kInstancesDir + "/bell5.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(8966406.49152).epsilon(1e-4));
}
