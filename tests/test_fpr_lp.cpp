#include "fpr_lp.h"
#include "Highs.h"
#include "test_common.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <string>

// ===================================================================
// fpr_lp dispatch tests
//
// fpr_lp runs LP-dependent FPR (paper Classes 2-3) during the B&B dive,
// after RINS/RENS, when the LP relaxation is at an optimal scaled state.
// It is a single heuristic family with one runner, so these tests pin
// that it exercises the dive path and finds the known optimum on bell5
// (small, non-trivial root LP where LP-dependent FPR contributes), and
// that every gate which should suppress the dispatch does.
// ===================================================================

namespace {
double solve_fpr_lp(const char* inst, int threads = 0) {
    // Only meaningful when `threads > 0`, but unconditional so the pin
    // and its teardown can never drift apart; a no-op reset is cheap.
    const ScopedThreadPin pin;
    Highs h;
    h.setOptionValue("output_flag", false);
    set_suite(h, "fpr");
    // bell5, the instance both callers use, is the one bundled instance
    // whose solve can stop on HiGHS's default `mip_rel_gap` (1e-4) short
    // of the optimum.  Require a proven-optimal solve so the objective
    // assertions are sound.
    h.setOptionValue("mip_rel_gap", 0.0);
    if (threads > 0) {
        h.setOptionValue("threads", threads);
    }
    REQUIRE(h.readModel(kInstancesDir + "/" + inst) == HighsStatus::kOk);
    REQUIRE(h.run() == HighsStatus::kOk);
    double obj;
    h.getInfoValue("objective_function_value", obj);
    return obj;
}
}  // namespace

TEST_CASE("fpr_lp: bell5 finds optimum and dispatches", "[fpr_lp][mode-matrix]") {
    fpr_lp::reset_dispatch_counts();
    REQUIRE(solve_fpr_lp("bell5.mps") == Catch::Approx(8966406.49152).epsilon(1e-4));
    REQUIRE(fpr_lp::dispatch_counts().dispatches >= 1);
}

// Regression tests for suite-aware gating: fpr_lp derives its enable flag
// via heuristics::effective_flags, so it runs only at `suite=fpr` and
// `suite=all`.  Before that gate existed, the "vanilla" benchmark config
// left fpr_lp running during the B&B dive and wasn't vanilla.
//
// `suite=local_mip` and `suite=scylla` disabling the dive-time heuristic is
// the deliberate consequence documented in README.md and docs/PARAMETERS.md:
// per-heuristic attribution has to cover fpr_lp too.  Both are pinned here
// so the property cannot regress silently.

namespace {
void require_no_fpr_lp_dispatch(const char* suite) {
    fpr_lp::reset_dispatch_counts();
    Highs h;
    h.setOptionValue("output_flag", false);
    set_suite(h, suite);
    REQUIRE(h.readModel(kInstancesDir + "/bell5.mps") == HighsStatus::kOk);
    REQUIRE(h.run() == HighsStatus::kOk);
    REQUIRE(fpr_lp::dispatch_counts().dispatches == 0);
}
}  // namespace

TEST_CASE("fpr_lp: suite=off disables fpr_lp dispatch", "[fpr_lp][mode-matrix][suite]") {
    require_no_fpr_lp_dispatch("off");
}

TEST_CASE("fpr_lp: suite=local_mip disables fpr_lp dispatch", "[fpr_lp][mode-matrix][suite]") {
    require_no_fpr_lp_dispatch("local_mip");
}

// Budget-integration regression: fpr_lp's per-call budget is capped at
// heuristic_effort_budget(nnz, mip_heuristic_effort), the shared vanilla
// B&B heuristic knob.  At effort=0 the cap is 0, so fpr_lp must never
// dispatch — even though the raw run_fpr flag is true and the
// moreHeuristicsAllowed() grace offset (+10000 LP iterations) would
// otherwise leave headroom.  Pins that fpr_lp draws its budget from
// mip_heuristic_effort (not mip_heuristic_presolve_effort) and that the
// cap actually gates dispatch.  Every sub-MIP-creating vanilla
// heuristic (RENS, RINS, rootReducedCost — the three solveSubMip
// callers) must be off here: sub-MIPs hard-set mip_heuristic_effort=0.8
// in the sub-MIP options (HighsPrimalHeuristics::solveSubMip), so
// fpr_lp legitimately dispatches inside a sub-MIP regardless of the
// parent's effort=0, and the dispatch counters are process-global.
TEST_CASE("fpr_lp: mip_heuristic_effort=0 disables fpr_lp via the budget cap",
          "[fpr_lp][mode-matrix][budget]") {
    fpr_lp::reset_dispatch_counts();
    Highs h;
    h.setOptionValue("output_flag", false);
    set_suite(h, "fpr");
    h.setOptionValue("mip_heuristic_run_rens", false);
    h.setOptionValue("mip_heuristic_run_rins", false);
    h.setOptionValue("mip_heuristic_run_root_reduced_cost", false);
    h.setOptionValue("mip_heuristic_effort", 0.0);
    REQUIRE(h.readModel(kInstancesDir + "/bell5.mps") == HighsStatus::kOk);
    REQUIRE(h.run() == HighsStatus::kOk);
    REQUIRE(fpr_lp::dispatch_counts().dispatches == 0);
}

TEST_CASE("fpr_lp: suite=scylla disables fpr_lp dispatch", "[fpr_lp][mode-matrix][suite]") {
    require_no_fpr_lp_dispatch("scylla");
}

// `run_workers` spawns `num_threads` workers with arm = w % kNumLpArms
// (10).  On a machine with threads > 10 the extra workers wrap around
// the arm list.  This test pins threads = 12 so workers 10 and 11 double
// up on arms 0 and 1 with distinct seeds — it must still find the
// optimum and must still dispatch (not crash on shared var_orders[arm]
// access, which is read-only).
TEST_CASE("fpr_lp: arm wrap-around with threads > kNumLpArms", "[fpr_lp][mode-matrix]") {
    fpr_lp::reset_dispatch_counts();
    REQUIRE(solve_fpr_lp("bell5.mps", /*threads=*/12) ==
            Catch::Approx(8966406.49152).epsilon(1e-4));
    REQUIRE(fpr_lp::dispatch_counts().dispatches >= 1);
}
