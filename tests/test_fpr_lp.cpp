#include "fpr_lp.h"
#include "Highs.h"
#include "test_common.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <string>

// ===================================================================
// fpr_lp 2-mode smoke tests
//
// fpr_lp runs LP-dependent FPR (paper Classes 2-3) during the B&B dive,
// after RINS/RENS, when the LP relaxation is at an optimal scaled state.
// It is a single heuristic family, so only the
// mip_heuristic_opportunistic flag selects between the two variants.
// Both variants must exercise the dive path and find the known optimum on
// bell5 (small, non-trivial root LP where LP-dependent FPR contributes).
// ===================================================================

namespace {
double solve_fpr_lp_mode(const char* inst, bool opp, int threads = 0) {
    Highs h;
    h.setOptionValue("output_flag", false);
    h.setOptionValue("mip_heuristic_run_fpr", true);
    h.setOptionValue("mip_heuristic_opportunistic", opp);
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

TEST_CASE("fpr_lp seq/det: bell5 finds optimum and dispatches", "[fpr_lp][mode-matrix]") {
    fpr_lp::reset_dispatch_counts();
    REQUIRE(solve_fpr_lp_mode("bell5.mps", false) == Catch::Approx(8966406.49152).epsilon(1e-4));
    const auto counts = fpr_lp::dispatch_counts();
    REQUIRE(counts.seq_det >= 1);
    REQUIRE(counts.seq_opp == 0);
}

TEST_CASE("fpr_lp seq/opp: bell5 finds optimum and dispatches",
          "[fpr_lp][mode-matrix][opportunistic]") {
    fpr_lp::reset_dispatch_counts();
    REQUIRE(solve_fpr_lp_mode("bell5.mps", true) == Catch::Approx(8966406.49152).epsilon(1e-4));
    const auto counts = fpr_lp::dispatch_counts();
    REQUIRE(counts.seq_opp >= 1);
    REQUIRE(counts.seq_det == 0);
}

// Regression tests for preset-aware gating: fpr_lp derives its enable
// flag via heuristics::effective_flags, so mip_heuristic_preset must
// reach it even though the raw mip_heuristic_run_fpr option (default
// true, restored by run_presolve's write-back before B&B starts) says
// otherwise.  Pre-split, preset=off left fpr_lp running during the
// B&B dive — the "vanilla" benchmark config wasn't vanilla.

TEST_CASE("fpr_lp: preset=off disables fpr_lp dispatch", "[fpr_lp][mode-matrix][preset]") {
    fpr_lp::reset_dispatch_counts();
    Highs h;
    h.setOptionValue("output_flag", false);
    h.setOptionValue("mip_heuristic_preset", "off");
    // Raw flag deliberately left at its default (true): the preset must win.
    REQUIRE(h.readModel(kInstancesDir + "/bell5.mps") == HighsStatus::kOk);
    REQUIRE(h.run() == HighsStatus::kOk);
    const auto counts = fpr_lp::dispatch_counts();
    REQUIRE(counts.seq_det == 0);
    REQUIRE(counts.seq_opp == 0);
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
    h.setOptionValue("mip_heuristic_run_fpr", true);
    h.setOptionValue("mip_heuristic_run_rens", false);
    h.setOptionValue("mip_heuristic_run_rins", false);
    h.setOptionValue("mip_heuristic_run_root_reduced_cost", false);
    h.setOptionValue("mip_heuristic_effort", 0.0);
    REQUIRE(h.readModel(kInstancesDir + "/bell5.mps") == HighsStatus::kOk);
    REQUIRE(h.run() == HighsStatus::kOk);
    const auto counts = fpr_lp::dispatch_counts();
    REQUIRE(counts.seq_det == 0);
    REQUIRE(counts.seq_opp == 0);
}

TEST_CASE("fpr_lp: preset=scylla disables fpr_lp dispatch", "[fpr_lp][mode-matrix][preset]") {
    fpr_lp::reset_dispatch_counts();
    Highs h;
    h.setOptionValue("output_flag", false);
    h.setOptionValue("mip_heuristic_preset", "scylla");
    REQUIRE(h.readModel(kInstancesDir + "/bell5.mps") == HighsStatus::kOk);
    REQUIRE(h.run() == HighsStatus::kOk);
    const auto counts = fpr_lp::dispatch_counts();
    REQUIRE(counts.seq_det == 0);
    REQUIRE(counts.seq_opp == 0);
}

// run_sequential_deterministic spawns `num_threads` workers with
// arm = w % kNumLpArms (10).  On a machine with threads > 10 the extra
// workers wrap around the arm list.  This test pins threads = 12 so
// workers 10 and 11 double up on arms 0 and 1 with distinct seeds —
// it must still find the optimum and must still dispatch via seq_det
// (not crash on shared var_orders[arm] access, which is read-only).
TEST_CASE("fpr_lp seq/det: arm wrap-around with threads > kNumLpArms", "[fpr_lp][mode-matrix]") {
    fpr_lp::reset_dispatch_counts();
    REQUIRE(solve_fpr_lp_mode("bell5.mps", /*opp=*/false,
                              /*threads=*/12) == Catch::Approx(8966406.49152).epsilon(1e-4));
    const auto counts = fpr_lp::dispatch_counts();
    REQUIRE(counts.seq_det >= 1);
}
