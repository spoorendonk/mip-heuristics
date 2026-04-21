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
// It is a single heuristic family (not a meta-portfolio), so only the
// mip_heuristic_opportunistic flag selects between the two variants —
// mip_heuristic_portfolio does not apply to fpr_lp dispatch.  Both
// variants must exercise the dive path and find the known optimum on
// bell5 (small, non-trivial root LP where LP-dependent FPR contributes).
// ===================================================================

namespace {
double solve_fpr_lp_mode(const char* inst, bool opp, bool portfolio = false, int threads = 0) {
    Highs h;
    h.setOptionValue("output_flag", false);
    h.setOptionValue("mip_heuristic_run_fpr", true);
    h.setOptionValue("mip_heuristic_opportunistic", opp);
    h.setOptionValue("mip_heuristic_portfolio", portfolio);
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

// Regression tests: the portfolio flag must not reach fpr_lp dispatch
// (fpr_lp is a single heuristic family, not a meta-portfolio).  Even when
// mip_heuristic_portfolio=true, fpr_lp must route to seq/det or seq/opp
// based only on mip_heuristic_opportunistic.  The assertions below lock
// that contract in so a future regression that re-introduces a portfolio
// branch inside fpr_lp::run would show up here rather than silently.

TEST_CASE("fpr_lp: portfolio flag does not affect seq/det dispatch",
          "[fpr_lp][mode-matrix][portfolio]") {
    fpr_lp::reset_dispatch_counts();
    REQUIRE(solve_fpr_lp_mode("bell5.mps", /*opp=*/false, /*portfolio=*/true) ==
            Catch::Approx(8966406.49152).epsilon(1e-4));
    const auto counts = fpr_lp::dispatch_counts();
    REQUIRE(counts.seq_det >= 1);
    REQUIRE(counts.seq_opp == 0);
}

TEST_CASE("fpr_lp: portfolio flag does not affect seq/opp dispatch",
          "[fpr_lp][mode-matrix][portfolio][opportunistic]") {
    fpr_lp::reset_dispatch_counts();
    REQUIRE(solve_fpr_lp_mode("bell5.mps", /*opp=*/true, /*portfolio=*/true) ==
            Catch::Approx(8966406.49152).epsilon(1e-4));
    const auto counts = fpr_lp::dispatch_counts();
    REQUIRE(counts.seq_opp >= 1);
    REQUIRE(counts.seq_det == 0);
}

// run_sequential_deterministic spawns `num_threads` workers with
// arm = w % kNumLpArms (10).  On a machine with threads > 10 the extra
// workers wrap around the arm list.  This test pins threads = 12 so
// workers 10 and 11 double up on arms 0 and 1 with distinct seeds —
// it must still find the optimum and must still dispatch via seq_det
// (not crash on shared var_orders[arm] access, which is read-only).
TEST_CASE("fpr_lp seq/det: arm wrap-around with threads > kNumLpArms", "[fpr_lp][mode-matrix]") {
    fpr_lp::reset_dispatch_counts();
    REQUIRE(solve_fpr_lp_mode("bell5.mps", /*opp=*/false, /*portfolio=*/false,
                              /*threads=*/12) == Catch::Approx(8966406.49152).epsilon(1e-4));
    const auto counts = fpr_lp::dispatch_counts();
    REQUIRE(counts.seq_det >= 1);
}
