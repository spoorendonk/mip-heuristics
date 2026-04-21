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
double solve_fpr_lp_mode(const char* inst, bool opp) {
    Highs h;
    h.setOptionValue("output_flag", false);
    h.setOptionValue("mip_heuristic_run_fpr", true);
    h.setOptionValue("mip_heuristic_opportunistic", opp);
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
