#include "fpr_lp.h"
#include "Highs.h"
#include "test_common.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <string>

// ===================================================================
// fpr_lp 4-mode smoke tests (issue #63)
//
// fpr_lp runs LP-dependent FPR (paper Classes 2-3) during the B&B dive,
// after RINS/RENS, when the LP relaxation is at an optimal scaled state.
// Each cell of the (portfolio × opportunistic) execution matrix must
// exercise the dive path and find the known optimum.  bell5 is chosen
// because it is small enough to run fast and has a non-trivial root LP
// where LP-dependent FPR contributes (also exercised by the existing
// "FPR strategies: portfolio multi-arm on bell5" characterization).
// ===================================================================

namespace {
double solve_fpr_lp_mode(const char* inst, bool portfolio, bool opp) {
    Highs h;
    h.setOptionValue("output_flag", false);
    h.setOptionValue("mip_heuristic_run_fpr", true);
    h.setOptionValue("mip_heuristic_portfolio", portfolio);
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
    REQUIRE(solve_fpr_lp_mode("bell5.mps", false, false) ==
            Catch::Approx(8966406.49152).epsilon(1e-4));
    const auto counts = fpr_lp::dispatch_counts();
    REQUIRE(counts.seq_det >= 1);
    REQUIRE(counts.seq_opp == 0);
    REQUIRE(counts.port_det == 0);
    REQUIRE(counts.port_opp == 0);
}

TEST_CASE("fpr_lp seq/opp: bell5 finds optimum and dispatches",
          "[fpr_lp][mode-matrix][opportunistic]") {
    fpr_lp::reset_dispatch_counts();
    REQUIRE(solve_fpr_lp_mode("bell5.mps", false, true) ==
            Catch::Approx(8966406.49152).epsilon(1e-4));
    const auto counts = fpr_lp::dispatch_counts();
    REQUIRE(counts.seq_opp >= 1);
    REQUIRE(counts.seq_det == 0);
    REQUIRE(counts.port_det == 0);
    REQUIRE(counts.port_opp == 0);
}

TEST_CASE("fpr_lp port/det: bell5 finds optimum and dispatches",
          "[fpr_lp][mode-matrix][portfolio]") {
    fpr_lp::reset_dispatch_counts();
    REQUIRE(solve_fpr_lp_mode("bell5.mps", true, false) ==
            Catch::Approx(8966406.49152).epsilon(1e-4));
    const auto counts = fpr_lp::dispatch_counts();
    REQUIRE(counts.port_det >= 1);
    REQUIRE(counts.seq_det == 0);
    REQUIRE(counts.seq_opp == 0);
    REQUIRE(counts.port_opp == 0);
}

TEST_CASE("fpr_lp port/opp: bell5 finds optimum and dispatches",
          "[fpr_lp][mode-matrix][portfolio][opportunistic]") {
    fpr_lp::reset_dispatch_counts();
    REQUIRE(solve_fpr_lp_mode("bell5.mps", true, true) ==
            Catch::Approx(8966406.49152).epsilon(1e-4));
    const auto counts = fpr_lp::dispatch_counts();
    REQUIRE(counts.port_opp >= 1);
    REQUIRE(counts.seq_det == 0);
    REQUIRE(counts.seq_opp == 0);
    REQUIRE(counts.port_det == 0);
}
