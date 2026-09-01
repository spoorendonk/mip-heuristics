#include "fpr_lp.h"
#include "fpr_lp_arms.h"
#include "Highs.h"
#include "test_common.h"

#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <string>
#include <vector>

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
// via heuristics::effective_flags, so it runs only at a `mip_heuristic_suite`
// value naming fpr — `fpr`, `all`, `fj,fpr` — and at no other.  Before that
// gate existed, the "vanilla" benchmark config left fpr_lp running during
// the B&B dive and wasn't vanilla.
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
// mip_heuristic_effort (not any presolve heuristic's option) and that the
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

// Issue #128: the `cliques2` arm used to sit in the full-obj-LP group even
// though its ranking (`fpr_var_order.cpp`'s `rank_cliques2`) reads its LP
// reference — for both the clique-tightness test and the per-clique
// ranking — as the paper's zero-objective vertex, not the full-objective
// LP solution. This test ties every LP-consuming arm's `ref_class`
// (`fpr_lp::lp_arm_table()`) to an expectation derived independently, from
// strategy identity, so the two facts cannot drift apart again: moving an
// arm's table entry to the wrong reference-class group — or wiring
// `build_setup` to the wrong pointer for a `ref_class` — fails here.
namespace {
bool same_strategy(const FprStrategyConfig& a, const FprStrategyConfig& b) {
    return a.var_strategy == b.var_strategy && a.val_strategy == b.val_strategy;
}
}  // namespace

TEST_CASE("fpr_lp: every LP arm's reference class matches what its strategy needs",
          "[fpr_lp][mode-matrix]") {
    const std::vector<fpr_lp::LpArmInfo> arms = fpr_lp::lp_arm_table();
    REQUIRE(arms.size() == 10);

    for (const auto& arm : arms) {
        INFO("arm = " << arm.name);
        const FprStrategyConfig& strat = arm.config.strat;

        // Paper Sect. 3 (Fig. 2 / zerocore value selection) and Sect. 4.1
        // (the plain `cliques` clique cover) are both defined against the
        // zero-obj analytic center.
        if (same_strategy(strat, kStratZerocore) || same_strategy(strat, kStratCliques)) {
            CHECK(arm.ref_class == fpr_lp::LpRefClass::kAnalyticCenter);
            // Paper Sect. 4.1: zerolp value selection and `cliques2` (Fig. 3's
            // dynamic clique cover) are both defined against the zero-obj
            // simplex vertex — the fact issue #128 exists to fix for cliques2.
        } else if (same_strategy(strat, kStratZerolp) || same_strategy(strat, kStratCliques2)) {
            CHECK(arm.ref_class == fpr_lp::LpRefClass::kZeroObjVertex);
            // The `lp` value strategy reads the full-objective LP solution.
        } else if (same_strategy(strat, kStratLp)) {
            CHECK(arm.ref_class == fpr_lp::LpRefClass::kFullObjLp);
        } else {
            FAIL("arm '" << arm.name << "' uses a strategy this test does not classify");
        }
    }

    // Directly pin the arm at the center of #128, by name rather than by
    // position: `cliques2`'s framework mode is `diveprop` in the paper's
    // Sect. 6.3 portfolio, so identify it by strategy pair rather than by
    // table index.
    const auto it = std::ranges::find_if(arms, [](const fpr_lp::LpArmInfo& arm) {
        return same_strategy(arm.config.strat, kStratCliques2);
    });
    REQUIRE(it != arms.end());
    CHECK(it->ref_class == fpr_lp::LpRefClass::kZeroObjVertex);
}

// ── Observability: fpr_lp reports its spend (#94) ──
//
// Before #94 `fpr_lp` charged the shared RENS/RINS envelope and booked
// `heuristic_effort_used`, but emitted no `[Sequential]` line — it did
// real work that no log and no benchmark script could see.  Routing it
// through `EffortLedger::charge_dive` fixed that, and this pins it: the
// dive-time heuristic must appear in the developer log alongside the
// four presolve heuristics.
//
// Asserted on non-zero effort via `heuristic_reported_effort`, so a
// regression that makes the ledger call conditional on `worker_effort >
// 0` — or drops it entirely — fails here rather than silently removing
// the observability.  `suite=fpr` because that is the narrowest value
// that still enables fpr_lp (see `heuristics::effective_flags`).
TEST_CASE("fpr_lp: emits a [Sequential] line for its dive-time spend",
          "[fpr_lp][mode-matrix][observability]") {
    fpr_lp::reset_dispatch_counts();
    const std::vector<std::string> lines = solve_capturing_log("bell5.mps", [](Highs& h) {
        h.setOptionValue("log_dev_level", 3);
        h.setOptionValue("mip_rel_gap", 0.0);
        set_suite(h, "fpr");
    });
    // Guard against a vacuous pass: no dispatch means nothing to report.
    REQUIRE(fpr_lp::dispatch_counts().dispatches >= 1);
    REQUIRE(heuristic_reported_effort(lines, "fpr_lp"));
}
