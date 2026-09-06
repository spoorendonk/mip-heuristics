#include "fpr_lp.h"
#include "fpr_lp_arms.h"
#include "heuristic_common.h"
#include "Highs.h"
#include "test_common.h"

#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
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
    set_suite(h, "fpr_lp");
    h.setOptionValue("mip_heuristic_fpr_lp_effort", 1.0);
    // `fpr_lp` ships off (effort default 0), so a fixture that wants it to
    // dispatch has to turn it on as well as name its token.
    h.setOptionValue("mip_heuristic_fpr_lp_effort", 1.0);
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
// value naming fpr_lp — `fpr_lp`, `all`, `fj,fpr_lp` — and at no other.
// Before that gate existed, the "vanilla" benchmark config left fpr_lp
// running during the B&B dive and wasn't vanilla.
//
// `suite=local_mip` and `suite=scylla` disabling the dive-time heuristic is
// the deliberate consequence documented in README.md and docs/PARAMETERS.md:
// per-heuristic attribution has to cover fpr_lp too.  Both are pinned here
// so the property cannot regress silently.
//
// `suite=fpr` joined them in #164, and that one is the issue's single
// intended behaviour change rather than a property that always held: the
// dive-time heuristic used to follow presolve FPR's token, so "presolve FPR
// without fpr_lp" — the row the ablation needs — had no spelling at all.

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
    set_suite(h, "fpr_lp");
    h.setOptionValue("mip_heuristic_fpr_lp_effort", 1.0);
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

// #164's intended behaviour change, asserted on the dispatch itself rather
// than on the flag: presolve FPR alone no longer drags the dive-time variant
// along with it.
TEST_CASE("fpr_lp: suite=fpr alone disables fpr_lp dispatch", "[fpr_lp][mode-matrix][suite]") {
    require_no_fpr_lp_dispatch("fpr");
}

// The other half of the same change, and the guard against a vacuous pass
// above: naming both tokens dispatches it, so the four cells of #164's
// matrix are all reachable.
TEST_CASE("fpr_lp: suite=fpr,fpr_lp dispatches it beside presolve FPR",
          "[fpr_lp][mode-matrix][suite]") {
    fpr_lp::reset_dispatch_counts();
    Highs h;
    h.setOptionValue("output_flag", false);
    set_suite(h, "fpr,fpr_lp");
    h.setOptionValue("mip_heuristic_fpr_lp_effort", 1.0);
    // `fpr_lp` ships off (effort default 0), so a fixture that wants it to
    // dispatch has to turn it on as well as name its token.
    h.setOptionValue("mip_heuristic_fpr_lp_effort", 1.0);
    REQUIRE(h.readModel(kInstancesDir + "/bell5.mps") == HighsStatus::kOk);
    REQUIRE(h.run() == HighsStatus::kOk);
    REQUIRE(fpr_lp::dispatch_counts().dispatches >= 1);
}

// The effort option's zero, which is the *other* way to disable fpr_lp and
// has to be indistinguishable from omitting the token (#164).
//
// It matters where the two return from, not just that they return: the
// suite gate sits above every read and write of `heuristic_lp_iterations` /
// `total_lp_iterations`, the counters `moreHeuristicsAllowed()` uses to
// decide whether RENS and RINS run, and a zero-effort disable that read or
// charged them would silently move those two heuristics — so an fpr_lp
// ablation would be measuring RENS/RINS as well.  The two spellings are
// therefore compared as whole solves below, not merely as dispatch counts.
TEST_CASE("fpr_lp: effort 0 disables the dispatch", "[fpr_lp][mode-matrix][budget]") {
    fpr_lp::reset_dispatch_counts();
    Highs h;
    h.setOptionValue("output_flag", false);
    set_suite(h, "fpr_lp");
    h.setOptionValue("mip_heuristic_fpr_lp_effort", 1.0);
    require_option(h, "mip_heuristic_fpr_lp_effort", 0.0);
    REQUIRE(h.readModel(kInstancesDir + "/bell5.mps") == HighsStatus::kOk);
    REQUIRE(h.run() == HighsStatus::kOk);
    REQUIRE(fpr_lp::dispatch_counts().dispatches == 0);
}

// The full equivalence #164 makes true for the first time: disabling fpr_lp
// by zeroing its effort and disabling it by leaving its token out of the
// suite are the same solve, RENS/RINS included.
//
// Compared on the solver's own counters rather than on a log: the primal
// bound says whether the search ended anywhere else, and the node and
// LP-iteration totals say whether it *got* there differently.  Those two
// are what a disabled fpr_lp that had already charged the shared envelope
// would move — `moreHeuristicsAllowed()` reads
// `heuristic_lp_iterations` against `total_lp_iterations`, so a charge
// changes which dives run RENS and RINS, and the node count changes with
// it.  `mip_rel_gap = 0` so both sides run to a proven optimum instead of
// stopping at two different incumbents.
TEST_CASE("fpr_lp: effort 0 is equivalent to omitting fpr_lp from the suite",
          "[fpr_lp][mode-matrix][budget][regression]") {
    struct Outcome {
        double objective = 0.0;
        int64_t nodes = 0;
        HighsInt lp_iterations = 0;
    };
    const auto solve = [](const char* suite, double fpr_lp_effort) {
        const ScopedThreadPin pin;
        Highs h;
        h.setOptionValue("output_flag", false);
        set_suite(h, suite);
        require_option(h, "mip_rel_gap", 0.0);
        require_option(h, "threads", 1);
        require_option(h, "random_seed", 1);
        require_option(h, "mip_heuristic_fpr_lp_effort", fpr_lp_effort);
        REQUIRE(h.readModel(kInstancesDir + "/bell5.mps") == HighsStatus::kOk);
        REQUIRE(h.run() == HighsStatus::kOk);
        Outcome out;
        h.getInfoValue("objective_function_value", out.objective);
        h.getInfoValue("mip_node_count", out.nodes);
        h.getInfoValue("simplex_iteration_count", out.lp_iterations);
        return out;
    };

    // Both disable fpr_lp; everything else about the two runs is identical.
    const Outcome zeroed = solve("all", 0.0);
    const Outcome omitted = solve("fj,fpr,local_mip,scylla", 1.0);
    CHECK(zeroed.objective == omitted.objective);
    CHECK(zeroed.nodes == omitted.nodes);
    CHECK(zeroed.lp_iterations == omitted.lp_iterations);
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
// strategy identity, so moving an arm's table entry to the wrong
// reference-class group fails here.
//
// This test alone does *not* catch `build_setup` wiring a `ref_class` to
// the wrong pointer — a cold review of the first version of this fix
// proved that empirically: reprogramming `fpr_lp::select_ref`'s
// `kZeroObjVertex` case to return the full-obj LP pointer reintroduces
// #128's bug and this test still passes, because `ref_class` is still the
// value `kLpArmTable` says. The class-to-pointer mapping is a second,
// independent fact, and the next test case below (`fpr_lp::select_ref`
// directly) is what pins it.
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

// Issue #128 cold review: exercises `fpr_lp::select_ref` — the
// class-to-pointer half of the arm's reference binding — directly, for
// every current `LpRefClass` enumerator. Distinct sentinel doubles (never
// dereferenced, only compared by address) stand in for the analytic
// center, zero-obj vertex and full-obj LP vectors `build_setup` would
// otherwise pass, so this fails exactly the two probes the review ran and
// the test above does not: reprogramming a case to return the wrong
// sentinel, or (with the `-Werror=switch` guard also added by this commit)
// adding a fourth `LpRefClass` enumerator with no case for it.
TEST_CASE("fpr_lp: select_ref returns the pointer its LpRefClass names", "[fpr_lp][mode-matrix]") {
    double ac_sentinel = 0.0;
    double zv_sentinel = 0.0;
    double lp_sentinel = 0.0;
    const double* const ac = &ac_sentinel;
    const double* const zv = &zv_sentinel;
    const double* const lp = &lp_sentinel;

    CHECK(fpr_lp::select_ref(fpr_lp::LpRefClass::kAnalyticCenter, ac, zv, lp) == ac);
    CHECK(fpr_lp::select_ref(fpr_lp::LpRefClass::kZeroObjVertex, ac, zv, lp) == zv);
    CHECK(fpr_lp::select_ref(fpr_lp::LpRefClass::kFullObjLp, ac, zv, lp) == lp);
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
// the observability.  `suite=fpr_lp` because that is the narrowest value
// that enables fpr_lp — since #164 it has its own token, and `fpr` selects
// presolve FPR alone (see `heuristics::effective_flags`).
TEST_CASE("fpr_lp: emits a [Sequential] line for its dive-time spend",
          "[fpr_lp][mode-matrix][observability]") {
    fpr_lp::reset_dispatch_counts();
    const std::vector<std::string> lines = solve_capturing_log("bell5.mps", [](Highs& h) {
        h.setOptionValue("log_dev_level", 3);
        h.setOptionValue("mip_rel_gap", 0.0);
        set_suite(h, "fpr_lp");
        h.setOptionValue("mip_heuristic_fpr_lp_effort", 1.0);
    });
    // Guard against a vacuous pass: no dispatch means nothing to report.
    REQUIRE(fpr_lp::dispatch_counts().dispatches >= 1);
    REQUIRE(heuristic_reported_effort(lines, "fpr_lp"));
}

// ── The effort share actually scales the budget (#164) ──
//
// A cold review of #164 deleted `* share` from the budget expression,
// rebuilt, and the whole suite passed: every test then touching the option
// used `0.0` (which the early return at the top of `run` catches) or `1.0`
// (a no-op factor), so nothing pinned the arithmetic in between.  The
// failure that would have escaped is not a crash — #165 sweeps this share,
// every point would run at the same budget, and the experiment would
// conclude the budget does not matter, from a green suite.
//
// Two cases, because either alone is passable by a wrong implementation.
// This one pins the arithmetic through `dive_budget`; the one below pins
// that a real dispatch still reads it, which a pure function nothing calls
// would not.
TEST_CASE("fpr_lp: the effort share scales the whole per-call budget",
          "[fpr_lp][budget][regression]") {
    // `nnz` and `mip_heuristic_effort` fix the cap; the headroom is then
    // chosen on either side of it, because the two sides fail differently
    // under the mutation this exists for.
    constexpr size_t kNnz = 4096;
    constexpr double kVanillaEffort = 0.05;
    const auto cap = static_cast<double>(vanilla_effort_budget(kNnz, kVanillaEffort));
    REQUIRE(cap > 0.0);

    SECTION("a headroom-bound call scales with the share") {
        // Headroom well under the cap, so the `min` picks it.
        const double headroom_iters = cap / static_cast<double>(kNnz) / 4.0;
        const double slice = headroom_iters * static_cast<double>(kNnz);
        REQUIRE(slice < cap);
        CHECK(fpr_lp::dive_budget(headroom_iters, kNnz, kVanillaEffort, 1.0) ==
              static_cast<size_t>(slice));
        CHECK(fpr_lp::dive_budget(headroom_iters, kNnz, kVanillaEffort, 0.5) ==
              static_cast<size_t>(0.5 * slice));
    }

    SECTION("a cap-bound call scales with the share too") {
        // Headroom far above the cap, so the `min` picks the cap.  This is
        // the half that dies under `min(headroom * share, cap)`, #164's
        // first spelling: there the cap is reached at every share >= 1 and
        // is *never* exceeded, so a search over the share sees one budget
        // on exactly the calls a large share was meant to deepen.
        const double headroom_iters = 1e6 * cap / static_cast<double>(kNnz);
        REQUIRE(headroom_iters * static_cast<double>(kNnz) > cap);
        CHECK(fpr_lp::dive_budget(headroom_iters, kNnz, kVanillaEffort, 1.0) ==
              static_cast<size_t>(cap));
        CHECK(fpr_lp::dive_budget(headroom_iters, kNnz, kVanillaEffort, 0.5) ==
              static_cast<size_t>(0.5 * cap));
        // Above 1.0 the budget grows without bound, which is what lets a
        // calibration hand fpr_lp a budget that cannot bind — the reason
        // the option's ceiling is 1e6, exactly as for the four presolve
        // effort options.  Overdrawing is self-correcting: the charge-back
        // depletes the counters the headroom is computed from.
        CHECK(fpr_lp::dive_budget(headroom_iters, kNnz, kVanillaEffort, 4.0) ==
              static_cast<size_t>(4.0 * cap));
    }

    SECTION("share 1.0 is the budget the call took before the option existed") {
        // The hard requirement of #164: the shipped default must not move.
        // `share * min(h, c)` and the pre-#164 `min(h, c)` are the same
        // number at 1.0, on both sides of the `min`.
        for (const double headroom_iters :
             {cap / static_cast<double>(kNnz) / 4.0, 1e6 * cap / static_cast<double>(kNnz)}) {
            INFO("headroom_iters " << headroom_iters);
            const double before = std::min(headroom_iters * static_cast<double>(kNnz), cap);
            CHECK(fpr_lp::dive_budget(headroom_iters, kNnz, kVanillaEffort, 1.0) ==
                  static_cast<size_t>(before));
        }
    }

    SECTION("the degenerate inputs yield no budget") {
        const double headroom_iters = cap / static_cast<double>(kNnz) / 4.0;
        CHECK(fpr_lp::dive_budget(headroom_iters, kNnz, kVanillaEffort, 0.0) == 0);
        CHECK(fpr_lp::dive_budget(0.0, kNnz, kVanillaEffort, 1.0) == 0);
        CHECK(fpr_lp::dive_budget(-1.0, kNnz, kVanillaEffort, 1.0) == 0);
        CHECK(fpr_lp::dive_budget(headroom_iters, 0, kVanillaEffort, 1.0) == 0);
    }

    SECTION("the product saturates instead of wrapping") {
        // Every factor is user-supplied — the share's ceiling is 1e6 and
        // `nnz` is whatever model was loaded — and `double -> size_t` is
        // undefined out of range.  A wrapped product would be a *small*
        // budget, which reads as "this parameter value is terrible" to
        // whatever is searching the space.
        constexpr size_t kHugeNnz = size_t{1} << 40;
        CHECK(fpr_lp::dive_budget(1e18, kHugeNnz, 1.0, 1e6) == SIZE_MAX);
    }
}

// The other half: the call site still reads the option.
//
// `dive_budget` above could be arithmetically perfect and unreferenced —
// which is the `select_ref` lesson from #128, where a correct table and an
// untested mapping onto it left the original bug in place.  So this asserts
// on a real solve, through the one observable that does not depend on how
// much work the dive then chooses to do: a share small enough that the
// derived budget falls under `run`'s own `nnz << 8` floor makes the
// dispatch not happen at all, while the shipped share dispatches on the
// same instance.  Under the deleted-`* share` mutation the small share
// yields the full budget and the dispatch happens, failing here.
TEST_CASE("fpr_lp: a small effort share suppresses the dispatch the default makes",
          "[fpr_lp][budget][regression]") {
    const auto dispatches_at = [](double share) {
        fpr_lp::reset_dispatch_counts();
        Highs h;
        h.setOptionValue("output_flag", false);
        set_suite(h, "fpr_lp");
        h.setOptionValue("mip_heuristic_fpr_lp_effort", 1.0);
        require_option(h, "mip_heuristic_fpr_lp_effort", share);
        REQUIRE(h.readModel(kInstancesDir + "/bell5.mps") == HighsStatus::kOk);
        REQUIRE(h.run() == HighsStatus::kOk);
        return fpr_lp::dispatch_counts().dispatches;
    };

    // Guard against a vacuous pass: the default share must dispatch, or
    // the zero below says nothing.
    REQUIRE(dispatches_at(1.0) >= 1);
    REQUIRE(dispatches_at(1e-4) == 0);
}
