#include "fpr_core.h"
#include "heuristic_common.h"
#include "prop_engine.h"
#include "repair_search.h"
#include "rng.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <limits>
#include <vector>

// ===================================================================
// sync_changes / RepairSearch tests (issue #125).
//
// Salvagnin, Roberti, Fischetti, MPC 17:111-139, 2025, Sect. 5.1: two
// domain-synchronization cases (`SyncChanges`, Fig. 5 line 13) that the
// pre-#125 code got wrong.
//
//   1. A column already fixed in the primary engine E, but whose value
//      the secondary engine R has since ruled out, must be re-fixed
//      ("flipped", for a binary) to R's domain -- this is what makes a
//      clique-constrained binary swap possible.  The old code skipped
//      every already-fixed column outright.
//   2. A column still unfixed in E whose domain is disjoint from R's
//      must be fixed to the endpoint of R's domain closer to E's (paper
//      worked example: D=[1,3], Dr=[4,5] -> 4). The old code instead
//      called `tighten_lb`/`tighten_ub`, which validates against E's
//      *current* domain and therefore rejects any value outside it --
//      i.e. always, for a genuinely disjoint pair -- so the node was
//      dropped instead of repaired.
//
// `E`/`R` throughout this file are the paper's own symbols for the
// primary/secondary propagation engines (Fig. 5), matching the existing
// NOLINT precedent in repair_search.cpp -- each local declaration below
// carries its own NOLINTNEXTLINE for the same reason.
// ===================================================================

TEST_CASE("sync_changes: flips an already-fixed binary via clique propagation",
          "[repair-search][sync-changes]") {
    // 2 binary variables, 1 clique constraint: x0 + x1 <= 1.
    const HighsInt ncol = 2;
    const HighsInt nrow = 1;
    std::vector<HighsInt> ar_start = {0, 2};
    std::vector<HighsInt> ar_index = {0, 1};
    std::vector<double> ar_value = {1.0, 1.0};
    std::vector<double> col_lb = {0.0, 0.0};
    std::vector<double> col_ub = {1.0, 1.0};
    std::vector<double> row_lo = {-kHighsInf};
    std::vector<double> row_hi = {1.0};
    std::vector<HighsVarType> integrality = {HighsVarType::kInteger, HighsVarType::kInteger};
    CscMatrix csc = build_csc(ncol, nrow, ar_start, ar_index, ar_value);
    const double feastol = 1e-6;

    // E: the primary engine, as a real DFS would leave it after fixing
    // x0=1 and propagating.  Clique propagation auto-fixes x1 to 0 with
    // its bounds genuinely narrowed to [0,0] -- this is the realistic
    // state (not a decision-fix, whose bounds stay wide), and the case
    // that defeats a naive `fix()`-based re-sync: [0,0] cannot hold 1.
    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine E(ncol, nrow, ar_start.data(), ar_index.data(), ar_value.data(), csc, col_lb.data(),
                 col_ub.data(), row_lo.data(), row_hi.data(), integrality.data(), feastol);
    REQUIRE(E.fix(0, 1.0));
    REQUIRE(E.propagate(0) == PropResult::kFixpoint);
    REQUIRE(E.var(0).fixed);
    REQUIRE(E.var(0).val == Catch::Approx(1.0));
    REQUIRE(E.var(1).fixed);
    REQUIRE(E.var(1).val == Catch::Approx(0.0));
    REQUIRE(E.var(1).lb == Catch::Approx(0.0));
    REQUIRE(E.var(1).ub == Catch::Approx(0.0));

    // R: the secondary engine, after RepairSearch applies the opposite
    // branch (fix x1=1) and propagates it.  Clique propagation forces
    // x0 down to 0 in R -- the swap RepairSearch is supposed to detect.
    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine R(ncol, nrow, ar_start.data(), ar_index.data(), ar_value.data(), csc, col_lb.data(),
                 col_ub.data(), row_lo.data(), row_hi.data(), integrality.data(), feastol);
    REQUIRE(R.fix(1, 1.0));
    REQUIRE(R.propagate(1) == PropResult::kFixpoint);
    REQUIRE(R.var(0).fixed);
    REQUIRE(R.var(0).val == Catch::Approx(0.0));
    REQUIRE(R.var(1).fixed);
    REQUIRE(R.var(1).val == Catch::Approx(1.0));

    REQUIRE(sync_changes(E, R));

    // The swap: x0 flips 1 -> 0, x1 flips 0 -> 1, both while already
    // fixed in E -- exactly the mechanism the paper gives as
    // RepairSearch's reason to exist over plain WalkSAT.
    REQUIRE(E.var(0).fixed);
    REQUIRE(E.var(0).val == Catch::Approx(0.0));
    REQUIRE(E.var(1).fixed);
    REQUIRE(E.var(1).val == Catch::Approx(1.0));
}

TEST_CASE("sync_changes: agreeing domains are left alone", "[repair-search][sync-changes]") {
    // Same clique model, but R agrees with E (x0=1, x1=0 in both) --
    // sync_changes must not touch anything.
    const HighsInt ncol = 2;
    const HighsInt nrow = 1;
    std::vector<HighsInt> ar_start = {0, 2};
    std::vector<HighsInt> ar_index = {0, 1};
    std::vector<double> ar_value = {1.0, 1.0};
    std::vector<double> col_lb = {0.0, 0.0};
    std::vector<double> col_ub = {1.0, 1.0};
    std::vector<double> row_lo = {-kHighsInf};
    std::vector<double> row_hi = {1.0};
    std::vector<HighsVarType> integrality = {HighsVarType::kInteger, HighsVarType::kInteger};
    CscMatrix csc = build_csc(ncol, nrow, ar_start, ar_index, ar_value);
    const double feastol = 1e-6;

    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine E(ncol, nrow, ar_start.data(), ar_index.data(), ar_value.data(), csc, col_lb.data(),
                 col_ub.data(), row_lo.data(), row_hi.data(), integrality.data(), feastol);
    REQUIRE(E.fix(0, 1.0));
    REQUIRE(E.propagate(0) == PropResult::kFixpoint);

    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine R(ncol, nrow, ar_start.data(), ar_index.data(), ar_value.data(), csc, col_lb.data(),
                 col_ub.data(), row_lo.data(), row_hi.data(), integrality.data(), feastol);
    REQUIRE(R.fix(0, 1.0));
    REQUIRE(R.propagate(0) == PropResult::kFixpoint);

    REQUIRE(sync_changes(E, R));
    REQUIRE(E.var(0).val == Catch::Approx(1.0));
    REQUIRE(E.var(1).val == Catch::Approx(0.0));
}

TEST_CASE("sync_changes: disjoint domains fix to the nearest endpoint of R's domain",
          "[repair-search][sync-changes]") {
    // 1 general-integer variable; the row is a slack constraint that
    // never binds -- only PropEngine::tighten_lb/tighten_ub, called
    // directly, drive the two domains here.
    const HighsInt ncol = 1;
    const HighsInt nrow = 1;
    std::vector<HighsInt> ar_start = {0, 1};
    std::vector<HighsInt> ar_index = {0};
    std::vector<double> ar_value = {1.0};
    std::vector<double> col_lb = {0.0};
    std::vector<double> col_ub = {10.0};
    std::vector<double> row_lo = {-kHighsInf};
    std::vector<double> row_hi = {100.0};
    std::vector<HighsVarType> integrality = {HighsVarType::kInteger};
    CscMatrix csc = build_csc(ncol, nrow, ar_start, ar_index, ar_value);
    const double feastol = 1e-6;

    // E: domain D = [1, 3], still unfixed.
    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine E(ncol, nrow, ar_start.data(), ar_index.data(), ar_value.data(), csc, col_lb.data(),
                 col_ub.data(), row_lo.data(), row_hi.data(), integrality.data(), feastol);
    REQUIRE(E.tighten_lb(0, 1.0));
    REQUIRE(E.tighten_ub(0, 3.0));
    REQUIRE_FALSE(E.var(0).fixed);
    REQUIRE(E.var(0).lb == Catch::Approx(1.0));
    REQUIRE(E.var(0).ub == Catch::Approx(3.0));

    // R: domain Dr = [4, 5] -- the paper's own worked example.
    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine R(ncol, nrow, ar_start.data(), ar_index.data(), ar_value.data(), csc, col_lb.data(),
                 col_ub.data(), row_lo.data(), row_hi.data(), integrality.data(), feastol);
    REQUIRE(R.tighten_lb(0, 4.0));
    REQUIRE(R.tighten_ub(0, 5.0));
    REQUIRE_FALSE(R.var(0).fixed);
    REQUIRE(R.var(0).lb == Catch::Approx(4.0));
    REQUIRE(R.var(0).ub == Catch::Approx(5.0));

    // Old code called E.tighten_lb(0, 4.0) here, which rejects (4 is
    // outside E's current [1, 3]) and the whole function returned
    // false -- the node the paper says should be *repaired* was instead
    // dropped.  The fix: sync_changes must succeed and fix to 4, the
    // endpoint of Dr closer to D.
    REQUIRE(sync_changes(E, R));
    REQUIRE(E.var(0).fixed);
    REQUIRE(E.var(0).val == Catch::Approx(4.0));
}

// ===================================================================
// Full repair_search() integration test (acceptance criterion): the
// SyncChanges flip fires with the whole DFS/backtrack machinery around
// it, not just in the isolated sync_changes call above.
//
// Scope, precisely: this model's *returned solution* does not by itself
// discriminate the fix -- reverting sync_changes to its pre-#125 skip
// still leaves `feasible` true and both row assertions passing, because
// WalkSAT's own move selection (driven by `lhs_cache`/`solution` against
// the columns' *global* bounds, independent of E) can rediscover "shift
// x0 back to 0" on its own once x1's flip makes row0 violated, in this
// small a model. What *does* discriminate -- and is what this test
// pins -- is `E.var(0).val`: only the #125 fix makes the primary engine's
// own bookkeeping reflect the swap, which is the fact `sync_changes` is
// specifically responsible for and the two isolated tests above already
// prove in the general case. The two direct sync_changes tests above are
// the ones that discriminate unconditionally.
// ===================================================================

TEST_CASE("RepairSearch: SyncChanges' flip is visible in E after a full search",
          "[repair-search]") {
    // Same clique model as the sync_changes swap test, plus a second row
    // that only x1 can satisfy (x1 >= 1), so the starting assignment
    // (x0=1, x1=0, matching a DFS that fixed x0=1 first) violates
    // exactly one row and the only useful repair move is to raise x1.
    const HighsInt ncol = 2;
    const HighsInt nrow = 2;
    std::vector<HighsInt> ar_start = {0, 2, 3};
    std::vector<HighsInt> ar_index = {0, 1, 1};
    std::vector<double> ar_value = {1.0, 1.0, 1.0};
    std::vector<double> col_lb = {0.0, 0.0};
    std::vector<double> col_ub = {1.0, 1.0};
    std::vector<double> row_lo = {-kHighsInf, 1.0};
    std::vector<double> row_hi = {1.0, kHighsInf};
    std::vector<HighsVarType> integrality = {HighsVarType::kInteger, HighsVarType::kInteger};
    CscMatrix csc = build_csc(ncol, nrow, ar_start, ar_index, ar_value);
    const double feastol = 1e-6;

    // E: x0 decision-fixed to 1 (wide bounds, as a DFS `fix()` call
    // leaves them); x1 left unfixed, as the paper allows ("if x_j is
    // still unfixed in the current node it is fixed to the same
    // value").  Deliberately *not* propagated here, so x1 stays open
    // for RepairSearch's own move selection to pick up.
    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine E(ncol, nrow, ar_start.data(), ar_index.data(), ar_value.data(), csc, col_lb.data(),
                 col_ub.data(), row_lo.data(), row_hi.data(), integrality.data(), feastol);
    REQUIRE(E.fix(0, 1.0));

    std::vector<double> solution = {1.0, 0.0};
    std::vector<double> lhs_cache = {1.0, 0.0};  // row0: 1*1+1*0=1; row1: 1*0=0

    FprScratch scratch;
    Rng rng(42);
    Deadline deadline;  // never expires
    size_t effort_out = 0;

    bool feasible = repair_search(
        E, solution, lhs_cache, col_lb.data(), col_ub.data(), row_lo.data(), row_hi.data(),
        /*repair_iterations=*/50, /*repair_noise=*/0.75,
        /*repair_track_best=*/true,
        /*max_effort=*/std::numeric_limits<size_t>::max(), rng, effort_out, scratch, deadline);

    REQUIRE(feasible);
    // Both rows genuinely satisfied.
    REQUIRE(solution[0] + solution[1] <= 1.0 + feastol);
    REQUIRE(solution[1] >= 1.0 - feastol);
    // The swap happened inside E: x0, decided (and left decided by
    // Phase 2) at 1, was flipped to 0 by sync_changes once the clique
    // forced it -- this is the assertion that discriminates the fixed
    // (#125) code from the pre-#125 code, which left E's x0 at 1
    // forever (its sync_changes skipped every already-fixed column).
    REQUIRE(E.var(0).fixed);
    REQUIRE(E.var(0).val == Catch::Approx(0.0));
}
