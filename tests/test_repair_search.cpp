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

// ===================================================================
// A deadline expiry inside a fixpoint must not prune (issue #151).
//
// `sync_changes` returns a `bool`, and the only thing a `false` can mean
// to `repair_search` is "this node is infeasible, drop it".  So the new
// `PropResult::kDeadlineExpired` must not reach that `false` -- the same
// misreading #127 removed for the work budget.  Promptness at this site
// is the RepairSearch loop's own per-node poll, not this return value.
// ===================================================================

TEST_CASE("sync_changes: an expired deadline truncates propagation without pruning",
          "[repair-search][sync-changes][deadline]") {
    // Two continuous columns, two rows: x - 2y >= 0 and y - 2x >= 0.
    // Feasible at x = y = 0, but each row visit only halves the other
    // column's upper bound, so a fixpoint takes ~log2(ub / feastol) row
    // visits -- long enough for either stopping rule to fire.  (Same
    // construction as the halving model in tests/test_prop_engine.cpp.)
    const HighsInt ncol = 2;
    const HighsInt nrow = 2;
    std::vector<HighsInt> ar_start = {0, 2, 4};
    std::vector<HighsInt> ar_index = {0, 1, 1, 0};
    std::vector<double> ar_value = {1.0, -2.0, 1.0, -2.0};
    std::vector<double> col_lb = {0.0, 0.0};
    std::vector<double> col_ub = {1e300, 1e300};
    std::vector<double> row_lo = {0.0, 0.0};
    std::vector<double> row_hi = {kHighsInf, kHighsInf};
    std::vector<HighsVarType> integrality = {HighsVarType::kContinuous, HighsVarType::kContinuous};
    CscMatrix csc = build_csc(ncol, nrow, ar_start, ar_index, ar_value);
    const double feastol = 1e-6;

    // R restricts column 0, which is what gives `sync_changes` something
    // to transfer into E and therefore something to propagate.
    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine R(ncol, nrow, ar_start.data(), ar_index.data(), ar_value.data(), csc, col_lb.data(),
                 col_ub.data(), row_lo.data(), row_hi.data(), integrality.data(), feastol);
    REQUIRE(R.tighten_ub(0, 1e299));

    // Control: with no deadline the sync succeeds having spent the whole
    // per-call work budget (100 * nnz = 400 counted accesses).
    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine E_free(ncol, nrow, ar_start.data(), ar_index.data(), ar_value.data(), csc,
                      col_lb.data(), col_ub.data(), row_lo.data(), row_hi.data(),
                      integrality.data(), feastol);
    REQUIRE(sync_changes(E_free, R));
    REQUIRE(E_free.effort() > 400);

    // Armed with a limit already behind the timer's origin: no dependence
    // on real elapsed time, and expired on the fixpoint's first poll.
    HighsTimer timer;
    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine E(ncol, nrow, ar_start.data(), ar_index.data(), ar_value.data(), csc, col_lb.data(),
                 col_ub.data(), row_lo.data(), row_hi.data(), integrality.data(), feastol);
    E.set_deadline(make_deadline(timer, -1.0));

    // The sync still succeeds -- the node is not pruned -- and it stopped
    // on the clock, well short of the control's spend.
    REQUIRE(sync_changes(E, R));
    REQUIRE(E.effort() > 0);
    REQUIRE(E.effort() < E_free.effort());
    REQUIRE(E.effort() < 400);
    // Sound: the domain E was left with is a legitimate partial fixpoint.
    REQUIRE(E.var(0).lb <= E.var(0).ub + feastol);
    REQUIRE(E.var(1).lb <= E.var(1).ub + feastol);
}

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
        /*repair_iterations=*/50, /*progress_threshold=*/kRepairProgressThreshold,
        /*repair_noise=*/0.75,
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

// ===================================================================
// RepairSearch arms the engines it propagates on (#151)
//
// `repair_search` builds (or reuses, out of the scratch) the secondary
// engine R and propagates on both it and E, so both have to be handed
// this call's `Deadline` or #151's in-fixpoint poll is dead on the whole
// Phase 3 path.  A cold review found the two `set_deadline` calls that
// do it uncovered: removing them left the suite green.  Asserted through
// the engines' own `deadline()` rather than through a timing, which
// nothing here could make decidable.
TEST_CASE("RepairSearch: E and R are armed with the call's deadline", "[repair-search][deadline]") {
    // Reuses the shape of the search test above; the model does not
    // matter here, only that `repair_search` is entered.
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

    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine E(ncol, nrow, ar_start.data(), ar_index.data(), ar_value.data(), csc, col_lb.data(),
                 col_ub.data(), row_lo.data(), row_hi.data(), integrality.data(), feastol);
    REQUIRE(E.fix(0, 1.0));
    // Un-armed going in, which is the state a production E is in only if
    // `acquire_engine` failed to arm it -- here it makes the assertions
    // below about `repair_search` and nothing else.
    REQUIRE(E.deadline().timer == nullptr);

    std::vector<double> solution = {1.0, 0.0};
    std::vector<double> lhs_cache = {1.0, 0.0};

    FprScratch scratch;
    Rng rng(42);
    HighsTimer timer;
    // A live limit, so the node loop actually runs: an expired one would
    // fail its own `!deadline.expired()` condition immediately, which
    // would still prove the arming but would test less of the path.
    const Deadline deadline = make_deadline(timer, 3600.0);
    size_t effort_out = 0;

    static_cast<void>(repair_search(E, solution, lhs_cache, col_lb.data(), col_ub.data(),
                                    row_lo.data(), row_hi.data(), /*repair_iterations=*/50,
                                    /*progress_threshold=*/kRepairProgressThreshold,
                                    /*repair_noise=*/0.75, /*repair_track_best=*/true,
                                    /*max_effort=*/std::numeric_limits<size_t>::max(), rng,
                                    effort_out, scratch, deadline));

    CHECK(E.deadline().timer == &timer);
    CHECK(E.deadline().limit == 3600.0);
    // Reached through an explicitly guarded pointer: `REQUIRE` is a Catch2
    // macro, and clang-tidy's optional dataflow does not read it as the
    // engagement check that a bare `->` or `.value()` would then need.
    const PropEngine* r_engine =
        scratch.repair_prop_engine_r.has_value() ? &scratch.repair_prop_engine_r.value() : nullptr;
    REQUIRE(r_engine != nullptr);
    CHECK(r_engine->deadline().timer == &timer);
    CHECK(r_engine->deadline().limit == 3600.0);
}

// ===================================================================
// The stall gate is the node loop's only steering (issue #130).
//
// Fig. 5 lines 18-19 / Sect. 5.1: "if we detect that we are not making
// enough progress in the current subtree, we backtrack directly to the
// most promising open node".  Until #130 `repair_search` also called the
// same `BacktrackBestOpen` unconditionally at the foot of every
// iteration -- annotated "paper line 27", which is the *post-loop*
// backtrack and is implemented separately -- so the node popped next was
// the lowest-violation open node on every step.  The paper's
// DFS-with-occasional-jumps was a per-node best-first search, and the
// threshold had almost nothing left to decide.
//
// The model below is what makes that decidable.  It was found by a
// randomized sweep over sparse +/-1 binary models (15 columns, 7 rows,
// one-sided row bounds inside the activity range) looking for one where
// the *first* subtree a pure DFS enters is a dead end it cannot leave
// within `repair_iterations` nodes, while a feasible repair sits
// elsewhere in the tree.  On it:
//
//   threshold 1     -> jumps out of the dead subtree, reaches a feasible
//                      assignment after 17 of the 50 permitted nodes;
//   threshold 10^6  -> the gate can never fire, the search is the pure
//                      DFS Fig. 5 describes, and it spends all 50 nodes
//                      in the dead subtree without finding anything.
//
// Both halves are load-bearing, and each fails against a different
// mutation.  Restoring the ungated call makes *both* thresholds fail to
// find a solution (measured: threshold 1 then also returns infeasible
// after all 50 nodes), because the steering no longer depends on the
// counter.  Deleting the gated `backtrack_best_open` while keeping the
// counter, or reading a hardcoded 10 instead of the parameter, collapses
// threshold 1 onto the threshold-10^6 run for the same reason.  An
// assertion that the two runs merely *differ* would not catch the first
// of those: the pre-#130 code's two runs differ too, since the gate still
// permuted Q and reset its own counter even when it changed nothing about
// which node came next.
// ===================================================================

namespace {

// 15 binaries, 7 rows of +/-1 coefficients, each row bounded on one side.
struct StallModel {
    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr HighsInt ncol = 15;
    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr HighsInt nrow = 7;
    std::vector<HighsInt> ar_start = {0, 7, 16, 26, 33, 39, 47, 56};
    std::vector<HighsInt> ar_index = {4,  6,  7, 9,  11, 13, 14, 0, 1,  2,  3,  4,  8,  11,
                                      12, 13, 0, 1,  3,  4,  5,  9, 10, 11, 12, 13, 0,  1,
                                      2,  4,  5, 13, 14, 4,  6,  8, 10, 12, 13, 0,  1,  5,
                                      6,  7,  8, 10, 14, 0,  3,  5, 6,  7,  8,  10, 11, 12};
    std::vector<double> ar_value = {1,  1,  -1, -1, -1, -1, 1,  1,  1,  -1, -1, -1, 1,  -1,
                                    -1, -1, -1, 1,  -1, 1,  -1, 1,  -1, -1, -1, 1,  -1, -1,
                                    -1, 1,  -1, 1,  1,  -1, -1, 1,  1,  1,  1,  -1, -1, 1,
                                    1,  -1, 1,  -1, -1, -1, -1, -1, 1,  -1, -1, 1,  -1, 1};
    std::vector<double> col_lb = std::vector<double>(ncol, 0.0);
    std::vector<double> col_ub = std::vector<double>(ncol, 1.0);
    std::vector<double> row_lo = {1.0, 0.0, -kHighsInf, -kHighsInf, 2.0, -kHighsInf, 0.0};
    std::vector<double> row_hi = {kHighsInf, kHighsInf, -2.0, -2.0, kHighsInf, -3.0, kHighsInf};
    std::vector<HighsVarType> integrality = std::vector<HighsVarType>(ncol, HighsVarType::kInteger);
};

// One `repair_search` run on the model above, from the all-zero
// assignment, with a fresh engine and a fresh RNG so the two runs differ
// in the threshold and in nothing else.
struct StallRun {
    bool feasible = false;
    RepairSearchStats stats;
    std::vector<double> solution;
    std::vector<double> lhs_cache;
};

StallRun run_stall_model(const StallModel& m, const CscMatrix& csc, HighsInt progress_threshold) {
    const double feastol = 1e-6;
    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine E(StallModel::ncol, StallModel::nrow, m.ar_start.data(), m.ar_index.data(),
                 m.ar_value.data(), csc, m.col_lb.data(), m.col_ub.data(), m.row_lo.data(),
                 m.row_hi.data(), m.integrality.data(), feastol);
    StallRun out;
    out.solution.assign(StallModel::ncol, 0.0);
    out.lhs_cache.assign(StallModel::nrow, 0.0);
    FprScratch scratch;
    Rng rng(42);
    Deadline deadline;  // never expires
    size_t effort_out = 0;
    out.feasible = repair_search(E, out.solution, out.lhs_cache, m.col_lb.data(), m.col_ub.data(),
                                 m.row_lo.data(), m.row_hi.data(), /*repair_iterations=*/50,
                                 progress_threshold, /*repair_noise=*/0.75,
                                 /*repair_track_best=*/true,
                                 /*max_effort=*/std::numeric_limits<size_t>::max(), rng, effort_out,
                                 scratch, deadline, &out.stats);
    return out;
}

}  // namespace

TEST_CASE("RepairSearch: the progress threshold decides the search", "[repair-search][progress]") {
    const StallModel m;
    CscMatrix csc =
        build_csc(StallModel::ncol, StallModel::nrow, m.ar_start, m.ar_index, m.ar_value);
    const double feastol = 1e-6;

    // Out of reach: 10^6 exceeds `repair_iterations`, so the gate cannot
    // fire even once and the node loop is a pure DFS.
    const StallRun dfs = run_stall_model(m, csc, /*progress_threshold=*/1000000);
    CHECK(dfs.stats.best_open_jumps == 0);
    CHECK(dfs.stats.nodes_visited == 50);  // the whole node budget, in one dead subtree
    REQUIRE_FALSE(dfs.feasible);

    // Threshold 1: abandon the subtree at the first node that fails to
    // improve the best violation.
    const StallRun jumping = run_stall_model(m, csc, /*progress_threshold=*/1);
    CHECK(jumping.stats.best_open_jumps > 0);
    CHECK(jumping.stats.nodes_visited < dfs.stats.nodes_visited);
    REQUIRE(jumping.feasible);

    // ... and what it returns really is feasible, checked against the rows
    // rather than trusting the return value.
    for (HighsInt i = 0; i < StallModel::nrow; ++i) {
        double lhs = 0.0;
        for (HighsInt k = m.ar_start[i]; k < m.ar_start[i + 1]; ++k) {
            lhs += m.ar_value[k] * jumping.solution[m.ar_index[k]];
        }
        INFO("row " << i);
        CHECK(lhs >= m.row_lo[i] - feastol);
        CHECK(lhs <= m.row_hi[i] + feastol);
    }
}

// ===================================================================
// MoveToDisjunction uses the repair move (issue #131).
//
// Sect. 5.1 builds the non-binary disjunction from the *shifted*
// interval [a, b] -- D translated by the repair move's shift s.
// `move_to_disjunction` built it from D itself and never read the move
// outside the binary branch, so both children re-imposed a bound R
// already held, `tighten_lb`/`tighten_ub` took their no-tightening early
// return, `sync_changes` found nothing to transfer, and the incumbent
// point never moved: a RepairSearch node on a general-integer variable
// cost two propagation fixpoints and moved nothing.
//
// The three models below are one general integer each with a single row,
// so `walksat_select_move` has exactly one violated row and exactly one
// candidate -- every assertion is on the disjunction, with no dependence
// on the RNG stream.  All three return `bound_branch_moves == 0` and no
// solution against the pre-#131 code.
// ===================================================================

namespace {

// x0 integer in [0, 10]; one row `x0 >= row_lo0`.  The caller sets up E
// and the incumbent point, which is what distinguishes the three cases.
struct OneIntModel {
    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr HighsInt ncol = 1;
    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr HighsInt nrow = 1;
    std::vector<HighsInt> ar_start = {0, 1};
    std::vector<HighsInt> ar_index = {0};
    std::vector<double> ar_value = {1.0};
    std::vector<double> col_lb = {0.0};
    std::vector<double> col_ub = {10.0};
    std::vector<double> row_lo;
    std::vector<double> row_hi = {kHighsInf};
    std::vector<HighsVarType> integrality = {HighsVarType::kInteger};

    explicit OneIntModel(double lo) : row_lo({lo}) {}
};

}  // namespace

TEST_CASE("MoveToDisjunction: a fixed general integer is moved by a bound branch",
          "[repair-search][disjunction]") {
    // E has x0 decision-fixed to 2 (so `[lb, ub]` stays the wide [0, 10],
    // as `fix()` leaves them) and the row wants x0 >= 7.  The single
    // repair move is the shift s = +5, so the shifted interval is the
    // singleton [7, 7]; against R's root domain [0, 10] that is
    // l = 7 > r = 3, so the preferred branch is `x0 <= 7`, R propagates
    // the row back to [7, 7], and `sync_changes` re-fixes E to 7.
    const OneIntModel m(7.0);
    CscMatrix csc =
        build_csc(OneIntModel::ncol, OneIntModel::nrow, m.ar_start, m.ar_index, m.ar_value);
    const double feastol = 1e-6;

    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine E(OneIntModel::ncol, OneIntModel::nrow, m.ar_start.data(), m.ar_index.data(),
                 m.ar_value.data(), csc, m.col_lb.data(), m.col_ub.data(), m.row_lo.data(),
                 m.row_hi.data(), m.integrality.data(), feastol);
    REQUIRE(E.fix(0, 2.0));
    REQUIRE(E.var(0).lb == Catch::Approx(0.0));   // wide, as `fix()` leaves them
    REQUIRE(E.var(0).ub == Catch::Approx(10.0));  // -- which is why D is the value

    std::vector<double> solution = {2.0};
    std::vector<double> lhs_cache = {2.0};
    FprScratch scratch;
    Rng rng(42);
    Deadline deadline;
    size_t effort_out = 0;
    RepairSearchStats stats;

    const bool feasible = repair_search(
        E, solution, lhs_cache, m.col_lb.data(), m.col_ub.data(), m.row_lo.data(), m.row_hi.data(),
        /*repair_iterations=*/50, /*progress_threshold=*/kRepairProgressThreshold,
        /*repair_noise=*/0.75, /*repair_track_best=*/true,
        /*max_effort=*/std::numeric_limits<size_t>::max(), rng, effort_out, scratch, deadline,
        &stats);

    REQUIRE(feasible);
    CHECK(stats.bound_branch_moves == 1);
    CHECK(solution[0] == Catch::Approx(7.0));
    CHECK(lhs_cache[0] == Catch::Approx(7.0));
    // E agrees: the sync is what moved the point, so it must have moved
    // E's own domain first.
    CHECK(E.var(0).fixed);
    CHECK(E.var(0).val == Catch::Approx(7.0));
}

TEST_CASE("MoveToDisjunction: the interval shifts, it does not collapse to the move value",
          "[repair-search][disjunction]") {
    // E has x0 unfixed on D = [2, 4] and the point at 3; the row wants
    // x0 >= 7, so the single move is s = +4 and the shifted interval is
    // [6, 8] -- width 2, not the singleton [7, 7].  That distinction is
    // the whole assertion: against R's [0, 10],
    //
    //   [6, 8] -> l = 6 > r = 2 -> `x0 <= 6` first (R propagates the row
    //             to lb 7 against ub 6 and the child is refuted), then
    //             `x0 >= 8` -> R = [8, 10], disjoint from D, so
    //             `sync_changes` fixes x0 to 8;
    //   [7, 7] -> l = 7 > r = 3 -> `x0 <= 7` -> R = [7, 7] -> x0 fixed
    //             to 7 on the *first* child.
    //
    // So an implementation that used the move value in place of the
    // shifted interval returns 7 here, and this test reads 8.
    const OneIntModel m(7.0);
    CscMatrix csc =
        build_csc(OneIntModel::ncol, OneIntModel::nrow, m.ar_start, m.ar_index, m.ar_value);
    const double feastol = 1e-6;

    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine E(OneIntModel::ncol, OneIntModel::nrow, m.ar_start.data(), m.ar_index.data(),
                 m.ar_value.data(), csc, m.col_lb.data(), m.col_ub.data(), m.row_lo.data(),
                 m.row_hi.data(), m.integrality.data(), feastol);
    REQUIRE(E.tighten_lb(0, 2.0));
    REQUIRE(E.tighten_ub(0, 4.0));
    REQUIRE_FALSE(E.var(0).fixed);

    std::vector<double> solution = {3.0};
    std::vector<double> lhs_cache = {3.0};
    FprScratch scratch;
    Rng rng(42);
    Deadline deadline;
    size_t effort_out = 0;
    RepairSearchStats stats;

    const bool feasible = repair_search(
        E, solution, lhs_cache, m.col_lb.data(), m.col_ub.data(), m.row_lo.data(), m.row_hi.data(),
        /*repair_iterations=*/50, /*progress_threshold=*/kRepairProgressThreshold,
        /*repair_noise=*/0.75, /*repair_track_best=*/true,
        /*max_effort=*/std::numeric_limits<size_t>::max(), rng, effort_out, scratch, deadline,
        &stats);

    REQUIRE(feasible);
    CHECK(stats.bound_branch_moves == 1);
    CHECK(solution[0] == Catch::Approx(8.0));
    CHECK(E.var(0).fixed);
    CHECK(E.var(0).val == Catch::Approx(8.0));
}

TEST_CASE("MoveToDisjunction: a sync that fixes inside E's own bounds still moves the point",
          "[repair-search][disjunction]") {
    // E leaves x0 unfixed on the full [0, 10] with the point at 0, and
    // the row wants x0 >= 8.  The move is s = +8, so the shifted interval
    // is [8, 18], clipped into R's [0, 10] as [8, 10]: l = 8 > r = 0, the
    // preferred branch is `x0 <= 8`, R propagates the row to [8, 8], and
    // `sync_changes` takes its *intersection* case -- D and Dr overlap in
    // the single point 8 -- which commits through `PropEngine::fix`.
    //
    // `fix()` leaves `[lb, ub]` at [0, 10], so a clamp reading the raw
    // bounds finds 0 already inside them and drops the move even though E
    // is fixed at 8.  The clamp reads the interval E holds instead, which
    // is what this test pins; the other two cases reach E through
    // `refix`, which narrows, and cannot see the difference.
    const OneIntModel m(8.0);
    CscMatrix csc =
        build_csc(OneIntModel::ncol, OneIntModel::nrow, m.ar_start, m.ar_index, m.ar_value);
    const double feastol = 1e-6;

    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine E(OneIntModel::ncol, OneIntModel::nrow, m.ar_start.data(), m.ar_index.data(),
                 m.ar_value.data(), csc, m.col_lb.data(), m.col_ub.data(), m.row_lo.data(),
                 m.row_hi.data(), m.integrality.data(), feastol);
    REQUIRE_FALSE(E.var(0).fixed);

    std::vector<double> solution = {0.0};
    std::vector<double> lhs_cache = {0.0};
    FprScratch scratch;
    Rng rng(42);
    Deadline deadline;
    size_t effort_out = 0;
    RepairSearchStats stats;

    const bool feasible = repair_search(
        E, solution, lhs_cache, m.col_lb.data(), m.col_ub.data(), m.row_lo.data(), m.row_hi.data(),
        /*repair_iterations=*/50, /*progress_threshold=*/kRepairProgressThreshold,
        /*repair_noise=*/0.75, /*repair_track_best=*/true,
        /*max_effort=*/std::numeric_limits<size_t>::max(), rng, effort_out, scratch, deadline,
        &stats);

    REQUIRE(feasible);
    CHECK(stats.bound_branch_moves == 1);
    CHECK(solution[0] == Catch::Approx(8.0));
    CHECK(E.var(0).fixed);
    CHECK(E.var(0).val == Catch::Approx(8.0));
    CHECK(E.var(0).lb == Catch::Approx(0.0));   // `fix()` left them wide ...
    CHECK(E.var(0).ub == Catch::Approx(10.0));  // ... which is the point of this case
}

// ===================================================================
// The shifted interval is clipped back into R's domain (issue #131).
//
// Sect. 5.1 states the containment "the shifted interval [a, b] in D is
// always contained in the interval [c, d] in Dr, by construction" as a
// precondition.  It does not hold here as written: `walksat_select_move`
// clips a shift to the column's *structural* bounds, not to R's current
// domain, so the translate can leave [c, d] -- and then one branch is a
// bound R already satisfies (a no-op) while the other is refuted
// outright, so the node makes no progress at all.  Clipping [a, b] back
// into [c, d] restores the precondition and turns both branches back
// into real restrictions.
//
// This only bites below the root, since R starts at the structural
// bounds and a shift cannot leave *those*: it needs an ancestor branch
// to have narrowed R first.  That makes it out of reach of the
// single-row, single-candidate models above -- reaching depth two needs
// enough rows for `walksat_select_move` to have a choice of violated
// row, and a choice is drawn from `rng`.  The model below is therefore a
// characterization test, found by sweeping random small integer models
// for one where the clip changes the outcome; with it, x0 is moved and
// every row ends satisfied, and without it no bound branch moves
// anything and the search returns nothing.
// ===================================================================

TEST_CASE("MoveToDisjunction: the shifted interval is clipped into R's domain",
          "[repair-search][disjunction]") {
    // 7 general integers on [0, 6], two >= rows; x6 is in no row.
    //   r0: 2*x0 + 2*x1 + 2*x2 - 3*x3 >= 11
    //   r1: 3*x0 + 2*x3 -   x4 + 3*x5 >= 23
    const HighsInt ncol = 7;
    const HighsInt nrow = 2;
    std::vector<HighsInt> ar_start = {0, 4, 8};
    std::vector<HighsInt> ar_index = {0, 1, 2, 3, 0, 3, 4, 5};
    std::vector<double> ar_value = {2.0, 2.0, 2.0, -3.0, 3.0, 2.0, -1.0, 3.0};
    std::vector<double> col_lb(ncol, 0.0);
    std::vector<double> col_ub(ncol, 6.0);
    std::vector<double> row_lo = {11.0, 23.0};
    std::vector<double> row_hi = {kHighsInf, kHighsInf};
    std::vector<HighsVarType> integrality(ncol, HighsVarType::kInteger);
    CscMatrix csc = build_csc(ncol, nrow, ar_start, ar_index, ar_value);
    const double feastol = 1e-6;

    // E as a dive would leave it: a mix of one-sided tightenings and two
    // decision fixes, so D is a strict subinterval on most columns.
    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine E(ncol, nrow, ar_start.data(), ar_index.data(), ar_value.data(), csc, col_lb.data(),
                 col_ub.data(), row_lo.data(), row_hi.data(), integrality.data(), feastol);
    REQUIRE(E.tighten_lb(0, 2.0));
    REQUIRE(E.fix(1, 2.0));
    REQUIRE(E.tighten_ub(2, 4.0));
    REQUIRE(E.tighten_ub(3, 4.0));
    REQUIRE(E.tighten_lb(4, 2.0));
    REQUIRE(E.fix(5, 2.0));
    REQUIRE(E.tighten_ub(6, 4.0));

    // The incumbent point, and its row activities: r0 = 7 (needs 11),
    // r1 = 18 (needs 23), so both rows are violated going in.
    std::vector<double> solution = {3.0, 2.0, 3.0, 3.0, 3.0, 2.0, 3.0};
    std::vector<double> lhs_cache = {7.0, 18.0};

    FprScratch scratch;
    Rng rng(42);
    Deadline deadline;
    size_t effort_out = 0;
    RepairSearchStats stats;

    const bool feasible = repair_search(
        E, solution, lhs_cache, col_lb.data(), col_ub.data(), row_lo.data(), row_hi.data(),
        /*repair_iterations=*/50, /*progress_threshold=*/kRepairProgressThreshold,
        /*repair_noise=*/0.75, /*repair_track_best=*/true,
        /*max_effort=*/std::numeric_limits<size_t>::max(), rng, effort_out, scratch, deadline,
        &stats);

    REQUIRE(feasible);
    CHECK(stats.bound_branch_moves == 1);
    CHECK(solution[0] == Catch::Approx(6.0));
    // Feasible against the rows themselves, not just per the return value.
    for (HighsInt i = 0; i < nrow; ++i) {
        double lhs = 0.0;
        for (HighsInt k = ar_start[i]; k < ar_start[i + 1]; ++k) {
            lhs += ar_value[k] * solution[ar_index[k]];
        }
        INFO("row " << i);
        CHECK(lhs >= row_lo[i] - feastol);
        CHECK(lhs <= row_hi[i] + feastol);
    }
}
