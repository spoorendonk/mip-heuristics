#include "fpr_core.h"
#include "heuristic_common.h"
#include "prop_engine.h"
#include "repair_search.h"
#include "rng.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <limits>
#include <optional>
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

    bool feasible = repair_search(E, solution, lhs_cache, col_lb.data(), col_ub.data(),
                                  row_lo.data(), row_hi.data(),
                                  /*repair_iterations=*/50, /*repair_noise=*/0.75,
                                  /*repair_track_best=*/true, rng, effort_out, scratch, deadline,
                                  /*stats=*/nullptr);

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
                                    /*repair_noise=*/0.75, /*repair_track_best=*/true, rng,
                                    effort_out, scratch, deadline, /*stats=*/nullptr));

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
// elsewhere in the tree.  At RNG seed 52 on it:
//
//   threshold 1     -> jumps out of the dead subtree, reaches a feasible
//                      assignment after 25 of the 50 permitted nodes;
//   threshold 10^6  -> the gate can never fire, the search is the pure
//                      DFS Fig. 5 describes, and it spends all 50 nodes
//                      in the dead subtree without finding anything.
//
// The shipped default of 10 sits on the escaping side too (feasible at
// node 25), which is what the third case below pins to the constant.
//
// The seed moved from 42 to 52 with issue #158, which made a jump
// *discard* the open nodes beneath it rather than permute them to the
// back: at seed 42 the threshold-1 arm now gives the dropped subtree up
// and returns infeasible after 11 nodes, which is the paper's own "cost
// of giving up on completeness" and is pinned by its own test below
// rather than hidden.  52 is the smallest seed that still separates the
// three thresholds, and the production arm is unmoved by #158 either way
// (threshold 10 at seed 42: 29 nodes, feasible, before and after).
//
// Both halves are load-bearing, and each fails against a different
// mutation -- re-measured at seed 52 on the post-#158 code.  Restoring
// the ungated call makes the steering independent of the counter, so
// every threshold from 9 upwards runs the same search and the 10^6 arm
// finds a solution in 19 nodes: `REQUIRE_FALSE(dfs.feasible)` fails.
// Deleting the gated `backtrack_best_open` while keeping the counter
// collapses threshold 1 onto the threshold-10^6 run (both 50 nodes, both
// infeasible), failing `REQUIRE(jumping.feasible)`.  Reading a hardcoded
// 10 instead of the parameter collapses *all* the thresholds onto the
// threshold-10 search (25 nodes, feasible), which again fails
// `REQUIRE_FALSE(dfs.feasible)`.  An assertion that the two runs merely
// *differ* would not catch the first of those: under the ungated call
// the two runs still differ (13 nodes against 19), so it is the
// feasibility verdicts and not the node counts that do the work.
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

// `use_default` calls `repair_search` without a `progress_threshold`
// argument, exactly as `fpr_core.cpp` does, so the third case below reads
// the production value rather than a copy of it.
StallRun run_stall_model(const StallModel& m, const CscMatrix& csc, HighsInt progress_threshold,
                         bool use_default = false, unsigned seed = 42) {
    const double feastol = 1e-6;
    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine E(StallModel::ncol, StallModel::nrow, m.ar_start.data(), m.ar_index.data(),
                 m.ar_value.data(), csc, m.col_lb.data(), m.col_ub.data(), m.row_lo.data(),
                 m.row_hi.data(), m.integrality.data(), feastol);
    StallRun out;
    out.solution.assign(StallModel::ncol, 0.0);
    out.lhs_cache.assign(StallModel::nrow, 0.0);
    FprScratch scratch;
    Rng rng(seed);
    Deadline deadline;  // never expires
    size_t effort_out = 0;
    if (use_default) {
        out.feasible = repair_search(
            E, out.solution, out.lhs_cache, m.col_lb.data(), m.col_ub.data(), m.row_lo.data(),
            m.row_hi.data(), /*repair_iterations=*/50, /*repair_noise=*/0.75,
            /*repair_track_best=*/true, rng, effort_out, scratch, deadline, &out.stats);
    } else {
        out.feasible = repair_search(E, out.solution, out.lhs_cache, m.col_lb.data(),
                                     m.col_ub.data(), m.row_lo.data(), m.row_hi.data(),
                                     /*repair_iterations=*/50, /*repair_noise=*/0.75,
                                     /*repair_track_best=*/true, rng, effort_out, scratch, deadline,
                                     &out.stats, progress_threshold);
    }
    return out;
}

}  // namespace

TEST_CASE("RepairSearch: the progress threshold decides the search", "[repair-search][progress]") {
    constexpr unsigned kSeed = 52;
    const StallModel m;
    CscMatrix csc =
        build_csc(StallModel::ncol, StallModel::nrow, m.ar_start, m.ar_index, m.ar_value);
    const double feastol = 1e-6;

    // Out of reach: 10^6 exceeds `repair_iterations`, so the gate cannot
    // fire even once and the node loop is a pure DFS.  `kSeed` is 52
    // rather than the fixture default of 42 -- see the section comment
    // above for why #158 moved it, and the give-up test below for what
    // 42 pins now.
    const StallRun dfs =
        run_stall_model(m, csc, /*progress_threshold=*/1000000, /*use_default=*/false, kSeed);
    CHECK(dfs.stats.best_open_jumps == 0);
    CHECK(dfs.stats.nodes_visited == 50);  // the whole node budget, in one dead subtree
    REQUIRE_FALSE(dfs.feasible);

    // Threshold 1: abandon the subtree at the first node that fails to
    // improve the best violation.
    const StallRun jumping =
        run_stall_model(m, csc, /*progress_threshold=*/1, /*use_default=*/false, kSeed);
    CHECK(jumping.stats.best_open_jumps > 0);
    CHECK(jumping.stats.nodes_visited < dfs.stats.nodes_visited);
    REQUIRE(jumping.feasible);

    // The value itself, which `docs/PARAMETERS.md` documents and nothing
    // else checks: the two runs below agree by construction whatever it
    // is, so without this a retune would move production's search
    // silently.  Same reason `tests/test_smoke.cpp` pins the four effort
    // defaults.
    STATIC_REQUIRE(kRepairProgressThreshold == 10);

    // The production call site names no threshold -- it takes the
    // parameter's default -- so this is what `fpr_core.cpp` actually runs,
    // and it must be `kRepairProgressThreshold`'s search and no other.
    // Neighbouring values are all distinguishable here (9 -> 24 nodes,
    // 10 -> 25, 11 -> 27, 3 -> 18), so moving the constant or the default
    // moves this.
    const StallRun production = run_stall_model(m, /*csc=*/csc, /*progress_threshold=*/0,
                                                /*use_default=*/true, kSeed);
    const StallRun named =
        run_stall_model(m, csc, kRepairProgressThreshold, /*use_default=*/false, kSeed);
    CHECK(production.feasible == named.feasible);
    CHECK(production.stats.nodes_visited == named.stats.nodes_visited);
    CHECK(production.stats.best_open_jumps == named.stats.best_open_jumps);
    CHECK(production.solution == named.solution);
    // Between the two ends, and strictly inside the node budget, so the
    // pin above is on a search that actually escapes.
    CHECK(named.stats.best_open_jumps > 0);
    CHECK(named.stats.nodes_visited < 50);
    CHECK(named.feasible);

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
// BacktrackBestOpen gives its subtree up (issue #158).
//
// Every `RepairSearchNode` restores its parent state by replaying an
// undo trail down to a mark, which is sound only while the marks along
// `Q` are non-decreasing front-to-back so that a pop always unwinds
// *downward*.  The pre-#158 jump was a plain `std::iter_swap` of the
// best node to the back, which can seat a deep node at an interior
// position; the search then unwinds beneath that node's mark, and
// popping it later hands `PropEngine::backtrack_to` /
// `backtrack_sol_lhs` a target above the live stack size, whose
// `resize` *grows* the stack into value-initialized entries -- a later
// backtrack replays them as `vs_[0] = VarState{}` and
// `solution_[0] = 0.0`.
//
// Sect. 5.1 prices the jump as "we backtrack directly to the most
// promising open node, at the cost of giving up on completeness", and a
// permutation costs no completeness at all: the fix is to discard the
// strictly-deeper open nodes, which is what these two cases pin
// directly.  They are unit tests on `backtrack_best_open` rather than
// assertions on a whole search because a search cannot separate
// "dropped the subtree" from "permuted it to the back" -- both leave the
// best node popped next, and the difference only surfaces nodes later.
// ===================================================================

namespace {

// A node carrying the same value in all seven undo marks, which is the
// shape the invariant is about; `var` is the label the cases identify it
// by and plays no part in the drop rule.
RepairSearchNode marked_node(HighsInt label, HighsInt mark, double violation) {
    return RepairSearchNode{/*var=*/label,
                            /*val=*/0.0,
                            /*is_fix=*/true,
                            /*is_lb=*/false,
                            /*e_vs_mark=*/mark,
                            /*e_sol_mark=*/mark,
                            /*e_pq_mark=*/mark,
                            /*r_vs_mark=*/mark,
                            /*r_sol_mark=*/mark,
                            /*sol_undo_mark=*/mark,
                            /*lhs_undo_mark=*/mark,
                            /*violation=*/violation};
}

// The seven marks, as assignable references, in the order
// `RepairSearchNode` declares them.  Used to raise exactly one of them.
std::vector<HighsInt*> marks_of(RepairSearchNode& n) {
    return {&n.e_vs_mark,  &n.e_sol_mark,    &n.e_pq_mark,    &n.r_vs_mark,
            &n.r_sol_mark, &n.sol_undo_mark, &n.lhs_undo_mark};
}

}  // namespace

TEST_CASE("BacktrackBestOpen: a jump drops the open nodes beneath it",
          "[repair-search][backtrack-best-open]") {
    // The issue's own trace: Q = [A(0), C(m1), E(m2)] with m1 < m2 and A
    // the lowest-violation node.  Under the pre-#158 swap this became
    // [E(m2), C(m1), A(0)]; A is then popped, the stacks unwind to 0, A's
    // own children are pushed at marks far below m1 and consumed, and the
    // next pop is C(m1) against stacks sitting below m1 -- the resize
    // path.  After #158 the two nodes beneath A are given up instead.
    std::vector<RepairSearchNode> q = {marked_node(/*label=*/10, /*mark=*/0, /*violation=*/1.0),
                                       marked_node(/*label=*/11, /*mark=*/4, /*violation=*/5.0),
                                       marked_node(/*label=*/12, /*mark=*/9, /*violation=*/7.0)};

    const size_t dropped = backtrack_best_open(q);

    CHECK(dropped == size_t{2});
    REQUIRE(q.size() == size_t{1});
    CHECK(q.front().var == 10);
    CHECK(q.front().violation == Catch::Approx(1.0));
}

TEST_CASE("BacktrackBestOpen: the alt/pref pair at the best node's own marks survives",
          "[repair-search][backtrack-best-open]") {
    // The two children of one parent are pushed together at the same
    // state, so their marks are *equal*, not deeper: they share the best
    // node's trail prefix and stay legally restorable.  Dropping them
    // would throw away the jump's own destination subtree, and would
    // also move the production search -- this equality case is what
    // keeps the threshold-10 node counts identical across #158.
    //
    // `std::ranges::min_element` returns the *first* minimum, so with the
    // pair tied on violation the alternative is chosen and the swap sends
    // it to the back, where the LIFO pop takes it next.  That tie rule is
    // load-bearing and unchanged.
    std::vector<RepairSearchNode> q = {marked_node(/*label=*/20, /*mark=*/0, /*violation=*/5.0),
                                       marked_node(/*label=*/21, /*mark=*/3, /*violation=*/1.0),
                                       marked_node(/*label=*/22, /*mark=*/3, /*violation=*/1.0)};

    const size_t dropped = backtrack_best_open(q);

    CHECK(dropped == size_t{0});
    REQUIRE(q.size() == size_t{3});
    CHECK(q[0].var == 20);
    CHECK(q[1].var == 22);  // pref, left where it was
    CHECK(q[2].var == 21);  // alt, the first minimum, swapped to the back
}

TEST_CASE("BacktrackBestOpen: any one deeper mark is enough to drop a node",
          "[repair-search][backtrack-best-open]") {
    // The drop rule is a *componentwise* comparison over all seven marks,
    // and the two cases above cannot see that: there every mark moves
    // together, so an implementation testing only `e_vs_mark` -- or only
    // `sol_undo_mark`, the one `backtrack_sol_lhs` reads -- passes both.
    // A stale mark in any single component is a corrupt restore, so each
    // is checked on its own here.
    RepairSearchNode probe = marked_node(/*label=*/0, /*mark=*/0, /*violation=*/0.0);
    const size_t num_marks = marks_of(probe).size();
    for (size_t k = 0; k < num_marks; ++k) {
        INFO("mark component " << k);
        RepairSearchNode best = marked_node(/*label=*/30, /*mark=*/2, /*violation=*/1.0);
        RepairSearchNode other = marked_node(/*label=*/31, /*mark=*/2, /*violation=*/9.0);
        *marks_of(other)[k] = 3;  // deeper in exactly one component

        std::vector<RepairSearchNode> q = {best, other};
        const size_t dropped = backtrack_best_open(q);

        CHECK(dropped == size_t{1});
        REQUIRE(q.size() == size_t{1});
        CHECK(q.front().var == 30);
    }
}

TEST_CASE("BacktrackBestOpen: a deeper node before the best one is dropped too",
          "[repair-search][backtrack-best-open]") {
    // The three cases above all place the best node at the front, where
    // the prefix is empty, so a suffix-only scan passes every one of
    // them.  It is not sufficient: this function is the only thing that
    // can break Q's sort order, and it runs many times in one search.  A
    // previous jump swapped its own B to the back, leaving that B's
    // deeper former neighbours to its *left*; when B is popped and its
    // children are pushed at the new, lower live marks, those older
    // nodes sit before shallower ones.  The next jump then promotes a
    // node with deeper nodes on its left -- and a suffix-only scan
    // strands exactly the node that later pops into the resize-grows
    // path.
    //
    // Q here is that state: a deep leftover at 9, the promoted node at 3
    // in the middle, and a deeper node after it.  Both deep nodes go.
    std::vector<RepairSearchNode> q = {marked_node(/*label=*/40, /*mark=*/9, /*violation=*/7.0),
                                       marked_node(/*label=*/41, /*mark=*/3, /*violation=*/1.0),
                                       marked_node(/*label=*/42, /*mark=*/5, /*violation=*/6.0)};

    const size_t dropped = backtrack_best_open(q);

    CHECK(dropped == size_t{2});
    REQUIRE(q.size() == size_t{1});
    CHECK(q.front().var == 41);

    // And the shallower prefix survives, in place, with the promoted node
    // still swapped to the back where the LIFO pop takes it next.
    std::vector<RepairSearchNode> q2 = {marked_node(/*label=*/50, /*mark=*/1, /*violation=*/7.0),
                                        marked_node(/*label=*/51, /*mark=*/3, /*violation=*/1.0),
                                        marked_node(/*label=*/52, /*mark=*/5, /*violation=*/6.0)};

    const size_t dropped2 = backtrack_best_open(q2);

    CHECK(dropped2 == size_t{1});
    REQUIRE(q2.size() == size_t{2});
    CHECK(q2[0].var == 50);
    CHECK(q2[1].var == 51);
}

TEST_CASE("RepairSearch: a jump abandons its subtree instead of corrupting the trail",
          "[repair-search][backtrack-best-open]") {
    // The whole-search half of #158, on the same `StallModel` and at the
    // fixture's own seed 42, where the defect is reachable: at
    // `progress_threshold` 1 the gate fires often enough to strand a
    // deeper open node above a rewritten trail.
    //
    // Measured on this fixture at seed 42, threshold 1, by temporarily
    // restoring the plain `std::iter_swap`:
    //
    //   pre-#158:  17 nodes,  9 jumps,  0 dropped,  1 mark overshoot,
    //              feasible -- and that one overshoot is the resize path,
    //              i.e. a node whose marks exceeded the live stacks;
    //   post-#158: 11 nodes,  7 jumps,  8 dropped,  0 overshoots,
    //              infeasible.
    //
    // The infeasible verdict is the point rather than a regression: the
    // eight discarded nodes are the completeness Sect. 5.1 says the jump
    // gives up, and the alternative is a search that keeps them and then
    // restores them against state that no longer exists.  Seed 2 is the
    // same defect an order of magnitude larger (15 overshoots pre-fix, 0
    // after), and the production threshold is untouched -- 10 at seed 42
    // visits 29 nodes and finds a solution both before and after.
    //
    // Mutation-sensitive twice over: dropping the discard from
    // `backtrack_best_open` turns `nodes_abandoned_by_jump` to 0 *and*
    // `mark_overshoots` nonzero, so neither assertion alone carries it.
    const StallModel m;
    CscMatrix csc =
        build_csc(StallModel::ncol, StallModel::nrow, m.ar_start, m.ar_index, m.ar_value);

    const StallRun run = run_stall_model(m, csc, /*progress_threshold=*/1);

    CHECK(run.stats.best_open_jumps > size_t{0});
    CHECK(run.stats.nodes_abandoned_by_jump > size_t{0});
    CHECK(run.stats.mark_overshoots == size_t{0});
}

// ===================================================================
// MoveToDisjunction (issue #131).
//
// Sect. 5.1's last paragraph, verbatim: "In the non-binary case, a
// repair move is always a shift s and the shifted interval [a, b] in D
// is always contained in the interval [c, d] in Dr, by construction. We
// compute the gaps to the left and to the right w.r.t. to [c, d], i.e.,
// l = a - c and r = d - b, and the disjunction is then as follows: if
// l <= r, we impose x_j <= b \/ x_j >= b, otherwise x_j <= a \/
// x_j >= a."
//
// Two facts to pin, and the second is why these are unit tests on
// `move_to_disjunction` rather than only end-to-end runs.
//
//   1. `[a, b]` is D *translated by the move's shift*, keeping D's
//      width.  It used to be D itself, so the move never reached the
//      disjunction for a non-binary column, both children re-imposed a
//      bound R already held, and the node moved nothing.
//   2. Each disjunction is a *point split*: both children name the same
//      endpoint.  Reading the repeated endpoint as a typo and widening
//      one side to the other endpoint puts the open interval (a, b) --
//      which holds the move value whenever the point is interior to D --
//      in neither child.
//
// An end-to-end `repair_search` run cannot separate the endpoints: the
// incumbent point ends up clamped to whatever bound propagation implies
// for the column, which is the row's bound and not the branch's, so two
// different split points routinely produce the same solution.  That is
// what makes the direct tests necessary; the runs below them then pin
// that the corrected disjunction actually moves the point.
//
// `move_to_disjunction` is declared in `repair_search.h` for this, the
// same way and for the same reason as `sync_changes`.
// ===================================================================

namespace {

// x0 integer in [0, 10]; one row `row_lo0 <= x0 <= row_hi0`.  The caller
// sets up E and the incumbent point, which is what distinguishes the
// cases.
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
    std::vector<double> row_hi;
    std::vector<HighsVarType> integrality = {HighsVarType::kInteger};

    explicit OneIntModel(double lo, double hi = kHighsInf) : row_lo({lo}), row_hi({hi}) {}

    [[nodiscard]] CscMatrix csc() const {
        return build_csc(ncol, nrow, ar_start, ar_index, ar_value);
    }
};

}  // namespace

TEST_CASE("MoveToDisjunction: a decision-fixed binary still gets the flip pair",
          "[repair-search][disjunction]") {
    // The binary case is the paper's own: "a repair move fixes x_j = b_j
    // on the preferred branch, and the other side of the disjunction is
    // simply x_j = 1 - b_j".  It is detected on the *raw* [lb, ub], which
    // `fix()` leaves wide, so a decision-fixed binary keeps it -- the
    // clique swap RepairSearch exists for depends on that.
    OneIntModel m(0.0);
    m.col_ub = {1.0};
    CscMatrix csc = m.csc();
    const double feastol = 1e-6;
    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine E(OneIntModel::ncol, OneIntModel::nrow, m.ar_start.data(), m.ar_index.data(),
                 m.ar_value.data(), csc, m.col_lb.data(), m.col_ub.data(), m.row_lo.data(),
                 m.row_hi.data(), m.integrality.data(), feastol);
    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine R(OneIntModel::ncol, OneIntModel::nrow, m.ar_start.data(), m.ar_index.data(),
                 m.ar_value.data(), csc, m.col_lb.data(), m.col_ub.data(), m.row_lo.data(),
                 m.row_hi.data(), m.integrality.data(), feastol);
    REQUIRE(E.fix(0, 0.0));

    auto [preferred, alternative] = move_to_disjunction(E, R, 0, /*cur_val=*/0.0,
                                                        /*move_val=*/1.0);
    CHECK(preferred.is_fix);
    CHECK(preferred.val == Catch::Approx(1.0));
    CHECK(alternative.is_fix);
    CHECK(alternative.val == Catch::Approx(0.0));
}

TEST_CASE("MoveToDisjunction: the split is a point split, at one endpoint of [a, b]",
          "[repair-search][disjunction]") {
    // D = [2, 4] unfixed, point 3, move to 7: s = +4, so the shifted
    // interval is [6, 8] -- width 2 -- against R's root domain [0, 10].
    // l = 6 > r = 2, so the paper splits at a = 6.
    //
    // Three things are wrong with any other answer, and each is a
    // separate CHECK below.  A disjunction built from D rather than from
    // the shifted interval names 2 and 4 (that is the pre-#131 code).
    // One built from [move_val, move_val] names 7 (that is the interval
    // shifted but collapsed to a point, losing D's width).  And a
    // disjunction whose two children name *different* endpoints leaves
    // the open interval (6, 8) -- which holds the move value 7 -- in
    // neither child.
    const OneIntModel m(7.0);
    CscMatrix csc = m.csc();
    const double feastol = 1e-6;
    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine E(OneIntModel::ncol, OneIntModel::nrow, m.ar_start.data(), m.ar_index.data(),
                 m.ar_value.data(), csc, m.col_lb.data(), m.col_ub.data(), m.row_lo.data(),
                 m.row_hi.data(), m.integrality.data(), feastol);
    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine R(OneIntModel::ncol, OneIntModel::nrow, m.ar_start.data(), m.ar_index.data(),
                 m.ar_value.data(), csc, m.col_lb.data(), m.col_ub.data(), m.row_lo.data(),
                 m.row_hi.data(), m.integrality.data(), feastol);
    REQUIRE(E.tighten_lb(0, 2.0));
    REQUIRE(E.tighten_ub(0, 4.0));

    auto [preferred, alternative] = move_to_disjunction(E, R, 0, /*cur_val=*/3.0,
                                                        /*move_val=*/7.0);
    // A point split: same value, opposite senses, neither a fix.
    CHECK(preferred.val == Catch::Approx(alternative.val));
    CHECK_FALSE(preferred.is_fix);
    CHECK_FALSE(alternative.is_fix);
    CHECK(preferred.is_lb != alternative.is_lb);
    // Split at a = D.lb + s = 6, which is neither D's own 2 nor the move
    // value 7.
    CHECK(preferred.val == Catch::Approx(6.0));
    // l > r, so the preferred child is the one holding [a, b]: x0 >= a.
    CHECK(preferred.is_lb);
    CHECK_FALSE(alternative.is_lb);
}

TEST_CASE("MoveToDisjunction: l <= r splits at b instead, with the senses swapped",
          "[repair-search][disjunction]") {
    // Same column, mirrored: D = [6, 8] unfixed, point 7, move to 3, so
    // s = -4 and the shifted interval is [2, 4].  Now l = 2 <= r = 6, so
    // the split is at b = 4 and the child holding [a, b] is x0 <= b.
    // Together with the case above this pins both arms of the `l <= r`
    // test, and that the preferred sense follows the arm rather than
    // being fixed.
    const OneIntModel m(0.0, 3.0);
    CscMatrix csc = m.csc();
    const double feastol = 1e-6;
    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine E(OneIntModel::ncol, OneIntModel::nrow, m.ar_start.data(), m.ar_index.data(),
                 m.ar_value.data(), csc, m.col_lb.data(), m.col_ub.data(), m.row_lo.data(),
                 m.row_hi.data(), m.integrality.data(), feastol);
    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine R(OneIntModel::ncol, OneIntModel::nrow, m.ar_start.data(), m.ar_index.data(),
                 m.ar_value.data(), csc, m.col_lb.data(), m.col_ub.data(), m.row_lo.data(),
                 m.row_hi.data(), m.integrality.data(), feastol);
    REQUIRE(E.tighten_lb(0, 6.0));
    REQUIRE(E.tighten_ub(0, 8.0));

    auto [preferred, alternative] = move_to_disjunction(E, R, 0, /*cur_val=*/7.0,
                                                        /*move_val=*/3.0);
    CHECK(preferred.val == Catch::Approx(alternative.val));
    CHECK(preferred.val == Catch::Approx(4.0));
    CHECK_FALSE(preferred.is_lb);
    CHECK(alternative.is_lb);
}

TEST_CASE("MoveToDisjunction: the shifted interval is clipped into R's domain",
          "[repair-search][disjunction]") {
    // Sect. 5.1 asserts [a, b] is inside [c, d] "by construction"; here
    // it is not, because `walksat_select_move` clips a shift to the
    // column's structural bounds and not to R's domain.  D is the
    // singleton {8} of a decision-fixed column and the point has drifted
    // to 0 -- which is what `apply_move` does, since it writes `solution`
    // and never E -- so a move to 3 is a shift of +3 and the shifted
    // interval is {11}, outside R's [0, 10].
    //
    // Clipped, the split is at 10 and both children are real bound
    // changes.  Unclipped it is at 11, where `x0 >= 11` is refuted on the
    // spot and `x0 <= 11` is a bound R already satisfies, so the node
    // moves nothing.
    const OneIntModel m(3.0);
    CscMatrix csc = m.csc();
    const double feastol = 1e-6;
    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine E(OneIntModel::ncol, OneIntModel::nrow, m.ar_start.data(), m.ar_index.data(),
                 m.ar_value.data(), csc, m.col_lb.data(), m.col_ub.data(), m.row_lo.data(),
                 m.row_hi.data(), m.integrality.data(), feastol);
    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine R(OneIntModel::ncol, OneIntModel::nrow, m.ar_start.data(), m.ar_index.data(),
                 m.ar_value.data(), csc, m.col_lb.data(), m.col_ub.data(), m.row_lo.data(),
                 m.row_hi.data(), m.integrality.data(), feastol);
    REQUIRE(E.fix(0, 8.0));

    auto [preferred, alternative] = move_to_disjunction(E, R, 0, /*cur_val=*/0.0,
                                                        /*move_val=*/3.0);
    CHECK(preferred.val == Catch::Approx(10.0));
    CHECK(alternative.val == Catch::Approx(10.0));
    CHECK(preferred.is_lb);
}

// ===================================================================
// ... and the same disjunction, driving a whole search.
//
// One general integer and one row, so `walksat_select_move` has exactly
// one violated row and exactly one candidate: every assertion below is
// on the disjunction, with no dependence on the RNG stream.  All three
// return `bound_branch_moves == 0` and no solution against the pre-#131
// code, which never let the move reach the disjunction at all.
// ===================================================================

namespace {

// One `repair_search` run on `m` from the given E state and point.
struct OneIntRun {
    bool feasible = false;
    RepairSearchStats stats;
    std::vector<double> solution;
    std::vector<double> lhs_cache;
};

OneIntRun run_one_int(const OneIntModel& m, PropEngine& e_engine, const CscMatrix& csc,
                      double point) {
    static_cast<void>(csc);
    OneIntRun out;
    out.solution = {point};
    out.lhs_cache = {point};
    FprScratch scratch;
    Rng rng(42);
    Deadline deadline;  // never expires
    size_t effort_out = 0;
    out.feasible =
        repair_search(e_engine, out.solution, out.lhs_cache, m.col_lb.data(), m.col_ub.data(),
                      m.row_lo.data(), m.row_hi.data(),
                      /*repair_iterations=*/50, /*repair_noise=*/0.75,
                      /*repair_track_best=*/true, rng, effort_out, scratch, deadline, &out.stats);
    return out;
}

}  // namespace

TEST_CASE("RepairSearch: a fixed general integer is moved by a bound branch",
          "[repair-search][disjunction]") {
    // x0 decision-fixed to 2 (so `[lb, ub]` stays the wide [0, 10]) and
    // the row wants x0 >= 7.  The one move is the shift +5, the shifted
    // interval is the singleton [7, 7], l = 7 > r = 3, so the split is at
    // 7 and the preferred child `x0 >= 7` drives R to [7, 10];
    // `sync_changes` then re-fixes E, which is what moves the point.
    const OneIntModel m(7.0);
    CscMatrix csc = m.csc();
    const double feastol = 1e-6;
    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine E(OneIntModel::ncol, OneIntModel::nrow, m.ar_start.data(), m.ar_index.data(),
                 m.ar_value.data(), csc, m.col_lb.data(), m.col_ub.data(), m.row_lo.data(),
                 m.row_hi.data(), m.integrality.data(), feastol);
    REQUIRE(E.fix(0, 2.0));
    REQUIRE(E.var(0).lb == Catch::Approx(0.0));   // wide, as `fix()` leaves them
    REQUIRE(E.var(0).ub == Catch::Approx(10.0));  // -- which is why D is the value

    const OneIntRun r = run_one_int(m, E, csc, /*point=*/2.0);
    REQUIRE(r.feasible);
    CHECK(r.stats.bound_branch_moves == 1);
    CHECK(r.solution[0] == Catch::Approx(7.0));
    CHECK(r.lhs_cache[0] == Catch::Approx(7.0));
    CHECK(E.var(0).fixed);
    CHECK(E.var(0).val == Catch::Approx(7.0));
}

TEST_CASE("RepairSearch: an unfixed general integer is moved by a bound branch",
          "[repair-search][disjunction]") {
    // The disjunction of the point-split test above, run: D = [2, 4],
    // point 3, row x0 >= 7, split at 6, preferred `x0 >= 6`.  R
    // propagates that to [7, 10], which is disjoint from D, so
    // `sync_changes` takes its disjoint case and fixes x0 to R's nearer
    // endpoint.
    const OneIntModel m(7.0);
    CscMatrix csc = m.csc();
    const double feastol = 1e-6;
    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine E(OneIntModel::ncol, OneIntModel::nrow, m.ar_start.data(), m.ar_index.data(),
                 m.ar_value.data(), csc, m.col_lb.data(), m.col_ub.data(), m.row_lo.data(),
                 m.row_hi.data(), m.integrality.data(), feastol);
    REQUIRE(E.tighten_lb(0, 2.0));
    REQUIRE(E.tighten_ub(0, 4.0));
    REQUIRE_FALSE(E.var(0).fixed);

    const OneIntRun r = run_one_int(m, E, csc, /*point=*/3.0);
    REQUIRE(r.feasible);
    CHECK(r.stats.bound_branch_moves == 1);
    CHECK(r.solution[0] == Catch::Approx(7.0));
    CHECK(E.var(0).fixed);
    CHECK(E.var(0).val == Catch::Approx(7.0));
}

TEST_CASE("RepairSearch: a sync that fixes inside E's own bounds still moves the point",
          "[repair-search][disjunction]") {
    // x0 unfixed on the full [0, 10], point 0, and the row is the
    // equality x0 = 8.  The move is +8, the shifted interval [8, 18]
    // clips to [8, 10], l = 8 > r = 0, so the preferred child is
    // `x0 >= 8`; R propagates the equality to [8, 8], and that intersects
    // D in the single point 8, which is `sync_changes`' intersection case
    // -- committed through `PropEngine::fix`, which leaves `[lb, ub]` at
    // [0, 10].
    //
    // So a clamp reading the raw bounds finds the point 0 already inside
    // them and drops the move even though E is fixed at 8.  The clamp
    // reads the interval E holds instead, which is what this case pins;
    // the two above reach E through `refix`, which narrows, and cannot
    // see the difference.
    const OneIntModel m(8.0, 8.0);
    CscMatrix csc = m.csc();
    const double feastol = 1e-6;
    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine E(OneIntModel::ncol, OneIntModel::nrow, m.ar_start.data(), m.ar_index.data(),
                 m.ar_value.data(), csc, m.col_lb.data(), m.col_ub.data(), m.row_lo.data(),
                 m.row_hi.data(), m.integrality.data(), feastol);
    REQUIRE_FALSE(E.var(0).fixed);

    const OneIntRun r = run_one_int(m, E, csc, /*point=*/0.0);
    REQUIRE(r.feasible);
    CHECK(r.stats.bound_branch_moves == 1);
    CHECK(r.solution[0] == Catch::Approx(8.0));
    CHECK(E.var(0).fixed);
    CHECK(E.var(0).val == Catch::Approx(8.0));
    CHECK(E.var(0).lb == Catch::Approx(0.0));   // `fix()` left them wide ...
    CHECK(E.var(0).ub == Catch::Approx(10.0));  // ... which is the point of this case
}

TEST_CASE("RepairSearch: the clip decides which node lands the move",
          "[repair-search][disjunction]") {
    // The clip case above, run.  x0 decision-fixed to 8, point drifted to
    // 0, row x0 >= 3: the shifted interval is {11}, outside R's [0, 10].
    //
    // Clipped, the split is at 10, the preferred child `x0 >= 10` drives
    // R to [10, 10], the sync re-fixes E, and the second node ends the
    // search at 10.  Unclipped the split is at 11: `x0 >= 11` is refuted
    // outright and `x0 <= 11` is a bound R already satisfies, so the
    // first two nodes move nothing and the search only lands -- on 8, out
    // of E's own singleton -- at the third.
    const OneIntModel m(3.0);
    CscMatrix csc = m.csc();
    const double feastol = 1e-6;
    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine E(OneIntModel::ncol, OneIntModel::nrow, m.ar_start.data(), m.ar_index.data(),
                 m.ar_value.data(), csc, m.col_lb.data(), m.col_ub.data(), m.row_lo.data(),
                 m.row_hi.data(), m.integrality.data(), feastol);
    REQUIRE(E.fix(0, 8.0));

    const OneIntRun r = run_one_int(m, E, csc, /*point=*/0.0);
    REQUIRE(r.feasible);
    CHECK(r.stats.bound_branch_moves == 1);
    CHECK(r.stats.nodes_visited == 2);
    CHECK(r.solution[0] == Catch::Approx(10.0));
}
