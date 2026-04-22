#include "heuristic_common.h"
#include "Highs.h"
#include "prop_engine.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <vector>

// ===================================================================
// PropEngine unit tests
// ===================================================================

// Helper: build a small test model for PropEngine tests.
// 3 variables: x0 (binary), x1 (integer [0,5]), x2 (continuous [0,10])
// 2 constraints:
//   row 0: x0 + x1 >= 2   (row_lo=2, row_hi=inf)
//   row 1: x1 + x2 <= 8   (row_lo=-inf, row_hi=8)
namespace {
struct SmallModel {
    static constexpr HighsInt ncol = 3;
    static constexpr HighsInt nrow = 2;
    std::vector<HighsInt> ar_start = {0, 2, 4};
    std::vector<HighsInt> ar_index = {0, 1, 1, 2};
    std::vector<double> ar_value = {1.0, 1.0, 1.0, 1.0};
    std::vector<double> col_lb = {0.0, 0.0, 0.0};
    std::vector<double> col_ub = {1.0, 5.0, 10.0};
    double row_lo[2] = {2.0, -kHighsInf};
    double row_hi[2] = {kHighsInf, 8.0};
    std::vector<HighsVarType> integrality = {HighsVarType::kInteger, HighsVarType::kInteger,
                                             HighsVarType::kContinuous};
    CscMatrix csc;

    SmallModel() { csc = build_csc(ncol, nrow, ar_start, ar_index, ar_value); }

    PropEngine make_engine(double feastol = 1e-6) {
        return PropEngine(ncol, nrow, ar_start.data(), ar_index.data(), ar_value.data(), csc,
                          col_lb.data(), col_ub.data(), row_lo, row_hi, integrality.data(),
                          feastol);
    }
};
}  // namespace

TEST_CASE("PropEngine: fix and propagate", "[prop-engine]") {
    SmallModel m;
    auto eng = m.make_engine();

    // Initial state: all variables unfixed with global bounds
    REQUIRE_FALSE(eng.var(0).fixed);
    REQUIRE(eng.var(1).lb == Catch::Approx(0.0));
    REQUIRE(eng.var(1).ub == Catch::Approx(5.0));

    // Fix x0 = 0, propagate: row 0 (x0+x1 >= 2) forces x1 >= 2
    REQUIRE(eng.fix(0, 0.0));
    REQUIRE(eng.propagate(0));
    REQUIRE(eng.var(0).fixed);
    REQUIRE(eng.var(0).val == Catch::Approx(0.0));
    REQUIRE(eng.var(1).lb >= 2.0 - 1e-6);

    // Fix x1 = 5, propagate: row 1 (x1+x2 <= 8) forces x2 <= 3
    REQUIRE(eng.fix(1, 5.0));
    REQUIRE(eng.propagate(1));
    REQUIRE(eng.var(2).ub <= 3.0 + 1e-6);
}

TEST_CASE("PropEngine: backtrack restores state", "[prop-engine]") {
    SmallModel m;
    auto eng = m.make_engine();

    HighsInt vs_m = eng.vs_mark();
    HighsInt sol_m = eng.sol_mark();

    REQUIRE(eng.fix(0, 1.0));
    REQUIRE(eng.propagate(0));
    REQUIRE(eng.var(0).fixed);

    eng.backtrack_to(vs_m, sol_m);
    REQUIRE_FALSE(eng.var(0).fixed);
    REQUIRE(eng.var(0).lb == Catch::Approx(0.0));
    REQUIRE(eng.var(0).ub == Catch::Approx(1.0));
    REQUIRE(eng.var(1).lb == Catch::Approx(0.0));
}

TEST_CASE("PropEngine: tighten bounds and auto-fix", "[prop-engine]") {
    SmallModel m;
    auto eng = m.make_engine();

    REQUIRE(eng.tighten_lb(1, 3.0));
    REQUIRE(eng.var(1).lb >= 3.0 - 1e-6);
    REQUIRE(eng.var(1).ub == Catch::Approx(5.0));

    // Tighten ub to match lb — should auto-fix
    REQUIRE(eng.tighten_ub(1, 3.0));
    REQUIRE(eng.var(1).fixed);
    REQUIRE(eng.var(1).val == Catch::Approx(3.0));
}

TEST_CASE("PropEngine: infeasible propagation", "[prop-engine]") {
    SmallModel m;
    auto eng = m.make_engine();

    // Fix x0=0, tighten x1 ub to 1. Propagation from row 0 (x0+x1 >= 2)
    // tries to tighten x1 lb to 2, but ub is 1 → lb > ub → infeasible.
    REQUIRE(eng.fix(0, 0.0));
    REQUIRE(eng.tighten_ub(1, 1.0));
    REQUIRE_FALSE(eng.propagate(0));
}

TEST_CASE("PropEngine: reset clears state", "[prop-engine]") {
    SmallModel m;
    auto eng = m.make_engine();

    eng.fix(0, 1.0);
    eng.propagate(0);
    REQUIRE(eng.var(0).fixed);

    eng.reset();
    REQUIRE_FALSE(eng.var(0).fixed);
    REQUIRE(eng.var(0).lb == Catch::Approx(0.0));
    REQUIRE(eng.var(0).ub == Catch::Approx(1.0));
    REQUIRE(eng.var(1).lb == Catch::Approx(0.0));
    REQUIRE(eng.var(1).ub == Catch::Approx(5.0));
}

TEST_CASE("PropEngine: effort tracking", "[prop-engine]") {
    SmallModel m;
    auto eng = m.make_engine();

    size_t before = eng.effort();
    eng.fix(0, 1.0);
    eng.propagate(0);
    REQUIRE(eng.effort() > before);

    eng.add_effort(100);
    REQUIRE(eng.effort() >= before + 100);
}

// ===================================================================
// Reset-equivalence tests: a reset() engine must behave identically
// to a freshly-constructed one. These guard the per-worker PropEngine
// reuse path in FprScratch: if reset leaves any stale internal state
// (undo stacks, activity caches, PQ, worklist flags, effort counter),
// the second fpr_attempt call would silently miscompute or stall.
// ===================================================================

namespace {
// Capture the full observable state of a PropEngine so tests can compare
// "fresh construction" against "reset after a prior attempt".
struct EngineSnapshot {
    std::vector<VarState> vars;
    std::vector<double> solution;
    std::vector<double> min_activity;
    std::vector<double> max_activity;
    size_t effort;
    HighsInt vs_mark;
    HighsInt sol_mark;
    HighsInt act_mark;
    bool activities_initialized;
    bool pq_initialized;
    HighsInt pq_top;

    static EngineSnapshot of(const PropEngine& eng) {
        EngineSnapshot s;
        s.vars.reserve(eng.ncol());
        s.solution.reserve(eng.ncol());
        for (HighsInt j = 0; j < eng.ncol(); ++j) {
            s.vars.push_back(eng.var(j));
            s.solution.push_back(eng.sol(j));
        }
        s.activities_initialized = eng.activities_initialized();
        if (s.activities_initialized) {
            s.min_activity.reserve(eng.nrow());
            s.max_activity.reserve(eng.nrow());
            for (HighsInt i = 0; i < eng.nrow(); ++i) {
                s.min_activity.push_back(eng.row_min_activity(i));
                s.max_activity.push_back(eng.row_max_activity(i));
            }
        }
        s.effort = eng.effort();
        s.vs_mark = eng.vs_mark();
        s.sol_mark = eng.sol_mark();
        s.act_mark = eng.act_mark();
        s.pq_initialized = eng.pq_initialized();
        s.pq_top = eng.pq_initialized() ? eng.pq_top() : -1;
        return s;
    }
};

void require_snapshots_equal(const EngineSnapshot& a, const EngineSnapshot& b) {
    REQUIRE(a.vars.size() == b.vars.size());
    for (size_t j = 0; j < a.vars.size(); ++j) {
        INFO("var " << j);
        REQUIRE(a.vars[j].lb == Catch::Approx(b.vars[j].lb));
        REQUIRE(a.vars[j].ub == Catch::Approx(b.vars[j].ub));
        REQUIRE(a.vars[j].val == Catch::Approx(b.vars[j].val));
        REQUIRE(a.vars[j].fixed == b.vars[j].fixed);
    }
    REQUIRE(a.solution.size() == b.solution.size());
    for (size_t j = 0; j < a.solution.size(); ++j) {
        INFO("sol " << j);
        REQUIRE(a.solution[j] == Catch::Approx(b.solution[j]));
    }
    REQUIRE(a.activities_initialized == b.activities_initialized);
    REQUIRE(a.min_activity.size() == b.min_activity.size());
    REQUIRE(a.max_activity.size() == b.max_activity.size());
    for (size_t i = 0; i < a.min_activity.size(); ++i) {
        INFO("row " << i);
        REQUIRE(a.min_activity[i] == Catch::Approx(b.min_activity[i]));
        REQUIRE(a.max_activity[i] == Catch::Approx(b.max_activity[i]));
    }
    REQUIRE(a.effort == b.effort);
    REQUIRE(a.vs_mark == b.vs_mark);
    REQUIRE(a.sol_mark == b.sol_mark);
    REQUIRE(a.act_mark == b.act_mark);
    REQUIRE(a.pq_initialized == b.pq_initialized);
    REQUIRE(a.pq_top == b.pq_top);
}
}  // namespace

TEST_CASE("PropEngine: reset matches fresh construction (post-propagate)", "[prop-engine][reset]") {
    SmallModel m;

    // Baseline: a freshly constructed engine, nothing done yet.
    auto fresh = m.make_engine();
    auto fresh_snap = EngineSnapshot::of(fresh);

    // Second engine: exercise fix/propagate/backtrack, then reset.
    auto reused = m.make_engine();
    REQUIRE(reused.fix(0, 1.0));
    REQUIRE(reused.propagate(0));
    REQUIRE(reused.fix(1, 4.0));
    REQUIRE(reused.propagate(1));
    reused.add_effort(4242);
    reused.reset();

    auto reused_snap = EngineSnapshot::of(reused);
    require_snapshots_equal(fresh_snap, reused_snap);

    // Explicit check: effort must be zeroed (guards the Phase 1-2 DFS
    // gate `E.effort() < cfg.max_effort` in fpr_attempt).
    REQUIRE(reused.effort() == 0);
}

TEST_CASE("PropEngine: reset matches fresh construction (with activities + PQ)",
          "[prop-engine][reset]") {
    SmallModel m;

    auto fresh = m.make_engine();
    fresh.init_activities();
    fresh.init_domain_pq();
    auto fresh_snap = EngineSnapshot::of(fresh);

    auto reused = m.make_engine();
    reused.init_activities();
    reused.init_domain_pq();
    REQUIRE(reused.fix(0, 1.0));
    REQUIRE(reused.propagate(0));
    REQUIRE(reused.fix(1, 3.0));
    REQUIRE(reused.propagate(1));
    reused.reset();
    // Match the fresh-construction path: caller re-initialises the opt-in
    // activity/PQ machinery after reset.
    reused.init_activities();
    reused.init_domain_pq();

    auto reused_snap = EngineSnapshot::of(reused);
    require_snapshots_equal(fresh_snap, reused_snap);
}

TEST_CASE("PropEngine: reset matches fresh construction (post-infeasibility)",
          "[prop-engine][reset]") {
    SmallModel m;

    auto fresh = m.make_engine();
    auto fresh_snap = EngineSnapshot::of(fresh);

    auto reused = m.make_engine();
    // Drive to infeasibility: propagate marks prop_in_wl_ entries and
    // clears them on the infeasibility path; reset() must still leave
    // the engine in a pristine state.
    REQUIRE(reused.fix(0, 0.0));
    REQUIRE(reused.tighten_ub(1, 1.0));
    REQUIRE_FALSE(reused.propagate(0));
    reused.reset();

    auto reused_snap = EngineSnapshot::of(reused);
    require_snapshots_equal(fresh_snap, reused_snap);
}

TEST_CASE("PropEngine: reset allows fresh fix+propagate sequence", "[prop-engine][reset]") {
    SmallModel m;
    auto eng = m.make_engine();

    // Attempt 1: fix x0=1, propagate, then backtrack isn't needed — we test
    // that reset() itself fully restores the engine for a second attempt
    // that produces the *same* outcome as a fresh engine would.
    REQUIRE(eng.fix(0, 1.0));
    REQUIRE(eng.propagate(0));
    // propagate should have touched x1 via row 0 (x0+x1 >= 2 with x0=1
    // forces x1 >= 1), so x1.lb should be tightened.
    REQUIRE(eng.var(1).lb >= 1.0 - 1e-6);

    // Capture the outcome for the "attempt 1" fix(0, 0) branch run on a
    // fresh engine, so we can compare the reset-then-attempt-2 path.
    SmallModel m2;
    auto fresh_run = m2.make_engine();
    REQUIRE(fresh_run.fix(0, 0.0));
    REQUIRE(fresh_run.propagate(0));
    auto expected = EngineSnapshot::of(fresh_run);

    // Reset and run the alternative branch on the reused engine; it
    // must land on exactly the same state the fresh engine reached.
    eng.reset();
    REQUIRE(eng.fix(0, 0.0));
    REQUIRE(eng.propagate(0));
    auto actual = EngineSnapshot::of(eng);

    // Full snapshot match: reset() zeroes prop_work_ and empties every
    // undo stack, so after the identical fix+propagate sequence the
    // reused engine must reach the same observable state as a fresh
    // engine — including effort, marks, and activity/PQ state.
    require_snapshots_equal(expected, actual);
    REQUIRE(actual.effort == expected.effort);
    REQUIRE(actual.vs_mark == expected.vs_mark);
    REQUIRE(actual.sol_mark == expected.sol_mark);
    REQUIRE(actual.act_mark == expected.act_mark);
}

// A-F2: direct assertion that reset() leaves prop_in_wl_ zeroed.
// The other reset tests only cover this transitively via the subsequent
// fix+propagate outcome.  If a future refactor left prop_worklist_
// non-empty on a propagate() early-exit path, this regression guard
// would fire on reset+reuse before any behaviour-visible miscompute.
TEST_CASE("PropEngine: reset leaves prop_worklist drained", "[prop-engine][reset]") {
    SmallModel m;
    auto eng = m.make_engine();

    // Drive propagate() on several rows so prop_worklist_ sees non-trivial
    // activity, then infeasibility to exercise the early-exit worklist
    // teardown path.
    REQUIRE(eng.fix(0, 0.0));
    REQUIRE(eng.tighten_ub(1, 1.0));
    REQUIRE_FALSE(eng.propagate(0));

    eng.reset();

    // Observable proxy for "prop_worklist_ empty + prop_in_wl_[*] == 0":
    // seed the worklist from a single var and propagate; if any stale row
    // flag survived reset, propagate would process it redundantly and the
    // effort counter would be non-zero even in the trivial no-deduction
    // case.  Fix x2 (continuous, not in any integer row) so propagate
    // does minimal real work; any effort above the single-var seed cost
    // indicates stale state.
    size_t baseline = eng.effort();
    REQUIRE(eng.fix(2, 5.0));
    REQUIRE(eng.propagate(2));
    size_t delta = eng.effort() - baseline;
    // Fresh baseline: run the same sequence on a brand-new engine and
    // verify the reused engine didn't do extra work.
    auto fresh = m.make_engine();
    size_t fresh_baseline = fresh.effort();
    REQUIRE(fresh.fix(2, 5.0));
    REQUIRE(fresh.propagate(2));
    size_t fresh_delta = fresh.effort() - fresh_baseline;
    REQUIRE(delta == fresh_delta);
}

// Regression test for fpr_core.cpp's pointer-identity guard used when a
// persistent FprScratch is reused across two distinct HighsMipSolver
// instances.  A cached PropEngine holds observer pointers into the *first*
// problem's data; reusing it against the second problem would silently
// miscompute.  The guard must compare every problem-data pointer and
// re-emplace on mismatch.  This test replays the exact comparison logic
// against two different SmallModel instances and asserts mismatch is
// detected.
TEST_CASE("PropEngine: pointer-identity guard detects problem swap",
          "[prop-engine][reset][guard]") {
    SmallModel m1;
    SmallModel m2;  // independent storage — every .data() differs from m1

    auto eng1 = m1.make_engine();

    // Mirror the guard at src/fpr_core.cpp:123-130 and the parallel guard
    // in repair_search.cpp.  If any pointer comparison is accidentally
    // dropped in a future refactor, this assertion fails.
    auto matches = [](const PropEngine& eng, const SmallModel& m, double feastol) {
        return eng.ncol() == m.ncol && eng.nrow() == m.nrow &&
               eng.ar_start() == m.ar_start.data() && eng.ar_index() == m.ar_index.data() &&
               eng.ar_value() == m.ar_value.data() && eng.csc_start() == m.csc.col_start.data() &&
               eng.csc_row() == m.csc.col_row.data() && eng.csc_val() == m.csc.col_val.data() &&
               eng.col_lb() == m.col_lb.data() && eng.col_ub() == m.col_ub.data() &&
               eng.row_lo() == m.row_lo && eng.row_hi() == m.row_hi &&
               eng.integrality() == m.integrality.data() && eng.feastol() == feastol;
    };

    REQUIRE(matches(eng1, m1, 1e-6));
    REQUIRE_FALSE(matches(eng1, m2, 1e-6));

    // Each feastol component is covered too: a mismatch on feastol alone
    // must be detected even when all pointers still match the first model.
    REQUIRE_FALSE(matches(eng1, m1, 1e-9));
}

// ===================================================================
// IndexedMinHeap unit tests — the vector-backed replacement for the
// prior std::set domain-PQ.  PropEngine's reset-equivalence tests
// exercise these transitively, but targeted invariants are easier to
// localize here if sift_up / sift_down / erase / update ever regress.
// ===================================================================

TEST_CASE("IndexedMinHeap: empty and single-element invariants", "[prop-engine][pq]") {
    IndexedMinHeap heap;
    heap.reserve(4);
    REQUIRE(heap.empty());
    REQUIRE(heap.size() == 0);
    REQUIRE_FALSE(heap.contains(0));

    heap.insert(3.0, 1);
    REQUIRE_FALSE(heap.empty());
    REQUIRE(heap.size() == 1);
    REQUIRE(heap.contains(1));
    REQUIRE(heap.top_var() == 1);

    heap.erase(1);
    REQUIRE(heap.empty());
    REQUIRE_FALSE(heap.contains(1));
}

TEST_CASE("IndexedMinHeap: ordering and tiebreak by var index", "[prop-engine][pq]") {
    IndexedMinHeap heap;
    heap.reserve(8);
    // Insert in non-sorted order; top_var must reflect min(key, var).
    heap.insert(5.0, 2);
    heap.insert(3.0, 0);
    heap.insert(3.0, 1);  // same key as var 0; var 0 wins the tiebreak
    heap.insert(7.0, 3);
    REQUIRE(heap.top_var() == 0);

    heap.erase(0);
    REQUIRE(heap.top_var() == 1);  // tiebreak now selects next-smallest var

    heap.erase(1);
    REQUIRE(heap.top_var() == 2);

    heap.erase(2);
    REQUIRE(heap.top_var() == 3);

    heap.erase(3);
    REQUIRE(heap.empty());
}

TEST_CASE("IndexedMinHeap: update moves entry in both directions", "[prop-engine][pq]") {
    IndexedMinHeap heap;
    heap.reserve(4);
    heap.insert(10.0, 0);
    heap.insert(20.0, 1);
    heap.insert(30.0, 2);
    REQUIRE(heap.top_var() == 0);

    // Decrease-key: var 2 beats var 0 after update.
    heap.update(2, 5.0);
    REQUIRE(heap.top_var() == 2);
    REQUIRE(heap.contains(0));
    REQUIRE(heap.contains(1));

    // Increase-key: var 2 drops past both peers.
    heap.update(2, 100.0);
    REQUIRE(heap.top_var() == 0);

    // No-op update (same key) must be a true no-op.
    heap.update(0, 10.0);
    REQUIRE(heap.top_var() == 0);
}

TEST_CASE("IndexedMinHeap: erase of the root re-heapifies correctly", "[prop-engine][pq]") {
    IndexedMinHeap heap;
    heap.reserve(8);
    // Distinct vars with distinct keys (keys chosen to force non-sorted
    // insertion order so sift_up is exercised on the build-up).
    const std::pair<double, HighsInt> entries[] = {{7.0, 0}, {3.0, 1}, {5.0, 2}, {1.0, 3},
                                                   {6.0, 4}, {4.0, 5}, {8.0, 6}, {2.0, 7}};
    for (const auto& [key, var] : entries) {
        heap.insert(key, var);
    }
    // Pop in ascending-key order: 1.0/3, 2.0/7, 3.0/1, 4.0/5, 5.0/2, 6.0/4,
    // 7.0/0, 8.0/6.  Each pop forces erase-of-root followed by sift_down of
    // the moved-last entry.
    REQUIRE(heap.top_var() == 3);
    heap.erase(3);
    REQUIRE(heap.top_var() == 7);
    heap.erase(7);
    REQUIRE(heap.top_var() == 1);
    heap.erase(1);
    REQUIRE(heap.top_var() == 5);
    heap.erase(5);
    REQUIRE(heap.top_var() == 2);
}

TEST_CASE("IndexedMinHeap: clear retains capacity and resets contains()", "[prop-engine][pq]") {
    IndexedMinHeap heap;
    heap.reserve(4);
    heap.insert(1.0, 0);
    heap.insert(2.0, 1);
    heap.insert(3.0, 2);

    heap.clear();
    REQUIRE(heap.empty());
    REQUIRE_FALSE(heap.contains(0));
    REQUIRE_FALSE(heap.contains(1));
    REQUIRE_FALSE(heap.contains(2));

    // Re-populating after clear must behave like a fresh heap.
    heap.insert(10.0, 2);
    heap.insert(5.0, 0);
    REQUIRE(heap.top_var() == 0);
    REQUIRE(heap.contains(2));
}
