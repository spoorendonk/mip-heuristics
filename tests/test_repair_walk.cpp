#include "fpr_core.h"
#include "fpr_strategies.h"
#include "heuristic_common.h"
#include "heuristic_context.h"
#include "Highs.h"
#include "parallel/HighsParallel.h"
#include "prop_engine.h"
#include "repair_walk.h"
#include "rng.h"
#include "test_common.h"
#include "walksat.h"

#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

// ===================================================================
// RepairWalk on a partial assignment (issue #124).
//
// Salvagnin, Roberti, Fischetti, MPC 17:111-139, 2025.  Fig. 1 lines
// 7-8 call `RepairWalk(P)` at *every* node whose processing left the
// problem infeasible, and Sect. 5 generalizes WalkSAT's violation and
// shift definitions from complete assignments to the domains that
// encode a partial one, precisely so that call is possible.  What
// shipped before #124 was the complete-assignment variant the paper
// reports abandoning ("we tested a version of WalkSAT generalized to
// work on MIPs ... the results were quite poor ... This is what
// prompted our strategy of applying solution repair not on complete
// assignments, but as a repair procedure within the fix-and-propagate
// search"), reachable only from the leaf.
//
// `E` throughout is the paper's own symbol for the primary propagation
// engine, matching the NOLINT precedent in `repair_search.cpp` and
// `tests/test_repair_search.cpp`.
// ===================================================================

namespace {

// x0 + x1 >= 1, 3*x0 + x1 <= 2, both binary.
//
// Fixing x0 = 1 makes the second row unsatisfiable by any completion, and
// the repair that recovers it is available on exactly one column: x0's
// singleton domain {1} can slide down to {0} inside its structural [0, 1],
// while x1's *unfixed* [0, 1] cannot slide at all -- the paper forbids
// widening it and translating it by the -1 the row asks for would leave
// the structural bounds.  So the walk is deterministic here: one
// candidate, zero damage, chosen greedily.
struct SwapModel {
    static constexpr HighsInt kNcol = 2;
    static constexpr HighsInt kNrow = 2;
    std::vector<HighsInt> ar_start = {0, 2, 4};
    std::vector<HighsInt> ar_index = {0, 1, 0, 1};
    std::vector<double> ar_value = {1.0, 1.0, 3.0, 1.0};
    std::vector<double> col_lb = {0.0, 0.0};
    std::vector<double> col_ub = {1.0, 1.0};
    std::array<double, 2> row_lo = {1.0, -kHighsInf};
    std::array<double, 2> row_hi = {kHighsInf, 2.0};
    std::vector<HighsVarType> integrality = {HighsVarType::kInteger, HighsVarType::kInteger};
    CscMatrix csc;

    SwapModel() { csc = build_csc(kNcol, kNrow, ar_start, ar_index, ar_value); }

    PropEngine make_engine(double feastol = 1e-6) {
        return {kNcol,           kNrow,         ar_start.data(),    ar_index.data(),
                ar_value.data(), csc,           col_lb.data(),      col_ub.data(),
                row_lo.data(),   row_hi.data(), integrality.data(), feastol};
    }
};

// One general integer column and one row `coeff * y {<=,>=} rhs`, so the
// only repair available is a shift of `y`'s own interval and a test can
// assert the *whole* resulting domain rather than a value.  The
// constructor's parameters are exactly the axes the shift arithmetic
// branches on: which side of the row is binding, the sign of the
// coefficient, and how much room the structural bounds leave.
struct OneRowModel {
    static constexpr HighsInt kNcol = 1;
    static constexpr HighsInt kNrow = 1;
    std::vector<HighsInt> ar_start = {0, 1};
    std::vector<HighsInt> ar_index = {0};
    std::vector<double> ar_value;
    std::vector<double> col_lb;
    std::vector<double> col_ub = {10.0};
    std::array<double, 1> row_lo{};
    std::array<double, 1> row_hi{};
    std::vector<HighsVarType> integrality = {HighsVarType::kInteger};
    CscMatrix csc;

    OneRowModel(double coeff, double lo, double hi, double structural_lb)
        : ar_value({coeff}), col_lb({structural_lb}), row_lo({lo}), row_hi({hi}) {
        csc = build_csc(kNcol, kNrow, ar_start, ar_index, ar_value);
    }

    PropEngine make_engine(double feastol = 1e-6) {
        return {kNcol,           kNrow,         ar_start.data(),    ar_index.data(),
                ar_value.data(), csc,           col_lb.data(),      col_ub.data(),
                row_lo.data(),   row_hi.data(), integrality.data(), feastol};
    }
};

// Narrow column 0 to `[lo, hi]` the way propagation would have.
void narrow(PropEngine& engine, double lo, double hi) {
    REQUIRE(engine.tighten_lb(0, lo));
    REQUIRE(engine.tighten_ub(0, hi));
}

bool walk(PropEngine& engine, HighsInt max_steps, Rng& rng, size_t& effort,
          RepairWalkScratch& scratch, double noise = 0.75) {
    return repair_walk(engine, max_steps, noise, rng, effort, scratch, Deadline{});
}

}  // namespace

TEST_CASE("repair_walk: repairs a partial assignment the DFS propagated to infeasibility",
          "[repair-walk][fpr]") {
    SwapModel m;
    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine E = m.make_engine();
    E.init_activities();

    // The node: x0 decided to 1, x1 still *unfixed*.  This is the state
    // Fig. 1 line 7 hands to repair, and it is a partial assignment --
    // there is no complete solution vector here to run the leaf-time
    // WalkSAT on.
    REQUIRE(E.fix(0, 1.0));
    REQUIRE(E.propagate(0) == PropResult::kInfeasible);
    REQUIRE_FALSE(E.var(1).fixed);

    RepairWalkScratch scratch;
    Rng rng(11);
    size_t effort = 0;
    REQUIRE(walk(E, /*max_steps=*/200, rng, effort, scratch));
    REQUIRE(effort > 0);

    // x0's singleton domain slid to {0}: the decision was overridden by
    // the repair, which is the point of repairing rather than pruning.
    REQUIRE(E.var(0).fixed);
    REQUIRE(E.var(0).val == Catch::Approx(0.0));
    // x1 was never a legal candidate (its [0, 1] cannot slide down inside
    // [0, 1]) and must therefore be untouched -- not fixed to something
    // convenient, and not widened.
    REQUIRE_FALSE(E.var(1).fixed);
    REQUIRE(E.var(1).lb == Catch::Approx(0.0));
    REQUIRE(E.var(1).ub == Catch::Approx(1.0));
}

TEST_CASE("repair_walk: shifts the node's current domain, never the global bounds",
          "[repair-walk][fpr]") {
    // y in [0, 10] structurally, narrowed to [5, 7] at this node.  The
    // row `y <= 2` wants the activity down to 2, which is a shift of -3.
    OneRowModel m(/*coeff=*/1.0, /*lo=*/-kHighsInf, /*hi=*/2.0, /*structural_lb=*/0.0);
    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine E = m.make_engine();
    E.init_activities();
    narrow(E, 5.0, 7.0);
    REQUIRE_FALSE(E.var(0).fixed);

    RepairWalkScratch scratch;
    Rng rng(3);
    size_t effort = 0;
    REQUIRE(walk(E, /*max_steps=*/200, rng, effort, scratch));

    // The paper's shift: the interval *translates*, keeping its width.
    // A repair that operated on the global bounds instead -- picking a
    // value anywhere in [0, 10], or reopening the domain to it -- would
    // not land on [2, 4], and the footnote rules that out explicitly:
    // domain enlargement "would lead to trivial repair actions where
    // fixings are just undone".
    REQUIRE_FALSE(E.var(0).fixed);
    REQUIRE(E.var(0).lb == Catch::Approx(2.0));
    REQUIRE(E.var(0).ub == Catch::Approx(4.0));
    REQUIRE((E.var(0).ub - E.var(0).lb) == Catch::Approx(2.0));
}

TEST_CASE("repair_walk: a shift is clipped by the structural bounds, not by the row's demand",
          "[repair-walk][fpr]") {
    // Same node domain [5, 7] and same demand for a -3 shift, but the
    // structural lower bound is now 3, so only -2 of it is available.
    // The walk must take the clipped shift ("clip it so that the variable
    // is still within its global bounds") and report the node still
    // infeasible, rather than either overshooting the structural bound or
    // refusing the partial improvement.
    OneRowModel m(/*coeff=*/1.0, /*lo=*/-kHighsInf, /*hi=*/2.0, /*structural_lb=*/3.0);
    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine E = m.make_engine();
    E.init_activities();
    narrow(E, 5.0, 7.0);

    RepairWalkScratch scratch;
    Rng rng(5);
    size_t effort = 0;
    REQUIRE_FALSE(walk(E, /*max_steps=*/200, rng, effort, scratch));

    REQUIRE(E.var(0).lb == Catch::Approx(3.0));
    REQUIRE(E.var(0).ub == Catch::Approx(5.0));
}

TEST_CASE("repair_walk: an undershooting row rounds the shift the other way",
          "[repair-walk][fpr]") {
    // `3y >= 8` with y narrowed to [1, 2]: the activity range [3, 6] sits
    // *below* the row, so the minimal shift is 2/3 of a unit and must
    // round **up** to 1.  Rounding it the way an overshooting row rounds
    // (down, to 0) yields no move at all and the node stays infeasible.
    // Every other case in this file is an overshoot with a positive
    // coefficient, i.e. one of the four (side, sign) combinations.
    OneRowModel m(/*coeff=*/3.0, /*lo=*/8.0, /*hi=*/kHighsInf, /*structural_lb=*/0.0);
    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine E = m.make_engine();
    E.init_activities();
    narrow(E, 1.0, 2.0);

    RepairWalkScratch scratch;
    Rng rng(13);
    size_t effort = 0;
    REQUIRE(walk(E, /*max_steps=*/200, rng, effort, scratch));

    REQUIRE(E.var(0).lb == Catch::Approx(2.0));
    REQUIRE(E.var(0).ub == Catch::Approx(3.0));
}

TEST_CASE("repair_walk: a negative coefficient flips which way the activity moves",
          "[repair-walk][fpr]") {
    // `-2y <= -9` with y narrowed to [1, 3]: the activity range is
    // [-6, -2], which overshoots the row's upper bound of -9 from above,
    // and the shift that fixes it is *positive* because the coefficient is
    // negative.  1.5 units of it, rounding up to 2.
    OneRowModel m(/*coeff=*/-2.0, /*lo=*/-kHighsInf, /*hi=*/-9.0, /*structural_lb=*/0.0);
    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine E = m.make_engine();
    E.init_activities();
    narrow(E, 1.0, 3.0);

    RepairWalkScratch scratch;
    Rng rng(17);
    size_t effort = 0;
    REQUIRE(walk(E, /*max_steps=*/200, rng, effort, scratch));

    REQUIRE(E.var(0).lb == Catch::Approx(3.0));
    REQUIRE(E.var(0).ub == Catch::Approx(5.0));
}

TEST_CASE("repair_walk: a zero step limit leaves the node exactly as it found it",
          "[repair-walk][fpr]") {
    SwapModel m;
    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine E = m.make_engine();
    E.init_activities();
    REQUIRE(E.fix(0, 1.0));
    REQUIRE(E.propagate(0) == PropResult::kInfeasible);

    RepairWalkScratch scratch;
    Rng rng(7);
    size_t effort = 0;
    REQUIRE_FALSE(walk(E, /*max_steps=*/0, rng, effort, scratch));
    REQUIRE(E.var(0).fixed);
    REQUIRE(E.var(0).val == Catch::Approx(1.0));
}

namespace {

// p + q <= 2, q >= 1, p >= 4, both columns general integers on [0, 10].
// With p and q both narrowed to [3, 4] the first row is violated by 4 and
// both columns can slide -3 to help it, but only q can do so for free:
// p's slide breaks `p >= 4` by 3, while q's leaves `q >= 1` satisfied.
// So the damage rule alone decides the move -- and it decides *against*
// the candidate the row enumerates first, so taking `cand[0]` is wrong
// here as well as scoring every candidate the same.
struct DamageModel {
    static constexpr HighsInt kNcol = 2;
    static constexpr HighsInt kNrow = 3;
    std::vector<HighsInt> ar_start = {0, 2, 3, 4};
    std::vector<HighsInt> ar_index = {0, 1, 1, 0};
    std::vector<double> ar_value = {1.0, 1.0, 1.0, 1.0};
    std::vector<double> col_lb = {0.0, 0.0};
    std::vector<double> col_ub = {10.0, 10.0};
    std::array<double, 3> row_lo = {-kHighsInf, 1.0, 4.0};
    std::array<double, 3> row_hi = {2.0, kHighsInf, kHighsInf};
    std::vector<HighsVarType> integrality = {HighsVarType::kInteger, HighsVarType::kInteger};
    CscMatrix csc;

    DamageModel() { csc = build_csc(kNcol, kNrow, ar_start, ar_index, ar_value); }

    PropEngine make_engine(double feastol = 1e-6) {
        return {kNcol,           kNrow,         ar_start.data(),    ar_index.data(),
                ar_value.data(), csc,           col_lb.data(),      col_ub.data(),
                row_lo.data(),   row_hi.data(), integrality.data(), feastol};
    }
};

}  // namespace

TEST_CASE("repair_walk: damage picks the move, not arrival order", "[repair-walk][fpr]") {
    // Swept over seeds rather than run once, and that is the assertion
    // doing the work: with the damage rule intact the choice is
    // deterministic (one strictly minimal candidate, taken greedily
    // because it costs zero), so every seed must give the same answer.
    // Score both candidates alike and the choice becomes a coin flip
    // instead -- which a single seed would report as a pass half the time.
    for (const uint64_t seed : {23U, 5U, 7U, 11U, 13U, 17U, 19U, 29U}) {
        DamageModel m;
        // NOLINTNEXTLINE(readability-identifier-naming)
        PropEngine E = m.make_engine();
        E.init_activities();
        for (HighsInt j = 0; j < 2; ++j) {
            REQUIRE(E.tighten_lb(j, 3.0));
            REQUIRE(E.tighten_ub(j, 4.0));
        }

        RepairWalkScratch scratch;
        Rng rng(seed);
        size_t effort = 0;
        // Exactly one step, so the assertion is about *this* move and not
        // about wherever a longer walk drifts to.
        static_cast<void>(walk(E, /*max_steps=*/1, rng, effort, scratch));

        // q slid; p did not.  Taking the row's first candidate would pick
        // p on every seed; scoring both at zero damage would pick p on
        // some of them.
        INFO("seed " << seed);
        REQUIRE(E.var(1).lb == Catch::Approx(0.0));
        REQUIRE(E.var(1).ub == Catch::Approx(1.0));
        REQUIRE(E.var(0).lb == Catch::Approx(3.0));
        REQUIRE(E.var(0).ub == Catch::Approx(4.0));
    }
}

namespace {

// `y <= 2` and `y >= 5` over one column narrowed to [5, 7]: the first row
// is violated, and the only shift that helps it breaks the second.  A walk
// with no tabu list oscillates between the two for its whole step limit;
// the paper's three-shift list stops it after one move.
struct OscillateModel {
    static constexpr HighsInt kNcol = 1;
    static constexpr HighsInt kNrow = 2;
    std::vector<HighsInt> ar_start = {0, 1, 2};
    std::vector<HighsInt> ar_index = {0, 0};
    std::vector<double> ar_value = {1.0, 1.0};
    std::vector<double> col_lb = {0.0};
    std::vector<double> col_ub = {10.0};
    std::array<double, 2> row_lo = {-kHighsInf, 5.0};
    std::array<double, 2> row_hi = {2.0, kHighsInf};
    std::vector<HighsVarType> integrality = {HighsVarType::kInteger};
    CscMatrix csc;

    OscillateModel() { csc = build_csc(kNcol, kNrow, ar_start, ar_index, ar_value); }

    PropEngine make_engine(double feastol = 1e-6) {
        return {kNcol,           kNrow,         ar_start.data(),    ar_index.data(),
                ar_value.data(), csc,           col_lb.data(),      col_ub.data(),
                row_lo.data(),   row_hi.data(), integrality.data(), feastol};
    }
};

}  // namespace

TEST_CASE("repair_walk: the tabu list stops a two-move cycle after one move",
          "[repair-walk][fpr]") {
    OscillateModel m;
    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine E = m.make_engine();
    E.init_activities();
    narrow(E, 5.0, 7.0);

    RepairWalkScratch scratch;
    Rng rng(29);
    size_t effort = 0;
    REQUIRE_FALSE(walk(E, /*max_steps=*/200, rng, effort, scratch));

    // One shift, then the column is tabu and no candidate remains.  The
    // effort bound is the assertion that matters: without the tabu list
    // this model shifts on every one of the 200 steps, each paying a row
    // scan plus two column walks, which is an order of magnitude more
    // than the 200 bare row scans a tabu'd walk pays.
    REQUIRE(E.var(0).lb == Catch::Approx(2.0));
    REQUIRE(E.var(0).ub == Catch::Approx(4.0));
    REQUIRE(effort < 500);
}

namespace {

// `y <= 2` and `10y >= 50` over one column narrowed to [5, 7].  The second
// row is satisfied where the walk starts and badly violated after the only
// move that helps the first, so the walk's single shift makes the total
// violation worse and the best state seen is the one it started in.
struct WorseningModel {
    static constexpr HighsInt kNcol = 1;
    static constexpr HighsInt kNrow = 2;
    std::vector<HighsInt> ar_start = {0, 1, 2};
    std::vector<HighsInt> ar_index = {0, 0};
    std::vector<double> ar_value = {1.0, 10.0};
    std::vector<double> col_lb = {0.0};
    std::vector<double> col_ub = {10.0};
    std::array<double, 2> row_lo = {-kHighsInf, 50.0};
    std::array<double, 2> row_hi = {2.0, kHighsInf};
    std::vector<HighsVarType> integrality = {HighsVarType::kInteger};
    CscMatrix csc;

    WorseningModel() { csc = build_csc(kNcol, kNrow, ar_start, ar_index, ar_value); }

    PropEngine make_engine(double feastol = 1e-6) {
        return {kNcol,           kNrow,         ar_start.data(),    ar_index.data(),
                ar_value.data(), csc,           col_lb.data(),      col_ub.data(),
                row_lo.data(),   row_hi.data(), integrality.data(), feastol};
    }
};

}  // namespace

TEST_CASE("repair_walk: a failed walk leaves the node in the best state it saw",
          "[repair-walk][fpr]") {
    WorseningModel m;
    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine E = m.make_engine();
    E.init_activities();
    narrow(E, 5.0, 7.0);

    RepairWalkScratch scratch;
    Rng rng(31);
    size_t effort = 0;
    REQUIRE_FALSE(walk(E, /*max_steps=*/200, rng, effort, scratch));

    // The walk shifted to [2, 4], which fixes `y <= 2` and breaks
    // `10y >= 50` by more, then had nothing left to try.  Total violation
    // 3 -> 10, so the state it must hand back is the one it entered with.
    // A non-backtracking mode carries whatever this leaves into the rest
    // of the dive, so a walk that could only make things worse must be
    // able to make things unchanged.
    REQUIRE(E.var(0).lb == Catch::Approx(5.0));
    REQUIRE(E.var(0).ub == Catch::Approx(7.0));
}

TEST_CASE("repair_walk: a backtrack past a repair restores every engine array",
          "[repair-walk][fpr][backtrack]") {
    // The in-tree repair is only safe because every change it makes goes
    // through `E`.  A backtracking mode (`dfsrep`) repairs at a node and
    // then backtracks to a sibling whose marks predate the repair; if any
    // one of the engine's four undo trails misses the shift, the state
    // that comes back is a hybrid.  The solution trail is the dangerous
    // one: `fpr_attempt_finish` copies `E.sol_data()` wholesale and Phase
    // 2.5 skips fixed columns, so a stale value on a column no later node
    // re-fixes is emitted verbatim.
    SwapModel m;
    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine E = m.make_engine();
    E.init_activities();
    E.init_domain_pq();
    REQUIRE(E.fix(0, 1.0));
    REQUIRE(E.propagate(0) == PropResult::kInfeasible);

    const HighsInt vs_m = E.vs_mark();
    const HighsInt sol_m = E.sol_mark();
    const HighsInt act_m = E.act_mark();
    const HighsInt pq_m = E.pq_mark();
    const double min_act_before = E.row_min_activity(1);
    REQUIRE(min_act_before == Catch::Approx(3.0));
    REQUIRE(E.sol(0) == Catch::Approx(1.0));
    REQUIRE(E.pq_top() == 1);

    RepairWalkScratch scratch;
    Rng rng(37);
    size_t effort = 0;
    REQUIRE(walk(E, /*max_steps=*/200, rng, effort, scratch));
    REQUIRE(E.var(0).val == Catch::Approx(0.0));
    REQUIRE(E.sol(0) == Catch::Approx(0.0));
    REQUIRE(E.row_min_activity(1) == Catch::Approx(0.0));

    E.backtrack_to(vs_m, sol_m, act_m, pq_m);

    // Variable state, solution, activities and the domain queue all back
    // where the node found them.
    REQUIRE(E.var(0).fixed);
    REQUIRE(E.var(0).val == Catch::Approx(1.0));
    REQUIRE(E.var(0).lb == Catch::Approx(0.0));
    REQUIRE(E.var(0).ub == Catch::Approx(1.0));
    REQUIRE(E.sol(0) == Catch::Approx(1.0));
    REQUIRE(E.row_min_activity(1) == Catch::Approx(min_act_before));
    REQUIRE_FALSE(E.var(1).fixed);
    REQUIRE(E.pq_top() == 1);
}

// ===================================================================
// The in-tree call site (Fig. 1 lines 4-8) and the branching that
// follows it (lines 9-18).
// ===================================================================

namespace {

// Mirrors `build_bare_mipsolver` (test_common.h) minus the `readModel`:
// these cases need models of an exact shape that no bundled instance is
// guaranteed to contain, so `configure` builds one directly through
// `Highs::addVar`/`addRow` and this stands a `HighsMipSolver` on it.
template <typename Configure>
std::unique_ptr<HighsMipSolver> bare_mipsolver_on(Highs& highs, HighsCallback& cb,
                                                  Configure&& configure) {
    configure(highs);
    // `Highs::addRow` leaves the matrix row-wise; round-trip through
    // `passModel`, whose master overload calls `ensureColwise()`.
    REQUIRE(highs.passModel(highs.getLp()) == HighsStatus::kOk);
    highs.setOptionValue("presolve", "off");
    require_option(highs, "time_limit", kHighsInf);
    auto mipsolver = std::make_unique<HighsMipSolver>(cb, highs.getOptions(), highs.getLp(),
                                                      highs.getSolution());
    mipsolver->timer_.start();
    mipsolver->improving_solution_file_ = nullptr;
    mipsolver->mipdata_ = std::make_unique<HighsMipSolverData>(*mipsolver);
    mipsolver->mipdata_->init();
    mipsolver->mipdata_->runMipPresolve(mipsolver->options_mip_->presolve_reduction_limit);
    mipsolver->mipdata_->runSetup();
    return mipsolver;
}

// The `SwapModel` rows again, this time as a real MIP so the whole
// begin/step/finish lifecycle runs on them: x0 + x1 >= 1 and
// 3*x0 + x1 <= 2, both binary, zero objective (so `greedy_1opt` cannot
// move the answer afterwards).
void build_swap_mip(Highs& highs) {
    highs.addVar(0.0, 1.0);
    highs.addVar(0.0, 1.0);
    highs.changeColIntegrality(0, HighsVarType::kInteger);
    highs.changeColIntegrality(1, HighsVarType::kInteger);
    const auto idx = std::to_array<HighsInt>({0, 1});
    const auto ge = std::to_array<double>({1.0, 1.0});
    highs.addRow(1.0, kHighsInf, 2, idx.data(), ge.data());
    const auto le = std::to_array<double>({3.0, 1.0});
    highs.addRow(-kHighsInf, 2.0, 2, idx.data(), le.data());
}

// Formulation order plus "always the upper bound", so the dive's first
// decision is forced to be x0 = 1 -- the decision that refutes the node.
// Both columns carry an up-lock and a down-lock (each appears in a `>=`
// and a `<=` row), so Phase 1's trivially-roundable pass fixes neither.
constexpr FprStrategyConfig kForcedUpLr{VarStrategy::kLR, ValStrategy::kUp};

struct SwapHarness {
    Highs highs;
    HighsCallback cb{&highs};
    std::unique_ptr<HighsMipSolver> mipsolver;
    CscMatrix csc;
    ProblemView problem;
    FprScratch scratch;
    FprConfig cfg{};

    explicit SwapHarness(FrameworkMode mode) {
        highs::parallel::initialize_scheduler();
        highs.setOptionValue("output_flag", false);
        mipsolver = bare_mipsolver_on(highs, cb, build_swap_mip);
        problem = make_problem(*mipsolver, csc);
        cfg.max_effort = std::numeric_limits<size_t>::max() / 2;
        cfg.csc = &csc;
        cfg.mode = mode;
        cfg.strategy = &kForcedUpLr;
        cfg.binary_mask = problem.binary.data();
        cfg.scratch = &scratch;
    }

    // Run the DFS to its verdict, one `step` call per `effort_remaining`
    // slice, and stop before `finish` so a caller can inspect what the
    // dive itself decided.
    void dive(Rng& rng, size_t slice = std::numeric_limits<size_t>::max() / 4) {
        fpr_attempt_begin(state, *mipsolver, cfg, rng, /*attempt_idx=*/0);
        int guard = 0;
        while (state.phase == FprAttemptState::Phase::kDfs && guard++ < 100) {
            static_cast<void>(fpr_attempt_step(state, *mipsolver, cfg, rng, slice));
        }
        REQUIRE(guard < 100);
    }

    FprAttemptState state;
};

}  // namespace

TEST_CASE("FPR diveprop: a mid-dive propagation failure is repaired, not fatal (#124)",
          "[repair-walk][fpr][diveprop]") {
    SwapHarness h(FrameworkMode::kDiveprop);
    Rng rng(1);

    // Driven through the lifecycle rather than the one-shot wrapper so
    // the assertion can land *between* the dive and Phase 3.  That
    // placement is the whole point: at the leaf, the pre-#124 code and
    // this one both end up reporting a feasible solution on this model,
    // because `fpr_attempt_finish`'s complete-assignment WalkSAT flips
    // x0 back down afterwards.  Deleting the in-tree call and asserting
    // only on `HeuristicResult` is therefore green either way -- the
    // abandoned variant covering for the missing one, which is exactly
    // the confusion issue #124 is about.  What only the in-tree repair
    // can produce is a dive that reaches its leaf with the *domain*
    // already feasible.
    h.dive(rng);

    // `diveprop` is the paper's best-performing parametrization and does
    // not backtrack, so before #124 this dive ended at its first refuted
    // node with an empty stack and `found_complete` false -- exactly
    // where the paper's diveprop begins repairing.
    REQUIRE(h.state.found_complete);
    // x0 was decided to 1 and refuted; the repair slid its singleton
    // domain to 0 in the tree, and the dive then carried on and fixed x1.
    REQUIRE(h.scratch.prop_engine->var(0).fixed);
    REQUIRE(h.scratch.prop_engine->var(0).val == Catch::Approx(0.0));
    REQUIRE(h.scratch.prop_engine->var(1).fixed);
    REQUIRE(h.scratch.prop_engine->var(1).val == Catch::Approx(1.0));

    const HeuristicResult result = fpr_attempt_finish(h.state, *h.mipsolver, h.cfg, rng);
    REQUIRE(result.found_feasible);
    REQUIRE(result.solution.size() == 2);
    REQUIRE(result.solution[0] == Catch::Approx(0.0));
    REQUIRE(result.solution[1] == Catch::Approx(1.0));
}

TEST_CASE("FPR dive: a fixing that violates a row is repaired, not dived past (#124)",
          "[repair-walk][fpr][dive]") {
    // `dive` is the mode the paper describes as "an incremental repair
    // strategy that constructs a complete solution in a single big dive",
    // and repair is the whole of it: propagation is off, so nothing else
    // can ever report an infeasible node.  `PropEngine::fix` answers only
    // "is the value inside this column's own domain", never "does a row
    // still admit a completion", so without the activity half of
    // `Apply(fixing, P)` this mode fixes blindly to the bottom and hands
    // an untouched violated assignment to the leaf-time WalkSAT.
    SwapHarness h(FrameworkMode::kDive);
    Rng rng(2);
    h.dive(rng);

    REQUIRE(h.state.found_complete);
    REQUIRE(h.scratch.prop_engine->var(0).val == Catch::Approx(0.0));
    REQUIRE(h.scratch.prop_engine->var(1).val == Catch::Approx(1.0));

    const HeuristicResult result = fpr_attempt_finish(h.state, *h.mipsolver, h.cfg, rng);
    REQUIRE(result.found_feasible);
    REQUIRE(result.solution[0] == Catch::Approx(0.0));
    REQUIRE(result.solution[1] == Catch::Approx(1.0));
}

TEST_CASE("FPR dfsrep: repair runs in a backtracking mode too (#124)",
          "[repair-walk][fpr][dfsrep]") {
    // `dfsrep` is `dfs` plus repair, and before #124 that was a difference
    // only at the leaf.  Here the repair fires at the same refuted node as
    // in `diveprop`, on a mode whose stack carries an untaken sibling
    // underneath it -- the marks that sibling holds predate the repair,
    // which is the configuration the backtrack unit case above pins.
    SwapHarness h(FrameworkMode::kDfsrep);
    Rng rng(3);
    h.dive(rng);

    REQUIRE(h.state.found_complete);
    REQUIRE(h.scratch.prop_engine->var(0).val == Catch::Approx(0.0));

    const HeuristicResult result = fpr_attempt_finish(h.state, *h.mipsolver, h.cfg, rng);
    REQUIRE(result.found_feasible);
    REQUIRE(result.solution[0] == Catch::Approx(0.0));
    REQUIRE(result.solution[1] == Catch::Approx(1.0));
}

TEST_CASE("FPR diveprop: an unrepaired node keeps diving instead of ending the attempt (#124)",
          "[repair-walk][fpr][diveprop]") {
    SwapHarness h(FrameworkMode::kDiveprop);
    // Zero repair steps: the walk still measures the node but can change
    // nothing, so the node stays infeasible.  This isolates Fig. 1 lines
    // 9-18 from line 8 -- with `backtrackOnInfeas` off, an infeasible
    // node must still `Branch` and carry the dive to the bottom.
    h.cfg.walksat_iterations = 0;
    Rng rng(1);
    h.dive(rng);

    // The dive branched past the refuted node and processed the next one.
    // Pruning instead -- what the pre-#124 bare `continue` did -- empties
    // the stack of a non-backtracking mode, so the attempt ends at the
    // refuted node having visited exactly one.
    REQUIRE(h.state.nodes_visited >= 2);
    // It reached the bottom with every integer fixed, which is all
    // `found_complete` claims, so `finish` runs its normal path -- but
    // with the repair budget at zero the leaf-time walk cannot help
    // either, and the row re-check rejects the assignment.
    REQUIRE(h.state.found_complete);
    Rng finish_rng(1);
    REQUIRE_FALSE(fpr_attempt_finish(h.state, *h.mipsolver, h.cfg, finish_rng).found_feasible);
}

TEST_CASE("FPR diveprop: the in-tree repair's effort reaches the engine's counter (#124)",
          "[repair-walk][fpr][diveprop]") {
    // The DFS budget gate, `state.effort_consumed` and every ledger
    // number downstream read `PropEngine::effort()`.  A repair that did
    // its work and reported it anywhere else would be invisible to all of
    // them -- unbounded search that no gate can see.
    //
    // Both runs execute exactly one node, and the repair happens after
    // that node's `Apply` and propagation, so the two differ by the walk's
    // own effort and by nothing else.
    auto effort_of_first_node = [](HighsInt steps) {
        SwapHarness h(FrameworkMode::kDiveprop);
        h.cfg.walksat_iterations = steps;
        Rng rng(1);
        fpr_attempt_begin(h.state, *h.mipsolver, h.cfg, rng, /*attempt_idx=*/0);
        REQUIRE(h.state.phase == FprAttemptState::Phase::kDfs);
        // A one-unit slice runs exactly one node: the gate is a delta from
        // the call's start, so it admits the first node and no more.
        static_cast<void>(fpr_attempt_step(h.state, *h.mipsolver, h.cfg, rng,
                                           /*effort_remaining=*/1));
        return h.state.effort_consumed;
    };

    REQUIRE(effort_of_first_node(200) > effort_of_first_node(0));
}

TEST_CASE("FPR diveprop: an in-tree repair leaves the attempt resumable (#77 x #124)",
          "[repair-walk][fpr][diveprop][resume]") {
    SwapHarness h(FrameworkMode::kDiveprop);
    Rng rng(1);
    // One node per call: the repair runs inside a `step` that is then
    // paused and re-entered, so anything it left half-applied to the DFS
    // stack or the engine's undo trail shows up as a wrong verdict here.
    h.dive(rng, /*slice=*/1);

    // Same mid-dive observation as the diveprop case above, for the same
    // reason: the leaf-time WalkSAT would otherwise cover for a repair
    // that never ran.
    REQUIRE(h.state.found_complete);
    REQUIRE(h.scratch.prop_engine->var(0).val == Catch::Approx(0.0));

    const HeuristicResult sliced = fpr_attempt_finish(h.state, *h.mipsolver, h.cfg, rng);
    REQUIRE(sliced.found_feasible);
    REQUIRE(sliced.solution.size() == 2);
    REQUIRE(sliced.solution[0] == Catch::Approx(0.0));
    REQUIRE(sliced.solution[1] == Catch::Approx(1.0));
}

namespace {

// A model whose propagation runs out of its per-call matrix-access budget
// (`kPropagateBudgetPerNnz * nnz`) at the dive's only node, with no row
// violated anywhere.
//
//   R0: x - 1.01*y + z >= 0
//   R1: y - 1.01*x     >= 0
//   R2:              z <= 1
//
// x and y are continuous on [0, 1e19].  R0/R1 shrink each other's upper
// bound by a factor 1.01 per visit and re-seed each other, so reaching a
// fixpoint takes thousands of row visits against a budget of a few
// hundred -- the same geometric-decay shape `tests/test_prop_engine.cpp`
// uses, slowed from halving to 1.01 because that file's 1e300 is not
// available here: `HighsLpUtils::assessBounds` rewrites any bound at or
// beyond the `infinite_bound` option (default 1e20) to infinite, and an
// infinite domain gives the cascade nothing to shrink.
//
// R2 exists only to give z an up-lock beside R0's down-lock, so Phase 1's
// trivially-roundable pass leaves it for the DFS to decide.  It is
// satisfied at z = 1, and so is R0, so `Apply` reports nothing and the
// node's only verdict comes from propagation -- which truncates.
void build_budget_burner(Highs& highs) {
    highs.addVar(0.0, 1e19);
    highs.addVar(0.0, 1e19);
    highs.addVar(0.0, 1.0);
    highs.changeColIntegrality(2, HighsVarType::kInteger);
    const auto r0_idx = std::to_array<HighsInt>({0, 1, 2});
    const auto r0_val = std::to_array<double>({1.0, -1.01, 1.0});
    highs.addRow(0.0, kHighsInf, 3, r0_idx.data(), r0_val.data());
    const auto r1_idx = std::to_array<HighsInt>({0, 1});
    const auto r1_val = std::to_array<double>({-1.01, 1.0});
    highs.addRow(0.0, kHighsInf, 2, r1_idx.data(), r1_val.data());
    const auto r2_idx = std::to_array<HighsInt>({2});
    const auto r2_val = std::to_array<double>({1.0});
    highs.addRow(-kHighsInf, 1.0, 1, r2_idx.data(), r2_val.data());
}

}  // namespace

TEST_CASE("FPR diveprop: a budget-exhausted fixpoint does not trigger repair (#124 x #127)",
          "[repair-walk][fpr][diveprop][budget]") {
    highs::parallel::initialize_scheduler();
    Highs highs;
    highs.setOptionValue("output_flag", false);
    HighsCallback cb(&highs);
    auto mipsolver = bare_mipsolver_on(highs, cb, build_budget_burner);

    CscMatrix csc;
    const ProblemView problem = make_problem(*mipsolver, csc);

    // Premise, asserted rather than assumed: fixing z = 1 and propagating
    // truncates on the work budget.  Anything else (a fixpoint, a
    // refutation) and the case below tests nothing.  `ref_effort` is what
    // that node costs when nothing but propagation runs on it.
    size_t ref_effort = 0;
    {
        // NOLINTNEXTLINE(readability-identifier-naming)
        PropEngine E(problem.ncol, problem.nrow, problem.mipdata->ARstart_.data(),
                     problem.mipdata->ARindex_.data(), problem.mipdata->ARvalue_.data(), csc,
                     problem.model->col_lower_.data(), problem.model->col_upper_.data(),
                     problem.model->row_lower_.data(), problem.model->row_upper_.data(),
                     problem.model->integrality_.data(), problem.mipdata->feastol);
        E.init_activities();
        REQUIRE(E.fix(2, 1.0));
        REQUIRE(E.propagate(2) == PropResult::kBudgetExhausted);
        ref_effort = E.effort();
    }

    FprScratch scratch;
    FprConfig cfg{};
    cfg.max_effort = std::numeric_limits<size_t>::max() / 2;
    cfg.csc = &csc;
    cfg.mode = FrameworkMode::kDiveprop;
    cfg.strategy = &kForcedUpLr;
    cfg.binary_mask = problem.binary.data();
    cfg.scratch = &scratch;
    Rng rng(1);

    FprAttemptState state;
    fpr_attempt_begin(state, *mipsolver, cfg, rng, /*attempt_idx=*/0);
    while (state.phase == FprAttemptState::Phase::kDfs) {
        static_cast<void>(fpr_attempt_step(state, *mipsolver, cfg, rng, cfg.max_effort));
    }

    // The node cost exactly propagation plus `Apply`'s scan of z's own
    // column, and nothing else ran on it.  A trigger widened from
    // `pr == kInfeasible` to anything weaker (`pr != kFixpoint`) would
    // have called `repair_walk`, whose entry scan alone charges one unit
    // per row before it can discover there is nothing to repair.
    const auto apply_charge = static_cast<size_t>(csc.col_start[3] - csc.col_start[2]);
    REQUIRE(state.effort_consumed == ref_effort + apply_charge);
}

namespace {

// u + v >= 3 with u, v binary: unsatisfiable, and unsatisfiable in the
// activity sense from the first fixing onwards, so every node of the dive
// is refuted and no shift can help (both columns want to move *up* and
// both are already at their structural upper bound).  The second row is
// there only to give both columns an up-lock beside the down-lock, so
// Phase 1's trivially-roundable pass leaves them for the DFS.
void build_unreachable_leaf_mip(Highs& highs) {
    highs.addVar(0.0, 1.0);
    highs.addVar(0.0, 1.0);
    highs.changeColIntegrality(0, HighsVarType::kInteger);
    highs.changeColIntegrality(1, HighsVarType::kInteger);
    const auto idx = std::to_array<HighsInt>({0, 1});
    const auto val = std::to_array<double>({1.0, 1.0});
    highs.addRow(3.0, kHighsInf, 2, idx.data(), val.data());
    highs.addRow(-kHighsInf, 5.0, 2, idx.data(), val.data());
}

}  // namespace

TEST_CASE("FPR dive: a dive refuted at its own leaf still runs Phase 2.5 and Phase 3 (#124)",
          "[repair-walk][fpr][dive]") {
    // Before the activity half of `Apply` existed, `dive` had no way to
    // set `infeas` at all, so every dive reached its leaf, set
    // `found_complete`, and got the Phase 2.5 fill plus the leaf-time
    // `walksat_repair` and `greedy_1opt`.  That leaf walk was the only
    // repair `dive` had, and it is what every recorded benchmark number
    // was measured with.  It works on a *point* rather than on activity
    // ranges and starts from its own RNG stream, so losing it would be a
    // real loss and not a de-duplication -- and the verdict deciding it
    // would be arbitrary, since `any_violated_row_in_column` scans only
    // the last-fixed column's rows.
    highs::parallel::initialize_scheduler();
    Highs highs;
    highs.setOptionValue("output_flag", false);
    HighsCallback cb(&highs);
    auto mipsolver = bare_mipsolver_on(highs, cb, build_unreachable_leaf_mip);

    CscMatrix csc;
    const ProblemView problem = make_problem(*mipsolver, csc);
    FprScratch scratch;
    FprConfig cfg{};
    cfg.max_effort = std::numeric_limits<size_t>::max() / 2;
    cfg.csc = &csc;
    cfg.mode = FrameworkMode::kDive;
    cfg.strategy = &kForcedUpLr;
    cfg.binary_mask = problem.binary.data();
    cfg.scratch = &scratch;
    Rng rng(41);

    FprAttemptState state;
    fpr_attempt_begin(state, *mipsolver, cfg, rng, /*attempt_idx=*/0);
    while (state.phase == FprAttemptState::Phase::kDfs) {
        static_cast<void>(fpr_attempt_step(state, *mipsolver, cfg, rng, cfg.max_effort));
    }

    // The dive was refuted at every node, including its last one, and no
    // shift was available anywhere -- yet it reached the bottom with every
    // integer fixed, which is all `found_complete` claims.
    REQUIRE(state.nodes_visited >= 2);
    REQUIRE(state.found_complete);

    const size_t dive_effort = state.effort_consumed;
    const HeuristicResult result = fpr_attempt_finish(state, *mipsolver, cfg, rng);

    // This model has no feasible solution, so the verdict is infeasible
    // either way; what distinguishes the two is whether `finish` did the
    // leaf's work before saying so.  Since #155 the `!found_complete`
    // path fills and rebuilds too, so what this pins is narrower than it
    // once was -- `found_complete` here buys Phase 3 on top of that, and
    // the `dive` case's leaf-time WalkSAT is the only repair `dive` has.
    REQUIRE_FALSE(result.found_feasible);
    REQUIRE(result.effort > dive_effort);
}

namespace {

// A model where propagation, and only propagation, refutes a node --
// which is what `Apply`'s arrival made hard to arrange, since for a
// *single* row the activity test and propagation's own bound derivation
// are the same test.  Refuting through propagation alone therefore needs
// a cascade across two rows:
//
//   R0: x0 + y <= 1     R1: y + w >= 2     R2: z <= 1
//   R3: x0 + z >= 1     R4: w <= 1
//
// Fixing x0 = 1 leaves both of x0's own rows satisfiable -- R0's range is
// [1, 2] against `hi = 1`, R3's is [1, 2] against `lo = 1` -- so `Apply`
// reports nothing.  Propagation then tightens y to 0 through R0, and R1
// derives an empty domain for w from that.  R2 and R4 exist only to give
// z and w an up-lock apiece so Phase 1 leaves all four columns alone.
void build_prop_refute_mip(Highs& highs) {
    for (HighsInt j = 0; j < 4; ++j) {
        highs.addVar(0.0, 1.0);
        highs.changeColIntegrality(j, HighsVarType::kInteger);
    }
    const auto ones = std::to_array<double>({1.0, 1.0});
    const auto one = std::to_array<double>({1.0});
    const auto r0_idx = std::to_array<HighsInt>({0, 1});
    highs.addRow(-kHighsInf, 1.0, 2, r0_idx.data(), ones.data());
    const auto r1_idx = std::to_array<HighsInt>({1, 3});
    highs.addRow(2.0, kHighsInf, 2, r1_idx.data(), ones.data());
    const auto r2_idx = std::to_array<HighsInt>({2});
    highs.addRow(-kHighsInf, 1.0, 1, r2_idx.data(), one.data());
    const auto r3_idx = std::to_array<HighsInt>({0, 2});
    highs.addRow(1.0, kHighsInf, 2, r3_idx.data(), ones.data());
    const auto r4_idx = std::to_array<HighsInt>({3});
    highs.addRow(-kHighsInf, 1.0, 1, r4_idx.data(), one.data());
}

struct PropRefuteHarness {
    Highs highs;
    HighsCallback cb{&highs};
    std::unique_ptr<HighsMipSolver> mipsolver;
    CscMatrix csc;
    ProblemView problem;
    FprScratch scratch;
    FprConfig cfg{};
    FprAttemptState state;

    PropRefuteHarness() {
        highs::parallel::initialize_scheduler();
        highs.setOptionValue("output_flag", false);
        mipsolver = bare_mipsolver_on(highs, cb, build_prop_refute_mip);
        problem = make_problem(*mipsolver, csc);
        cfg.max_effort = std::numeric_limits<size_t>::max() / 2;
        cfg.csc = &csc;
        cfg.mode = FrameworkMode::kDiveprop;
        cfg.strategy = &kForcedUpLr;
        cfg.binary_mask = problem.binary.data();
        cfg.scratch = &scratch;
    }
};

}  // namespace

TEST_CASE("FPR diveprop: a propagation-only refutation triggers the repair (#124 x #127)",
          "[repair-walk][fpr][diveprop]") {
    PropRefuteHarness h;

    // Premise, asserted rather than assumed: `Apply` is silent on this
    // node and propagation is what refutes it.  Without it the case would
    // be a third copy of the `Apply` path, and deleting
    // `infeas = pr == PropResult::kInfeasible` would be unopposed in the
    // file that owns the behaviour.
    {
        // NOLINTNEXTLINE(readability-identifier-naming)
        PropEngine E(h.problem.ncol, h.problem.nrow, h.problem.mipdata->ARstart_.data(),
                     h.problem.mipdata->ARindex_.data(), h.problem.mipdata->ARvalue_.data(), h.csc,
                     h.problem.model->col_lower_.data(), h.problem.model->col_upper_.data(),
                     h.problem.model->row_lower_.data(), h.problem.model->row_upper_.data(),
                     h.problem.model->integrality_.data(), h.problem.mipdata->feastol);
        E.init_activities();
        REQUIRE(E.fix(0, 1.0));
        size_t probe_effort = 0;
        REQUIRE_FALSE(any_violated_row_in_column(E, 0, probe_effort));
        REQUIRE(E.propagate(0) == PropResult::kInfeasible);
    }

    Rng rng(43);
    fpr_attempt_begin(h.state, *h.mipsolver, h.cfg, rng, /*attempt_idx=*/0);
    while (h.state.phase == FprAttemptState::Phase::kDfs) {
        static_cast<void>(fpr_attempt_step(h.state, *h.mipsolver, h.cfg, rng, h.cfg.max_effort));
    }

    REQUIRE(h.state.found_complete);
    // The repair undid the decision propagation refuted and lifted y with
    // it, and the dive then completed feasibly on top of that.
    REQUIRE(h.scratch.prop_engine->var(0).val == Catch::Approx(0.0));
    REQUIRE(h.scratch.prop_engine->var(1).val == Catch::Approx(1.0));

    const HeuristicResult result = fpr_attempt_finish(h.state, *h.mipsolver, h.cfg, rng);
    REQUIRE(result.found_feasible);
}

TEST_CASE("FPR diveprop: a child node starts from the repaired state, not the refuted one (#124)",
          "[repair-walk][fpr][diveprop]") {
    // The undo marks a node records for its children are read *after* the
    // in-tree repair.  Read them before it and every descent silently
    // undoes its parent's repair, which defeats most of #124 -- and on a
    // two-column model the repair simply re-fires at the next node and the
    // same answer comes out, so nothing notices.  Here the second node's
    // own column is incident to no violated row, so it runs no repair of
    // its own: whatever state it inherits is the state it leaves.
    PropRefuteHarness h;
    Rng rng(47);

    fpr_attempt_begin(h.state, *h.mipsolver, h.cfg, rng, /*attempt_idx=*/0);
    // One node per `step` call, so the engine can be read between them.
    REQUIRE(h.state.phase == FprAttemptState::Phase::kDfs);
    static_cast<void>(fpr_attempt_step(h.state, *h.mipsolver, h.cfg, rng, /*effort_remaining=*/1));
    REQUIRE(h.scratch.prop_engine->var(0).val == Catch::Approx(0.0));
    REQUIRE(h.scratch.prop_engine->var(1).val == Catch::Approx(1.0));

    REQUIRE(h.state.phase == FprAttemptState::Phase::kDfs);
    static_cast<void>(fpr_attempt_step(h.state, *h.mipsolver, h.cfg, rng, /*effort_remaining=*/1));

    // Node 2 fixed z and nothing else; x0 and y still carry what node 1's
    // repair decided.  Marks taken before the repair would show x0 back at
    // the refuted 1 and y back at propagation's 0.
    REQUIRE(h.scratch.prop_engine->var(2).fixed);
    REQUIRE(h.scratch.prop_engine->var(0).val == Catch::Approx(0.0));
    REQUIRE(h.scratch.prop_engine->var(1).val == Catch::Approx(1.0));
}

namespace {

// Twelve columns, each with its own row `a_i <= 0`, plus one shared row
// `3 * sum(a_i) >= 36`.  Fixed at all-ones the twelve small rows are each
// violated by 1 and the shared row is exactly satisfied, so every repair
// move trades one unit of violation for three: the walk drifts steadily
// away from the state it started in and never finds a better one.  That
// makes it the only model here that both applies more than
// `kSoftRestartPeriod` shifts and has somewhere to be restored to.
struct DriftModel {
    static constexpr HighsInt kNcol = 12;
    static constexpr HighsInt kNrow = kNcol + 1;
    std::vector<HighsInt> ar_start;
    std::vector<HighsInt> ar_index;
    std::vector<double> ar_value;
    std::vector<double> col_lb = std::vector<double>(kNcol, 0.0);
    std::vector<double> col_ub = std::vector<double>(kNcol, 1.0);
    std::vector<double> row_lo;
    std::vector<double> row_hi;
    std::vector<HighsVarType> integrality =
        std::vector<HighsVarType>(kNcol, HighsVarType::kInteger);
    CscMatrix csc;

    DriftModel() {
        ar_start.push_back(0);
        for (HighsInt j = 0; j < kNcol; ++j) {
            ar_index.push_back(j);
            ar_value.push_back(1.0);
            ar_start.push_back(static_cast<HighsInt>(ar_index.size()));
            row_lo.push_back(-kHighsInf);
            row_hi.push_back(0.0);
        }
        for (HighsInt j = 0; j < kNcol; ++j) {
            ar_index.push_back(j);
            ar_value.push_back(3.0);
        }
        ar_start.push_back(static_cast<HighsInt>(ar_index.size()));
        row_lo.push_back(3.0 * kNcol);
        row_hi.push_back(kHighsInf);
        csc = build_csc(kNcol, kNrow, ar_start, ar_index, ar_value);
    }

    PropEngine make_engine(double feastol = 1e-6) {
        return {kNcol,           kNrow,         ar_start.data(),    ar_index.data(),
                ar_value.data(), csc,           col_lb.data(),      col_ub.data(),
                row_lo.data(),   row_hi.data(), integrality.data(), feastol};
    }
};

}  // namespace

TEST_CASE("repair_walk: a long drifting walk is restored to the state it started in",
          "[repair-walk][fpr][restart]") {
    DriftModel m;
    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine E = m.make_engine();
    E.init_activities();
    for (HighsInt j = 0; j < DriftModel::kNcol; ++j) {
        REQUIRE(E.fix(j, 1.0));
    }

    RepairWalkScratch scratch;
    Rng rng(53);
    size_t effort = 0;
    REQUIRE_FALSE(walk(E, /*max_steps=*/200, rng, effort, scratch));

    // Every one of the twelve columns is back where it started.  Reaching
    // this state requires `restore_best` to undo an arbitrary run of
    // shifts and rebuild the violated set from the engine's activities --
    // and this is the one model in the file that exercises the soft
    // restart at all, since the walk applies far more than
    // `kSoftRestartPeriod` shifts before it gives up.
    for (HighsInt j = 0; j < DriftModel::kNcol; ++j) {
        INFO("column " << j);
        REQUIRE(E.var(j).fixed);
        REQUIRE(E.var(j).val == Catch::Approx(1.0));
    }
}

// ===================================================================
// Phase 3 no longer takes an effort cap from its caller (issue #156).
//
// `fpr_attempt_finish` used to size both leaf-time repairs as
// `cfg.max_effort - total_prop_work`, but under the lifecycle API
// `cfg.max_effort` bounds nothing: the DFS gate is the per-call slice and
// an attempt spans calls, so `PropEngine::effort()` outgrows it and the
// subtraction arrives as 0 -- a Phase 3 that returns without a step while
// still paying its entry scan, on precisely the long attempts on hard
// models it exists for.  #124 had already removed the same caller-supplied
// cap from the in-tree `repair_walk` above; these cases are the leaf-time
// half of that.
// ===================================================================

namespace {

// One binary x0 carrying both an up- and a down-lock row, plus two
// zero-cost continuous columns c1, c2 in [0, 1] joined by `c1 + c2 >= 1`.
//
// The shape exists for *when* the violation appears.  x0's two rows are
// satisfied by either of its values, so `Apply`'s activity scan -- which
// looks at the fixed column's own rows and nothing else -- reports
// nothing, and propagation deduces nothing from `c1 + c2 >= 1` while both
// columns are open.  The row only becomes violated in Phase 2.5, which
// fills a zero-cost continuous column from `cfg.cont_fallback` (null here,
// so 0.0).  So every in-tree check has already passed when the leaf turns
// out to be infeasible, and Phase 3 is the only repair that can run --
// which is what makes the two cases below about Phase 3's budget and
// nothing else.
//
// x0's rows are what keep it out of Phase 1's batch: a column with only an
// up-lock or only a down-lock is trivially roundable and never reaches the
// DFS, and a dive with no nodes is not the attempt these cases need.
void build_late_violation_mip(Highs& highs) {
    highs.addVar(0.0, 1.0);
    highs.changeColIntegrality(0, HighsVarType::kInteger);
    highs.addVar(0.0, 1.0);
    highs.addVar(0.0, 1.0);
    const auto x_idx = std::to_array<HighsInt>({0});
    const auto one = std::to_array<double>({1.0});
    highs.addRow(-kHighsInf, 1.0, 1, x_idx.data(), one.data());
    highs.addRow(0.0, kHighsInf, 1, x_idx.data(), one.data());
    const auto c_idx = std::to_array<HighsInt>({1, 2});
    const auto ones = std::to_array<double>({1.0, 1.0});
    highs.addRow(1.0, kHighsInf, 2, c_idx.data(), ones.data());
}

struct LateViolationHarness {
    Highs highs;
    HighsCallback cb{&highs};
    std::unique_ptr<HighsMipSolver> mipsolver;
    CscMatrix csc;
    ProblemView problem;
    FprScratch scratch;
    FprConfig cfg{};
    FprAttemptState state;

    explicit LateViolationHarness(FrameworkMode mode) {
        highs::parallel::initialize_scheduler();
        highs.setOptionValue("output_flag", false);
        mipsolver = bare_mipsolver_on(highs, cb, build_late_violation_mip);
        problem = make_problem(*mipsolver, csc);
        // The defect's premise, spelled at its smallest: an attempt whose
        // accumulated effort has passed `cfg.max_effort`.  A worker
        // reaches the same state with a realistic budget and a long
        // attempt; one is the cheap way to reproduce it.
        cfg.max_effort = 1;
        cfg.csc = &csc;
        cfg.mode = mode;
        cfg.strategy = &kForcedUpLr;
        cfg.binary_mask = problem.binary.data();
        cfg.scratch = &scratch;
    }

    // Run the DFS to its verdict on a slice far larger than the attempt
    // needs, so `cfg.max_effort` is the only small number in play.
    void dive(Rng& rng) {
        fpr_attempt_begin(state, *mipsolver, cfg, rng, /*attempt_idx=*/0);
        int guard = 0;
        while (state.phase == FprAttemptState::Phase::kDfs && guard++ < 100) {
            static_cast<void>(fpr_attempt_step(state, *mipsolver, cfg, rng,
                                               std::numeric_limits<size_t>::max() / 4));
        }
        REQUIRE(guard < 100);
    }
};

}  // namespace

TEST_CASE(
    "FPR dive: Phase 3's walk runs on an attempt whose effort has passed cfg.max_effort "
    "(#156)",
    "[repair-walk][walksat_repair][fpr][dive]") {
    LateViolationHarness h(FrameworkMode::kDive);
    Rng rng(42);
    h.dive(rng);

    REQUIRE(h.state.found_complete);
    // The premise, asserted rather than assumed: the attempt has already
    // spent more than `cfg.max_effort`, which is what made the old
    // `cfg.max_effort - total_prop_work` cap arrive as 0.
    REQUIRE(h.state.effort_consumed > h.cfg.max_effort);

    const size_t effort_before_finish = h.scratch.prop_engine->effort();
    const auto nnz = static_cast<size_t>(h.problem.mipdata->ARindex_.size());

    const HeuristicResult result = fpr_attempt_finish(h.state, *h.mipsolver, h.cfg, rng);

    // Phase 2.5 filled c1 = c2 = 0, so `c1 + c2 >= 1` is violated at the
    // leaf and only the leaf-time `walksat_repair` can recover it.
    REQUIRE(result.found_feasible);
    REQUIRE(result.solution.size() == 3);
    REQUIRE(result.solution[1] + result.solution[2] >= 1.0 - 1e-6);
    // `finish` charges the dive's effort plus one `nnz` row-activity
    // rebuild before Phase 3 starts, so anything above that is the walk's
    // own spend.  With the old cap it was exactly that sum: the walk broke
    // out of its loop before its first step.
    REQUIRE(result.effort > effort_before_finish + nnz);
}

TEST_CASE(
    "FPR repairsearch: Phase 3's search runs on an attempt whose effort has passed "
    "cfg.max_effort (#156)",
    "[repair-walk][repair-search][fpr][repairsearch]") {
    // The other Phase 3 branch, and it needs its own case: the two repairs
    // took the same expression from two separate arguments, so a cap
    // restored on either one alone is invisible to the other's test.
    LateViolationHarness h(FrameworkMode::kRepairSearch);
    Rng rng(42);
    h.dive(rng);

    REQUIRE(h.state.found_complete);
    REQUIRE(h.state.effort_consumed > h.cfg.max_effort);

    const size_t effort_before_finish = h.scratch.prop_engine->effort();
    const auto nnz = static_cast<size_t>(h.problem.mipdata->ARindex_.size());

    const HeuristicResult result = fpr_attempt_finish(h.state, *h.mipsolver, h.cfg, rng);

    REQUIRE(result.found_feasible);
    REQUIRE(result.solution.size() == 3);
    REQUIRE(result.solution[1] + result.solution[2] >= 1.0 - 1e-6);
    REQUIRE(result.effort > effort_before_finish + nnz);
}

namespace {

// `y0 + y1 >= 5` with both columns binary: unsatisfiable, and the walk
// cannot even stall quietly on it.  The first two steps push both columns
// to their upper bound; from then on every candidate clips to the bound it
// already holds, so `walksat_select_move` returns no move -- after
// charging the row's length -- and the loop `continue`s.  With
// `max_iterations` at 10^9 nothing but the internal valve can stop it.
// One binary row `sum y_j >= ncol + 4`, which no assignment satisfies, so
// every step finds a candidate, charges a row length and makes no progress.
// `ncol` is a parameter because the valve is `kWalkSatBudgetPerNnz * nnz`
// and a single width cannot tell that apart from a flat constant.
struct UnrepairableRowModel {
    static constexpr HighsInt kNrow = 1;
    HighsInt ncol;
    std::vector<HighsInt> ar_start;
    std::vector<HighsInt> ar_index;
    std::vector<double> ar_value;
    std::vector<double> col_lb;
    std::vector<double> col_ub;
    std::array<double, 1> row_lo;
    std::array<double, 1> row_hi = {kHighsInf};
    std::vector<HighsVarType> integrality;
    CscMatrix csc;

    explicit UnrepairableRowModel(HighsInt n)
        : ncol(n),
          ar_start{0, n},
          ar_value(static_cast<size_t>(n), 1.0),
          col_lb(static_cast<size_t>(n), 0.0),
          col_ub(static_cast<size_t>(n), 1.0),
          row_lo{static_cast<double>(n) + 4.0},
          integrality(static_cast<size_t>(n), HighsVarType::kInteger) {
        ar_index.reserve(static_cast<size_t>(n));
        for (HighsInt j = 0; j < n; ++j) {
            ar_index.push_back(j);
        }
        csc = build_csc(ncol, kNrow, ar_start, ar_index, ar_value);
    }

    PropEngine make_engine(double feastol = 1e-6) {
        return {ncol,
                kNrow,
                ar_start.data(),
                ar_index.data(),
                ar_value.data(),
                csc,
                col_lb.data(),
                col_ub.data(),
                row_lo.data(),
                row_hi.data(),
                integrality.data(),
                feastol};
    }
};

}  // namespace

TEST_CASE("walksat_repair: the internal per-nnz valve bounds a walk that cannot converge (#156)",
          "[repair-walk][walksat_repair][fpr]") {
    // Two widths, because the valve scales with `nnz`: a single width would
    // be satisfied by a flat constant just as well.
    const HighsInt width = GENERATE(2, 8);
    CAPTURE(width);

    UnrepairableRowModel m(width);
    PropEngine engine = m.make_engine();
    std::vector<double> solution(static_cast<size_t>(width), 0.0);
    std::vector<double> lhs_cache = {0.0};
    WalkSatScratch scratch;
    Rng rng(42);
    size_t effort = 0;

    const bool feasible = walksat_repair(
        engine, solution, lhs_cache, m.col_lb.data(), m.col_ub.data(),
        /*max_iterations=*/1000000000, /*noise=*/0.75, /*track_best=*/true, rng, effort, scratch);

    REQUIRE_FALSE(feasible);

    // `kWalkSatBudgetPerNnz` is file-local to `src/walksat.cpp`, so the
    // value is spelled out here -- the same way `tests/test_prop_engine.cpp`
    // spells `100 * nnz` for `kPropagateBudgetPerNnz`.  The gate is checked
    // at the top of a step, and one step charges at most a row length plus
    // the columns it scores, so the overrun past the budget is under one
    // step -- well inside `2 * nnz`.  Without the valve this call runs 10^9
    // steps.
    const auto nnz = static_cast<size_t>(m.ar_start[UnrepairableRowModel::kNrow]);
    REQUIRE(effort >= 100 * nnz);
    REQUIRE(effort < (100 * nnz) + (2 * nnz));
}

// ===================================================================
// A failed attempt hands back its point (issue #155).
//
// Mexi, Besancon, Bolusani, Chmiela, Hoen, Gleixner, *Scylla: a
// matrix-free fix-propagate-and-project heuristic*, arXiv 2307.03466v2,
// Sect. 2.3: "This is repeated until all such variables are fixed or
// infeasibility is detected by some domain becoming empty. In the latter
// case, fix-and-propagate continues in order to produce an integer vector
// by ignoring any constraint that would lead to empty domains. At the end
// of the propagation, all remaining unfixed variables are fixed to their
// values in the fractional reference solution or its projection on to
// their domain. The procedure always produces an integer-feasible, but
// not necessarily LP-feasible, solution."
//
// Algorithm 1.1 then spends that vector on lines 14-16 -- cycling /
// perturb, the alpha_K decay, the objective blend -- with no branch that
// discards an infeasible rounding.  `fpr_attempt` could not express one,
// so `ScyllaWorker` skipped the whole round and re-solved a byte-identical
// LP; `tests/test_scylla.cpp` pins that half.  This one pins the contract
// underneath it.
// ===================================================================

namespace {

// `2*x0 + 2*x1 = 1`, both binary: LP-feasible (x0 = 0.5) and
// integer-infeasible, so no rounding of it can ever succeed and every
// `fpr_attempt_finish` failure path is reachable on the same model.  The
// equality gives both columns an up-lock and a down-lock apiece, so Phase
// 1's trivially-roundable pass leaves both to the DFS.
void build_parity_mip(Highs& highs) {
    highs.addVar(0.0, 1.0);
    highs.addVar(0.0, 1.0);
    highs.changeColIntegrality(0, HighsVarType::kInteger);
    highs.changeColIntegrality(1, HighsVarType::kInteger);
    const auto idx = std::to_array<HighsInt>({0, 1});
    const auto val = std::to_array<double>({2.0, 2.0});
    highs.addRow(1.0, 1.0, 2, idx.data(), val.data());
}

struct ParityHarness {
    Highs highs;
    HighsCallback cb{&highs};
    std::unique_ptr<HighsMipSolver> mipsolver;
    CscMatrix csc;
    ProblemView problem;
    FprScratch scratch;
    FprConfig cfg{};
    FprAttemptState state;

    explicit ParityHarness(FrameworkMode mode) {
        highs::parallel::initialize_scheduler();
        highs.setOptionValue("output_flag", false);
        mipsolver = bare_mipsolver_on(highs, cb, build_parity_mip);
        problem = make_problem(*mipsolver, csc);
        cfg.max_effort = std::numeric_limits<size_t>::max() / 2;
        cfg.csc = &csc;
        cfg.mode = mode;
        cfg.strategy = &kForcedUpLr;
        cfg.binary_mask = problem.binary.data();
        cfg.scratch = &scratch;
    }
};

// The whole of criterion 2: a failed result still carries a complete
// integer point inside the structural bounds.  Asserted on the *result*,
// not on the engine, because that is what a caller sees.
void require_complete_integer_point(const HeuristicResult& result, const ProblemView& problem) {
    REQUIRE_FALSE(result.found_feasible);
    REQUIRE(std::cmp_equal(result.solution.size(), problem.ncol));
    for (HighsInt j = 0; j < problem.ncol; ++j) {
        const auto idx = static_cast<size_t>(j);
        CAPTURE(j, result.solution[idx]);
        REQUIRE(result.solution[idx] >= problem.model->col_lower_[idx]);
        REQUIRE(result.solution[idx] <= problem.model->col_upper_[idx]);
        if (is_integer(problem.model->integrality_, j)) {
            REQUIRE(result.solution[idx] ==
                    Catch::Approx(std::round(result.solution[idx])).margin(1e-9));
        }
    }
}

}  // namespace

TEST_CASE("fpr_attempt hands back a complete integer point on a failed rounding (#155)",
          "[fpr][repair-walk][fpr-core]") {
    SECTION("the DFS was cut by the per-call budget") {
        // Shape (i), and the *common* one on Scylla: three of the four
        // `kFprConfigs` entries backtrack, and a single attempt routinely
        // does not finish inside the `ncol + 1` node budget (see the #121
        // case in tests/test_scylla.cpp).  A fix that produced the point
        // only for a completed dive would leave most Scylla rounds still
        // skipping, which is why this shape is pinned first.
        ParityHarness h(FrameworkMode::kDfs);
        Rng rng(5);
        fpr_attempt_begin(h.state, *h.mipsolver, h.cfg, rng, /*attempt_idx=*/0);
        REQUIRE(h.state.phase == FprAttemptState::Phase::kDfs);

        // A one-unit slice admits exactly one node: the gate is a delta
        // from this call's start.
        const FprStepResult outcome =
            fpr_attempt_step(h.state, *h.mipsolver, h.cfg, rng, /*effort_remaining=*/1);
        REQUIRE(outcome == FprStepResult::kBudgetGate);
        REQUIRE_FALSE(h.state.found_complete);

        // What the one-shot `fpr_attempt` wrapper does with a paused
        // attempt, reproduced here so `finish` sees the same state it
        // sees in production.
        h.state.phase = FprAttemptState::Phase::kReadyToFinish;
        const size_t effort_before = h.state.effort_consumed;
        const auto nnz = static_cast<size_t>(h.problem.mipdata->ARindex_.size());

        const HeuristicResult result = fpr_attempt_finish(h.state, *h.mipsolver, h.cfg, rng);
        require_complete_integer_point(result, h.problem);

        // The `!found_complete` path used to return `E.effort()`
        // untouched.  It now fills Phase 2.5 and rebuilds the row
        // activities, and every gate in the solve reads this number, so
        // the work has to be charged: a fill that reported nothing would
        // be search no budget can see.
        REQUIRE(result.effort >= effort_before + nnz);
        REQUIRE(h.state.effort_consumed == result.effort);
    }

    SECTION("the backtracking stack was exhausted") {
        // Shape (ii): `kDfs` propagates and backtracks, and propagation
        // refutes both branches of the first decision, so the DFS runs out
        // of stack rather than out of budget or nodes.  `kDfsrep` reaches
        // the same `!found_complete` verdict here but does it at the node
        // limit, because its in-tree repair rescues the first branch and
        // buys the dive a third node.
        ParityHarness h(FrameworkMode::kDfs);
        Rng rng(6);
        fpr_attempt_begin(h.state, *h.mipsolver, h.cfg, rng, /*attempt_idx=*/0);
        int guard = 0;
        while (h.state.phase == FprAttemptState::Phase::kDfs && guard++ < 100) {
            static_cast<void>(fpr_attempt_step(h.state, *h.mipsolver, h.cfg, rng,
                                               std::numeric_limits<size_t>::max() / 4));
        }
        REQUIRE(guard < 100);
        REQUIRE(h.state.phase == FprAttemptState::Phase::kReadyToFinish);
        REQUIRE_FALSE(h.state.found_complete);
        // Not the node limit: the stack really did empty.
        REQUIRE(h.state.nodes_visited < h.problem.ncol + 1);

        const HeuristicResult result = fpr_attempt_finish(h.state, *h.mipsolver, h.cfg, rng);
        require_complete_integer_point(result, h.problem);
    }

    SECTION("the leaf was reached and Phase 3 failed") {
        // Shape (iii): `kDive` never backtracks, so it reaches a leaf with
        // every integer fixed, the row re-check rejects it, and the
        // leaf-time WalkSAT cannot repair a parity violation.  This is the
        // `!feasible` return *after* Phase 3, and what it hands back is
        // `scratch.solution` as Phase 3 left it.
        ParityHarness h(FrameworkMode::kDive);
        Rng rng(7);
        fpr_attempt_begin(h.state, *h.mipsolver, h.cfg, rng, /*attempt_idx=*/0);
        int guard = 0;
        while (h.state.phase == FprAttemptState::Phase::kDfs && guard++ < 100) {
            static_cast<void>(fpr_attempt_step(h.state, *h.mipsolver, h.cfg, rng,
                                               std::numeric_limits<size_t>::max() / 4));
        }
        REQUIRE(guard < 100);
        REQUIRE(h.state.found_complete);

        const HeuristicResult result = fpr_attempt_finish(h.state, *h.mipsolver, h.cfg, rng);
        require_complete_integer_point(result, h.problem);
    }
}
