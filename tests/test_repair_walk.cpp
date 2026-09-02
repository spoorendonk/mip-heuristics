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

#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <limits>
#include <memory>
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
// Fixing x0 = 1 is refuted by propagation (the second row forces
// x1 <= -1), and the repair that recovers it is available on exactly one
// column: x0's singleton domain {1} can slide down to {0} inside its
// structural [0, 1], while x1's *unfixed* [0, 1] cannot slide at all --
// the paper forbids widening it and translating it by the -1 the row
// asks for would leave the structural bounds.  So the walk is
// deterministic here: one candidate, zero damage, chosen greedily.
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

// A single row `y <= 2` over one general integer column, so the only
// repair available is a shift of `y`'s own interval and the test can
// assert the *whole* resulting domain rather than a value.
struct OneRowModel {
    static constexpr HighsInt kNcol = 1;
    static constexpr HighsInt kNrow = 1;
    std::vector<HighsInt> ar_start = {0, 1};
    std::vector<HighsInt> ar_index = {0};
    std::vector<double> ar_value = {1.0};
    std::vector<double> col_lb;
    std::vector<double> col_ub = {10.0};
    std::array<double, 1> row_lo = {-kHighsInf};
    std::array<double, 1> row_hi = {2.0};
    std::vector<HighsVarType> integrality = {HighsVarType::kInteger};
    CscMatrix csc;

    explicit OneRowModel(double structural_lb) : col_lb({structural_lb}) {
        csc = build_csc(kNcol, kNrow, ar_start, ar_index, ar_value);
    }

    PropEngine make_engine(double feastol = 1e-6) {
        return {kNcol,           kNrow,         ar_start.data(),    ar_index.data(),
                ar_value.data(), csc,           col_lb.data(),      col_ub.data(),
                row_lo.data(),   row_hi.data(), integrality.data(), feastol};
    }
};

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
    REQUIRE(repair_walk(E, /*max_steps=*/200, /*noise=*/0.75,
                        /*max_effort=*/std::numeric_limits<size_t>::max(), rng, effort, scratch,
                        Deadline{}));
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
    // row wants the activity down to 2, which is a shift of -3.
    OneRowModel m(/*structural_lb=*/0.0);
    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine E = m.make_engine();
    E.init_activities();
    REQUIRE(E.tighten_lb(0, 5.0));
    REQUIRE(E.tighten_ub(0, 7.0));
    REQUIRE_FALSE(E.var(0).fixed);

    RepairWalkScratch scratch;
    Rng rng(3);
    size_t effort = 0;
    REQUIRE(repair_walk(E, /*max_steps=*/200, /*noise=*/0.75,
                        /*max_effort=*/std::numeric_limits<size_t>::max(), rng, effort, scratch,
                        Deadline{}));

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
    // The walk must take the clipped shift and report the node still
    // infeasible rather than either overshooting the structural bound or
    // refusing the partial improvement.
    OneRowModel m(/*structural_lb=*/3.0);
    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine E = m.make_engine();
    E.init_activities();
    REQUIRE(E.tighten_lb(0, 5.0));
    REQUIRE(E.tighten_ub(0, 7.0));

    RepairWalkScratch scratch;
    Rng rng(5);
    size_t effort = 0;
    REQUIRE_FALSE(repair_walk(E, /*max_steps=*/200, /*noise=*/0.75,
                              /*max_effort=*/std::numeric_limits<size_t>::max(), rng, effort,
                              scratch, Deadline{}));

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
    REQUIRE_FALSE(repair_walk(E, /*max_steps=*/0, /*noise=*/0.75,
                              /*max_effort=*/std::numeric_limits<size_t>::max(), rng, effort,
                              scratch, Deadline{}));
    REQUIRE(E.var(0).fixed);
    REQUIRE(E.var(0).val == Catch::Approx(1.0));
}

// ===================================================================
// The in-tree call site (Fig. 1 lines 7-8) and the branching that
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
// decision is forced to be x0 = 1 -- the decision propagation refutes.
// Both columns carry an up-lock and a down-lock (each appears in a `>=`
// and a `<=` row), so Phase 1's trivially-roundable pass fixes neither.
constexpr FprStrategyConfig kForcedUpLr{VarStrategy::kLR, ValStrategy::kUp};

}  // namespace

TEST_CASE("FPR diveprop: a mid-dive propagation failure is repaired, not fatal (#124)",
          "[repair-walk][fpr][diveprop]") {
    highs::parallel::initialize_scheduler();
    Highs highs;
    highs.setOptionValue("output_flag", false);
    HighsCallback cb(&highs);
    auto mipsolver = bare_mipsolver_on(highs, cb, build_swap_mip);

    CscMatrix csc;
    const ProblemView problem = make_problem(*mipsolver, csc);
    FprScratch scratch;
    FprConfig cfg{};
    cfg.max_effort = std::numeric_limits<size_t>::max() / 2;
    cfg.csc = &csc;
    cfg.mode = FrameworkMode::kDiveprop;
    cfg.strategy = &kForcedUpLr;
    cfg.binary_mask = problem.binary.data();
    cfg.scratch = &scratch;
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
    FprAttemptState state;
    fpr_attempt_begin(state, *mipsolver, cfg, rng, /*attempt_idx=*/0);
    while (state.phase == FprAttemptState::Phase::kDfs) {
        static_cast<void>(fpr_attempt_step(state, *mipsolver, cfg, rng, cfg.max_effort));
    }

    // `diveprop` is the paper's best-performing parametrization and does
    // not backtrack, so before #124 this dive ended at its first
    // propagation failure with an empty stack and `found_complete` false
    // -- exactly where the paper's diveprop begins repairing.
    REQUIRE(state.found_complete);
    // x0 was decided to 1 by the dive and refuted by propagation; the
    // repair slid its singleton domain to 0 in the tree, and the dive
    // then carried on and fixed x1.
    REQUIRE(scratch.prop_engine->var(0).fixed);
    REQUIRE(scratch.prop_engine->var(0).val == Catch::Approx(0.0));
    REQUIRE(scratch.prop_engine->var(1).fixed);
    REQUIRE(scratch.prop_engine->var(1).val == Catch::Approx(1.0));

    const HeuristicResult result = fpr_attempt_finish(state, *mipsolver, cfg, rng);
    REQUIRE(result.found_feasible);
    REQUIRE(result.solution.size() == 2);
    REQUIRE(result.solution[0] == Catch::Approx(0.0));
    REQUIRE(result.solution[1] == Catch::Approx(1.0));
}

TEST_CASE("FPR diveprop: an unrepaired node keeps diving instead of ending the attempt (#124)",
          "[repair-walk][fpr][diveprop]") {
    highs::parallel::initialize_scheduler();
    Highs highs;
    highs.setOptionValue("output_flag", false);
    HighsCallback cb(&highs);
    auto mipsolver = bare_mipsolver_on(highs, cb, build_swap_mip);

    CscMatrix csc;
    const ProblemView problem = make_problem(*mipsolver, csc);
    FprScratch scratch;
    FprConfig cfg{};
    cfg.max_effort = std::numeric_limits<size_t>::max() / 2;
    cfg.csc = &csc;
    cfg.mode = FrameworkMode::kDiveprop;
    cfg.strategy = &kForcedUpLr;
    cfg.binary_mask = problem.binary.data();
    cfg.scratch = &scratch;
    // Zero repair steps: the walk still measures the node but can change
    // nothing, so the node stays infeasible.  This isolates Fig. 1 lines
    // 9-18 from line 8 -- with `backtrackOnInfeas` off, an infeasible
    // node must still `Branch` and carry the dive to the bottom.
    cfg.walksat_iterations = 0;
    Rng rng(1);

    FprAttemptState state;
    fpr_attempt_begin(state, *mipsolver, cfg, rng, /*attempt_idx=*/0);
    while (state.phase == FprAttemptState::Phase::kDfs) {
        static_cast<void>(fpr_attempt_step(state, *mipsolver, cfg, rng, cfg.max_effort));
    }
    // The dive reached a leaf with every integer fixed.  Pruning the
    // refuted node instead -- what the pre-#124 bare `continue` did --
    // empties the stack of a non-backtracking mode and leaves this false.
    REQUIRE(state.found_complete);
}

TEST_CASE("FPR diveprop: an in-tree repair leaves the attempt resumable (#77 x #124)",
          "[repair-walk][fpr][diveprop][resume]") {
    highs::parallel::initialize_scheduler();
    Highs highs;
    highs.setOptionValue("output_flag", false);
    HighsCallback cb(&highs);
    auto mipsolver = bare_mipsolver_on(highs, cb, build_swap_mip);

    CscMatrix csc;
    const ProblemView problem = make_problem(*mipsolver, csc);
    FprScratch scratch;
    FprConfig cfg{};
    cfg.max_effort = std::numeric_limits<size_t>::max() / 2;
    cfg.csc = &csc;
    cfg.mode = FrameworkMode::kDiveprop;
    cfg.strategy = &kForcedUpLr;
    cfg.binary_mask = problem.binary.data();
    cfg.scratch = &scratch;
    Rng rng(1);

    // One node per call: the repair runs inside a `step` that is then
    // paused and re-entered, so anything it left half-applied to the DFS
    // stack or the engine's undo trail shows up as a wrong verdict here.
    FprAttemptState state;
    fpr_attempt_begin(state, *mipsolver, cfg, rng, /*attempt_idx=*/0);
    int guard = 0;
    while (state.phase == FprAttemptState::Phase::kDfs && guard++ < 100) {
        static_cast<void>(fpr_attempt_step(state, *mipsolver, cfg, rng, /*effort_remaining=*/1));
    }
    REQUIRE(guard < 100);
    // Same mid-dive observation as the case above, for the same reason:
    // the leaf-time WalkSAT would otherwise cover for a repair that never
    // ran.
    REQUIRE(state.found_complete);
    REQUIRE(scratch.prop_engine->var(0).val == Catch::Approx(0.0));
    const HeuristicResult sliced = fpr_attempt_finish(state, *mipsolver, cfg, rng);

    REQUIRE(sliced.found_feasible);
    REQUIRE(sliced.solution.size() == 2);
    REQUIRE(sliced.solution[0] == Catch::Approx(0.0));
    REQUIRE(sliced.solution[1] == Catch::Approx(1.0));
}

namespace {

// A model whose propagation runs out of its per-call matrix-access
// budget (`kPropagateBudgetPerNnz * nnz`) at the dive's only node, while
// a row is simultaneously violated in the activity sense that
// `repair_walk` measures.
//
//   R0: x - 1.01*y + z >= 0
//   R1: y - 1.01*x     >= 0
//   R2:              z <= 0
//
// x and y are continuous on [0, 1e19].  R0/R1 shrink each other's upper
// bound by a factor 1.01 per visit and re-seed each other, so reaching a
// fixpoint takes thousands of row visits against a budget of a few
// hundred -- the same geometric-decay shape `tests/test_prop_engine.cpp`
// uses, slowed from halving to 1.01 because that file's 1e300 is not
// available here: `HighsLpUtils::assessBounds` rewrites any bound at or
// beyond the `infinite_bound` option (default 1e20) to infinite, and an
// infinite domain gives the cascade nothing to shrink.
// z is the dive's only integer; fixing it to 1 makes R2 violated.
//
// So a node where propagation returns `kBudgetExhausted` -- a truncated
// but *sound* fixpoint, never a refutation (issues #127, #151) -- carries
// an activity-violated row.  Repair must not run there.
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
    highs.addRow(-kHighsInf, 0.0, 1, r2_idx.data(), r2_val.data());
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
    // refutation) and the case below tests nothing.
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
        // ... and R2 really is violated in the activity sense repair reads,
        // so a repair invoked here would have something to shift.
        REQUIRE(E.row_min_activity(2) > problem.model->row_upper_[2] + problem.mipdata->feastol);
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

    // z is still at the value the dive fixed it to.  A trigger widened
    // from `pr == kInfeasible` to anything weaker (`pr != kFixpoint`)
    // would have repaired this node and slid z's singleton domain down to
    // 0 -- the only legal shift R2 admits.
    REQUIRE(scratch.prop_engine->var(2).fixed);
    REQUIRE(scratch.prop_engine->var(2).val == Catch::Approx(1.0));
}
