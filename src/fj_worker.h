#pragma once

#include "heuristic_context.h"
#include "worker_base.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <vector>

class SolutionSink;
struct ExactKRow;

// Upstream FJ's `CALLBACK_EFFORT` (`feasibilityjump.hh`): FjWorker reads
// its budgets, its stall counter and the clock only at FJ's callback, so
// no FJ attempt stops sooner than this many effort units.  A caller whose
// budget is smaller than one period cannot hold FJ to it.
constexpr size_t kFjCallbackEffort = 500000;

// FeasibilityJump worker.  Owns a FeasibilityJumpSolver
// and supports pause/resume across attempt boundaries via the `resume`
// parameter on FJ's solve() method.
//
// First call to run_attempt() builds the solver, adds vars/constraints,
// and calls solve(initial, callback, /*resume=*/false).
// Subsequent calls resume via solve(nullptr, callback, /*resume=*/true).
//
// Finished when FJ stalls (effortSinceLastImprovement exceeds threshold)
// or spends its total budget.
class FjWorker {
public:
    // `start` seeds the initial assignment: the best solution known when
    // this worker was created, or empty for a bound-based start.  Taken by
    // value because the caller resolves it per worker (pool first, then the
    // dispatch's incumbent snapshot) — a reference to the live
    // `mipdata->incumbent` is what issue #98 was about, and a reference to
    // a caller local would dangle.
    // `problem` must outlive the worker.
    //
    // `objective_cutoff`, when finite, adds one row to FJ's model,
    // `col_cost . x <= objective_cutoff - offset`, so FJ counts a point
    // feasible only if its objective (offset included) is at most the
    // cutoff, up to FJ's absolute violation tolerance `feastol`.  The ruin
    // and repair LNS (#207) repairs under its incumbent's objective this
    // way.  No row when every cost is zero: no point could improve.
    //
    // `objective_cut`, when on, is the objective cut of #208 (#222): the
    // worker polls `sink.best_objective()` before each attempt and at every
    // FJ callback that lets the solve continue (its own offers' included;
    // one that ends the attempt leaves it to the next), and tightens the
    // same row to `objective_cut_target` of it in place — no rebuild, every
    // FJ weight kept (`FeasibilityJumpSolver::setObjectiveCut`, patched
    // in).  The row is added at the first finite cutoff, so a worker with
    // the cut on and an empty pool searches exactly as one with it off.
    // The tighter of the two cutoffs is in force.
    //
    // `exchange_rows`, when not empty, are the exact-k rows (#206's
    // `detect_structure`) whose binary columns FJ may exchange (#218): one
    // from 1 to 0 and another from 0 to 1 in one step (the patched
    // `FeasibilityJumpSolver::setExchangeGroups`).  They must outlive the
    // worker's first attempt.
    FjWorker(const ProblemView& problem, const ExecutionContext& exec, SolutionSink& sink,
             size_t total_budget, size_t stale_budget, uint32_t seed, std::vector<double> start,
             WorkerTrace trace, double objective_cutoff = std::numeric_limits<double>::infinity(),
             ObjectiveCutParams objective_cut = {}, std::span<const ExactKRow> exchange_rows = {});
    ~FjWorker();

    // Run FJ for up to attempt_budget effort, then pause via callback.
    AttemptResult run_attempt(size_t attempt_budget);

    [[nodiscard]] bool finished() const { return base_.finished; }

    // The best feasible point this worker's FJ reported over its lifetime
    // and its objective (offset included), or empty and +infinity.  FJ
    // reports a point only when it beats its own previous one, so this is
    // the last one reported.
    [[nodiscard]] const std::vector<double>& best_found() const { return best_found_; }
    [[nodiscard]] double best_found_objective() const { return best_found_objective_; }

    // Monotone charged effort for the `[HeurSol]` trace (#106): this
    // worker's own counter plus what the slot's retired occupants spent.
    // `fj::run` reads it off the outgoing worker to seed the replacement's
    // `WorkerTrace`, which is what keeps the emitted `effort_at` rising
    // across a rebuild.
    [[nodiscard]] size_t traced_effort() const { return trace_.at(base_.total_effort); }

private:
    // Set FJ's cutoff row to the tighter of the fixed and the polled
    // cutoff when that is below the one in force.
    void apply_cut();

    struct Impl;
    std::unique_ptr<Impl> impl_;

    const ProblemView& problem_;
    // Read only for `past_deadline()`, from inside upstream FJ's callback:
    // one attempt at a large effort option runs far past the solve's
    // `time_limit` otherwise (issue #114).
    const ExecutionContext& exec_;
    SolutionSink& sink_;
    const std::vector<double> start_;
    const uint32_t seed_;
    const double objective_cutoff_;
    const ObjectiveCutParams objective_cut_;
    const std::span<const ExactKRow> exchange_rows_;
    // The cutoff FJ's row carries, offset included; +inf while there is no
    // row.  Only ever lowered.
    double cut_in_force_ = std::numeric_limits<double>::infinity();
    std::vector<double> best_found_;
    double best_found_objective_ = std::numeric_limits<double>::infinity();

    // Effort / finished bookkeeping.  Both budgets come from the caller:
    // FJ's patience has always been an absolute `nnz << 8` rather than a
    // fraction of the allowance, and since issue #111 the other three
    // heuristics derive theirs the same way, so it is sized once in
    // `mode_dispatch` (`HeuristicBudget::worker_stale`) instead of
    // re-derived from the FJ solver's own copy of the matrix.
    // `stale_budget` is read against upstream FJ's own
    // `effortSinceLastImprovement` only (#191), so
    // `effort_since_improvement` is never charged.
    WorkerBudgetState base_;
    // Trace-only slot identity; see `WorkerTrace` in worker_base.h.
    const WorkerTrace trace_;
    bool initialized_ = false;
    bool first_solve_done_ = false;
};
