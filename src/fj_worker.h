#pragma once

#include "heuristic_context.h"
#include "structure.h"
#include "worker_base.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <vector>

class SolutionSink;

// Upstream FJ's `CALLBACK_EFFORT` (`feasibilityjump.hh`): FjWorker reads
// its budgets, its stall counter and the clock only at FJ's callback, so
// an attempt stops no sooner than this many effort units, unless its caller
// holds it to its budget with `FjWorker::hold_to_budget` (#309).
constexpr size_t kFjCallbackEffort = 500000;

// FeasibilityJump worker.  Owns a FeasibilityJumpSolver
// and supports pause/resume across attempt boundaries via the `resume`
// parameter on FJ's solve() method.
//
// First call to run_attempt() builds the solver, adds vars/constraints,
// and calls solve(initial, callback, /*resume=*/false).
// Subsequent calls resume via solve(nullptr, callback, /*resume=*/true).
// `reset` starts a new search on the model already built (#316).
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
    // `cutoff`, when finite, adds one row to FJ's model,
    // `col_cost . x <= cutoff - offset`, so FJ counts a point
    // feasible only if its objective (offset included) is at most the
    // cutoff, up to FJ's absolute violation tolerance `feastol`.  The ruin
    // and repair LNS (#207) repairs under its incumbent's objective this
    // way.  No row when every cost is zero: no point could improve.
    //
    // `exchange_rows`, when not empty, are the exact-k rows (#206's
    // `detect_structure`) whose binary columns FJ may exchange (#218): one
    // from 1 to 0 and another from 0 to 1 in one step (the patched
    // `FeasibilityJumpSolver::setExchangeGroups`).  They must outlive the
    // worker's first attempt.
    FjWorker(const ProblemView& problem, const ExecutionContext& exec, SolutionSink& sink,
             size_t total_budget, size_t stale_budget, uint32_t seed, std::vector<double> start,
             WorkerTrace trace, double cutoff = std::numeric_limits<double>::infinity(),
             std::span<const ExactKRow> exchange_rows = {});
    ~FjWorker();

    // A new search on this worker's model, as a worker constructed with the
    // same problem, exec, sink, cutoff's finiteness and exchange rows and
    // with these arguments would run it, budgets and effort counted afresh
    // (#316): ruin's repair and recombine's local search keep one worker
    // per slot instead of building FJ's model for every attempt.  FJ's
    // constraint weights go back to 1 too.  `test_fj_reuse.cpp` checks this
    // without exchange rows only, as ruin and recombine pass none.  The
    // cutoff row is in the model or not from construction on, so `cutoff`
    // must be finite exactly when the constructor's was, else the reset is
    // refused and the worker finishes; a finite one moves the row's RHS.
    // `hold_to_budget` stays set.  `start` is copied into the worker's own
    // buffer, so a reset allocates nothing once that has grown.
    void reset(std::span<const double> start, uint32_t seed, size_t total_budget,
               size_t stale_budget, WorkerTrace trace,
               double cutoff = std::numeric_limits<double>::infinity());

    // Run FJ for up to attempt_budget effort, then pause via callback.
    AttemptResult run_attempt(size_t attempt_budget);

    // Holds each attempt to its budget (#309): FJ's callback period is set
    // to what is left of it, at most `kFjCallbackEffort`, and set again at
    // every callback that lets FJ continue, so an attempt stops within one
    // FJ step of `attempt_budget` past FJ's setup.  The setup, two passes
    // over FJ's nonzeros that its `init()` charges, is added to the first
    // attempt's budget and to the total, so a budget below it still
    // searches; a held attempt spends at most setup + budget + one step.
    // For the one-shot callers (ruin's repair, recombine's local search,
    // Scylla's cycle escape); FJ's own dispatch keeps upstream's period.
    void hold_to_budget() { hold_to_budget_ = true; }

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
    struct Impl;
    // Builds FJ's model into `impl_`, or leaves it null when a column's
    // bounds are unusable.  Its parts, in the order FJ needs them.
    void build_model();
    [[nodiscard]] bool add_columns();
    void add_rows();
    void add_cutoff_row();
    void add_exchange_groups();
    // FJ's start, `start_` clamped into FJ's bounds, into `impl_->col_value`.
    void fill_start();

    std::unique_ptr<Impl> impl_;

    const ProblemView& problem_;
    // Read only for `past_deadline()`, from inside upstream FJ's callback:
    // one attempt at a large effort option runs far past the solve's
    // `time_limit` otherwise (issue #114).
    const ExecutionContext& exec_;
    SolutionSink& sink_;
    std::vector<double> start_;
    uint32_t seed_;
    double cutoff_;
    const std::span<const ExactKRow> exchange_rows_;
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
    WorkerTrace trace_;
    // The nonzeros handed to FJ's model, cutoff row included.
    size_t fj_nonzeros_ = 0;
    // The cutoff row's index in FJ's model, or -1 for none.
    int cutoff_row_ = -1;
    bool hold_to_budget_ = false;
    // The model was built (or its build refused): `impl_` is final.
    bool initialized_ = false;
    // This search has run, so the next attempt resumes it.
    bool first_solve_done_ = false;
};
