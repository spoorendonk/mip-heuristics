#pragma once

#include "worker_base.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

class HighsMipSolver;
class IncumbentSink;

// FeasibilityJump worker.  Owns a FeasibilityJumpSolver
// and supports pause/resume across attempt boundaries via the `resume`
// parameter on FJ's solve() method.
//
// First call to run_attempt() builds the solver, adds vars/constraints,
// and calls solve(initial, callback, /*resume=*/false).
// Subsequent calls resume via solve(nullptr, callback, /*resume=*/true).
//
// Finished when FJ stalls (effortSinceLastImprovement exceeds threshold).
class FjWorker {
public:
    // `start` seeds the initial assignment: the best solution known when
    // this worker was created, or empty for a bound-based start.  Taken by
    // value because the caller resolves it per worker (pool first, then the
    // dispatch's incumbent snapshot) — a reference to the live
    // `mipdata->incumbent` is what issue #98 was about, and a reference to
    // a caller local would dangle.
    FjWorker(HighsMipSolver& mipsolver, IncumbentSink& sink, size_t total_budget,
             size_t stale_budget, uint32_t seed, std::vector<double> start, WorkerTrace trace);
    ~FjWorker();

    // Run FJ for up to attempt_budget effort, then pause via callback.
    AttemptResult run_attempt(size_t attempt_budget);

    [[nodiscard]] bool finished() const { return base_.finished; }

    // Monotone charged effort for the `[HeurSol]` trace (#106): this
    // worker's own counter plus what the slot's retired occupants spent.
    // `fj::run` reads it off the outgoing worker to seed the replacement's
    // `WorkerTrace`, which is what keeps the emitted `effort_at` rising
    // across a rebuild.
    [[nodiscard]] size_t traced_effort() const { return trace_.at(base_.total_effort); }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    HighsMipSolver& mipsolver_;
    IncumbentSink& sink_;
    const std::vector<double> start_;
    const uint32_t seed_;

    // Effort / staleness / finished bookkeeping.  Both budgets come from
    // the caller: FJ's stall threshold has always been an absolute
    // `nnz << 8` rather than a fraction of the allowance, and since issue
    // #111 the other three heuristics derive theirs the same way, so it
    // is sized once in `mode_dispatch` (`HeuristicBudget::worker_stale`)
    // instead of re-derived from the FJ solver's own copy of the matrix.
    WorkerBudgetState base_;
    // Trace-only slot identity; see `WorkerTrace` in worker_base.h.
    const WorkerTrace trace_;
    bool initialized_ = false;
    bool first_solve_done_ = false;
};
