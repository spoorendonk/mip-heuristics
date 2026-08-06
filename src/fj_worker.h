#pragma once

#include "worker_base.h"

#include <cstddef>
#include <cstdint>
#include <memory>

class HighsMipSolver;
class SolutionPool;

// Epoch-gated FeasibilityJump worker.  Owns a FeasibilityJumpSolver
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
    FjWorker(HighsMipSolver &mipsolver, SolutionPool &pool, size_t total_budget, uint32_t seed);
    ~FjWorker();

    // Run FJ for up to attempt_budget effort, then pause via callback.
    AttemptResult run_attempt(size_t attempt_budget);

    bool finished() const { return base_.finished; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    HighsMipSolver &mipsolver_;
    SolutionPool &pool_;
    const uint32_t seed_;

    // Effort / staleness / finished bookkeeping.  FJ's `stale_budget` is
    // derived from the constraint matrix nonzero count (see run_attempt)
    // rather than the generic `total_budget >> 2` default.
    WorkerBudgetState base_;
    bool initialized_ = false;
    bool first_solve_done_ = false;
};
