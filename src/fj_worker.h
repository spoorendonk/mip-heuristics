#pragma once

#include <cstddef>
#include <cstdint>

#include "epoch_runner.h"

class HighsMipSolver;
class SolutionPool;

// Wraps a single feasibilityJumpCapture call with epoch-based execution.
// FJ is single-shot per construction: one call to run_epoch runs the
// solver to completion (or budget exhaustion), after which finished()
// returns true.  Restart = construct a new FjWorker with a different seed.
//
// Satisfies the EpochWorker concept from epoch_runner.h.
class FjWorker {
 public:
  FjWorker(HighsMipSolver &mipsolver, SolutionPool &pool, size_t total_budget,
           uint32_t seed);

  // Run FJ to completion within epoch_budget.  Single-shot: after the
  // first call, finished() returns true.
  EpochResult run_epoch(size_t epoch_budget);

  bool finished() const { return finished_; }

  // No-op for FJ: single-shot workers have no staleness to reset.
  void reset_staleness() {}

 private:
  HighsMipSolver &mipsolver_;
  SolutionPool &pool_;
  const size_t total_budget_;
  const uint32_t seed_;
  bool finished_ = false;
};

static_assert(EpochWorker<FjWorker>,
              "FjWorker must satisfy EpochWorker concept");
