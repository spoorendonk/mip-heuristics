#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "epoch_runner.h"

class HighsMipSolver;
class SolutionPool;

// Epoch-gated FeasibilityJump worker.  Owns a FeasibilityJumpSolver
// and supports pause/resume across epoch boundaries via the `resume`
// parameter on FJ's solve() method.
//
// First call to run_epoch() builds the solver, adds vars/constraints,
// and calls solve(initial, callback, /*resume=*/false).
// Subsequent calls resume via solve(nullptr, callback, /*resume=*/true).
//
// Finished when FJ stalls (effortSinceLastImprovement exceeds threshold).
//
// Satisfies the EpochWorker concept from epoch_runner.h.
class FjWorker {
 public:
  FjWorker(HighsMipSolver &mipsolver, SolutionPool &pool, size_t total_budget,
           uint32_t seed);
  ~FjWorker();

  // Run FJ for up to epoch_budget effort, then pause via callback.
  EpochResult run_epoch(size_t epoch_budget);

  bool finished() const { return finished_; }

  // Reset the improvement staleness counter (called at epoch boundary
  // when another worker found an improvement).
  void reset_staleness();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  HighsMipSolver &mipsolver_;
  SolutionPool &pool_;
  const size_t total_budget_;
  const uint32_t seed_;

  size_t total_effort_ = 0;
  size_t effort_since_improvement_ = 0;
  size_t stale_budget_ = 0;
  bool initialized_ = false;
  bool first_solve_done_ = false;
  bool finished_ = false;
};

// static_assert in fj_worker.cpp to avoid leaking C++23 concepts into
// headers that HiGHS (C++17) might transitively include.
