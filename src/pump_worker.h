#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "Highs.h"
#include "epoch_runner.h"
#include "util/HighsInt.h"

class HighsMipSolver;
struct CscMatrix;
class SolutionPool;

// Encapsulates a PDLP pump chain (Mexi et al. 2023, Algorithm 1.1)
// with epoch-based execution.  Each worker owns its own PDLP solver
// instance, warm-start vectors, cycling history, and RNG.
//
// Satisfies the EpochWorker concept from epoch_runner.h.
class PumpWorker {
 public:
  PumpWorker(HighsMipSolver &mipsolver, const CscMatrix &csc,
             SolutionPool &pool, size_t total_budget, uint32_t seed);

  // Run pump chain iterations until epoch_budget effort is consumed.
  // Sets finished_ when the worker cannot make further progress.
  EpochResult run_epoch(size_t epoch_budget);

  bool finished() const { return finished_; }
  size_t total_effort() const { return total_effort_; }

  // Reset the improvement staleness counter (called at epoch boundary
  // when another worker found an improvement).
  void reset_staleness() { effort_since_improvement_ = 0; }

 private:
  HighsMipSolver &mipsolver_;
  const CscMatrix &csc_;
  SolutionPool &pool_;
  const size_t total_budget_;
  const uint32_t seed_;

  HighsInt ncol_ = 0;
  HighsInt nrow_ = 0;
  double cost_scale_ = 1.0;
  size_t nnz_lp_ = 0;
  size_t stale_budget_ = 0;

  Highs highs_;
  HighsSolution warm_start_;
  int pdlp_stall_count_ = 0;

  double epsilon_;
  double alpha_K_ = 1.0;
  int K_ = 0;

  size_t total_effort_ = 0;
  size_t effort_since_improvement_ = 0;
  bool finished_ = false;

  std::vector<std::vector<double>> cycle_history_;
  std::vector<double> scores_;
  std::vector<double> modified_cost_;
  std::mt19937 rng_;
};
