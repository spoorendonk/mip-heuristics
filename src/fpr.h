#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "epoch_runner.h"
#include "fpr_strategies.h"
#include "util/HighsInt.h"

class HighsMipSolver;
struct CscMatrix;
class SolutionPool;

namespace fpr {

void run(HighsMipSolver &mipsolver, size_t max_effort);

// Parallel mode with epoch-gated synchronization: N FprWorkers run in
// parallel, synchronizing at epoch boundaries.  Each epoch increments
// the attempt index (new random init, shuffled variable order).  After
// K stale epochs a worker randomizes its config from the full
// strategy x mode space.
void run_parallel(HighsMipSolver &mipsolver, size_t max_effort);

// Worker that wraps a single fpr_attempt call per epoch.  Satisfies
// the EpochWorker concept from epoch_runner.h.
class FprWorker {
 public:
  FprWorker(HighsMipSolver &mipsolver, const CscMatrix &csc,
            SolutionPool &pool, FprStrategyConfig strat, FrameworkMode mode,
            uint32_t seed);

  EpochResult run_epoch(size_t epoch_budget);

  // FPR can always retry with a new attempt — never "finished".
  bool finished() const { return false; }

  void reset_staleness() { epochs_without_improvement_ = 0; }

 private:
  // Randomize config from the full strategy x mode space.
  void randomize_config();
  // Pre-compute variable order (must be called before parallel epoch).
  void recompute_var_order();

  HighsMipSolver &mipsolver_;
  const CscMatrix &csc_;
  SolutionPool &pool_;

  FprStrategyConfig strat_;
  FrameworkMode mode_;

  int attempt_idx_ = 0;
  int epochs_without_improvement_ = 0;

  std::vector<HighsInt> var_order_;
  std::mt19937 rng_;
};

static_assert(EpochWorker<FprWorker>,
              "FprWorker must satisfy EpochWorker concept");

}  // namespace fpr
