#include "scylla.h"

#include <memory>
#include <vector>

#include "epoch_runner.h"
#include "heuristic_common.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "pump_worker.h"
#include "solution_pool.h"

namespace scylla {

namespace {

// Convenience wrapper for the sequential (single-worker) code path.
// Runs a single pump chain to completion within the given budget.
size_t pump_worker(HighsMipSolver &mipsolver, const CscMatrix &csc,
                   SolutionPool &pool, size_t worker_budget, uint32_t seed) {
  PumpWorker worker(mipsolver, csc, pool, worker_budget, seed);
  auto result = worker.run_epoch(worker_budget);
  return result.effort;
}

} // namespace

void run(HighsMipSolver &mipsolver, size_t max_effort) {
  const auto *model = mipsolver.model_;
  auto *mipdata = mipsolver.mipdata_.get();
  const HighsInt ncol = model->num_col_;
  const HighsInt nrow = model->num_row_;
  if (ncol == 0 || nrow == 0) return;

  const bool minimize = (model->sense_ == ObjSense::kMinimize);

  auto csc = build_csc(ncol, nrow, mipdata->ARstart_, mipdata->ARindex_,
                       mipdata->ARvalue_);

  SolutionPool pool(kPoolCapacity, minimize);
  seed_pool(pool, mipsolver);

  size_t effort = pump_worker(mipsolver, csc, pool, max_effort, 42);

  mipdata->heuristic_effort_used += effort;

  for (auto &entry : pool.sorted_entries()) {
    mipdata->trySolution(entry.solution, kSolutionSourceScylla);
  }
}

void run_parallel(HighsMipSolver &mipsolver, size_t max_effort) {
  const auto *model = mipsolver.model_;
  auto *mipdata = mipsolver.mipdata_.get();
  const HighsInt ncol = model->num_col_;
  const HighsInt nrow = model->num_row_;
  if (ncol == 0 || nrow == 0) return;

  const bool minimize = (model->sense_ == ObjSense::kMinimize);
  const int N = highs::parallel::num_threads();

  auto csc = build_csc(ncol, nrow, mipdata->ARstart_, mipdata->ARindex_,
                       mipdata->ARvalue_);

  SolutionPool pool(kPoolCapacity, minimize);
  seed_pool(pool, mipsolver);

  // Per-worker total budget and epoch budget.
  // Each epoch gives every worker a small slice; ~10 epochs per worker
  // provides meaningful synchronization without excessive overhead.
  const size_t worker_budget = max_effort / static_cast<size_t>(N);
  constexpr int kEpochsPerWorker = 10;
  const size_t epoch_budget =
      std::max<size_t>(worker_budget / kEpochsPerWorker, 1);

  uint32_t base_seed =
      static_cast<uint32_t>(mipdata->numImprovingSols + kBaseSeedOffset);

  // Create per-worker PumpWorker instances (sequential — deterministic
  // initialization order with deterministic seeds).
  std::vector<std::unique_ptr<PumpWorker>> workers;
  workers.reserve(N);
  for (int w = 0; w < N; ++w) {
    uint32_t seed = base_seed + static_cast<uint32_t>(w) * kSeedStride;
    workers.push_back(
        std::make_unique<PumpWorker>(mipsolver, csc, pool, worker_budget, seed));
  }

  size_t total_effort = run_epoch_loop(
      mipsolver, workers, max_effort, epoch_budget,
      [](int) { /* PumpWorkers are not restartable */ },
      max_effort >> 2);

  mipdata->heuristic_effort_used += total_effort;

  for (auto &entry : pool.sorted_entries()) {
    mipdata->trySolution(entry.solution, kSolutionSourceScylla);
  }
}

} // namespace scylla
