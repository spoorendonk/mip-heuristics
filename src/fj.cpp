#include "fj.h"

#include <memory>
#include <vector>

#include "epoch_runner.h"
#include "fj_worker.h"
#include "heuristic_common.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "solution_pool.h"

namespace fj {

bool run(HighsMipSolver &mipsolver, size_t max_effort) {
  auto *mipdata = mipsolver.mipdata_.get();
  std::vector<double> sol;
  double obj = 0.0;
  size_t effort = 0;
  auto status =
      mipdata->feasibilityJumpCapture(sol, obj, effort, max_effort, nullptr);
  if (!sol.empty()) {
    mipdata->trySolution(sol, kSolutionSourceHeuristic);
  }
  mipdata->heuristic_effort_used += effort;
  return status == HighsModelStatus::kInfeasible;
}

bool run_parallel(HighsMipSolver &mipsolver, size_t max_effort) {
  const auto *model = mipsolver.model_;
  auto *mipdata = mipsolver.mipdata_.get();
  if (model->num_col_ == 0 || model->num_row_ == 0) return false;

  const bool minimize = (model->sense_ == ObjSense::kMinimize);
  const int N = highs::parallel::num_threads();

  SolutionPool pool(kPoolCapacity, minimize);
  seed_pool(pool, mipsolver);

  const size_t worker_budget = max_effort / static_cast<size_t>(N);
  constexpr int kEpochsPerWorker = 20;
  const size_t epoch_budget =
      std::max<size_t>(worker_budget / kEpochsPerWorker, 1);

  uint32_t base_seed =
      static_cast<uint32_t>(mipdata->numImprovingSols + kBaseSeedOffset);

  // Restart counter for generating fresh seeds when workers finish.
  uint32_t restart_counter = 0;

  std::vector<std::unique_ptr<FjWorker>> workers;
  workers.reserve(N);
  for (int w = 0; w < N; ++w) {
    uint32_t seed = base_seed + static_cast<uint32_t>(w) * kSeedStride;
    workers.push_back(
        std::make_unique<FjWorker>(mipsolver, pool, worker_budget, seed));
  }

  size_t total_effort = run_epoch_loop(
      mipsolver, workers, max_effort, epoch_budget,
      [&](int w) {
        // Restart finished FjWorker with a new seed.
        ++restart_counter;
        uint32_t seed =
            base_seed + (static_cast<uint32_t>(N) + restart_counter) * kSeedStride;
        workers[w] =
            std::make_unique<FjWorker>(mipsolver, pool, worker_budget, seed);
      },
      max_effort >> 2);

  mipdata->heuristic_effort_used += total_effort;

  for (auto &entry : pool.sorted_entries()) {
    mipdata->trySolution(entry.solution, kSolutionSourceHeuristic);
  }

  return false;
}

}  // namespace fj
