#include "fj_worker.h"

#include <vector>

#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "solution_pool.h"

FjWorker::FjWorker(HighsMipSolver &mipsolver, SolutionPool &pool,
                   size_t total_budget, uint32_t seed)
    : mipsolver_(mipsolver),
      pool_(pool),
      total_budget_(total_budget),
      seed_(seed) {}

EpochResult FjWorker::run_epoch(size_t epoch_budget) {
  if (finished_) return {};

  // FJ is single-shot: run once, then mark finished.
  finished_ = true;

  auto *mipdata = mipsolver_.mipdata_.get();
  const size_t budget = std::min(epoch_budget, total_budget_);

  std::vector<double> sol;
  double obj = 0.0;
  size_t effort = 0;

  mipdata->feasibilityJumpCapture(sol, obj, effort, budget, nullptr,
                                  static_cast<int>(seed_));

  EpochResult result{};
  result.effort = effort;

  if (!sol.empty()) {
    pool_.try_add(obj, sol);
    result.found_improvement = true;
  }

  return result;
}
