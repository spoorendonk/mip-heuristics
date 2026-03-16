#include "adaptive/portfolio.h"

#include <cmath>
#include <random>
#include <vector>

#include "adaptive/solution_pool.h"
#include "adaptive/thompson_sampler.h"
#include "fpr.h"
#include "fpr_core.h"
#include "heuristic_common.h"
#include "local_mip.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "parallel/HighsParallel.h"
#include "scylla_fpr.h"

namespace portfolio {

namespace {

// Arm indices for presolve portfolio
enum PresolveArm { kArmFPR = 0, kArmLocalMIP = 1 };

// Arm indices for LP-based portfolio
enum LpArm { kArmScyllaFPR = 0 };

// MIPLIB sweep priors
constexpr double kFprAlpha = 2.5;
constexpr double kLocalMipAlpha = 3.0;
constexpr double kScyllaFprAlpha = 2.0;

constexpr double kBaseEpochEffort = 64.0;
constexpr double kMinEpochScale = 0.25;
constexpr double kMaxEpochScale = 4.0;
constexpr double kGrowFactor = 1.2;
constexpr double kShrinkFactor = 0.75;
constexpr int kStaleThreshold = 2;
constexpr int kPoolCapacity = 10;

bool objective_better(bool minimize, double lhs, double rhs) {
  constexpr double kTol = 1e-9;
  return minimize ? lhs < rhs - kTol : lhs > rhs + kTol;
}

int compute_reward(SolutionPool::Snapshot before, SolutionPool::Snapshot after,
                   const HeuristicResult& result, bool minimize) {
  if (!result.found_feasible) return 0;
  if (!before.has_solution) {
    // First feasible ever
    return after.has_solution &&
                   !objective_better(minimize, after.best_objective,
                                     result.objective)
               ? 2
               : 1;
  }
  // Had solution before — check if we improved global best
  bool improved =
      after.has_solution &&
      objective_better(minimize, after.best_objective, before.best_objective) &&
      !objective_better(minimize, after.best_objective, result.objective);
  return improved ? 3 : 1;
}

HeuristicResult run_presolve_arm(HighsMipSolver& mipsolver, int arm_type,
                                 std::mt19937& rng, int attempt_idx,
                                 const double* restart_sol,
                                 const CscMatrix& csc) {
  switch (arm_type) {
    case kArmFPR: {
      const auto* model = mipsolver.model_;
      auto* mipdata = mipsolver.mipdata_.get();
      const auto& integrality = model->integrality_;
      const auto& col_cost = model->col_cost_;
      const HighsInt ncol = model->num_col_;

      // Ranking: degree * (1 + |cost|)
      std::vector<double> scores(ncol);
      for (HighsInt j = 0; j < ncol; ++j) {
        if (!is_integer(integrality, j))
          scores[j] = -1.0;
        else {
          double degree = static_cast<double>(csc.col_start[j + 1] -
                                              csc.col_start[j]);
          scores[j] = degree * (1.0 + std::abs(col_cost[j]));
        }
      }

      const double* hint =
          mipdata->incumbent.empty() ? nullptr : mipdata->incumbent.data();
      std::vector<double> cont_fallback(ncol, 0.0);

      FprConfig cfg{};
      cfg.max_attempts = 1;
      cfg.rng_seed_offset = 42;
      cfg.hint = hint;
      cfg.scores = scores.data();
      cfg.cont_fallback = cont_fallback.data();
      cfg.csc = &csc;

      return fpr_attempt(mipsolver, cfg, rng, attempt_idx, restart_sol);
    }
    case kArmLocalMIP: {
      return local_mip::worker(mipsolver, csc, rng, restart_sol);
    }
    default:
      return {};
  }
}

HeuristicResult run_lp_arm(HighsMipSolver& mipsolver, int arm_type,
                           std::mt19937& rng) {
  switch (arm_type) {
    case kArmScyllaFPR:
      return scylla_fpr::attempt(mipsolver, rng);
    default:
      return {};
  }
}

}  // namespace

void run_presolve(HighsMipSolver& mipsolver) {
  const auto* model = mipsolver.model_;
  auto* mipdata = mipsolver.mipdata_.get();
  const auto* options = mipsolver.options_mip_;
  const HighsInt ncol = model->num_col_;
  const HighsInt nrow = model->num_row_;
  if (ncol == 0 || nrow == 0) return;

  const bool minimize = (model->sense_ == ObjSense::kMinimize);
  const int N = highs::parallel::num_threads();

  // Determine enabled arms
  std::vector<int> enabled_arms;
  std::vector<double> priors;
  if (options->mip_heuristic_run_fpr) {
    enabled_arms.push_back(kArmFPR);
    priors.push_back(kFprAlpha);
  }
  if (options->mip_heuristic_run_local_mip) {
    enabled_arms.push_back(kArmLocalMIP);
    priors.push_back(kLocalMipAlpha);
  }
  if (enabled_arms.empty()) return;

  // Build CSC once for all workers
  auto csc = build_csc(ncol, nrow, mipdata->ARstart_, mipdata->ARindex_,
                        mipdata->ARvalue_);

  ThompsonSampler bandit(static_cast<int>(enabled_arms.size()), priors.data(),
                         false);
  SolutionPool pool(kPoolCapacity, minimize);

  // Seed incumbent into pool if available
  if (!mipdata->incumbent.empty()) {
    double obj = 0.0;
    for (HighsInt j = 0; j < ncol; ++j)
      obj += model->col_cost_[j] * mipdata->incumbent[j];
    pool.try_add(obj, mipdata->incumbent);
  }

  const size_t nnz = mipdata->ARindex_.size();
  const size_t budget = nnz * 1024;
  size_t total_effort = 0;

  // Deterministic per-worker state
  uint32_t base_seed = static_cast<uint32_t>(mipdata->numImprovingSols + 42);
  std::vector<std::mt19937> rngs(N);
  for (int w = 0; w < N; ++w) rngs[w].seed(base_seed + w * 997);
  std::vector<double> epoch_efforts(N, kBaseEpochEffort);
  std::vector<int> stale(N, 0);
  std::vector<int> attempt_counters(N, 0);

  for (int epoch = 0; total_effort < budget; ++epoch) {
    if (mipdata->terminatorTerminated()) break;

    // Pre-epoch: snapshot + get restarts (sequential, deterministic)
    auto pool_snap = pool.snapshot();
    std::vector<std::vector<double>> restarts(N);
    for (int w = 0; w < N; ++w) pool.get_restart(rngs[w], restarts[w]);

    // Parallel: each worker selects arm and runs one attempt
    std::vector<HeuristicResult> results(N);
    std::vector<int> arms(N);

    highs::parallel::for_each(
        0, static_cast<HighsInt>(N),
        [&](HighsInt lo, HighsInt hi) {
          for (HighsInt w = lo; w < hi; ++w) {
            arms[w] = bandit.select(rngs[w]);
            const double* restart_ptr =
                restarts[w].empty() ? nullptr : restarts[w].data();
            results[w] = run_presolve_arm(mipsolver, enabled_arms[arms[w]],
                                          rngs[w], attempt_counters[w],
                                          restart_ptr, csc);
          }
        },
        1);

    // Post-epoch: merge in deterministic worker order
    for (int w = 0; w < N; ++w) {
      if (results[w].found_feasible)
        pool.try_add(results[w].objective, results[w].solution);

      auto after_snap = pool.snapshot();
      int reward = compute_reward(pool_snap, after_snap, results[w], minimize);
      bandit.update(arms[w], reward);
      total_effort += results[w].effort;
      attempt_counters[w]++;

      if (reward >= 2) {
        stale[w] = 0;
        epoch_efforts[w] = std::min(kBaseEpochEffort * kMaxEpochScale,
                                    epoch_efforts[w] * kGrowFactor);
      } else if (++stale[w] >= kStaleThreshold) {
        epoch_efforts[w] = std::max(kBaseEpochEffort * kMinEpochScale,
                                    epoch_efforts[w] * kShrinkFactor);
        stale[w] = 0;
      }

      // Update snapshot for next worker's reward computation
      pool_snap = after_snap;
    }
  }

  // Flush pool solutions to HiGHS (best first)
  for (auto& entry : pool.sorted_entries())
    mipdata->trySolution(entry.solution, kSolutionSourceHeuristic);
}

void run_lp_based(HighsMipSolver& mipsolver) {
  const auto* model = mipsolver.model_;
  auto* mipdata = mipsolver.mipdata_.get();
  const auto* options = mipsolver.options_mip_;
  const HighsInt ncol = model->num_col_;
  if (ncol == 0) return;

  const bool minimize = (model->sense_ == ObjSense::kMinimize);
  const int N = highs::parallel::num_threads();

  std::vector<int> enabled_arms;
  std::vector<double> priors;
  if (options->mip_heuristic_run_scylla_fpr) {
    enabled_arms.push_back(kArmScyllaFPR);
    priors.push_back(kScyllaFprAlpha);
  }
  if (enabled_arms.empty()) return;

  ThompsonSampler bandit(static_cast<int>(enabled_arms.size()), priors.data(),
                         false);
  SolutionPool pool(kPoolCapacity, minimize);

  if (!mipdata->incumbent.empty()) {
    double obj = 0.0;
    for (HighsInt j = 0; j < ncol; ++j)
      obj += model->col_cost_[j] * mipdata->incumbent[j];
    pool.try_add(obj, mipdata->incumbent);
  }

  const size_t nnz = mipdata->ARindex_.size();
  const size_t budget = nnz * 256;  // Smaller budget for LP-based (per dive)
  size_t total_effort = 0;

  uint32_t base_seed = static_cast<uint32_t>(mipdata->numImprovingSols + 137);
  std::vector<std::mt19937> rngs(N);
  for (int w = 0; w < N; ++w) rngs[w].seed(base_seed + w * 997);
  std::vector<int> stale(N, 0);

  for (int epoch = 0; total_effort < budget; ++epoch) {
    if (mipdata->terminatorTerminated()) break;

    auto pool_snap = pool.snapshot();

    std::vector<HeuristicResult> results(N);
    std::vector<int> arms(N);

    highs::parallel::for_each(
        0, static_cast<HighsInt>(N),
        [&](HighsInt lo, HighsInt hi) {
          for (HighsInt w = lo; w < hi; ++w) {
            arms[w] = bandit.select(rngs[w]);
            results[w] =
                run_lp_arm(mipsolver, enabled_arms[arms[w]], rngs[w]);
          }
        },
        1);

    for (int w = 0; w < N; ++w) {
      if (results[w].found_feasible)
        pool.try_add(results[w].objective, results[w].solution);

      auto after_snap = pool.snapshot();
      int reward = compute_reward(pool_snap, after_snap, results[w], minimize);
      bandit.update(arms[w], reward);
      total_effort += results[w].effort;

      pool_snap = after_snap;
    }
  }

  for (auto& entry : pool.sorted_entries())
    mipdata->trySolution(entry.solution, kSolutionSourceHeuristic);
}

}  // namespace portfolio
