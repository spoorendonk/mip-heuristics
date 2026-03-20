#include "adaptive/portfolio.h"

#include <algorithm>
#include <atomic>
#include <random>
#include <vector>

#include "adaptive/solution_pool.h"
#include "adaptive/thompson_sampler.h"
#include "fpr_core.h"
#include "heuristic_common.h"
#include "local_mip.h"
#include "mip/HighsLpRelaxation.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "parallel/HighsParallel.h"

namespace portfolio {

namespace {

// Arm indices for presolve portfolio (values are arbitrary, used as arm_type)
enum PresolveArm { kArmFPR = 0, kArmLocalMIP = 1, kArmFJ = 2 };

// MIPLIB sweep priors
constexpr double kFjAlpha = 2.0;
constexpr double kFprAlpha = 2.5;
constexpr double kLocalMipAlpha = 3.0;

constexpr int kPoolCapacity = 10;

void seed_pool(SolutionPool& pool, const HighsMipSolver& mipsolver) {
  const auto* model = mipsolver.model_;
  auto* mipdata = mipsolver.mipdata_.get();
  if (mipdata->incumbent.empty()) return;
  const HighsInt ncol = model->num_col_;
  double obj = model->offset_;
  for (HighsInt j = 0; j < ncol; ++j)
    obj += model->col_cost_[j] * mipdata->incumbent[j];
  pool.try_add(obj, mipdata->incumbent);
}

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
                                 const CscMatrix& csc,
                                 const std::vector<double>& incumbent_snapshot,
                                 double deadline) {
  if (mipsolver.mipdata_->terminatorTerminated())
    return {};
  if (mipsolver.timer_.read() >= std::min(mipsolver.options_mip_->time_limit,
                                          deadline))
    return {};
  switch (arm_type) {
    case kArmFPR: {
      const auto* model = mipsolver.model_;
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
          incumbent_snapshot.empty() ? nullptr : incumbent_snapshot.data();
      std::vector<double> cont_fallback(ncol, 0.0);

      FprConfig cfg{};
      cfg.max_attempts = 1;
      cfg.rng_seed_offset = 42;
      cfg.hint = hint;
      cfg.scores = scores.data();
      cfg.cont_fallback = cont_fallback.data();
      cfg.csc = &csc;
      cfg.deadline = deadline;

      return fpr_attempt(mipsolver, cfg, rng, attempt_idx, restart_sol);
    }
    case kArmLocalMIP: {
      const double* init = restart_sol;
      if (!init && !incumbent_snapshot.empty())
        init = incumbent_snapshot.data();
      return local_mip::worker(mipsolver, csc, rng, init, deadline);
    }
    case kArmFJ: {
      auto* mipdata = mipsolver.mipdata_.get();
      const HighsInt ncol = mipsolver.model_->num_col_;
      HeuristicResult result;
      std::vector<double> captured_sol;
      double captured_obj = 0.0;

      // Prefer pool restart over static incumbent snapshot.
      // restart_sol is a raw pointer from the pool, so we must copy into a
      // vector to satisfy the FJ signature (const vector<double>*).
      const std::vector<double>* hint = nullptr;
      std::vector<double> restart_vec;
      if (restart_sol) {
        restart_vec.assign(restart_sol, restart_sol + ncol);
        hint = &restart_vec;
      } else if (!incumbent_snapshot.empty()) {
        hint = &incumbent_snapshot;
      }

      mipdata->feasibilityJumpCapture(captured_sol, captured_obj, hint);
      if (!captured_sol.empty()) {
        result.found_feasible = true;
        result.solution = std::move(captured_sol);
        result.objective = captured_obj;
      }
      result.effort = mipdata->ARindex_.size();  // approximate
      return result;
    }
    default:
      return {};
  }
}

void run_presolve_opportunistic(HighsMipSolver& mipsolver,
                                 const std::vector<int>& enabled_arms,
                                 const std::vector<double>& priors,
                                 const CscMatrix& csc, bool minimize) {
  auto* mipdata = mipsolver.mipdata_.get();
  const int N = highs::parallel::num_threads();
  const int num_arms = static_cast<int>(enabled_arms.size());

  ThompsonSampler bandit(num_arms, priors.data(), true);  // mutex-protected
  SolutionPool pool(kPoolCapacity, minimize);
  seed_pool(pool, mipsolver);

  // Snapshot incumbent once (read-only for all workers)
  std::vector<double> incumbent_snapshot = mipdata->incumbent;

  const size_t nnz = mipdata->ARindex_.size();
  const size_t budget = nnz << 10;
  const size_t stale_budget = nnz << 8;

  const double wall_deadline =
      heuristic_deadline(mipsolver.options_mip_->time_limit,
                         mipsolver.timer_.read());

  std::atomic<size_t> total_effort{0};
  std::atomic<size_t> effort_since_improvement{0};
  std::atomic<bool> stop{false};

  uint32_t base_seed = static_cast<uint32_t>(mipdata->numImprovingSols + 42);

  highs::parallel::for_each(
      0, static_cast<HighsInt>(N),
      [&](HighsInt lo, HighsInt hi) {
        for (HighsInt w = lo; w < hi; ++w) {
          std::mt19937 rng(base_seed + static_cast<uint32_t>(w) * 997);
          int attempt_counter = 0;

          while (!stop.load(std::memory_order_relaxed)) {
            // Worker 0 periodically checks termination (not thread-safe
            // to call from multiple workers)
            if (w == 0 && attempt_counter % 8 == 0) {
              if (mipdata->terminatorTerminated() ||
                  mipsolver.timer_.read() >= wall_deadline)
                stop.store(true, std::memory_order_relaxed);
            }
            if (stop.load(std::memory_order_relaxed)) break;

            int arm = bandit.select(rng);
            auto before = pool.snapshot();

            std::vector<double> restart;
            pool.get_restart(rng, restart);
            const double* restart_ptr =
                restart.empty() ? nullptr : restart.data();

            auto result =
                run_presolve_arm(mipsolver, enabled_arms[arm], rng,
                                 attempt_counter++, restart_ptr, csc,
                                 incumbent_snapshot, wall_deadline);

            if (result.found_feasible)
              pool.try_add(result.objective, result.solution);

            auto after = pool.snapshot();
            int reward = compute_reward(before, after, result, minimize);
            bandit.update(arm, reward);

            if (reward >= 2)
              effort_since_improvement.store(0, std::memory_order_relaxed);
            else
              effort_since_improvement.fetch_add(result.effort,
                                                  std::memory_order_relaxed);

            if (effort_since_improvement.load(std::memory_order_relaxed) >=
                stale_budget)
              stop.store(true, std::memory_order_relaxed);

            size_t new_total =
                total_effort.fetch_add(result.effort) + result.effort;
            if (new_total >= budget)
              stop.store(true, std::memory_order_relaxed);
          }
        }
      },
      1);

  // Flush pool solutions to HiGHS (sequential, use generic H tag since
  // pool mixes arms)
  for (auto& entry : pool.sorted_entries())
    mipdata->trySolution(entry.solution, kSolutionSourceHeuristic);
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
  if (options->mip_heuristic_run_feasibility_jump) {
    enabled_arms.push_back(kArmFJ);
    priors.push_back(kFjAlpha);
  }
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

  // Dispatch to opportunistic mode if requested
  if (options->mip_heuristic_portfolio_opportunistic) {
    run_presolve_opportunistic(mipsolver, enabled_arms, priors, csc, minimize);
    return;
  }

  ThompsonSampler bandit(static_cast<int>(enabled_arms.size()), priors.data(),
                         false);
  SolutionPool pool(kPoolCapacity, minimize);
  seed_pool(pool, mipsolver);

  // Snapshot incumbent before parallel work (read-only for all workers)
  std::vector<double> incumbent_snapshot = mipdata->incumbent;

  // FJ-style budget: nnz << 10 total, nnz << 8 since last improvement
  const size_t nnz = mipdata->ARindex_.size();
  const size_t budget = nnz << 10;
  const size_t stale_budget = nnz << 8;
  size_t total_effort = 0;
  size_t effort_since_improvement = 0;

  const double wall_deadline =
      heuristic_deadline(mipsolver.options_mip_->time_limit,
                         mipsolver.timer_.read());

  // Deterministic per-worker state
  uint32_t base_seed = static_cast<uint32_t>(mipdata->numImprovingSols + 42);
  std::vector<std::mt19937> rngs(N);
  for (int w = 0; w < N; ++w) rngs[w].seed(base_seed + w * 997);
  std::vector<int> attempt_counters(N, 0);

  for (int epoch = 0; total_effort < budget; ++epoch) {
    if (mipdata->terminatorTerminated() ||
        mipsolver.timer_.read() >= wall_deadline)
      break;
    if (effort_since_improvement > stale_budget) break;

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
                                          restart_ptr, csc,
                                          incumbent_snapshot, wall_deadline);
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

      if (reward >= 2)
        effort_since_improvement = 0;
      else
        effort_since_improvement += results[w].effort;

      // Update snapshot for next worker's reward computation
      pool_snap = after_snap;
    }
  }

  // Flush pool solutions to HiGHS (best first)
  for (auto& entry : pool.sorted_entries())
    mipdata->trySolution(entry.solution, kSolutionSourceHeuristic);
}

void run_scylla_parallel(HighsMipSolver& mipsolver) {
  const auto* model = mipsolver.model_;
  auto* mipdata = mipsolver.mipdata_.get();
  const HighsInt ncol = model->num_col_;
  const HighsInt nrow = model->num_row_;
  if (ncol == 0 || nrow == 0) return;

  // Guard: need an optimal LP relaxation
  auto lp_status = mipdata->lp.getStatus();
  if (!HighsLpRelaxation::scaledOptimal(lp_status)) return;

  const bool minimize = (model->sense_ == ObjSense::kMinimize);
  const int N = highs::parallel::num_threads();
  constexpr int kMaxEpochs = 4;

  SolutionPool pool(kPoolCapacity, minimize);
  seed_pool(pool, mipsolver);

  const double deadline =
      heuristic_deadline(mipsolver.options_mip_->time_limit,
                         mipsolver.timer_.read());

  // Snapshot LP solution once (read-only for all workers)
  const auto& lp_sol = mipdata->lp.getLpSolver().getSolution().col_value;
  const auto& integrality = model->integrality_;
  const auto& col_lb = model->col_lower_;
  const auto& col_ub = model->col_upper_;

  // Build CSC once for all workers
  auto csc = build_csc(ncol, nrow, mipdata->ARstart_, mipdata->ARindex_,
                        mipdata->ARvalue_);

  // Pre-allocate per-worker scores to avoid repeated allocation in epoch loop
  std::vector<std::vector<double>> worker_scores(N);
  for (int w = 0; w < N; ++w) worker_scores[w].resize(ncol);

  for (int epoch = 0; epoch < kMaxEpochs; ++epoch) {
    if (mipsolver.timer_.read() >= deadline) break;
    if (mipdata->terminatorTerminated()) break;

    std::vector<HeuristicResult> results(N);
    highs::parallel::for_each(0, static_cast<HighsInt>(N),
        [&](HighsInt lo, HighsInt hi) {
          for (HighsInt w = lo; w < hi; ++w) {
            std::mt19937 rng(42 + epoch * N + static_cast<int>(w));

            // Compute scores: LP fractionality + per-worker noise
            auto& scores = worker_scores[w];
            for (HighsInt j = 0; j < ncol; ++j) {
              if (!is_integer(integrality, j)) {
                scores[j] = -1.0;
              } else {
                double s = std::abs(lp_sol[j] - std::round(lp_sol[j]));
                if (w > 0) {  // perturb for workers 1..N-1
                  double range = col_ub[j] - col_lb[j];
                  double proximity = (range > 0 && range < 1e8)
                      ? std::min(lp_sol[j] - col_lb[j],
                                 col_ub[j] - lp_sol[j]) / range
                      : 0.0;
                  double noise_scale = 0.3 + 0.6 * proximity;
                  s *= 1.0 + std::uniform_real_distribution<>(
                      -noise_scale, noise_scale)(rng);
                }
                scores[j] = s;
              }
            }

            FprConfig cfg{};
            cfg.max_attempts = 1;
            cfg.rng_seed_offset = 42 + epoch * N + static_cast<int>(w);
            cfg.hint = lp_sol.data();
            cfg.scores = scores.data();
            cfg.cont_fallback = lp_sol.data();
            cfg.csc = &csc;
            cfg.deadline = deadline;

            results[w] = fpr_attempt(mipsolver, cfg, rng, 0, nullptr);
          }
        }, 1);

    for (int w = 0; w < N; ++w)
      if (results[w].found_feasible)
        pool.try_add(results[w].objective, results[w].solution);
  }

  // Submit best solutions to solver
  for (auto& entry : pool.sorted_entries())
    mipdata->trySolution(entry.solution, kSolutionSourceScyllaFPR);
}

}  // namespace portfolio
