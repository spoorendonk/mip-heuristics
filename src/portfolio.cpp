#include "portfolio.h"

#include <atomic>
#include <chrono>
#include <random>
#include <vector>

#include "fpr_core.h"
#include "fpr_strategies.h"
#include "heuristic_common.h"
#include "io/HighsIO.h"
#include "local_mip.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include <optional>

#include "parallel/HighsParallel.h"
#include "solution_pool.h"
#include "thompson_sampler.h"

namespace portfolio {

namespace {

// Arm indices for presolve portfolio (used as arm_type).
// FPR arms 0-5 correspond to the paper's 6 LP-free configs.
enum PresolveArm {
  kArmFprDfsBadobjcl = 0,
  kArmFprDfsLocks2,
  kArmFprDiveLocks2,
  kArmFprDfsrepLocks,
  kArmFprDfsrepBadobjcl,
  kArmFprDivepropRandom,
  kArmFprRepairSearchLocks,
  kArmLocalMIP,
  kArmFJ,
};

// Strategy configs for each FPR arm (matching fpr.cpp's kLpFreeConfigs)
struct FprArmConfig {
  int arm_id;
  FprStrategyConfig strat;
  FrameworkMode mode;
};
constexpr FprArmConfig kFprArms[] = {
    {kArmFprDfsBadobjcl, kStratBadobjcl, FrameworkMode::kDfs},
    {kArmFprDfsLocks2, kStratLocks2, FrameworkMode::kDfs},
    {kArmFprDiveLocks2, kStratLocks2, FrameworkMode::kDive},
    {kArmFprDfsrepLocks, kStratLocks, FrameworkMode::kDfsrep},
    {kArmFprDfsrepBadobjcl, kStratBadobjcl, FrameworkMode::kDfsrep},
    {kArmFprDivepropRandom, kStratRandom, FrameworkMode::kDiveprop},
    {kArmFprRepairSearchLocks, kStratLocks, FrameworkMode::kRepairSearch},
};
constexpr int kNumFprArms = static_cast<int>(std::size(kFprArms));

// Returns the FprArmConfig for the given arm_type, or nullptr if not FPR.
// Arm IDs are contiguous 0..kNumFprArms-1, so direct index suffices.
const FprArmConfig *find_fpr_arm(int arm_type) {
  if (arm_type >= 0 && arm_type < kNumFprArms) return &kFprArms[arm_type];
  return nullptr;
}

// Uniform priors (α=1, β=1): let the bandit learn from scratch.
constexpr double kFjAlpha = 1.0;
constexpr double kFprArmAlpha = 1.0;
constexpr double kLocalMipAlpha = 1.0;

bool objective_better(bool minimize, double lhs, double rhs) {
  constexpr double kTol = 1e-9;
  return minimize ? lhs < rhs - kTol : lhs > rhs + kTol;
}

int compute_reward(SolutionPool::Snapshot before, SolutionPool::Snapshot after,
                   const HeuristicResult &result, bool minimize) {
  if (!result.found_feasible) {
    return 0;
  }
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

constexpr const char* kArmNames[] = {
    "FprDfsBadobjcl",      // kArmFprDfsBadobjcl
    "FprDfsLocks2",        // kArmFprDfsLocks2
    "FprDiveLocks2",       // kArmFprDiveLocks2
    "FprDfsrepLocks",      // kArmFprDfsrepLocks
    "FprDfsrepBadobjcl",   // kArmFprDfsrepBadobjcl
    "FprDivepropRandom",   // kArmFprDivepropRandom
    "FprRepairSearchLocks",// kArmFprRepairSearchLocks
    "LocalMIP",            // kArmLocalMIP
    "FJ",                  // kArmFJ
};
static_assert(std::size(kArmNames) == kArmFJ + 1,
              "kArmNames must match PresolveArm enum");

const char* arm_name(int arm_type) {
  if (arm_type >= 0 && arm_type < static_cast<int>(std::size(kArmNames)))
    return kArmNames[arm_type];
  return "Unknown";
}

void log_arm_effort(const HighsLogOptions &log_options, int arm_type,
                    size_t effort, double wall_ms) {
  double effort_per_ms =
      wall_ms > 0.0 ? static_cast<double>(effort) / wall_ms : 0.0;
  highsLogDev(log_options, HighsLogType::kVerbose,
              "[Portfolio] arm=%s effort=%zu wall_ms=%.1f "
              "effort_per_ms=%.0f\n",
              arm_name(arm_type), effort, wall_ms, effort_per_ms);
}

// Pre-computed variable orders for FPR arms (avoids cliquePartition data race).
// Indexed by FprArmConfig index (0..kNumFprArms-1), not by arm_type enum.
using FprVarOrders = std::vector<std::vector<HighsInt>>;

FprVarOrders precompute_fpr_var_orders(const HighsMipSolver &mipsolver);

// Setup state shared between deterministic and opportunistic portfolio modes.
struct PresolveSetup {
  std::vector<int> enabled_arms;
  std::vector<double> priors;
  CscMatrix csc;
  FprVarOrders fpr_var_orders;
  std::vector<double> incumbent_snapshot;
  size_t budget;
  size_t stale_budget;
  bool minimize;
};

std::optional<PresolveSetup> build_presolve_setup(
    HighsMipSolver &mipsolver, size_t max_effort) {
  const auto *model = mipsolver.model_;
  auto *mipdata = mipsolver.mipdata_.get();
  const auto *options = mipsolver.options_mip_;
  const HighsInt ncol = model->num_col_;
  const HighsInt nrow = model->num_row_;
  if (ncol == 0 || nrow == 0) return std::nullopt;

  PresolveSetup s;
  s.minimize = (model->sense_ == ObjSense::kMinimize);
  s.budget = max_effort;
  s.stale_budget = max_effort >> 2;

  if (options->mip_heuristic_run_feasibility_jump) {
    s.enabled_arms.push_back(kArmFJ);
    s.priors.push_back(kFjAlpha);
  }
  if (options->mip_heuristic_run_fpr) {
    for (int i = 0; i < kNumFprArms; ++i) {
      s.enabled_arms.push_back(kFprArms[i].arm_id);
      s.priors.push_back(kFprArmAlpha);
    }
  }
  if (options->mip_heuristic_run_local_mip) {
    s.enabled_arms.push_back(kArmLocalMIP);
    s.priors.push_back(kLocalMipAlpha);
  }
  if (s.enabled_arms.empty()) return std::nullopt;

  s.csc = build_csc(ncol, nrow, mipdata->ARstart_, mipdata->ARindex_,
                     mipdata->ARvalue_);

  if (options->mip_heuristic_run_fpr) {
    s.fpr_var_orders = precompute_fpr_var_orders(mipsolver);
  }

  s.incumbent_snapshot = mipdata->incumbent;
  return s;
}

FprVarOrders precompute_fpr_var_orders(const HighsMipSolver &mipsolver) {
  FprVarOrders orders(kNumFprArms);
  for (int i = 0; i < kNumFprArms; ++i) {
    std::mt19937 rng(42 + static_cast<uint32_t>(i));
    orders[i] = compute_var_order(mipsolver, kFprArms[i].strat.var_strategy,
                                  rng, nullptr);
  }
  return orders;
}

HeuristicResult run_presolve_arm(HighsMipSolver &mipsolver, int arm_type,
                                 std::mt19937 &rng, int attempt_idx,
                                 const double *restart_sol,
                                 const CscMatrix &csc,
                                 const std::vector<double> &incumbent_snapshot,
                                 size_t max_effort,
                                 const FprVarOrders &fpr_var_orders) {
  if (mipsolver.mipdata_->terminatorTerminated()) {
    return {};
  }
  if (mipsolver.timer_.read() >= mipsolver.options_mip_->time_limit) {
    return {};
  }
  // Check if this is an FPR config arm
  const FprArmConfig *fpr_arm = find_fpr_arm(arm_type);
  if (fpr_arm) {
    // Find the index into kFprArms for this arm
    int fpr_idx = static_cast<int>(fpr_arm - kFprArms);

    FprConfig cfg{};
    cfg.max_effort = max_effort;
    cfg.hint =
        incumbent_snapshot.empty() ? nullptr : incumbent_snapshot.data();
    cfg.scores = nullptr;
    cfg.cont_fallback = nullptr;
    cfg.csc = &csc;
    cfg.mode = fpr_arm->mode;
    cfg.strategy = &fpr_arm->strat;
    cfg.lp_ref = nullptr;
    // Use pre-computed var order to avoid cliquePartition data race
    cfg.precomputed_var_order = fpr_var_orders[fpr_idx].data();
    cfg.precomputed_var_order_size =
        static_cast<HighsInt>(fpr_var_orders[fpr_idx].size());
    return fpr_attempt(mipsolver, cfg, rng, attempt_idx, restart_sol);
  }

  switch (arm_type) {
  case kArmLocalMIP: {
    const double *init = restart_sol;
    if (!init && !incumbent_snapshot.empty()) {
      init = incumbent_snapshot.data();
    }
    return local_mip::worker(mipsolver, csc, rng, init, max_effort);
  }
  case kArmFJ: {
    auto *mipdata = mipsolver.mipdata_.get();
    const HighsInt ncol = mipsolver.model_->num_col_;
    HeuristicResult result;
    std::vector<double> captured_sol;
    double captured_obj = 0.0;

    // Prefer pool restart over static incumbent snapshot.
    // restart_sol is a raw pointer from the pool, so we must copy into a
    // vector to satisfy the FJ signature (const vector<double>*).
    const std::vector<double> *hint = nullptr;
    std::vector<double> restart_vec;
    if (restart_sol) {
      restart_vec.assign(restart_sol, restart_sol + ncol);
      hint = &restart_vec;
    } else if (!incumbent_snapshot.empty()) {
      hint = &incumbent_snapshot;
    }

    size_t fj_effort = 0;
    mipdata->feasibilityJumpCapture(captured_sol, captured_obj, fj_effort,
                                    max_effort, hint);
    if (!captured_sol.empty()) {
      result.found_feasible = true;
      result.solution = std::move(captured_sol);
      result.objective = captured_obj;
    }
    result.effort = fj_effort;
    return result;
  }
  default:
    return {};
  }
}

void run_presolve_opportunistic(HighsMipSolver &mipsolver,
                                const PresolveSetup &setup) {
  auto *mipdata = mipsolver.mipdata_.get();
  const HighsLogOptions &log_options =
      mipsolver.options_mip_->log_options;
  const int N = highs::parallel::num_threads();
  const int num_arms = static_cast<int>(setup.enabled_arms.size());

  ThompsonSampler bandit(num_arms, setup.priors.data(), true);
  SolutionPool pool(kPoolCapacity, setup.minimize);
  seed_pool(pool, mipsolver);

  const auto &enabled_arms = setup.enabled_arms;
  const auto &incumbent_snapshot = setup.incumbent_snapshot;
  const auto &csc = setup.csc;
  const auto &fpr_var_orders = setup.fpr_var_orders;
  const bool minimize = setup.minimize;
  const size_t budget = setup.budget;
  const size_t stale_budget = setup.stale_budget;

  const double time_limit = mipsolver.options_mip_->time_limit;

  std::atomic<size_t> total_effort{0};
  std::atomic<size_t> effort_since_improvement{0};
  std::atomic<bool> stop{false};

  uint32_t base_seed =
      static_cast<uint32_t>(mipdata->numImprovingSols + kBaseSeedOffset);

  highs::parallel::for_each(
      0, static_cast<HighsInt>(N),
      [&](HighsInt lo, HighsInt hi) {
        for (HighsInt w = lo; w < hi; ++w) {
          std::mt19937 rng(base_seed + static_cast<uint32_t>(w) * kSeedStride);
          int attempt_counter = 0;

          while (!stop.load(std::memory_order_relaxed)) {
            // Worker 0 periodically checks termination (not thread-safe
            // to call from multiple workers)
            if (w == 0 && attempt_counter % 8 == 0) {
              if (mipdata->terminatorTerminated() ||
                  mipsolver.timer_.read() >= time_limit) {
                stop.store(true, std::memory_order_relaxed);
              }
            }
            if (stop.load(std::memory_order_relaxed)) {
              break;
            }

            int arm = bandit.select_effort_aware(rng);
            auto before = pool.snapshot();

            std::vector<double> restart;
            pool.get_restart(rng, restart);
            const double *restart_ptr =
                restart.empty() ? nullptr : restart.data();

            size_t remaining =
                budget -
                std::min(budget, total_effort.load(std::memory_order_relaxed));
            auto t0 = std::chrono::steady_clock::now();
            auto result = run_presolve_arm(mipsolver, enabled_arms[arm], rng,
                                           attempt_counter++, restart_ptr, csc,
                                           incumbent_snapshot, remaining,
                                           fpr_var_orders);
            auto t1 = std::chrono::steady_clock::now();

            if (result.found_feasible) {
              pool.try_add(result.objective, result.solution);
            }

            auto after = pool.snapshot();
            int reward = compute_reward(before, after, result, minimize);
            bandit.update(arm, reward);
            bandit.record_effort(arm, result.effort);

            if (result.effort > 0) {
              double wall_ms =
                  std::chrono::duration<double, std::milli>(t1 - t0).count();
              log_arm_effort(log_options, enabled_arms[arm], result.effort,
                             wall_ms);
            }

            if (reward >= 2) {
              effort_since_improvement.store(0, std::memory_order_relaxed);
            } else {
              effort_since_improvement.fetch_add(result.effort,
                                                 std::memory_order_relaxed);
            }

            if (effort_since_improvement.load(std::memory_order_relaxed) >=
                stale_budget) {
              stop.store(true, std::memory_order_relaxed);
            }

            size_t new_total =
                total_effort.fetch_add(result.effort) + result.effort;
            if (new_total >= budget) {
              stop.store(true, std::memory_order_relaxed);
            }
          }
        }
      },
      1);

  mipdata->heuristic_effort_used +=
      total_effort.load(std::memory_order_relaxed);

  // Flush pool solutions to HiGHS (sequential, use generic H tag since
  // pool mixes arms)
  for (auto &entry : pool.sorted_entries()) {
    mipdata->trySolution(entry.solution, kSolutionSourceHeuristic);
  }
}

} // namespace

void run_presolve(HighsMipSolver &mipsolver, size_t max_effort) {
  auto setup_opt = build_presolve_setup(mipsolver, max_effort);
  if (!setup_opt) return;
  auto &setup = *setup_opt;

  // Dispatch to opportunistic mode if requested
  if (mipsolver.options_mip_->mip_heuristic_portfolio_opportunistic) {
    run_presolve_opportunistic(mipsolver, setup);
    return;
  }

  auto *mipdata = mipsolver.mipdata_.get();
  const int N = highs::parallel::num_threads();

  const auto &enabled_arms = setup.enabled_arms;
  const auto &incumbent_snapshot = setup.incumbent_snapshot;
  const auto &csc = setup.csc;
  const auto &fpr_var_orders = setup.fpr_var_orders;
  const bool minimize = setup.minimize;

  ThompsonSampler bandit(static_cast<int>(enabled_arms.size()),
                         setup.priors.data(), false);
  SolutionPool pool(kPoolCapacity, minimize);
  seed_pool(pool, mipsolver);

  size_t total_effort = 0;
  size_t effort_since_improvement = 0;

  const double time_limit = mipsolver.options_mip_->time_limit;

  uint32_t base_seed =
      static_cast<uint32_t>(mipdata->numImprovingSols + kBaseSeedOffset);
  std::vector<std::mt19937> rngs(N);
  for (int w = 0; w < N; ++w) {
    rngs[w].seed(base_seed + w * kSeedStride);
  }
  std::vector<int> attempt_counters(N, 0);

  const HighsLogOptions &log_options = mipsolver.options_mip_->log_options;

  // Pre-allocate per-worker vectors outside epoch loop
  std::vector<int> arms(N);
  std::vector<HeuristicResult> results(N);
  std::vector<double> wall_ms_vec(N);
  std::vector<std::vector<double>> restarts(N);

  for (int epoch = 0; total_effort < setup.budget; ++epoch) {
    if (mipdata->terminatorTerminated() ||
        mipsolver.timer_.read() >= time_limit) {
      break;
    }
    if (effort_since_improvement > setup.stale_budget) {
      break;
    }

    // Pre-epoch: snapshot + get restarts (sequential, deterministic)
    auto pool_snap = pool.snapshot();
    for (int w = 0; w < N; ++w) {
      restarts[w].clear();
      pool.get_restart(rngs[w], restarts[w]);
    }

    // Select arms deterministically (sequential, uses effort-aware sampling)
    for (int w = 0; w < N; ++w) {
      arms[w] = bandit.select_effort_aware(rngs[w]);
    }

    highs::parallel::for_each(
        0, static_cast<HighsInt>(N),
        [&](HighsInt lo, HighsInt hi) {
          for (HighsInt w = lo; w < hi; ++w) {
            const double *restart_ptr =
                restarts[w].empty() ? nullptr : restarts[w].data();
            size_t remaining = setup.budget - std::min(setup.budget, total_effort);
            auto t0 = std::chrono::steady_clock::now();
            results[w] = run_presolve_arm(
                mipsolver, enabled_arms[arms[w]], rngs[w], attempt_counters[w],
                restart_ptr, csc, incumbent_snapshot, remaining,
                fpr_var_orders);
            auto t1 = std::chrono::steady_clock::now();
            wall_ms_vec[w] =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
          }
        },
        1);

    // Post-epoch: merge in deterministic worker order
    for (int w = 0; w < N; ++w) {
      if (results[w].found_feasible) {
        pool.try_add(results[w].objective, results[w].solution);
      }

      auto after_snap = pool.snapshot();
      int reward = compute_reward(pool_snap, after_snap, results[w], minimize);
      bandit.update(arms[w], reward);
      bandit.record_effort(arms[w], results[w].effort);
      total_effort += results[w].effort;
      attempt_counters[w]++;

      if (results[w].effort > 0) {
        log_arm_effort(log_options, enabled_arms[arms[w]],
                       results[w].effort, wall_ms_vec[w]);
      }

      if (reward >= 2) {
        effort_since_improvement = 0;
      } else {
        effort_since_improvement += results[w].effort;
      }

      // Update snapshot for next worker's reward computation
      pool_snap = after_snap;
    }
  }

  mipdata->heuristic_effort_used += total_effort;

  // Flush pool solutions to HiGHS (best first)
  for (auto &entry : pool.sorted_entries()) {
    mipdata->trySolution(entry.solution, kSolutionSourceHeuristic);
  }
}

} // namespace portfolio
